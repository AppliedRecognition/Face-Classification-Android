#pragma once

#include <dlib/dnn/core.h>
#include <dlib/dnn/layers.h>
#include <dlib/dnn/utilities.h>
#include "dnn_bias_mode.hpp"
#include "conv.hpp"
#include "qmat.hpp"
#include "bfloat16.hpp"
#include "tensor_conv.hpp"
#include "library_init.hpp"
#include <optional>


namespace dlibx {

    // older versions of dlib did not have bias_is_disabled() member method
    template <typename CON, typename = void>
    struct has_bias_is_disabled : std::false_type {};

    template <long nf, long nr, long nc, int sy, int sx, int py, int px>
    struct has_bias_is_disabled<dlib::con_<nf,nr,nc,sy,sx,py,px>, std::void_t<decltype(std::declval<dlib::con_<nf,nr,nc,sy,sx,py,px> >().bias_is_disabled())> > : std::true_type {};

    template <long nf, long nr, long nc, int sy, int sx, int py, int px>
    inline std::enable_if_t<has_bias_is_disabled<dlib::con_<nf,nr,nc,sy,sx,py,px> >::value, bool>
    bias_is_disabled(const dlib::con_<nf,nr,nc,sy,sx,py,px>& c) {
        return c.bias_is_disabled();
    }

    template <long nf, long nr, long nc, int sy, int sx, int py, int px>
    constexpr std::enable_if_t<!has_bias_is_disabled<dlib::con_<nf,nr,nc,sy,sx,py,px> >::value, bool>
    bias_is_disabled(const dlib::con_<nf,nr,nc,sy,sx,py,px>&) {
        return false;
    }


    /** \brief Low memory version of dlib::con_<...>.
     *
     * This object does cpu convolution using significanly less memory
     * than the standard version.
     */
    template <
        long _num_filters,
        long _nr,
        long _nc,
        int _stride_y,
        int _stride_x,
        int _padding_y = _stride_y!=1 ? 0 : int(_nr/2),
        int _padding_x = _stride_x!=1 ? 0 : int(_nc/2),
        int _dilate_y = 1,
        int _dilate_x = 1
        >
    class lm_con_ {
        static_assert(_num_filters > 0, "The number of filters must be > 0");
        static_assert(_nr >= 0, "The number of rows in a filter must be >= 0");
        static_assert(_nc >= 0, "The number of columns in a filter must be >= 0");
        static_assert(_stride_y > 0, "The filter stride must be > 0");
        static_assert(_stride_x > 0, "The filter stride must be > 0");
        static_assert(_dilate_y > 0, "The filter dilation must be > 0");
        static_assert(_dilate_x > 0, "The filter dilation must be > 0");

        static constexpr auto _window_nr = 1 + (_nr-1) * _dilate_y;
        static constexpr auto _window_nc = 1 + (_nc-1) * _dilate_x;

        static_assert(_nr==0 || (0 <= _padding_y && _padding_y < _window_nr),
                      "The padding must be smaller than the filter size.");
        static_assert(_nc==0 || (0 <= _padding_x && _padding_x < _window_nc),
                      "The padding must be smaller than the filter size.");
        static_assert(_nr!=0 || 0 == _padding_y,
                      "If _nr==0 then the padding must be set to 0 as well.");
        static_assert(_nc!=0 || 0 == _padding_x,
                      "If _nr==0 then the padding must be set to 0 as well.");

        static constexpr auto dilate = _dilate_y > 1 || _dilate_x > 1;
        static constexpr auto version_float4 = dilate ? "con_4d" : "con_4";
        static constexpr auto version_float5 = dilate ? "con_5d" : "con_5";
        static constexpr auto version_quant = dilate ? "qcon_2" : "qcon_1";

    public:
        lm_con_(dlib::num_con_outputs o = _num_filters)
            : learning_rate_multiplier(1),
              weight_decay_multiplier(1),
              bias_learning_rate_multiplier(1),
              bias_weight_decay_multiplier(0),
              use_bias(true),
              num_filters_(long(o.num_outputs)) {
            DLIB_CASSERT(num_filters_ > 0);
            library_init();
        }

        template <long _nf>
        lm_con_(const dlib::con_<_nf,_nr,_nc,_stride_y,_stride_x,_padding_y,_padding_x>& other)
            : params(std::make_shared<dlib::resizable_tensor>(other.get_layer_params())),
              learning_rate_multiplier(other.get_learning_rate_multiplier()),
              weight_decay_multiplier(other.get_weight_decay_multiplier()),
              bias_learning_rate_multiplier(other.get_bias_learning_rate_multiplier()),
              bias_weight_decay_multiplier(other.get_bias_weight_decay_multiplier()),
              use_bias(!dlibx::bias_is_disabled(other)),
              num_filters_(other.num_filters()) {
            library_init();
            DLIB_CASSERT(!dilate);
            DLIB_CASSERT(other.padding_y() == _padding_y);
            DLIB_CASSERT(other.padding_x() == _padding_x);
            DLIB_CASSERT(num_filters_ > 0);
            if (const auto size = params->size()) {
                const auto nf = std::size_t(num_filters_);
                long num_inputs;
                if (use_bias) {
                    num_inputs = long(size/nf)-1;
                    DLIB_CASSERT(size == std::size_t(num_inputs+1)*nf);
                    biases = { 1, num_filters_ };
                }
                else { // no bias
                    num_inputs = long(size/nf);
                    DLIB_CASSERT(size == std::size_t(num_inputs)*nf);
                    biases = {};
                }
                DLIB_CASSERT(num_inputs > 0);
                const auto filt_nr = other.nr();
                const auto filt_nc = other.nc();
                const auto k = num_inputs / filt_nr / filt_nc;
                DLIB_CASSERT(num_inputs == k*filt_nr*filt_nc);
                filters = { num_filters_, k, filt_nr, filt_nc };
            }
        }

        // returns true if bias was not already enabled
        bool add_biases() {
            if (use_bias)
                return false;
            else if (qfilt) {
                DLIB_CASSERT(!params || params->size() == 0);
                DLIB_CASSERT(filters.size() == 0);
                auto new_params =
                    std::make_shared<dlib::resizable_tensor>(qfilt->nr());
                biases = dlib::alias_tensor(1,qfilt->nr());
                biases(*new_params,0) = 0;
                params = move(new_params);
                conv.reset();
            }
            else if (params && params->size() > 0) {
                DLIB_CASSERT(filters.size() == params->size(),
                             "Inconsistent filter size in condw.");
                const auto num_filters = filters.num_samples();
                DLIB_CASSERT(0 < num_filters && num_filters == num_filters_,
                             "Inconsistent number of filters in condw.");
                auto new_params =
                    std::make_shared<dlib::resizable_tensor>(
                        static_cast<long long>(params->size()) + num_filters);
                const dlib::tensor& old_params = *params;
                memcpy(new_params->host_write_only(), old_params.host(),
                       params->size() * sizeof(float));
                biases = dlib::alias_tensor(1,num_filters);
                biases(*new_params,filters.size()) = 0;
                params = move(new_params);
                conv.setup(_nr, _nc,
                           _dilate_y, _dilate_x,
                           _stride_y, _stride_x,
                           _padding_y, _padding_x,
                           filters(*params,0));
            }
            // else not setup yet so nothing to do
            return use_bias = true;
        }

        void enable_bias() { add_biases(); }
        void disable_bias() { // only if not already setup
            DLIB_CASSERT(biases.size() == 0);
            use_bias = false;
        }
        inline bool bias_is_disabled() const { return !use_bias; }
        inline auto get_bias_mode() const {
            return use_bias ? HAS_BIAS : NO_BIAS;
        }

        long num_filters() const { return num_filters_; }
        long nr() const { return _nr == 0 ? filters.nr() : _nr; }
        long nc() const { return _nc == 0 ? filters.nc() : _nc; }
        auto window_nr() const { return 1 + (nr()-1) * _dilate_y; }
        auto window_nc() const { return 1 + (nc()-1) * _dilate_x; }
        constexpr auto dilate_y() const { return _dilate_y; }
        constexpr auto dilate_x() const { return _dilate_x; }
        constexpr auto stride_y() const { return _stride_y; }
        constexpr auto stride_x() const { return _stride_x; }
        constexpr auto padding_y() const { return _padding_y; }
        constexpr auto padding_x() const { return _padding_x; }

        void set_num_filters(long num) {
            DLIB_CASSERT(num > 0);
            if (num != num_filters_) {
                DLIB_CASSERT(!params || params->size() == 0,
                             "You can't change the number of filters in con_ if the parameter tensor has already been allocated.");
                num_filters_ = num;
            }
        }

        double get_learning_rate_multiplier() const {
            return learning_rate_multiplier;
        }
        double get_weight_decay_multiplier() const {
            return weight_decay_multiplier;
        }
        void set_learning_rate_multiplier(double val) {
            learning_rate_multiplier = val;
        }
        void set_weight_decay_multiplier(double val) {
            weight_decay_multiplier = val;
        }

        double get_bias_learning_rate_multiplier() const {
            return bias_learning_rate_multiplier;
        }
        double get_bias_weight_decay_multiplier() const {
            return bias_weight_decay_multiplier;
        }
        void set_bias_learning_rate_multiplier(double val) {
            bias_learning_rate_multiplier = val;
        }
        void set_bias_weight_decay_multiplier(double val) {
            bias_weight_decay_multiplier = val;
        }

        inline auto map_input_to_output(dlib::dpoint p) const {
            p.x() = (p.x()+(padding_x()-window_nc()/2))/stride_x();
            p.y() = (p.y()+(padding_y()-window_nr()/2))/stride_y();
            return p;
        }
        inline auto map_output_to_input(dlib::dpoint p) const {
            p.x() = p.x()*double(stride_x()) + double(window_nc()/2 - padding_x());
            p.y() = p.y()*double(stride_y()) + double(window_nr()/2 - padding_y());
            return p;
        }

        template <typename SUBNET>
        void setup(const SUBNET& sub) {
            DLIB_CASSERT(!qfilt);
            const auto& input = sub.get_output();

            // note: the following doesn't work if dilate > 1 and dim == 0
            const auto filt_nr = _nr!=0 ? _nr : long(input.nr());
            const auto filt_nc = _nc!=0 ? _nc : long(input.nc());

            auto num_inputs = filt_nr*filt_nc*long(input.k());
            auto num_outputs = num_filters_;
            // allocate params for the filters and bias values
            auto p = std::make_shared<dlib::resizable_tensor>(
                num_inputs*num_filters_ + (use_bias ? num_filters_ : 0));

            dlib::rand rnd(std::rand());
            randomize_parameters(
                *p, std::size_t(num_inputs+num_outputs), rnd);

            filters = { num_filters_, input.k(), filt_nr, filt_nc };

            if (use_bias) {
                // set the initial bias values to zero
                biases = { 1, num_filters_ };
                biases(*p,filters.size()) = 0;
            }
            else biases = {};

            params = move(p);
            conv.setup(_nr, _nc,
                       _dilate_y, _dilate_x,
                       _stride_y, _stride_x,
                       _padding_y, _padding_x,
                       filters(*params,0));
        }

        template <typename SUBNET>
        void forward(const SUBNET& sub, dlib::resizable_tensor& output) {
            DLIB_CASSERT(params, "lmcon layer is not setup");
            auto&& data = sub.get_output();

            if (!qfilt) {
                // floating point
                if (!conv)
                    conv.setup(_nr, _nc,
                               _dilate_y, _dilate_x,
                               _stride_y, _stride_x,
                               _padding_y, _padding_x,
                               filters(*params,0));
                conv(data, output);
            }

            else if (_nr == 1 && _nc == 1 &&
                     _stride_y == 1 && _stride_x == 1 &&
                     _padding_y == 0 && _padding_x == 0)
                qfilt->conv1x1(data, output);

            else if (0 < _padding_y || 0 < _padding_x) {
                auto padded = apply_padding(data, _padding_y, _padding_x);
                qfilt->conv(*padded, output,
                            _nr, _nc, _dilate_y, _dilate_x,
                            _stride_y, _stride_x);
            }

            else  // no padding
                qfilt->conv(data, output,
                            _nr, _nc, _dilate_y, _dilate_x,
                            _stride_y, _stride_x);

            if (use_bias)
                dlib::tt::add(1,output,1,biases(*params,filters.size()));
        }

        template <typename SUBNET>
        void backward(const dlib::tensor& input,
                      SUBNET& sub, dlib::tensor& params_grad) {
            DLIB_CASSERT(params, "lmcon layer is not setup");
            DLIB_CASSERT(0 < filters.nr() && 0 < filters.nc(),
                         "lmcon layer is not setup");
            DLIB_CASSERT(!qfilt, "cannot train quantized lmcon layer");
            DLIB_CASSERT(!dilate, "training with dilation not supported");
            auto filt = filters(*params,0);
            auto&& data = sub.get_output();
            tconv.setup(data, filt,
                        _stride_y, _stride_x,
                        _padding_y, _padding_x);

            if (learning_rate_multiplier <= 0)
                tconv.backward_conv(
                    filt, input, sub.get_gradient_input());
            else {
                auto pg = filters(params_grad,0);
                if (use_bias) {
                    auto bg = biases(params_grad, filters.size());
                    tconv.backward_conv(
                        filt, input, sub.get_gradient_input(),
                        &data, &pg, &bg);
                }
                else
                    tconv.backward_conv(
                        filt, input, sub.get_gradient_input(), &data, &pg);
            }
        }

        inline auto get_num_params() const {
            return (qfilt ? qfilt->size() : 0) + (params ? params->size() : 0);
        }
        const dlib::tensor& get_layer_params() const {
            return params ? *params : empty_tensor;
        }
        dlib::tensor& get_layer_params() {
            if (!params)
                params = std::make_shared<dlib::resizable_tensor>();
            else if (params.use_count() > 1) { // make copy
                params = std::make_shared<dlib::resizable_tensor>(*params);
                conv.reset();
            }
            return const_cast<dlib::resizable_tensor&>(*params);
        }
        inline auto get_shared_params() const { return params; }
        inline auto get_shared_qfilt() const { return qfilt; }

        friend auto serialize_format(const lm_con_& item) {
            if (item.qfilt)
                return item.qfilt->empty() ?
                    pf::native : quantize(item.qfilt->serialize_bits());
            if (!item.params || item.params->size() <= 0)
                return pf::native;
            return is_bfloat16(*item.params) ? pf::bfloat16 : pf::float32;
        }

        friend void serialize(const lm_con_& item, std::ostream& out) {
            using error = dlib::serialization_error;
            const auto format = get_parameter_format(out);
            switch (format) {
            case pf::native:
                if (item.qfilt) {
                    DLIB_CASSERT(item.filters.size() == 0);
                    if (item.use_bias && item.params)
                        item.serialize_qfilt(
                            out, *item.qfilt, item.params.get());
                    else
                        item.serialize_qfilt(out, *item.qfilt);
                }
                else
                    item.serialize_float(
                        out, is_bfloat16(item.get_layer_params()));
                break;

            case pf::float32:
                if (item.qfilt)
                    throw error("Conversion from quantization to floating point not supported in lmcon layer.");
                else
                    item.serialize_float(out, false);
                break;

            case pf::bfloat16:
                if (item.qfilt)
                    throw error("Conversion from quantization to floating point not supported in lmcon layer.");
                else
                    item.serialize_float(out, true);
                break;

            default:
                if (const auto bits = bits_per_element(format)) {
                    if (item.qfilt) {
                        DLIB_CASSERT(item.filters.size() == 0);
                        if (item.use_bias && item.params)
                            item.serialize_qfilt(
                                out, *item.qfilt, item.params.get());
                        else
                            item.serialize_qfilt(out, *item.qfilt);
                    }
                    else {
                        // quantize from float
                        DLIB_CASSERT(item.params);
                        // use 16-bit regardless of bits
                        // it'll deserialize to 8-bit if bits <= 8
                        qmat16 qm;
                        qm.assign_lhs(mat(item.filters(*item.params,0)), bits);
                        if (item.use_bias) {
                            DLIB_CASSERT(0 < item.biases.size());
                            auto b =
                                item.biases(*item.params, item.filters.size());
                            const dlib::tensor& t = b;
                            item.serialize_qfilt(out, qm, &t);
                        }
                        else
                            item.serialize_qfilt(out, qm);
                    }
                }
                else
                    throw error("Invalid serialization format.");
            }
        }

        friend void deserialize(lm_con_& item, std::istream& in) {
            std::string version;
            dlib::deserialize(version, in);
            if (version == version_float4)
                item.deserialize_float(in, false);
            else if (version == version_float5)
                item.deserialize_float(in, true);
            else if (version == version_quant)
                item.deserialize_quant(in);
            else
                throw dlib::serialization_error("Unexpected version '"+version+"' found while deserializing lm_con_.");
        }

        friend auto& operator<<(std::ostream& out, const lm_con_& item) {
            out << "con\t ("
                << "num_filters="<<item.num_filters_
                << ", nr="<<item.nr()
                << ", nc="<<item.nc();
            if (dilate) {
                out << ", dilate_y="<<_dilate_y
                    << ", dilate_x="<<_dilate_x;
            }
            out << ", stride_y="<<_stride_y
                << ", stride_x="<<_stride_x
                << ", padding_y="<<_padding_y
                << ", padding_x="<<_padding_x
                << ")";
            out << " learning_rate_mult="<<item.learning_rate_multiplier;
            out << " weight_decay_mult="<<item.weight_decay_multiplier;
            if (item.use_bias) {
                out << " bias_learning_rate_mult="<<item.bias_learning_rate_multiplier;
                out << " bias_weight_decay_mult="<<item.bias_weight_decay_multiplier;
            }
            else
                out << " use_bias=false";
            return out;
        }

        friend void to_xml(const lm_con_& item, std::ostream& out) {
            out << "<con"
                << " num_filters='"<<item.num_filters_<<"'"
                << " nr='"<<item.nr()<<"'"
                << " nc='"<<item.nc()<<"'";
            if (dilate) {
                out << " dilate_y='"<<_dilate_y<<"'"
                    << " dilate_x='"<<_dilate_x<<"'";
            }
            out << " stride_y='"<<_stride_y<<"'"
                << " stride_x='"<<_stride_x<<"'"
                << " padding_y='"<<_padding_y<<"'"
                << " padding_x='"<<_padding_x<<"'"
                << " learning_rate_mult='"<<item.learning_rate_multiplier<<"'"
                << " weight_decay_mult='"<<item.weight_decay_multiplier<<"'"
                << " bias_learning_rate_mult='"<<item.bias_learning_rate_multiplier<<"'"
                << " bias_weight_decay_mult='"<<item.bias_weight_decay_multiplier<<"'"
                << " use_bias='"<<(item.use_bias?"true":"false")<<"'>\n";
            out << mat(item.get_layer_params());
            out << "</con>";
        }

    private:
        std::shared_ptr<const dlib::resizable_tensor> params;
        dlib::alias_tensor filters, biases;

        std::shared_ptr<const qmat> qfilt;

        double learning_rate_multiplier;
        double weight_decay_multiplier;
        double bias_learning_rate_multiplier;
        double bias_weight_decay_multiplier;
        bool use_bias;

        long num_filters_;

        forward_conv conv;
        tensor_conv tconv;

        void serialize_qfilt(std::ostream& out, const qmat& qm,
                             const dlib::tensor* biases = nullptr) const {
            using dlib::serialize;
            serialize(std::string(version_quant), out);
            serialize(num_filters_, out);
            serialize(_nr, out);
            serialize(_nc, out);
            if (dilate) {
                serialize(_dilate_y, out);
                serialize(_dilate_x, out);
            }
            serialize(_stride_y, out);
            serialize(_stride_x, out);
            serialize(_padding_y, out);
            serialize(_padding_x, out);

            serialize(qm.rhs_limit(), out);
            serialize(qm, out);
            if (biases)
                serialize_bfloat16(*biases, out);
            else
                serialize(empty_tensor, out); // use_bias == false

            serialize(learning_rate_multiplier, out);
            serialize(weight_decay_multiplier, out);
            serialize(bias_learning_rate_multiplier, out);
            serialize(bias_weight_decay_multiplier, out);
        }

        void serialize_float(std::ostream& out, bool bfloat16) const {
            using dlib::serialize;
            serialize(use_bias ? std::string(version_float4) :
                      std::string(version_float5), out);
            if (bfloat16)
                serialize_bfloat16(get_layer_params(), out);
            else
                serialize(get_layer_params(), out);
            serialize(num_filters_, out);
            serialize(_nr, out);
            serialize(_nc, out);
            if (dilate) {
                serialize(_dilate_y, out);
                serialize(_dilate_x, out);
            }
            serialize(_stride_y, out);
            serialize(_stride_x, out);
            serialize(_padding_y, out);
            serialize(_padding_x, out);
            serialize(filters, out);
            serialize(biases, out);
            serialize(learning_rate_multiplier, out);
            serialize(weight_decay_multiplier, out);
            serialize(bias_learning_rate_multiplier, out);
            serialize(bias_weight_decay_multiplier, out);
            if (!use_bias)
                serialize(use_bias, out);
        }

        static void deserialize_dims(std::istream& in) {
            using error = dlib::serialization_error;
            using dlib::deserialize;

            long nr, nc;
            deserialize(nr, in);
            deserialize(nc, in);
            if (nr != _nr)
                throw error("Wrong nr found deserializing lm_con_");
            if (nc != _nc)
                throw error("Wrong nc found deserializing lm_con_");

            if (dilate) {
                int dilate_y, dilate_x;
                deserialize(dilate_y, in);
                deserialize(dilate_x, in);
                if (dilate_y != _dilate_y)
                    throw error("Wrong dilate_y found deserializing lm_con_");
                if (dilate_x != _dilate_x)
                    throw error("Wrong dilate_x found deserializing lm_con_");
            }

            int stride_y, stride_x;
            deserialize(stride_y, in);
            deserialize(stride_x, in);
            if (stride_y != _stride_y)
                throw error("Wrong stride_y found deserializing lm_con_");
            if (stride_x != _stride_x)
                throw error("Wrong stride_x found deserializing lm_con_");

            int padding_y, padding_x;
            deserialize(padding_y, in);
            deserialize(padding_x, in);
            if (padding_y != _padding_y)
                throw error("Wrong padding_y found deserializing lm_con_");
            if (padding_x != _padding_x)
                throw error("Wrong padding_x found deserializing lm_con_");
        }

        void deserialize_float(std::istream& in, bool has_use_bias) {
            qfilt = nullptr;
            if (auto p = std::make_shared<dlib::resizable_tensor>()) {
                dlibx::deserialize(*p, in); // might be bfloat16
                params = move(p);
            }

            using dlib::deserialize;
            deserialize(num_filters_, in);
            deserialize_dims(in);

            deserialize(filters, in);
            deserialize(biases, in);

            conv.setup(_nr, _nc,
                       _dilate_y, _dilate_x,
                       _stride_y, _stride_x,
                       _padding_y, _padding_x,
                       filters(*params,0));

            deserialize(learning_rate_multiplier, in);
            deserialize(weight_decay_multiplier, in);
            deserialize(bias_learning_rate_multiplier, in);
            deserialize(bias_weight_decay_multiplier, in);
            if (has_use_bias)
                deserialize(use_bias, in);
            else
                use_bias = true;
        }

        void deserialize_quant(std::istream& in) {
            using dlib::deserialize;
            deserialize(num_filters_, in);
            deserialize_dims(in);

            int rhs_limit_no_longer_used;
            deserialize(rhs_limit_no_longer_used, in);
            qfilt = qmat::deserialize_shared(in);
            filters = {0};

            // biases as resizable_tensor
            if (auto p = std::make_shared<dlib::resizable_tensor>()) {
                dlibx::deserialize(*p, in); // bfloat16
                params = move(p);
            }
            if (auto n = params->size()) {
                if (n != std::size_t(num_filters_))
                    throw dlib::serialization_error(
                        "Wrong bias tensor size found deserializing lm_con_");
                biases = { 1, num_filters_ };
                use_bias = true;
            }
            else {
                biases = {};
                use_bias = false;
            }
            conv.reset();

            deserialize(learning_rate_multiplier, in);
            deserialize(weight_decay_multiplier, in);
            deserialize(bias_learning_rate_multiplier, in);
            deserialize(bias_weight_decay_multiplier, in);
        }
    };

    template <long nf, long nr, long nc,
              int sy, int sx, int py, int px, int dy, int dx>
    inline bool
    bias_is_disabled(const lm_con_<nf,nr,nc,sy,sx,py,px,dy,dx>& c) {
        return c.bias_is_disabled();
    }

    template <
        long num_filters,
        long nr,
        long nc,
        int stride_y,
        int stride_x,
        typename SUBNET
        >
    using lmcon = dlib::add_layer<lm_con_<num_filters,nr,nc,stride_y,stride_x>, SUBNET>;
}
