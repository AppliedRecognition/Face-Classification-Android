#pragma once

#include <dlib/dnn/core.h>
#include <dlib/dnn/layers.h>
#include <dlib/dnn/utilities.h>
#include "dnn_bias_mode.hpp"
#include "qmat.hpp"
#include "conv.hpp"
#include "bfloat16.hpp"
#include "tensor_conv.hpp"
#include "library_init.hpp"
#include <optional>


namespace dlibx {

    /** \brief Depth-wise convolution with or without bias.
     *
     * This class implements per-channel convolution with optional multipler.
     * The number of output channels is in_channels * multipler.
     *
     * For a complete depthwise separable convolution,
     * use the con_ class for the following pointwise 1x1 convolution.
     */
    template <
        bias_mode _default_bias,
        long _default_multiplier,
        long _nr,
        long _nc,
        int _stride_y,
        int _stride_x,
        int _padding_y = _stride_y!=1 ? 0 : int(_nr/2),
        int _padding_x = _stride_x!=1 ? 0 : int(_nc/2)
        >
    class condw_ {
        static_assert(_default_multiplier > 0, "The depth multiplier must be > 0");
        static_assert(_nr >= 0, "The number of rows in a filter must be >= 0");
        static_assert(_nc >= 0, "The number of columns in a filter must be >= 0");
        static_assert(_stride_y > 0, "The filter stride must be > 0");
        static_assert(_stride_x > 0, "The filter stride must be > 0");
        static_assert(_nr==0 || (0 <= _padding_y && _padding_y < _nr), "The padding must be smaller than the filter size.");
        static_assert(_nc==0 || (0 <= _padding_x && _padding_x < _nc), "The padding must be smaller than the filter size.");
        static_assert(_nr!=0 || 0 == _padding_y, "If _nr==0 then the padding must be set to 0 as well.");
        static_assert(_nc!=0 || 0 == _padding_x, "If _nr==0 then the padding must be set to 0 as well.");

    public:
        condw_(bias_mode mode = _default_bias, long mult = _default_multiplier)
            : mode(mode), multiplier(mult),
              learning_rate_multiplier(1),
              weight_decay_multiplier(1),
              bias_learning_rate_multiplier(1),
              bias_weight_decay_multiplier(0) {
            if (multiplier < 1)
                throw std::invalid_argument("depth multipler must be > 0");
            library_init();
        }

        // generalized copy
        // note: if _default_bias is HAS_BIAS and other does not have bias,
        //       then zero bias is added
        template <bias_mode _other_bias, long _other_mult>
        condw_(const condw_<_other_bias,_other_mult,_nr,_nc,_stride_y,_stride_x,_padding_y,_padding_x>& other)
            : mode(other.get_bias_mode()),
              multiplier(other.get_depth_multiplier()),
              params(other.get_shared_params()),
              qfilt(other.get_shared_qfilt()),
              learning_rate_multiplier(other.get_learning_rate_multiplier()),
              weight_decay_multiplier(other.get_weight_decay_multiplier()),
              bias_learning_rate_multiplier(other.get_bias_learning_rate_multiplier()),
              bias_weight_decay_multiplier(other.get_bias_weight_decay_multiplier()) {
            if (qfilt) {
                if (params && params->size() > 0) {
                    DLIB_CASSERT(mode == HAS_BIAS);
                    DLIB_CASSERT(qfilt->nr() == long(params->size()));
                    biases = dlib::alias_tensor(1,qfilt->nr());
                }
                else
                    DLIB_CASSERT(mode == NO_BIAS);
            }
            else if (params && params->size() > 0) {
                const auto filter_size = std::size_t(other.nr() * other.nc());
                const auto ob = (mode == HAS_BIAS ? 1u : 0u);
                const auto num_filters = params->size() / (filter_size + ob);
                filters = dlib::alias_tensor(
                    long(num_filters), 1, other.nr(), other.nc());
                if (mode == HAS_BIAS)
                    biases = dlib::alias_tensor(1,long(num_filters));
                DLIB_CASSERT(params->size() == filters.size() + biases.size(),
                             "Inconsistent params size in condw.");
            }
            if (_default_bias == HAS_BIAS &&
                _other_bias == NO_BIAS && mode == NO_BIAS)
                add_biases();
        }

        // returns true if bias was not already enabled
        bool add_biases() {
            if (mode == HAS_BIAS)
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
                DLIB_CASSERT(num_filters > 0,
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
                conv.setup(_nr, _nc, 1, 1, _stride_y, _stride_x,
                           _padding_y, _padding_x, filters(*params,0));
            }
            // else not setup yet so nothing to do
            mode = HAS_BIAS;
            return true;
        }

        void enable_bias() { add_biases(); }
        void disable_bias() { // only if not already setup
            if (mode != NO_BIAS) {
                DLIB_CASSERT(biases.size() == 0);
                mode = NO_BIAS;
            }
        }
        inline bool bias_is_disabled() const { return mode == NO_BIAS; }
        inline auto get_bias_mode() const { return mode; }

        long nr() const {
            if (_nr==0)
                return filters.nr();
            else
                return _nr;
        }
        long nc() const {
            if (_nc==0)
                return filters.nc();
            else
                return _nc;
        }
        constexpr auto stride_y() const { return _stride_y; }
        constexpr auto stride_x() const { return _stride_x; }
        constexpr auto padding_y() const { return _padding_y; }
        constexpr auto padding_x() const { return _padding_x; }

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
            p.x() = (p.x()+(padding_x()-nc()/2))/stride_x();
            p.y() = (p.y()+(padding_y()-nr()/2))/stride_y();
            return p;
        }
        inline auto map_output_to_input(dlib::dpoint p) const {
            p.x() = p.x()*double(stride_x()) + double(nc()/2 - padding_x());
            p.y() = p.y()*double(stride_y()) + double(nr()/2 - padding_y());
            return p;
        }

        inline auto get_depth_multiplier() const { return multiplier; }
        inline auto num_filters() const {
            return qfilt ? qfilt->nr() : long(filters.num_samples());
        }

        template <typename SUBNET>
        void setup(const SUBNET& sub) {
            DLIB_CASSERT(!qfilt);

            const auto& input = sub.get_output();

            const auto filt_nr = _nr!=0 ? _nr : long(input.nr());
            const auto filt_nc = _nc!=0 ? _nc : long(input.nc());

            const auto num_inputs = filt_nr * filt_nc;
            const auto num_outputs = multiplier * long(input.k());

            // allocate params for the filters
            auto p = std::make_shared<dlib::resizable_tensor>(
                num_inputs*num_outputs + (mode == HAS_BIAS ? num_outputs : 0));

            dlib::rand rnd(std::rand());
            randomize_parameters(*p, std::size_t(num_inputs + multiplier), rnd);

            filters = dlib::alias_tensor(num_outputs, 1, filt_nr, filt_nc);

            if (mode == HAS_BIAS) {
                // set the initial bias values to zero
                biases = dlib::alias_tensor(1,num_outputs);
                biases(*p,filters.size()) = 0;
            }
            params = move(p);
            conv.setup(_nr, _nc, 1, 1, _stride_y, _stride_x,
                       _padding_y, _padding_x, filters(*params,0));
        }

        template <typename SUBNET>
        void forward(const SUBNET& sub, dlib::resizable_tensor& output) {
            auto&& data = sub.get_output();
            if (!qfilt) {
                // floating point
                DLIB_CASSERT(params, "condw layer is not setup");
                if (!conv)
                    conv.setup(_nr, _nc, 1, 1, _stride_y, _stride_x,
                               _padding_y, _padding_x, filters(*params,0));
                conv(data, output);
            }
            else if (0 < _padding_y || 0 < _padding_x) {
                auto padded = apply_padding(data, _padding_y, _padding_x);
                qfilt->convdw(*padded, output, _nr, _nc, 1, 1,
                              _stride_y, _stride_x);
            }
            else  // no padding
                qfilt->convdw(data, output, _nr, _nc, 1, 1,
                              _stride_y, _stride_x);

            if (mode == HAS_BIAS) {
                DLIB_CASSERT(params, "condw layer is not setup");
                dlib::tt::add(1,output,1,biases(*params,filters.size()));
            }
        }

        template <typename SUBNET>
        void backward(const dlib::tensor& input,
                      SUBNET& sub, dlib::tensor& params_grad) {
            DLIB_CASSERT(params, "condw layer is not setup");
            DLIB_CASSERT(!qfilt, "cannot train quantized condw layer");
            auto filt = filters(*params,0);
            auto&& data = sub.get_output();
            const auto data_channel =
                dlib::alias_tensor(1,1,data.nr(),data.nc());
            const auto channel_filters =
                dlib::alias_tensor(multiplier,1,filters.nr(),filters.nc());
            tconv.setup(data_channel(data,0), channel_filters(*params,0),
                        _stride_y, _stride_x,
                        _padding_y, _padding_x);

            if (learning_rate_multiplier <= 0)
                tconv.backward_dw(
                    filt, input, sub.get_gradient_input());
            else if (mode != HAS_BIAS)
                tconv.backward_dw(
                    filt, input, sub.get_gradient_input(),
                    &data, &params_grad);
            else {
                auto pg = filters(params_grad,0);
                auto bg = biases(params_grad, filters.size());
                tconv.backward_dw(
                    filt, input, sub.get_gradient_input(),
                    &data, &pg, &bg);
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

        friend auto serialize_format(const condw_& item) {
            if (item.qfilt)
                return item.qfilt->empty() ?
                    pf::native : quantize(item.qfilt->serialize_bits());
            if (!item.params || item.params->size() <= 0)
                return pf::native;
            return is_bfloat16(*item.params) ? pf::bfloat16 : pf::float32;
        }

        friend void serialize(const condw_& item, std::ostream& out) {
            using error = dlib::serialization_error;
            const auto format = get_parameter_format(out);
            switch (format) {
            case pf::native:
                if (item.qfilt) {
                    DLIB_CASSERT(item.filters.size() == 0);
                    item.serialize_qfilt(
                        out, *item.qfilt, item.get_layer_params());
                }
                else
                    item.serialize_float(
                        out, is_bfloat16(item.get_layer_params()));
                break;

            case pf::float32:
                if (item.qfilt)
                    throw error("Conversion from quantization to floating point not supported in condw layer.");
                else
                    item.serialize_float(out, false);
                break;

            case pf::bfloat16:
                if (item.qfilt)
                    throw error("Conversion from quantization to floating point not supported in condw layer.");
                else
                    item.serialize_float(out, true);
                break;

            default:
                if (const auto bits = bits_per_element(format)) {
                    if (item.qfilt) {
                        DLIB_CASSERT(item.filters.size() == 0);
                        item.serialize_qfilt(
                            out, *item.qfilt, item.get_layer_params());
                    }
                    else {
                        // quantize from float
                        DLIB_CASSERT(item.params);
                        // use 16-bit regardless of bits
                        // it'll deserialize to 8-bit if bits <= 8
                        qmat16 qm;
                        qm.assign_lhs(mat(item.filters(*item.params,0)), bits);
                        auto b = item.biases(*item.params, item.filters.size());
                        item.serialize_qfilt(out, qm, b);
                    }
                }
                else
                    throw error("Invalid serialization format.");
            }
        }

        friend void deserialize(condw_& item, std::istream& in) {
            std::string version;
            dlib::deserialize(version, in);
            if (version == "condw_1" || version == "sepcon_1")
                item.deserialize_condw1(in);
            else if (version == "qdw_1")
                item.deserialize_qdw1(in);
            else
                throw dlib::serialization_error("Unexpected version '"+version+"' found while deserializing condw_.");
        }

        friend std::ostream& operator<<(std::ostream& out, const condw_& item) {
            out << (item.mode == HAS_BIAS ? "condw\t (" : "condw_no_bias\t (")
                << "multiplier="<<item.multiplier
                << ", nr="<<item.nr()
                << ", nc="<<item.nc()
                << ", stride_y="<<_stride_y
                << ", stride_x="<<_stride_x
                << ", padding_y="<<_padding_y
                << ", padding_x="<<_padding_x
                << ")";
            out << " learning_rate_mult="<<item.learning_rate_multiplier;
            out << " weight_decay_mult="<<item.weight_decay_multiplier;
            if (item.mode == HAS_BIAS) {
                out << " bias_learning_rate_mult="<<item.bias_learning_rate_multiplier;
                out << " bias_weight_decay_mult="<<item.bias_weight_decay_multiplier;
            }
            return out;
        }

        friend void to_xml(const condw_& item, std::ostream& out) {
            out << (item.mode == HAS_BIAS ? "<condw" : "<condw_no_bias")
                << " multiplier='"<<item.multiplier<<"'"
                << " nr='"<<item.nr()<<"'"
                << " nc='"<<item.nc()<<"'"
                << " stride_y='"<<_stride_y<<"'"
                << " stride_x='"<<_stride_x<<"'"
                << " padding_y='"<<_padding_y<<"'"
                << " padding_x='"<<_padding_x<<"'"
                << " learning_rate_mult='"<<item.learning_rate_multiplier<<"'"
                << " weight_decay_mult='"<<item.weight_decay_multiplier<<"'";
            if (item.mode == HAS_BIAS)
                out << " bias_learning_rate_mult='"<<item.bias_learning_rate_multiplier<<"'"
                    << " bias_weight_decay_mult='"<<item.bias_weight_decay_multiplier<<"'";
            out << ">\n";
            out << mat(item.get_layer_params());
            out << "</condw>";
        }

    private:
        bias_mode mode;
        long multiplier;

        std::shared_ptr<const dlib::resizable_tensor> params;
        dlib::alias_tensor filters, biases;

        std::shared_ptr<const qmat> qfilt;

        double learning_rate_multiplier;
        double weight_decay_multiplier;
        double bias_learning_rate_multiplier;
        double bias_weight_decay_multiplier;

        forward_convdw conv;
        tensor_conv tconv;

        void serialize_float(std::ostream& out, bool bfloat16) const {
            using dlib::serialize;
            serialize("condw_1", out);
            if (bfloat16)
                serialize_bfloat16(get_layer_params(), out);
            else
                serialize(get_layer_params(), out);
            serialize(multiplier, out);
            serialize(_nr, out);
            serialize(_nc, out);
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
        }

        void serialize_qfilt(std::ostream& out, const qmat& qm,
                             const dlib::tensor& biases) const {
            using dlib::serialize;
            serialize("qdw_1", out);
            serialize(multiplier, out);
            serialize(_nr, out);
            serialize(_nc, out);
            serialize(_stride_y, out);
            serialize(_stride_x, out);
            serialize(_padding_y, out);
            serialize(_padding_x, out);

            serialize(qm, out);
            serialize_bfloat16(biases, out);

            serialize(learning_rate_multiplier, out);
            serialize(weight_decay_multiplier, out);
            serialize(bias_learning_rate_multiplier, out);
            serialize(bias_weight_decay_multiplier, out);
        }

        static void deserialize_dims(std::istream& in) {
            using error = dlib::serialization_error;
            using dlib::deserialize;

            long nr, nc;
            deserialize(nr, in);
            deserialize(nc, in);
            if (nr != _nr)
                throw error("Wrong nr found deserializing condw_");
            if (nc != _nc)
                throw error("Wrong nc found deserializing condw_");

            int stride_y, stride_x;
            deserialize(stride_y, in);
            deserialize(stride_x, in);
            if (stride_y != _stride_y)
                throw error("Wrong stride_y found deserializing condw_");
            if (stride_x != _stride_x)
                throw error("Wrong stride_x found deserializing condw_");

            int padding_y, padding_x;
            deserialize(padding_y, in);
            deserialize(padding_x, in);
            if (padding_y != _padding_y)
                throw error("Wrong padding_y found deserializing condw_");
            if (padding_x != _padding_x)
                throw error("Wrong padding_x found deserializing condw_");
        }

        void deserialize_condw1(std::istream& in) {
            qfilt = nullptr;
            if (auto p = std::make_shared<dlib::resizable_tensor>()) {
                dlibx::deserialize(*p, in); // might be bfloat16
                params = move(p);
            }

            using dlib::deserialize;
            deserialize(multiplier, in);
            if (multiplier < 1)
                throw dlib::serialization_error("Invalid multiplier found while deserializing dlibx::condw_");

            deserialize_dims(in);

            deserialize(filters, in);
            deserialize(biases, in);
            mode = biases.size() > 0 ? HAS_BIAS : NO_BIAS;

            conv.setup(_nr, _nc, 1, 1, _stride_y, _stride_x,
                       _padding_y, _padding_x, filters(*params,0));

            deserialize(learning_rate_multiplier, in);
            deserialize(weight_decay_multiplier, in);
            deserialize(bias_learning_rate_multiplier, in);
            deserialize(bias_weight_decay_multiplier, in);
        }

        void deserialize_qdw1(std::istream& in) {
            using dlib::deserialize;
            deserialize(multiplier, in);
            if (multiplier < 1)
                throw dlib::serialization_error("Invalid multiplier found while deserializing dlibx::condw_");

            deserialize_dims(in);

            qfilt = qmat::deserialize_shared(in);
            filters = {0};
            conv.reset();

            // biases as resizable_tensor
            if (auto p = std::make_shared<dlib::resizable_tensor>()) {
                dlibx::deserialize(*p, in); // bfloat16
                params = move(p);
                if (params->size() > 0) {
                    biases = { 1, long(params->size()) };
                    mode = HAS_BIAS;
                }
                else {
                    biases = {0};
                    mode = NO_BIAS;
                }
            }

            deserialize(learning_rate_multiplier, in);
            deserialize(weight_decay_multiplier, in);
            deserialize(bias_learning_rate_multiplier, in);
            deserialize(bias_weight_decay_multiplier, in);
        }
    };

    template <
        long nr,
        long nc,
        int stride_y,
        int stride_x,
        typename SUBNET
        >
    using condw = dlib::add_layer<
        condw_<HAS_BIAS,1,nr,nc,stride_y,stride_x>, SUBNET>;

    template <
        long nr,
        long nc,
        int stride_y,
        int stride_x,
        typename SUBNET
        >
    using condw_no_bias = dlib::add_layer<
        condw_<NO_BIAS,1,nr,nc,stride_y,stride_x>, SUBNET>;
}
