#pragma once

#include <dlib/dnn/core.h>
#include <dlib/dnn/layers.h>
#include "dnn_bias_mode.hpp"
#include "qmat.hpp"
#include "bfloat16.hpp"
#include "tensor.hpp"


namespace dlibx {

    using dlib::num_fc_outputs;

    /** \brief Same as dlib::fc_ but bias mode is runtime dynamic.
     */
    template <unsigned long num_outputs_, bias_mode default_bias_mode>
    class fc_dynamic_ {
        static_assert(num_outputs_ > 0,
                      "The number of outputs from a fc_ layer must be > 0");

    public:
        fc_dynamic_(num_fc_outputs o, bias_mode mode = default_bias_mode)
            : mode(mode),
              num_outputs(long(o.num_outputs)),
              num_inputs(0),
              learning_rate_multiplier(1),
              weight_decay_multiplier(1),
              bias_learning_rate_multiplier(1),
              bias_weight_decay_multiplier(0)
        {}

        fc_dynamic_() : fc_dynamic_(num_fc_outputs(num_outputs_)) {}

        template <unsigned long K, bias_mode BM>
        fc_dynamic_(const dlib::fc_<K,BM>& other)
            : mode(other.get_bias_mode()),
              num_outputs(other.get_num_outputs()),
              params(std::make_shared<dlib::resizable_tensor>(other.get_layer_params())),
              learning_rate_multiplier(
                  other.get_learning_rate_multiplier()),
              weight_decay_multiplier(
                  other.get_weight_decay_multiplier()),
              bias_learning_rate_multiplier(
                  other.get_bias_learning_rate_multiplier()),
              bias_weight_decay_multiplier(
                  other.get_bias_weight_decay_multiplier()) {
            if ((num_inputs = params->num_samples()) > 0) {
                DLIB_CASSERT(num_outputs > 0, "num_outputs must be positive");
                if (mode == HAS_BIAS) {
                    --num_inputs;
                    DLIB_CASSERT(num_inputs > 0, "num_inputs must be positive");
                    biases = dlib::alias_tensor(1,num_outputs);
                }
                weights = dlib::alias_tensor(num_inputs, num_outputs);
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
            weight_decay_multiplier  = val;
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
            bias_weight_decay_multiplier  = val;
        }

        unsigned long get_num_outputs() const {
            return static_cast<unsigned long>(num_outputs);
        }

        void set_num_outputs(long num)  {
            DLIB_CASSERT(num > 0);
            if (num != num_outputs) {
                DLIB_CASSERT(!params || params->size() == 0,
                             "You can't change the number of filters in fc_ if the parameter tensor has already been allocated.");
                num_outputs = num;
            }
        }

        auto get_bias_mode() const { return mode; }
        bool bias_is_disabled() const { return mode == NO_BIAS; }
        //void disable_bias() { use_bias = false; }

        template <typename SUBNET>
        void setup(const SUBNET& sub) {
            DLIB_CASSERT(!qfilt);
            const auto& input = sub.get_output();
            num_inputs = long(input.k()*input.nr()*input.nc());
            DLIB_CASSERT(num_inputs > 0, "num_inputs must be positive");
            DLIB_CASSERT(num_outputs > 0, "num_outputs must be positive");
            auto p = std::make_shared<dlib::resizable_tensor>(
                mode == HAS_BIAS ? num_inputs+1 : num_inputs, num_outputs);

            dlib::rand rnd(std::rand());
            randomize_parameters(*p, std::size_t(num_inputs+num_outputs), rnd);

            weights = dlib::alias_tensor(num_inputs, num_outputs);

            if (mode == HAS_BIAS) {
                biases = dlib::alias_tensor(1,num_outputs);
                // set the initial bias values to zero
                biases(*p,weights.size()) = 0;
            }
            params = move(p);
        }

        template <typename SUBNET>
        void forward(const SUBNET& sub, dlib::resizable_tensor& output) {
            DLIB_CASSERT(params, "fc_dynamic layer is not setup");
            const dlib::tensor& input = sub.get_output();
            DLIB_CASSERT(num_inputs == input.nr()*input.nc()*input.k(), "The size of the input tensor to this fc layer doesn't match the size the fc layer was trained with.");
            output.set_size(input.num_samples(), num_outputs);
            if (qfilt)
                qfilt->fc(input, output);
            else {
                auto w = weights(*params, 0);
                dlib::tt::gemm(0,output, 1,input,false, w,false);
            }
            if (mode == HAS_BIAS) {
                auto b = biases(*params, weights.size());
                dlib::tt::add(1,output,1,b);
            }
        } 

        template <typename SUBNET>
        void backward(const dlib::tensor& gradient_input, SUBNET& sub,
                      dlib::tensor& params_grad) {
            DLIB_CASSERT(params, "fc_dynamic layer is not setup");
            DLIB_CASSERT(!qfilt, "cannot train quantized fc layer");
            // no point computing the parameter gradients if they won't be used.
            if (learning_rate_multiplier > 0) {
                // compute the gradient of the weight parameters.  
                auto pw = weights(params_grad, 0);
                dlib::tt::gemm(0,pw, 1,sub.get_output(),true, gradient_input,false);

                if (mode == HAS_BIAS) {
                    // compute the gradient of the bias parameters.  
                    auto pb = biases(params_grad, weights.size());
                    dlib::tt::assign_bias_gradient(pb, gradient_input);
                }
            }

            // compute the gradient for the data
            auto w = weights(*params, 0);
            dlib::tt::gemm(1,sub.get_gradient_input(), 1,gradient_input,false, w,true);
        }

        auto get_weights() { return weights(get_layer_params(), 0); }
        auto get_weights() const { return weights(get_layer_params(), 0); }

        auto get_biases() {
            DLIB_CASSERT(mode == HAS_BIAS, "This fc_ layer doesn't have a bias vector to be retrieved.");
            return biases(get_layer_params(), weights.size());
        }
        auto get_biases() const {
            DLIB_CASSERT(mode == HAS_BIAS, "This fc_ layer doesn't have a bias vector to be retrieved.");
            return biases(get_layer_params(), weights.size());
        }

        void add_biases() {
            DLIB_CASSERT(mode != HAS_BIAS, "This fc_ layer already has biases.");
            DLIB_CASSERT(!qfilt, "cannot add biases after quantization");
            mode = HAS_BIAS;
            if (params && params->size() > 0) {
                DLIB_CASSERT(num_inputs > 0, "num_inputs must be positive");
                DLIB_CASSERT(num_outputs > 0, "num_outputs must be positive");
                auto new_params = std::make_shared<dlib::resizable_tensor>(
                    num_inputs+1, num_outputs);
                weights(*new_params, 0) = mat(weights(*params, 0));
                biases = dlib::alias_tensor(1,num_outputs);
                biases(*new_params,weights.size()) = 0;
                params = move(new_params);
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
            else if (params.use_count() > 1) // make copy
                params = std::make_shared<dlib::resizable_tensor>(*params);
            return const_cast<dlib::resizable_tensor&>(*params);
        }

        friend auto serialize_format(const fc_dynamic_& item) {
            if (item.qfilt)
                return item.qfilt->empty() ?
                    pf::native : quantize(item.qfilt->serialize_bits());
            if (!item.params || item.params->size() <= 0)
                return pf::native;
            return is_bfloat16(*item.params) ? pf::bfloat16 : pf::float32;
        }

        friend void serialize(const fc_dynamic_& item, std::ostream& out) {
            using error = dlib::serialization_error;
            const auto format = get_parameter_format(out);
            switch (format) {
            case pf::native:
                if (item.qfilt) {
                    DLIB_CASSERT(item.params && item.weights.size() == 0);
                    item.serialize_qfilt(out, *item.qfilt, *item.params);
                }
                else
                    item.serialize_float(
                        out, is_bfloat16(item.get_layer_params()));
                break;

            case pf::float32:
                if (item.qfilt)
                    throw error("Conversion from quantization to floating point not supported in fc layer.");
                else
                    item.serialize_float(out, false);
                break;

            case pf::bfloat16:
                if (item.qfilt)
                    throw error("Conversion from quantization to floating point not supported in fc layer.");
                else
                    item.serialize_float(out, true);
                break;

            default:
                if (const auto bits = bits_per_element(format)) {
                    if (item.qfilt) {
                        DLIB_CASSERT(item.params && item.weights.size() == 0);
                        item.serialize_qfilt(out, *item.qfilt, *item.params);
                    }
                    else {
                        // quantize from float
                        DLIB_CASSERT(item.params);
                        // use 16-bit regardless of bits
                        // it'll deserialize to 8-bit if bits <= 8
                        qmat16 qm;
                        qm.assign_lhs(
                            trans(mat(item.weights(*item.params,0))), bits);
                        item.serialize_qfilt(
                            out, qm,
                            item.biases(*item.params,item.weights.size()));
                    }
                }
                else
                    throw error("Invalid serialization format.");
            }
        }

        friend void deserialize(fc_dynamic_& item, std::istream& in) {
            std::string version;
            dlib::deserialize(version, in);
            if (version == "fc_2")
                item.deserialize_fc2(in);
            else if (version == "fc_3") {
                item.deserialize_fc2(in);
                bool use_bias;
                dlib::deserialize(use_bias, in);
            }
            else if (version == "qfc_1")
                item.deserialize_qfc1(in);
            else
                throw dlib::serialization_error("Unexpected version '"+version+"' found while deserializing fc_dynamic.");
        }

        friend std::ostream& operator<<(std::ostream& out, const fc_dynamic_& item) {
            if (item.mode == HAS_BIAS) {
                out << "fc\t ("
                    << "num_outputs="<<item.num_outputs
                    << ")";
                out << " learning_rate_mult="<<item.learning_rate_multiplier;
                out << " weight_decay_mult="<<item.weight_decay_multiplier;
                out << " bias_learning_rate_mult="<<item.bias_learning_rate_multiplier;
                out << " bias_weight_decay_mult="<<item.bias_weight_decay_multiplier;
            }
            else {
                out << "fc_no_bias ("
                    << "num_outputs="<<item.num_outputs
                    << ")";
                out << " learning_rate_mult="<<item.learning_rate_multiplier;
                out << " weight_decay_mult="<<item.weight_decay_multiplier;
            }
            return out;
        }

        friend void to_xml(const fc_dynamic_& item, std::ostream& out) {
            if (item.mode==HAS_BIAS) {
                out << "<fc"
                    << " num_outputs='"<<item.num_outputs<<"'"
                    << " learning_rate_mult='"<<item.learning_rate_multiplier<<"'"
                    << " weight_decay_mult='"<<item.weight_decay_multiplier<<"'"
                    << " bias_learning_rate_mult='"<<item.bias_learning_rate_multiplier<<"'"
                    << " bias_weight_decay_mult='"<<item.bias_weight_decay_multiplier<<"'";
                out << ">\n";
                out << mat(item.get_layer_params());
                out << "</fc>\n";
            }
            else {
                out << "<fc_no_bias"
                    << " num_outputs='"<<item.num_outputs<<"'"
                    << " learning_rate_mult='"<<item.learning_rate_multiplier<<"'"
                    << " weight_decay_mult='"<<item.weight_decay_multiplier<<"'";
                out << ">\n";
                out << mat(item.get_layer_params());
                out << "</fc_no_bias>\n";
            }
        }

    private:
        bias_mode mode;
        long num_outputs;
        long num_inputs;

        std::shared_ptr<const dlib::resizable_tensor> params;
        dlib::alias_tensor weights, biases;

        std::shared_ptr<const qmat> qfilt;

        double learning_rate_multiplier;
        double weight_decay_multiplier;
        double bias_learning_rate_multiplier;
        double bias_weight_decay_multiplier;

        void serialize_qfilt(std::ostream& out, const qmat& qm,
                             const dlib::tensor& biases) const {
            using dlib::serialize;
            serialize("qfc_1", out);
            serialize(num_outputs, out);
            serialize(num_inputs, out);
            serialize(qm, out);
            serialize_bfloat16(biases, out); // mode is inferred from size
            serialize(learning_rate_multiplier, out);
            serialize(weight_decay_multiplier, out);
            serialize(bias_learning_rate_multiplier, out);
            serialize(bias_weight_decay_multiplier, out);
        }

        void serialize_float(std::ostream& out, bool bfloat16) const {
            using dlib::serialize;
            serialize("fc_2", out);
            serialize(num_outputs, out);
            serialize(num_inputs, out);
            if (bfloat16)
                serialize_bfloat16(get_layer_params(), out);
            else
                serialize(get_layer_params(), out);
            serialize(weights, out);
            serialize(biases, out);
            serialize(int(mode), out);
            serialize(learning_rate_multiplier, out);
            serialize(weight_decay_multiplier, out);
            serialize(bias_learning_rate_multiplier, out);
            serialize(bias_weight_decay_multiplier, out);
        }

        void deserialize_fc2(std::istream& in) {
            using dlib::deserialize;
            deserialize(num_outputs, in);
            deserialize(num_inputs, in);
            qfilt = nullptr;
            if (auto p = std::make_shared<dlib::resizable_tensor>()) {
                dlibx::deserialize(*p, in); // might be bfloat16
                params = move(p);
            }
            deserialize(weights, in);
            deserialize(biases, in);
            if (params->size() != weights.size() + biases.size())
                throw dlib::serialization_error(
                    "Parameters size doesn't match weights and biases.");
            int bmode;
            deserialize(bmode, in);
            if (0 < params->size())
                mode = biases.size() == 0 ? NO_BIAS : HAS_BIAS;
            else
                mode = bias_mode(bmode);
            deserialize(learning_rate_multiplier, in);
            deserialize(weight_decay_multiplier, in);
            deserialize(bias_learning_rate_multiplier, in);
            deserialize(bias_weight_decay_multiplier, in);
        }

        void deserialize_qfc1(std::istream& in) {
            using dlib::deserialize;
            deserialize(num_outputs, in);
            deserialize(num_inputs, in);
            qfilt = qmat::deserialize_shared(in);
            if (qfilt->nr() != num_outputs ||
                qfilt->nc() != num_inputs)
                throw dlib::serialization_error("Incorrect matrix size found while deserializing fc_dynamic.");

            // biases as resizable_tensor
            if (auto p = std::make_shared<dlib::resizable_tensor>()) {
                dlibx::deserialize(*p, in); // bfloat16
                params = move(p);
            }
            weights = {0};
            if (params->size() == 0) {
                mode = NO_BIAS;
                biases = {0};
            }
            else if (params->size() == std::size_t(num_outputs)) {
                mode = HAS_BIAS;
                biases = { 1, num_outputs };
            }
            else
                throw dlib::serialization_error("Incorrect bias vector size found while deserializing fc_dynamic.");

            deserialize(learning_rate_multiplier, in);
            deserialize(weight_decay_multiplier, in);
            deserialize(bias_learning_rate_multiplier, in);
            deserialize(bias_weight_decay_multiplier, in);
        }
    };


    template <unsigned long num_outputs, typename SUBNET>
    using fc = dlib::add_layer<fc_dynamic_<num_outputs,HAS_BIAS>, SUBNET>;
    template <unsigned long num_outputs, typename SUBNET>
    using fc_no_bias = dlib::add_layer<fc_dynamic_<num_outputs,NO_BIAS>, SUBNET>;
}



