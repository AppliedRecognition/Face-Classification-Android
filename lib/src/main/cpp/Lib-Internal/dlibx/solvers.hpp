#pragma once

#include <dlib/dnn/solvers.h>

#include "dnn_fc_dynamic.hpp"
#include "dnn_lmcon.hpp"
#include "dnn_condw.hpp"

namespace dlibx {

    /** \brief Same calculation as dlib::sgd but with support for dlibx layers.
     */
    class sgd {
        using tensor = dlib::tensor;

    public:
        explicit sgd(float weight_decay,
                     float momentum = 0.9f)
            : weight_decay(weight_decay),
              momentum(momentum) {
        }
        sgd() : sgd(0.0005f, 0.9f) {}

        inline float get_momentum () const { return momentum; }
        inline float get_weight_decay () const { return weight_decay; }

        // general case (for layers without bias)
        template <typename layer_type> 
        inline const tensor& operator()(const float learning_rate,
                                        const layer_type& l,
                                        const tensor& params_grad) {
            update_general(learning_rate, l, params_grad);
            return v;
        }

        // convolution
        template <long K, long NR, long NC, int... Is>
        inline const tensor& operator()(const float learning_rate,
                                        const dlib::con_<K,NR,NC,Is...>& l,
                                        const tensor& params_grad) {
            update_with_bias(learning_rate, l, params_grad,
                             params_grad.size() - l.num_filters());
            return v;
        }

        template <long K, long NR, long NC, int... Is>
        inline const tensor& operator()(const float learning_rate,
                                        const lm_con_<K,NR,NC,Is...>& l,
                                        const tensor& params_grad) {
            update_with_bias(learning_rate, l, params_grad,
                             params_grad.size() - l.num_filters());
            return v;
        }

        // depth-wise convolution
        template <bias_mode BM, long MULT, long NR, long NC, int... Is>
        inline const tensor& operator()(const float learning_rate,
                                        const condw_<BM,MULT,NR,NC,Is...>& l,
                                        const tensor& params_grad) {
            if (l.get_bias_mode() == HAS_BIAS)
                update_with_bias(learning_rate, l, params_grad,
                                 params_grad.size() - l.num_filters());
            else
                update_general(learning_rate, l, params_grad);
            return v;
        }

        // transposed convolution
        template <long K, long NR, long NC, int... Is>
        inline const tensor& operator()(const float learning_rate,
                                        const dlib::cont_<K,NR,NC,Is...>& l,
                                        const tensor& params_grad) {
            update_with_bias(learning_rate, l, params_grad,
                             params_grad.size() - l.num_filters());
            return v;
        }

        // fully-connected
        template <unsigned long N>
        inline const tensor& operator()(const float learning_rate,
                                        const dlib::fc_<N,dlib::FC_HAS_BIAS>& l,
                                        const tensor& params_grad) {
            update_with_bias(learning_rate, l, params_grad,
                             params_grad.size() - l.get_num_outputs());
            return v;
        }

        template <unsigned long K, bias_mode BM>
        inline const tensor& operator()(const float learning_rate,
                                        const fc_dynamic_<K,BM>& l,
                                        const tensor& params_grad) {
            if (l.get_bias_mode() == HAS_BIAS)
                update_with_bias(learning_rate, l, params_grad,
                                 params_grad.size() - l.get_num_outputs());
            else
                update_general(learning_rate, l, params_grad);
            return v;
        }

        // batch normalization
        template <dlib::layer_mode mode>
        inline const tensor& operator()(const float learning_rate,
                                        const dlib::bn_<mode>& l,
                                        const tensor& params_grad) {
            update_with_bias(learning_rate, l, params_grad,
                             params_grad.size() / 2);
            return v;
        }


        friend void serialize(const sgd& item, std::ostream& out) {
            dlib::serialize("sgd2", out);
            dlib::serialize(item.v, out);
            dlib::serialize(item.weight_decay, out);
            dlib::serialize(item.momentum, out);
        }

        friend void deserialize(sgd& item, std::istream& in) {
            std::string version;
            dlib::deserialize(version, in);
            if (version != "sgd2")
                throw dlib::serialization_error(
                    "Unexpected version found while deserializing dlibx::sgd.");
            dlib::deserialize(item.v, in);
            dlib::deserialize(item.weight_decay, in);
            dlib::deserialize(item.momentum, in);
        }

        friend std::ostream& operator<<(std::ostream& out, const sgd& item) {
            out << "sgd: weight_decay=" << item.get_weight_decay()
                << ", momentum=" << item.get_momentum(); 
            return out;
        }

    private:
        template <typename layer_type> 
        void update_general(const float learning_rate,
                            const layer_type& l,
                            const tensor& params_grad) {
            const tensor& params = l.get_layer_params();
            DLIB_CASSERT(params.size() != 0);
            if (v.size() == 0) {
                v.copy_size(params_grad);
                v = 0;
            }

            using namespace dlib;
            const auto lr = learning_rate*get_learning_rate_multiplier(l);
            const auto wd = weight_decay*get_weight_decay_multiplier(l);
            
            // v = momentum*mat(v) - wd*lr*mat(params) - lr*mat(params_grad);

            dlib::tt::affine_transform(v, v, params, params_grad,
                                       momentum, -float(wd*lr), -float(lr));
        }

        template <typename T>
        static constexpr auto is_one(T x) {
            return 1 <= x && x <= 1;
        }

        template <typename layer_type> 
        void update_with_bias(const float learning_rate,
                              const layer_type& l,
                              const tensor& params_grad,
                              unsigned long bias_offset) {
            const tensor& params = l.get_layer_params();
            DLIB_CASSERT(params.size() != 0);
            if (v.size() == 0) {
                v.copy_size(params_grad);
                v = 0;
            }

            using namespace dlib;
            auto lr = learning_rate * get_learning_rate_multiplier(l);
            auto wd = weight_decay * get_weight_decay_multiplier(l);
            
            // v = momentum*mat(v) - wd*lr*mat(params) - lr*mat(params_grad);

            if (is_one(l.get_bias_learning_rate_multiplier()) &&
                is_one(l.get_bias_weight_decay_multiplier()))
                dlib::tt::affine_transform(
                    v, v, params, params_grad,
                    momentum, -float(wd*lr), -float(lr));
            else {
                dlib::tt::affine_transform_range(
                    0, bias_offset, v, v, params, params_grad,
                    momentum, -float(wd*lr), -float(lr));

                // now update the biases but apply their multipliers
                lr *= l.get_bias_learning_rate_multiplier();
                wd *= l.get_bias_weight_decay_multiplier();
                dlib::tt::affine_transform_range(
                    bias_offset, v.size(), v, v, params, params_grad,
                    momentum, -float(wd*lr), -float(lr));
            }
        }

        dlib::resizable_tensor v;
        float weight_decay;
        float momentum;
    };
}
