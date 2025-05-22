#pragma once

#include "net_layer_impl_common.hpp"
#include "dnn_fc_dynamic.hpp"

namespace dlibx {
    namespace net {

        // **** dlib::fc

        template <unsigned long K, dlib::fc_bias_mode BM>
        constexpr auto layer_has_bias(const dlib::fc_<K,BM>&) {
            return BM == dlib::FC_HAS_BIAS;
        }
        template <unsigned long K, dlib::fc_bias_mode BM>
        constexpr auto layer_add_bias(const dlib::fc_<K,BM>&) {
            return BM == dlib::FC_HAS_BIAS;
        }
        template <unsigned long K, dlib::fc_bias_mode BM>
        constexpr auto layer_code(const dlib::fc_<K,BM>&) {
            static_assert(BM == dlib::FC_HAS_BIAS || BM == dlib::FC_NO_BIAS);
            return BM == dlib::FC_HAS_BIAS ? "fc+bias" : "fcnb";
        }
        template <unsigned long K, dlib::fc_bias_mode BM>
        constexpr auto layer_type(const dlib::fc_<K,BM>&) { return "fc"; }
        template <unsigned long K, dlib::fc_bias_mode BM>
        inline auto layer_concise(const dlib::fc_<K,BM>&) {
            static_assert(BM == dlib::FC_HAS_BIAS || BM == dlib::FC_NO_BIAS);
            return BM == dlib::FC_HAS_BIAS ? "bias|fc" : "fc";
        }
        template <unsigned long K, dlib::fc_bias_mode BM>
        inline auto layer_output_size(const dlib::fc_<K,BM>& fc) {
            return unsigned(fc.get_num_outputs());
        }
        template <unsigned long K, dlib::fc_bias_mode BM>
        auto layer_json(const dlib::fc_<K,BM>& fc) {
            json::object config;
            config["activation"] = "linear";
            config["trainable"] = true;
            config["units"] = fc.get_num_outputs();
            config["use_bias"] = fc.get_bias_mode() == dlib::FC_HAS_BIAS;
            config["dtype"] = "float32";
            json::object obj;
            obj["class_name"] = "Dense";
            obj["config"] = move(config);
            return json::array(1,move(obj));
        }
        template <unsigned long K, dlib::fc_bias_mode BM>
        constexpr auto serialize_format(const dlib::fc_<K,BM>&) {
            return pf::float32;
        }

        // **** dlibx::fc_dynamic

        template <unsigned long K, bias_mode BM>
        inline auto layer_has_bias(const dlibx::fc_dynamic_<K,BM>& fc) {
            return fc.get_bias_mode() == HAS_BIAS;
        }
        template <unsigned long K, dlib::fc_bias_mode BM>
        inline auto layer_add_bias(dlibx::fc_dynamic_<K,BM>& fc) {
            if (fc.get_bias_mode() != HAS_BIAS)
                fc.add_biases();
            return true;
        }
        template <unsigned long K, bias_mode BM>
        inline auto layer_code(const dlibx::fc_dynamic_<K,BM>& fc) {
            return fc.get_bias_mode() == HAS_BIAS ? "fc+bias" : "fcnb";
        }
        template <unsigned long K, bias_mode BM>
        constexpr auto layer_type(const dlibx::fc_dynamic_<K,BM>&) { return "fc"; }
        template <unsigned long K, bias_mode BM>
        inline auto layer_concise(const dlibx::fc_dynamic_<K,BM>& fc) {
            return fc.get_bias_mode() == HAS_BIAS ? "bias|fc" : "fc";
        }
        template <unsigned long K, bias_mode BM>
        inline auto layer_output_size(const dlibx::fc_dynamic_<K,BM>& fc) {
            return unsigned(fc.get_num_outputs());
        }
        template <unsigned long K, bias_mode BM>
        auto layer_json(const dlibx::fc_dynamic_<K,BM>& fc) {
            json::object config;
            config["activation"] = "linear";
            config["trainable"] = true;
            config["units"] = fc.get_num_outputs();
            config["use_bias"] = fc.get_bias_mode() == HAS_BIAS;
            config["dtype"] = "float32";
            json::object obj;
            obj["class_name"] = "Dense";
            obj["config"] = move(config);
            return json::array(1,move(obj));
        }


        /** \brief Specialization for fc_ (to allow special operations).
         */
        struct layer_fc : public layer {
            virtual unsigned long get_num_outputs() const = 0;
            virtual bool has_bias() const = 0;
            virtual bool add_bias() = 0;
        };
        template <typename FC>
        struct layer_fc_t : layer_fc {
            FC detail;

            template <typename... Args>
            layer_fc_t(Args&&... args)
                : detail(std::forward<Args>(args)...) {}

            layer_ptr copy_detail() const override {
                return std::make_unique<layer_fc_t>(detail);
            }

            dlib::tensor& get_layer_params() override {
                return detail.get_layer_params();
            }
            const dlib::tensor& get_layer_params() const override {
                return detail.get_layer_params();
            }

            unsigned long get_num_outputs() const override {
                return detail.get_num_outputs();
            }

            bool has_bias() const override {
                return layer_has_bias(detail);
            }
            bool add_bias() override {
                return layer_add_bias(detail);
            }
            
            void forward_const(dlib::tensor const* const* inputs,
                               std::size_t num_inputs) override {
                if (num_inputs != 1 || !inputs || !*inputs)
                    throw std::invalid_argument("single input expected");
                auto& out = allocate_output();
                detail.forward(tagged_input<0>(inputs), out);
            }

            json::array keras_array() const override {
                return layer_json(detail);
            }
            description layer_description() const override {
                using ulong = unsigned long;
                return { layer_type(detail),
                         layer_concise(detail),
                         ulong(layer_output_size(detail)),
                         ulong(layer_parameter_count(detail)) };
            }
            std::string code() const override {
                return layer_code(detail);
            }
            void serialize_detail(std::ostream& out) const override {
                serialize(detail, out);
            }
            dlibx::parameter_format parameter_format() const override {
                return serialize_format(detail);
            }
        };
        template <unsigned long K, dlib::fc_bias_mode BM>
        struct layer_regular<dlib::fc_<K,BM> > final
            : layer_fc_t<dlib::fc_<K,BM> > {
            using base = layer_fc_t<dlib::fc_<K,BM> >;
            using base::base;
        };
        template <unsigned long K, bias_mode BM>
        struct layer_regular<dlibx::fc_dynamic_<K,BM> > final
            : layer_fc_t<dlibx::fc_dynamic_<K,BM> > {
            using base = layer_fc_t<dlibx::fc_dynamic_<K,BM> >;
            using base::base;
        };
    }
}
