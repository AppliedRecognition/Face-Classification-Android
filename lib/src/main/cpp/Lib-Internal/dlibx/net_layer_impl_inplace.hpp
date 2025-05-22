#pragma once

#include "net_layer_impl_common.hpp"

#include "dnn_invdropout.hpp"
#include "dnn_lambda.hpp"
#include "dnn_batch_centering.hpp"
#include "dnn_sum_neighbours.hpp"
#include "dnn_prelu.hpp"
#include <dlib/dnn/layers.h>

// This file contains details for "in place" regular layers.

namespace dlibx {
    namespace net {

        // **** sig

        constexpr auto layer_code(const dlib::sig_&) {
            return "sig";
        }
        inline auto layer_json(const dlib::sig_&) {
            json::object config;
            //config["activation"] = "sig";
            //config["dtype"] = "float32";
            //config["trainable"] = true;
            json::object obj;
            //obj["class_name"] = "Activation";
            obj["config"] = move(config);
            return json::array(1,move(obj));
        }

        // **** softmax

        constexpr auto layer_code(const dlib::softmax_&) {
            return "softmax";
        }
        inline auto layer_json(const dlib::softmax_&) {
            json::object config;
            //config["activation"] = "softmax";
            //config["dtype"] = "float32";
            //config["trainable"] = true;
            json::object obj;
            //obj["class_name"] = "Activation";
            obj["config"] = move(config);
            return json::array(1,move(obj));
        }

        // **** relu

        constexpr auto layer_code(const dlib::relu_&) {
            return "relu";
        }
        inline auto layer_json(const dlib::relu_&) {
            json::object config;
            config["activation"] = "relu";
            config["dtype"] = "float32";
            config["trainable"] = true;
            json::object obj;
            obj["class_name"] = "Activation";
            obj["config"] = move(config);
            return json::array(1,move(obj));
        }

        // **** prelu

        constexpr auto layer_code(const dlib::prelu_&) {
            return "prelu";
        }
        inline auto layer_json(const dlib::prelu_&) {
            json::object config;
            config["activation"] = "prelu";
            config["dtype"] = "float32";
            config["trainable"] = true;
            json::object obj;
            obj["class_name"] = "Activation";
            obj["config"] = move(config);
            return json::array(1,move(obj));
        }

        constexpr auto layer_code(const dlibx::prelu_&) {
            return "prelu";
        }
        inline auto layer_json(const dlibx::prelu_&) {
            json::object config;
            config["activation"] = "prelu";
            config["dtype"] = "float32";
            config["trainable"] = true;
            json::object obj;
            obj["class_name"] = "Activation";
            obj["config"] = move(config);
            return json::array(1,move(obj));
        }


        // **** invdropout

        template <typename INIT>
        constexpr auto layer_code(const dlibx::invdropout_<INIT>&) {
            return "invdropout";
        }
        template <typename INIT>
        inline auto layer_concise(const dlibx::invdropout_<INIT>& item) {
            auto s = std::to_string(item.get_drop_rate());
            if (s.compare(0,2,"0.") == 0)
                s.erase(0,2);
            s.insert(0,"idrop");
            return s;
        }
        template <typename INIT>
        inline auto layer_json(const dlibx::invdropout_<INIT>& item) {
            json::object config;
            config["rate"] = item.get_drop_rate();
            config["trainable"] = true;
            json::object obj;
            obj["class_name"] = "Dropout"; // not sure if this is correct ?
            obj["config"] = move(config);
            return json::array(1,move(obj));
        }


        // **** dropout

        constexpr auto layer_code(const dlib::dropout_&) {
            return "dropout";
        }
        inline auto layer_concise(const dlib::dropout_& item) {
            auto s = std::to_string(item.get_drop_rate());
            if (s.compare(0,2,"0.") == 0)
                s.erase(0,1);
            s.insert(0,"drop");
            while (s.back() == '0')
                s.pop_back();
            return s;
        }
        inline auto layer_json(const dlib::dropout_& item) {
            json::object config;
            config["rate"] = item.get_drop_rate();
            config["trainable"] = true;
            json::object obj;
            obj["class_name"] = "Dropout"; // not sure if this is correct ?
            obj["config"] = move(config);
            return json::array(1,move(obj));
        }


        // **** multiply

        constexpr auto layer_code(const dlib::multiply_&) {
            return "multiply";
        }
        inline auto layer_concise(const dlib::multiply_& item) {
            auto s = std::to_string(item.get_multiply_value());
            unsigned d = 0, e = 0;
            for (auto c : s)
                if (c == '.') ++d;
                else if (c == 'e' || c == 'E') ++e;
            if (d == 1 && e == 0) {
                while (s.back() == '0') s.pop_back();
                if (s.back() == '.') s.pop_back();
            }
            s.insert(0,"scale");
            return s;
        }
        inline auto layer_json(const dlib::multiply_& m) {
            json::object config;
            config["arguments"] = json::object {
                { "scale", m.get_multiply_value() }
            };
            config["dtype"] = "float32";
            config["trainable"] = true;
            json::object obj;
            obj["class_name"] = "Lambda";  //"MultiplyLayer";
            obj["config"] = move(config);
            return json::array(1,move(obj));
        }


        // **** l2normalize

        constexpr auto layer_code(const dlib::l2normalize_&) {
            return "l2norm";
        }
        constexpr auto layer_concise(const dlib::l2normalize_&) {
            return "l2norm";
        }
        inline auto layer_json(const dlib::l2normalize_&) {
            json::object config;
            config["arguments"] = json::object {
                //{ "scale", m.get_multiply_value() }
            };
            config["dtype"] = "float32";
            config["trainable"] = true;
            json::object obj;
            obj["class_name"] = "Lambda";  //"L2Regularizer";
            obj["config"] = move(config);
            return json::array(1,move(obj));
        }


        // **** lambda

        template <typename... FUNCS>
        inline auto layer_code(const dlibx::lambda_<FUNCS...>&) {
            return dlibx::lambda_<FUNCS...>::name();
        }
        template <typename FUNC0>
        inline void lambda_csv_names(std::string& s) { s += FUNC0::name(); }
        template <typename FUNC0, typename FUNC1, typename... FUNCS>
        inline auto lambda_csv_names(std::string& s) {
            s += FUNC0::name();
            s += ',';
            lambda_csv_names<FUNC1,FUNCS...>(s);
        }
        template <typename... FUNCS>
        auto layer_concise(const dlibx::lambda_<FUNCS...>&) {
            auto s = std::string("lambda[");
            lambda_csv_names<FUNCS...>(s);
            s += ']';
            return s;
        }
        template <typename... FUNCS>
        auto layer_json(const dlibx::lambda_<FUNCS...>& item) {
            json::array arr;
            item.impl.visit_tail_first(
                [&](const auto& fn) {
                    json::object config;
                    auto args = fn.args();
                    if (!args.empty())
                        config["arguments"] = move(args);
                    config["function"] = fn.name();
                    config["function_type"] = "lambda";
                    config["output_shape_type"] = "raw";
                    config["trainable"] = true;
                    config["dtype"] = "float32";
                    config["module"] = "inception_resnet_v1";
                    json::object obj;
                    obj["class_name"] = "Lambda";
                    obj["config"] = move(config);
                    obj["name"] = fn.name();
                    arr.push_back(move(obj));
                });
            if (arr.size() == 1)
                get_object(arr.front()).erase("name");
            return arr;
        }


        // **** affine

        constexpr auto layer_code(const dlib::affine_&) {
            return "affine";
        }
        inline auto layer_json(const dlib::affine_&) {
            json::object config;
            config["dtype"] = "float32";
            config["trainable"] = true;
            json::object obj;
            obj["class_name"] = "ScaleLayer";
            obj["config"] = move(config);
            return json::array(1,move(obj));
        }


        // **** bn

        template <dlib::layer_mode mode>
        constexpr auto layer_code(const dlib::bn_<mode>&) {
            static_assert(mode == dlib::CONV_MODE || mode == dlib::FC_MODE);
            return mode == dlib::CONV_MODE ? "bncon" : "bnfc";
        }
        template <dlib::layer_mode mode>
        auto layer_json(const dlib::bn_<mode>&) {
            json::object config;
            config["axis"] = json::array{mode == dlib::CONV_MODE ? 3 : 1};
            config["center"] = true;
            config["epsilon"] = 0.001;
            config["momentum"] = 0.995;
            config["scale"] = false;
            config["dtype"] = "float32";
            config["trainable"] = true;
            json::object obj;
            obj["class_name"] = "BatchNormalization";
            obj["config"] = move(config);
            return json::array(1,move(obj));
        }


        // **** batch_centering

        constexpr inline auto layer_code(const dlibx::batch_centering_&) {
            return "bcenter";
        }
        inline auto layer_json(const dlibx::batch_centering_&) {
            json::object config;
            config["axis"] = json::array{1};
            config["center"] = true;
            config["epsilon"] = 0.001;
            config["momentum"] = 0.995;
            config["scale"] = false;
            config["dtype"] = "float32";
            config["trainable"] = true;
            json::object obj;
            obj["class_name"] = "BatchCentering";
            obj["config"] = move(config);
            return json::array(1,move(obj));
        }


        // **** sum_neighbours

        template <long SIZE>
        auto layer_code(const dlibx::sum_neighbours_<SIZE>&) {
            return "sum_neighbours_" + std::to_string(SIZE);
        }
        template <long SIZE>
        auto layer_json(const dlibx::sum_neighbours_<SIZE>&) {
            return json::array{};
        }


        /** \brief General in place capable layer.
         *
         * Any layer implementing the forward_inplace() method.
         */
        struct layer_inplace_base : layer {
        };
        template <typename DETAILS>
        struct layer_inplace final : layer_inplace_base {
            DETAILS detail;
            template <typename... Args>
            layer_inplace(Args&&... args)
                : detail(std::forward<Args>(args)...) {}

            layer_ptr copy_detail() const override {
                return std::make_unique<layer_inplace>(detail);
            }

            dlib::tensor& forward_inplace(dlib::tensor& in) override {
                detail.forward_inplace(in, in);
                return in;
            }
            
            void forward_const(dlib::tensor const* const* inputs,
                               std::size_t num_inputs) override {
                if (num_inputs != 1 || !inputs || !*inputs)
                    throw std::invalid_argument("single input expected");
                auto& in = **inputs;
                auto& out = allocate_output();
                out.copy_size(in);
                detail.forward_inplace(in, out);
            }

            dlib::tensor& get_layer_params() override {
                return detail.get_layer_params();
            }
            const dlib::tensor& get_layer_params() const override {
                return detail.get_layer_params();
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
            json::array keras_array() const override {
                return layer_json(detail);
            }
            void serialize_detail(std::ostream& out) const override {
                serialize(detail, out);
            }
        };
    }
}
