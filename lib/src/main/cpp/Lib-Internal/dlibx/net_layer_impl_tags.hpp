#pragma once

#include "net_layer_impl_common.hpp"
#include "dnn_add_cropped.hpp"

// Layers that require multiple inputs (tags).

namespace dlibx {
    namespace net {

        template <std::size_t COUNT, template<typename> class... TAGS>
        struct dlib_concat_;

        template <template<typename> class... TAGS>
        struct dlib_concat_<0, TAGS...> {
            using type = dlib::concat_<TAGS...>;
        };

        template <std::size_t COUNT, template<typename> class... TAGS>
        struct dlib_concat_ {
            template <typename SUBNET>
            using tag = tagged_input_<COUNT-1,SUBNET>;
            using type = typename dlib_concat_<COUNT-1, TAGS..., tag>::type;
        };

        /** \brief Specialization for concat.
         */
        template <std::size_t COUNT>
        struct layer_concat : public layer {
            typename dlib_concat_<COUNT>::type detail;

            layer_concat() = default;

            layer_ptr copy_detail() const override {
                return std::make_unique<layer_concat>();
            }

            void forward_const(dlib::tensor const* const* inputs,
                               std::size_t num_inputs) override {
                if (num_inputs != COUNT || !inputs)
                    throw std::invalid_argument(
                        "incorrect number of inputs to concat");
                auto& out = allocate_output();
                detail.forward(tagged_input<COUNT-1>(inputs), out);
            }

            json::object keras_object() const override {
                json::object config;
                config["axis"] = 3;
                config["dtype"] = "float32";
                config["trainable"] = true;
                json::object obj;
                obj["class_name"] = "Concatenate";
                obj["config"] = config;
                return obj;
            }
            std::string code() const override {
                return "concat_" + std::to_string(COUNT);
            }
            description layer_description() const override {
                return { "concat", "concat", 0, 0 };
            }
        };
        template <template<typename> class... TAGS>
        struct layer_regular<dlib::concat_<TAGS...> > final
            : layer_concat<sizeof...(TAGS)> {
            layer_regular(const dlib::concat_<TAGS...>&) {}
        };


        /** \brief Specialization for add_cropped.
         */
        struct layer_add_cropped : public layer {
            dlibx::add_cropped_<input_tag_0> detail;

            layer_add_cropped() = default;

            layer_ptr copy_detail() const override {
                return std::make_unique<layer_add_cropped>();
            }

            void forward_const(dlib::tensor const* const* inputs,
                               std::size_t num_inputs) override {
                if (num_inputs != 2 || !inputs)
                    throw std::invalid_argument("add_cropped requires 2 inputs");
                auto& out = allocate_output();
                detail.forward(tagged_input<1>(inputs), out);
            }

            json::object keras_object() const override {
                json::object config;
                config["dtype"] = "float32";
                config["trainable"] = true;
                json::object obj;
                obj["class_name"] = "AddCropped";
                obj["config"] = move(config);
                return obj;
            }
            std::string code() const override {
                return "add_cropped";
            }
            description layer_description() const override {
                return { "addcrop", "addcrop", 0, 0 };
            }
        };
        template <template<typename> class TAG>
        struct layer_regular<dlibx::add_cropped_<TAG> > final : layer_add_cropped {
            layer_regular(const dlibx::add_cropped_<TAG>&) {}
        };


        /** \brief Specialization for add_prev.
         */
        struct layer_add_prev : public layer {
            dlib::add_prev_<input_tag_0> detail;

            layer_add_prev() = default;

            layer_ptr copy_detail() const override {
                return std::make_unique<layer_add_prev>();
            }

            void forward_const(dlib::tensor const* const* inputs,
                               std::size_t num_inputs) override {
                if (num_inputs != 2 || !inputs)
                    throw std::invalid_argument("add_prev requires 2 inputs");
                auto& out = allocate_output();
                detail.forward(tagged_input<1>(inputs), out);
            }

            json::object keras_object() const override {
                json::object config;
                config["dtype"] = "float32";
                config["trainable"] = true;
                json::object obj;
                obj["class_name"] = "Add";
                obj["config"] = move(config);
                return obj;
            }
            std::string code() const override {
                return "add_prev";
            }
            description layer_description() const override {
                return { "add", "add", 0, 0 };
            }
        };
        template <template<typename> class TAG>
        struct layer_regular<dlib::add_prev_<TAG> > final : layer_add_prev {
            layer_regular(const dlib::add_prev_<TAG>&) {}
        };


        /** \brief Specialization for mult_prev.
         */
        struct layer_mult_prev : public layer {
            dlib::mult_prev_<input_tag_0> detail;

            layer_mult_prev() = default;

            layer_ptr copy_detail() const override {
                return std::make_unique<layer_mult_prev>();
            }

            void forward_const(dlib::tensor const* const* inputs,
                               std::size_t num_inputs) override {
                if (num_inputs != 2 || !inputs)
                    throw std::invalid_argument("mult_prev requires 2 inputs");
                auto& out = allocate_output();
                detail.forward(tagged_input<1>(inputs), out);
            }

            json::object keras_object() const override {
                json::object config;
                config["dtype"] = "float32";
                config["trainable"] = true;
                json::object obj;
                obj["class_name"] = "Mult";
                obj["config"] = move(config);
                return obj;
            }
            std::string code() const override {
                return "mult_prev";
            }
            description layer_description() const override {
                return { "mult", "mult", 0, 0 };
            }
        };
        template <template<typename> class TAG>
        struct layer_regular<dlib::mult_prev_<TAG> > final : layer_mult_prev {
            layer_regular(const dlib::mult_prev_<TAG>&) {}
        };
    }
}
