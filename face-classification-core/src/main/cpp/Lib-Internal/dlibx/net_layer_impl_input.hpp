#pragma once

#include "net_layer_impl_common.hpp"
#include "dnn_input_generic_image.hpp"
#include "raw_image.hpp"

namespace dlibx {
    namespace net {

        // input layers

        template <typename T> const char* pixel_code();
        template <> constexpr const char*
        pixel_code<unsigned char>() { return "u8"; }
        template <> constexpr const char*
        pixel_code<float>() { return "float"; }
        template <> constexpr const char*
        pixel_code<dlib::rgb_pixel>() { return "rgb"; }
        template <> constexpr const char*
        pixel_code<dlib::rgb_alpha_pixel>() { return "rgba"; }

        template <typename IMAGE>
        auto layer_code(const dlibx::input_generic_image<IMAGE>&) {
            using PIXEL = typename dlib::image_traits<IMAGE>::pixel_type;
            return std::string("input_image_") + pixel_code<PIXEL>();
        }
        template <typename IMAGE>
        constexpr auto layer_concise(const dlibx::input_generic_image<IMAGE>&) {
            return "u8";
        }
        template <typename IMAGE>
        auto layer_output_size(const dlibx::input_generic_image<IMAGE>&) {
            using PIXEL = typename dlib::image_traits<IMAGE>::pixel_type;
            return unsigned(dlib::pixel_traits<PIXEL>::num);
        }

        template <typename T, typename MM>
        auto layer_code(const dlib::input<dlib::array2d<T,MM> >&) {
            return std::string("input_array2d_") + pixel_code<T>();
        }
        template <typename T, typename MM>
        auto layer_concise(const dlib::input<dlib::array2d<T,MM> >&) {
            return pixel_code<typename dlib::pixel_traits<T>::basic_pixel_type>();
        }
        template <typename T, typename MM>
        auto layer_output_size(const dlib::input<dlib::array2d<T,MM> >&) {
            return unsigned(dlib::pixel_traits<T>::num);
        }

        template <typename T, long NR, long NC, typename MM, typename L>
        auto layer_code(const dlib::input<dlib::matrix<T,NR,NC,MM,L> >&) {
            return std::string("input_matrix_") + pixel_code<T>();
        }
        template <typename T, long NR, long NC, typename MM, typename L>
        auto layer_concise(const dlib::input<dlib::matrix<T,NR,NC,MM,L> >&) {
            return pixel_code<typename dlib::pixel_traits<T>::basic_pixel_type>();
        }
        template <typename T, long NR, long NC, typename MM, typename L>
        auto layer_output_size(const dlib::input<dlib::matrix<T,NR,NC,MM,L> >&) {
            return unsigned(dlib::pixel_traits<T>::num);
        }

        template <typename T, long NR, long NC, typename MM, typename L, size_t K>
        auto layer_code(
            const dlib::input<std::array<dlib::matrix<T,NR,NC,MM,L>,K> >&) {
            return "input_array_" + std::to_string(K) + '_' + pixel_code<T>();
        }
        template <typename T, long NR, long NC, typename MM, typename L, size_t K>
        constexpr auto layer_concise(
            const dlib::input<std::array<dlib::matrix<T,NR,NC,MM,L>,K> >&) {
            return pixel_code<T>();
        }
        template <typename T, long NR, long NC, typename MM, typename L, size_t K>
        constexpr auto layer_output_size(
            const dlib::input<std::array<dlib::matrix<T,NR,NC,MM,L>,K> >&) {
            return K;
        }

        constexpr auto layer_code(const dlib::input_rgb_image&) {
            return "input_rgb_image";
        }
        constexpr auto layer_concise(const dlib::input_rgb_image&) {
            return "rgb";
        }
        constexpr auto layer_output_size(const dlib::input_rgb_image&) {
            return 3u;
        }
        template <std::size_t NR, std::size_t NC>
        auto layer_code(const dlib::input_rgb_image_sized<NR,NC>&) {
            std::stringstream ss;
            ss << "input_rgb_image_" << NR;
            if (NC != NR) ss << '_' << NC;
            return ss.str();
        }
        template <std::size_t NR, std::size_t NC>
        constexpr auto layer_concise(const dlib::input_rgb_image_sized<NR,NC>&) {
            return "rgb";
        }
        template <std::size_t NR, std::size_t NC>
        constexpr auto layer_output_size(const dlib::input_rgb_image_sized<NR,NC>&) {
            return 3u;
        }

        /** \brief Any type of input layer.
         */
        template <typename INPUT>
        struct layer_input final : public layer {
            INPUT detail;
            layer_input() = default;
            layer_input(const INPUT& detail) : detail(detail) {}

            layer_ptr copy_detail() const override {
                return std::make_unique<layer_input>(detail);
            }

            template <typename input_type, typename = void>
            struct to_tensor_helper {
                layer_input& obj;
                template <typename... Args>
                void operator()(Args&&...) const {
                    throw std::runtime_error("input layer does not support conversion from raw_image");
                }
            };

            template <typename input_type>
            struct to_tensor_helper<input_type, std::void_t<typename dlib::image_traits<input_type>::pixel_type> > {
                layer_input& obj;
                void operator()(
                    stdx::forward_iterator<const raw_image::plane&> first,
                    stdx::forward_iterator<const raw_image::plane&> last) {
                    using pixel_type = typename dlib::image_traits<input_type>::pixel_type;
                    std::vector<raw_image::fixed_dlib_image<pixel_type> > imgs;
                    imgs.reserve(std::size_t(distance(first,last)));
                    imgs.assign(first,last);
                    auto& out = obj.allocate_output();
                    obj.detail.to_tensor(imgs.data(), imgs.data()+imgs.size(), out);
                }
            };

            void to_tensor(
                stdx::forward_iterator<const raw_image::plane&> first,
                stdx::forward_iterator<const raw_image::plane&> last) override {
                to_tensor_helper<typename INPUT::input_type>{*this}(first,last);
            }

            json::object keras_object() const override {
                json::object config;
                // config["batch_input_shape"] = [null,150,150,3];
                config["dtype"] = "float32";
                config["sparse"] = false;
                json::object obj;
                obj["class_name"] = "InputLayer";
                obj["config"] = move(config);
                return obj;
            }
            std::string code() const override {
                return layer_code(detail);
            }
            description layer_description() const override {
                return { "input", // should this be layer_type(detail) ?
                         layer_concise(detail),
                         layer_output_size(detail),
                         0 };
            }
            void serialize_detail(std::ostream& out) const override {
                serialize(detail, out);
            }
        };
    }
}
