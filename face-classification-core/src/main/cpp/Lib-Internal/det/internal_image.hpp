#pragma once

#include "internal.hpp"

#include <raw_image/core.hpp>

namespace det {
    struct image_struct {
        struct color_record {
            std::vector<raw_image::plane> multiplane;
            std::unique_ptr<const raw_image::plane> u;
            std::shared_ptr<const raw_image::plane> s;

            template <typename T>
            color_record(const T& img, std::enable_if_t<std::is_same_v<T,raw_image::multi_plane_arg> >* = nullptr)
                : multiplane(img.begin(), img.end()) {
            }
            color_record(const raw_image::plane& img)
                : multiplane{img} {}
            color_record(std::unique_ptr<raw_image::plane> img)
                : multiplane{*img}, u(move(img)) {
            }
            color_record(std::shared_ptr<const raw_image::plane> img)
                : multiplane{*img}, s(move(img)) {
            }
        };
        color_record color;

        struct gray_record {
            raw_image::plane plane;
            std::unique_ptr<const raw_image::plane> u;

            gray_record(const raw_image::plane& img) : plane(img) {}
            gray_record(std::unique_ptr<raw_image::plane> img)
                : plane(*img), u(move(img)) {
            }
        };
        gray_record gray;

        raw_image::image_size size;

        template <typename T>
        image_struct(const T& img, std::enable_if_t<std::is_same_v<T,raw_image::multi_plane_arg> >* = nullptr);

        image_struct(const raw_image::plane& img);
        image_struct(std::unique_ptr<raw_image::plane> img);
        image_struct(std::shared_ptr<const raw_image::plane> img);

        template <typename C, typename G>
        image_struct(C&& color, G&& gray);
    };
}
