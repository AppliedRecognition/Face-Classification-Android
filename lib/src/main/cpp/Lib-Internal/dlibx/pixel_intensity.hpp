#pragma once

#include <dlib/image_processing.h>
#include <raw_image/core.hpp>

namespace dlibx {
    template <typename T>
    struct pixel_intensity_base {
        virtual ~pixel_intensity_base() = default;
        virtual T operator()(long row, long col, T def = 0) const = 0;
    };

    template <typename T, typename image_type>
    struct pixel_intensity_helper final : public pixel_intensity_base<T> {
        const dlib::const_image_view<image_type> img;
        const dlib::rectangle area;

        explicit pixel_intensity_helper(const image_type& img)
            : img(img), area(dlib::get_rect(img)) {
        }

        T operator()(long row, long col, T def) const override {
            return area.contains(col,row) ?
                dlib::get_pixel_intensity(img[row][col]) : def;
        }
    };

    template <typename T>
    struct pixel_intensity_helper<T, raw_image::plane> final
        : public pixel_intensity_base<T> {

        const raw_image::plane img;
        const unsigned bpp;
        const decltype(gray8_from_pixel(img.layout)) g8;
        
        explicit pixel_intensity_helper(const raw_image::plane& img)
            : img(img),
              bpp(bytes_per_pixel(img.layout)),
              g8(gray8_from_pixel(img.layout)) {
        }

        T operator()(long row, long col, T def) const override {
            if (row >= 0 && unsigned(row) < img.height &&
                col >= 0 && unsigned(col) < img.width)
                return g8(img.data + row*img.bytes_per_line + col*bpp);
            else
                return def;
        }
    };
}
