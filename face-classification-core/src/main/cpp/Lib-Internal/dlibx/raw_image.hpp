#pragma once

#include <raw_image/core.hpp>
#include <dlib/pixel.h>
#include <dlib/image_processing/generic_image.h>
#include <dlib/matrix/matrix_mat.h>
#include <stdext/rounding.hpp>
#include <stdext/convert.hpp>
#include <stdexcept>

namespace dlib {
    struct chip_details;
}
namespace raw_image {
    /// rgb pixel from gray8 image
    struct rgb_from_gray8 {
        union {
            unsigned char red;
            unsigned char green;
            unsigned char blue;
        };
        inline operator dlib::rgb_pixel() const {
            return { red, green, blue };
        }
        inline operator dlib::bgr_pixel() const {
            return { blue, green, red };
        }
    };
    static_assert(sizeof(rgb_from_gray8) == 1);

    template <typename pixel_type>
    raw_image::pixel_layout to_layout();
    template <> constexpr raw_image::pixel_layout
    to_layout<uint8_t>() { return pixel::gray8; }
    template <> constexpr raw_image::pixel_layout
    to_layout<rgb_from_gray8>() { return pixel::gray8; }
    template <> constexpr raw_image::pixel_layout
    to_layout<uint16_t>() { return pixel::a16_le; }
    template <> constexpr raw_image::pixel_layout
    to_layout<dlib::rgb_pixel>() { return pixel::rgb24; }
    template <> constexpr raw_image::pixel_layout
    to_layout<dlib::bgr_pixel>() { return pixel::bgr24; }
    template <> constexpr raw_image::pixel_layout
    to_layout<dlib::rgb_alpha_pixel>() { return pixel::rgba32; }

    template <> inline raw_image::pixel_layout to_layout<float>() {
        throw std::logic_error("cannot form raw_image with float pixels");
    }


    /** \brief Dlib compatible image with raw_image storage.
     */
    template <typename pixel_type>
    class dlib_image {
        plane_ptr raw;

    public:
        dlib_image() = default;
        dlib_image(long rows, long cols) {
            if (rows > 0 && cols > 0)
                raw = create(
                    stdx::round_from(cols), stdx::round_from(rows),
                    to_layout<pixel_type>());
        }

        operator plane_ptr() && { return std::move(raw); }
        
        inline long nc() const { return raw ? raw->width : 0; }
        inline long nr() const { return raw ? raw->height : 0; }

        pixel_type& operator()(long row, long col) {
            static constexpr auto bpp = long(sizeof(pixel_type));
            return *reinterpret_cast<pixel_type*>(
                raw->data + row*long(raw->bytes_per_line) + col*bpp);
        }
        const pixel_type& operator()(long row, long col) const {
            static constexpr auto bpp = long(sizeof(pixel_type));
            return *reinterpret_cast<const pixel_type*>(
                raw->data + row*long(raw->bytes_per_line) + col*bpp);
        }

        friend long num_rows(const dlib_image& img) {
            return img.nr();
        }
        friend long num_columns(const dlib_image& img) {
            return img.nc();
        }
        friend long width_step(const dlib_image& img) {
            return img.raw ? img.raw->bytes_per_line : 0;
        }
        friend void* image_data(dlib_image& img) {
            return img.raw ? img.raw->data : nullptr;
        }
        friend const void* image_data(const dlib_image& img) {
            return img.raw ? img.raw->data : nullptr;
        }
        friend void set_image_size(dlib_image& img, long rows, long cols) {
            if (rows > 0 && cols > 0 && (rows != img.nr() || cols != img.nc()))
                img.raw = create(
                    stdx::round_from(cols), stdx::round_from(rows),
                    to_layout<pixel_type>());
        }
        friend void swap(dlib_image& a, dlib_image& b) {
            std::swap(a.raw, b.raw);
        }
        friend const auto& to_raw_image(const dlib_image& img) {
            return *img.raw;
        }
    };


    /** \brief Dlib compatible image wrapper for raw_image.
     *
     * With this object the image metadata is fixed, but
     * the pixels may be modified.
     */
    template <typename pixel_type>
    class fixed_dlib_image {
        plane raw;

        static auto& verify_image(const plane& raw) {
            if (sizeof(pixel_type) == 1) {
                if (bytes_per_pixel(raw.layout) != 1)
                    throw std::invalid_argument(
                        "fixed_dlib_image: expected 1 byte per pixel"
                        " but image has " + to_string(raw.layout) + " pixels");
            }
            else if (raw.layout != to_layout<pixel_type>())
                throw std::invalid_argument(
                    "fixed_dlib_image: expected " +
                    to_string(to_layout<pixel_type>()) +
                    " but image has " + to_string(raw.layout) + " pixels");
            return raw;
        }

    public:
        fixed_dlib_image() = default;
        fixed_dlib_image(const fixed_dlib_image&) = default;
        fixed_dlib_image& operator=(const fixed_dlib_image&) = default;

        fixed_dlib_image(single_plane_arg rawp)
            : raw(verify_image(*rawp)) {
        }
        fixed_dlib_image& operator=(single_plane_arg rawp) {
            raw = verify_image(*rawp);
            return *this;
        }

        inline auto nc() const { return long(raw.width); }
        inline auto nr() const { return long(raw.height); }

        pixel_type& operator()(long row, long col) {
            static constexpr auto bpp = long(sizeof(pixel_type));
            return *reinterpret_cast<pixel_type*>(
                raw.data + row*long(raw.bytes_per_line) + col*bpp);
        }
        const pixel_type& operator()(long row, long col) const {
            static constexpr auto bpp = long(sizeof(pixel_type));
            return *reinterpret_cast<const pixel_type*>(
                raw.data + row*long(raw.bytes_per_line) + col*bpp);
        }

        friend auto num_rows(const fixed_dlib_image& img) {
            return img.nr();
        }
        friend auto num_columns(const fixed_dlib_image& img) {
            return img.nc();
        }
        friend auto width_step(const fixed_dlib_image& img) {
            return long(img.raw.bytes_per_line);
        }
        friend void* image_data(fixed_dlib_image& img) {
            return img.raw.data;
        }
        friend const void* image_data(const fixed_dlib_image& img) {
            return img.raw.data;
        }
        friend void set_image_size(
            const fixed_dlib_image& img, long rows, long cols) {
            if (rows != img.nr() || cols != img.nc())
                throw std::runtime_error("cannot change fixed_dlib_image dimensions");
        }
        friend void swap(fixed_dlib_image& a, fixed_dlib_image& b) {
            std::swap(a.raw, b.raw);
        }
        friend const auto& to_raw_image(const fixed_dlib_image& img) {
            return img.raw;
        }
    };


    /** \brief Dlib matrix_exp object from raw_image.
     */
    template <typename pixel_type>
    auto mat(single_plane_arg rawp) {
        // note: the temporary fixed_dlib_image object is used to
        // construct a dlib::image_view object which holds the pointer
        // to the pixels and image dimensions
        return dlib::mat(fixed_dlib_image<pixel_type>(rawp));
    }


    /** \brief Like dlib::extract_image_chip but works with raw_image.
     */
    plane_ptr
    extract_image_chip(const multi_plane_arg& image,
                       const dlib::chip_details& cd,
                       pixel_layout layout);
}

namespace dlib {
    /** \brief Create raw_image::plane from dlib compatible image.
     *
     * The raw_image::plane shares the same pixels as the source object.
     * Lifetime of the pixel data is managed by the source object.
     */
    template <typename image_type>
    auto to_raw_image(const image_type& image) {
        raw_image::plane r;
        using pixel_type = typename dlib::image_traits<image_type>::pixel_type;
        r.layout = raw_image::to_layout<pixel_type>();
        r.width = stdx::convert_from(num_columns(image));
        r.height = stdx::convert_from(num_rows(image));
        r.bytes_per_line = stdx::convert_from(width_step(image));
        r.data = static_cast<unsigned char*>(
            const_cast<void*>(image_data(image)));
        return r;
    }


    template <>
    struct pixel_traits<raw_image::rgb_from_gray8> {
        static constexpr bool rgb  = true;
        static constexpr bool rgb_alpha  = false;
        static constexpr bool grayscale = false;
        static constexpr bool hsi = false;
        static constexpr bool lab = false;
        enum { num = 1 };
        using basic_pixel_type = unsigned char;
        static constexpr basic_pixel_type min() { return 0; }
        static constexpr basic_pixel_type max() { return 255; }
        static constexpr bool is_unsigned = true;
        static constexpr bool has_alpha = false;
    };
    template <typename PT> 
    struct image_traits<raw_image::dlib_image<PT> > {
        using pixel_type = PT;
    };
    template <typename PT> 
    struct image_traits<raw_image::fixed_dlib_image<PT> > {
        using pixel_type = PT;
    };
}
