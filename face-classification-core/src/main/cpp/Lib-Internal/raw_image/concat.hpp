#pragma once

#include "core.hpp"
#include <cassert>
#include <cstring>
#include <stdexcept>

namespace raw_image {

    const auto justify_left =
        [](auto, auto) { return 0u; };
    const auto justify_right =
        [](auto dim, auto space) { return space - dim; };
    const auto justify_top =
        [](auto, auto) { return 0u; };
    const auto justify_bottom =
        [](auto dim, auto space) { return space - dim; };
    const auto justify_center =
        [](auto dim, auto space) { return (space - dim) / 2; };
    

    /** \brief Concatenate images horizontally.
     *
     * Height of destination image must be equal or greater than the height
     * of each source image.
     * Width of destination image must be equal or greater than the sum
     * of the widths of the source images.
     * pixel layout is converted as necessary to destination pixel layout.
     */
    template <typename ITER, typename JUSTIFY = decltype(justify_top)>
    void concat_horz(stdx::arg<const plane> dest, ITER first, ITER last,
                     const JUSTIFY& justify = justify_top,
                     unsigned padding = 0) {
        for (unsigned x = 0; first != last; ++first) {
            const auto img = single_plane_arg(*first);
            const auto y = justify(img->height, dest->height);
            copy_pixels(
                *img, crop(dest, x, y, img->width, img->height));
            x += img->width;
            x += padding;
        }
    }

    /** \brief Concatenate images horizontally.
     *
     * A new image of sufficient size is constructed and zeroed.
     * The pixel layout of the new image is the maximum (bytes per pixel)
     * of the pixel layouts of the source images.
     */
    template <typename ITER, typename JUSTIFY = decltype(justify_top)>
    plane_ptr concat_horz(ITER first, ITER last,
                          const JUSTIFY& justify = justify_top) {
        if (first == last) return nullptr;
        unsigned width = 0;
        unsigned height = 0;
        auto layout = pixel::gray8;
        for (auto it = first; it != last; ++it) {
            const auto img = single_plane_arg(*it);
            width += img->width;
            height = std::max(height, img->height);
            layout = std::max(layout, img->layout);
        }
        auto r = create(width, height, layout);
        memset(r->data, 0, r->height*r->bytes_per_line);
        concat_horz(*r, first, last, justify);
        return r;
    }


    /** \brief Concatenate images vertically.
     *
     * Width of destination image must be equal or greater than the width
     * of each source image.
     * Height of destination image must be equal or greater than the sum
     * of the heights of the source images.
     * pixel layout is converted as necessary to destination pixel layout.
     */
    template <typename ITER, typename JUSTIFY = decltype(justify_left)>
    void concat_vert(stdx::arg<const plane> dest, ITER first, ITER last,
                     const JUSTIFY& justify = justify_left,
                     unsigned padding = 0) {
        for (unsigned y = 0; first != last; ++first) {
            const auto img = single_plane_arg(*first);
            const auto x = justify(img->width, dest->width);
            copy_pixels(
                *img, crop(dest, x, y, img->width, img->height));
            y += img->height;
            y += padding;
        }
    }


    /** \brief Concatenate images vertically.
     *
     * A new image of sufficient size is constructed and zeroed.
     * The pixel layout of the new image is the maximum (bytes per pixel)
     * of the pixel layouts of the source images.
     */
    template <typename ITER, typename JUSTIFY = decltype(justify_left)>
    plane_ptr concat_vert(ITER first, ITER last,
                          const JUSTIFY& justify = justify_left) {
        if (first == last) return nullptr;
        unsigned width = 0;
        unsigned height = 0;
        auto layout = pixel::gray8;
        for (auto it = first; it != last; ++it) {
            const auto img = single_plane_arg(*it);
            height += img->height;
            width = std::max(width, img->width);
            layout = std::max(layout, img->layout);
        }
        auto r = create(width, height, layout);
        memset(r->data, 0, r->height*r->bytes_per_line);
        concat_vert(*r, first, last, justify);
        return r;
    }


    /** \brief Pad image to minimum width and height.
     *
     * If the input image already has both the minimum width and height,
     * then no new image is created (nothing is done).
     *
     * The returned tuple includes the x and y offset of the image.
     * That is, the left and top padding in pixels.
     */
    template <typename JUSTIFY_HORZ = decltype(justify_center),
              typename JUSTIFY_VERT = decltype(justify_center)>
    std::tuple<plane_ptr,unsigned,unsigned>
    pad_image(single_plane_arg src,
              unsigned min_width, unsigned min_height, int fill = 0,
              const JUSTIFY_HORZ& justify_horz = justify_center,
              const JUSTIFY_VERT& justify_vert = justify_center) {
        std::tuple<plane_ptr,unsigned,unsigned> r;
        if (src && (src->width < min_width || src->height < min_height)) {
            const auto w = std::max(src->width, min_width);
            const auto h = std::max(src->height, min_height);
            const auto x = std::get<1>(r) = justify_horz(src->width,w);
            const auto y = std::get<2>(r) = justify_vert(src->height,h);
            auto& dest =
                *(std::get<0>(r) = create(w, h, src->layout));
            if (0 <= fill)
                memset(dest.data, fill, dest.height*dest.bytes_per_line);
            copy_pixels(
                src, crop(dest, x, y, src->width, src->height));
        }
        return r;
    }


    /** \brief Make montage (grid) of same size images.
     *
     * All images must have the same width and height,
     * but may vary in layout.
     *
     * An attempt is made to find a number of rows and columns that
     * matches the specified aspect ratio.  The specific values of aspect_w
     * and aspect_h don't matter, only the ratio of them.
     *
     * In the case of a tie on aspect ratio, then an attempt is made to find
     * a number of rows and columns such as to minimize the empty spaces.
     * Ideally, rows * columns == num_images.
     *
     * The padding value is the number of pixels between images.
     */
    template <long aspect_w, long aspect_h, typename ITER>
    plane_ptr
    make_montage(ITER first, ITER last, unsigned padding = 0, int fill = 0) {
        static_assert(aspect_w > 0 && aspect_h > 0);
        if (first == last) return nullptr;

        const auto img0 = single_plane_arg(*first);
        auto layout = img0->layout;
        int n = 1;
        for (auto it = std::next(first); it != last; ++it, ++n) {
            const auto img = single_plane_arg(*it);
            if (img->width != img0->width || img->height != img0->height)
                throw std::invalid_argument("images must have same dimensions");
            layout = std::max(layout, img->layout);
        }
        const auto col_coeff = long(img0->width + padding) * aspect_h;
        const auto row_coeff = long(img0->height + padding) * aspect_w;
        
        int rows = 0, cols = 0;
        long aspect = n * (col_coeff + row_coeff);
        int empty_spaces = n;
        for (auto r = 1; r <= n; ++r) {
            const auto c = (n + r - 1) / r;
            assert(0 < c && c <= n);
            assert(n <= r*c);
            const auto e = r*c - n;
            if (c <= e) continue;
            const auto a = std::abs(c * col_coeff - r * row_coeff);
            if (aspect > a) {
                aspect = a;
                empty_spaces = e;
                rows = r, cols = c;
            }
            else if (aspect == a &&
                     empty_spaces > e) {
                empty_spaces = e;
                rows = r, cols = c;
            }
        }

        auto r = create(cols*(img0->width+padding) - padding,
                        rows*(img0->height+padding) - padding,
                        layout);
        if (fill >= 0)
            memset(r->data, fill, r->height*r->bytes_per_line);

        for (unsigned y_ofs = 0; rows > 0; --rows,
                 y_ofs += img0->height + padding) {
            unsigned x_ofs = 0;
            for (auto j = std::min(n,cols); j > 0; --j, --n, ++first,
                     x_ofs += img0->width + padding) {
                const auto roi = crop(
                    r, x_ofs, y_ofs, img0->width, img0->height);
                copy_pixels(*first, roi);
            }
        }
        assert(first == last);
        
        return r;
    }
}
