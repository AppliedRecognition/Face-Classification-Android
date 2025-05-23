#pragma once

#include "types.hpp"
#include "point_rounding.hpp"

namespace raw_image {

    /** \brief Translate "original" point to scaled and rotated raw_image.
     *
     * The "original" image is the raw_image after scale and rotate.
     */
    template <typename PT>
    PT to_image_point(const PT& p, const plane& image) {
        const auto r = round_from(p);
        auto x = r.x();
        auto y = r.y();
        const auto w1 = decltype(x)(image.width - 1);
        const auto h1 = decltype(y)(image.height - 1);
        if (image.scale) {
            x = stdx::round_from(std::ldexp(x, -image.scale));
            y = stdx::round_from(std::ldexp(y, -image.scale));
        }
        if (image.rotate & 1) {
            std::swap(x,y);
            x = w1 - x;
        }
        if (image.rotate & 2) {
            x = w1 - x;
            y = h1 - y;
        }
        if (image.rotate & 4)
            x = w1 - x;
        return { x, y };
    }

    /** \brief Translate point on scaled and rotated raw_image
     * to "original" location.
     *
     * The "original" image is the raw_image after scale and rotate.
     */
    template <typename PT>
    PT to_original_point(const PT& p, const plane& image) {
        const auto r = round_from(p);
        auto x = r.x();
        auto y = r.y();
        const auto w1 = decltype(x)(image.width - 1);
        const auto h1 = decltype(y)(image.height - 1);
        if (image.rotate & 4)
            x = w1 - x;
        if (image.rotate & 2) {
            x = w1 - x;
            y = h1 - y;
        }
        if (image.rotate & 1) {
            x = w1 - x;
            std::swap(x,y);
        }
        if (image.scale) {
            x = stdx::round_from(std::ldexp(x, image.scale));
            y = stdx::round_from(std::ldexp(y, image.scale));
        }
        return { x, y };
    }

}


