#pragma once

#include "types.hpp"
#include <cmath>

namespace det {

    /** \brief Compute coordinate system relative to landmarks.
     *
     * The origin (0,0) is at the center point between the eyes.
     * A unit is the distance between the eyes.
     * The space is rotated such that the eyes are horizontal.
     * In this space, the eyes are located at (-0.5,0) and (+0.5,0).
     */
    template <typename PT = coordinate_type>
    struct landmark_standardize {
        static_assert(
            std::is_floating_point<decltype(std::declval<PT>().x)>::value,
            "floating point coordinates required");

        using point_type = PT;

        const point_type eye_left, eye_right;
        const point_type eye_center;
        const point_type eye_vec;
        const float eye_dist;
        const point_type right, down;

        static inline auto dot(point_type a, point_type b) {
            return a.x*b.x + a.y*b.y;
        }
        static inline auto rotate_clockwise_90(point_type p) {
            return point_type { -p.y, p.x };
        }

        landmark_standardize(point_type eye_left, point_type eye_right)
            : eye_left(eye_left),
              eye_right(eye_right),
              eye_center(0.5f*(eye_left+eye_right)),
              eye_vec(eye_right - eye_left),
              eye_dist(std::sqrt(dot(eye_vec,eye_vec))),
              right{eye_vec.x/eye_dist, eye_vec.y/eye_dist},
              down(rotate_clockwise_90(right)) {
        }

        /** \brief Map point from image space to landmark space.
         */
        inline point_type operator()(point_type p) const {
            p -= eye_center;
            p.x /= eye_dist;
            p.y /= eye_dist;
            return { dot(p,right), dot(p,down) };
        }

        /** \brief Reverse map point from landmark space back to image space.
         */
        inline point_type recover(float x, float y) const {
            return eye_center + eye_dist * (x*right + y*down);
        }
        template <typename T>
        inline point_type recover(const T& r) const {
            return recover(r.x,r.y);
        }
    };
}
