#pragma once

#include "point3.hpp"
#include <stdexcept>
#include <vector>

namespace dlib {
    class tensor;
}

namespace raw_image {

    /** \brief Helper for dense 3d landmark model.
     *
     * The dense landmark model takes a 120x120 RGB chip as input and
     * has two sections.  The first section computes the following values:
     *   + 12 element camera matrix (rotation and translation)
     *   + 40 element base descriptor representing the 3d shape of the subject
     *   + 10 element facial expression descriptor
     *
     * The second section of the model then takes the two descriptors and
     * produces either the 30,000+ dense landmarks or the standard 68 landmarks.
     * Note that the 68 landmarks are a subset of the larger set.
     *
     * The landmarks produced must be aligned using the camera matrix to
     * account for the subject's pose and get landmarks relative to the
     * 120x120 chip.

     * The landmarks may need to be further translated to be relative to
     * the larger image the chip came from.  This is outside the scope of
     * this object.
     *
     * In addtion to landmark alignment, an accurate assessment of the
     * subject's yaw, pitch and roll can be computed from the camera matrix.
     *
     * (ECCV 2020) Towards Fast, Accurate and Stable 3D Dense Face Alignment
     * https://github.com/cleardusk/3DDFA_V2
     * https://github.com/1996scarlet/Dense-Head-Pose-Estimation
     */
    class dense_landmark_align {
        template <typename TENSOR>
        static constexpr auto is_tensor =
            std::is_convertible_v<const TENSOR&, const dlib::tensor&>;

    public:
        static constexpr auto chip_width = 120;
        static constexpr auto chip_height = 120;

        matrix3x3f rotation;
        point3f translation;

        dense_landmark_align() = default;

        /** \brief Construct from 12 floats.
         *
         * Rotation | Translation
         *  0  1  2 |   3
         *  4  5  6 |   7
         *  8  9 10 |  11
         */
        dense_landmark_align(const float* matrix)
            : rotation {
                    point3f{ matrix[0], matrix[1], matrix[2] },
                    point3f{ matrix[4], matrix[5], matrix[6] },
                    point3f{ matrix[8], matrix[9], matrix[10] },
                },
              translation { matrix[3], matrix[7], matrix[11] } {
        }
        template <typename TENSOR, typename =
                  std::enable_if_t<is_tensor<TENSOR> > >
        dense_landmark_align(const TENSOR& t)
            : dense_landmark_align(t.host()) {
            if (t.size() != 12 && t.size() != 62)
                throw std::invalid_argument("tensor has incorrect size");
        }


        /** \brief Yaw, pitch and roll.
         */
        std::array<float,3> yaw_pitch_roll_radians() const {
            const auto z = rotation.rows[2].z;
            return {
                std::atan2(rotation.rows[0].z, z),
                std::atan2(rotation.rows[1].z, z),
                std::atan2(rotation.rows[0].y, rotation.rows[1].y)
            };
        }
        std::array<float,3> yaw_pitch_roll_degrees() const {
            auto a = yaw_pitch_roll_radians();
            for (auto& x : a) x *= float(180/M_PI);
            return a;
        }


        /** \brief Align points to chip.
         *
         * Resulting x and y coordinates will be in [0,1] cooresponding
         * to position within 120x120 chip fed into model.
         *
         * The point3f -> point3f overload can operate in place (dest == src).
         */
        void align_to(const point3f* src, unsigned n, point3f* dest) const {
            for ( ; 0 < n; --n, ++src, ++dest) {
                *dest = translation + rotation * *src;
                dest->x = (dest->x - 1.0f) / chip_width;
                dest->y = 1.0f - dest->y / chip_height;
            }
        }
        template <typename TENSOR>
        std::enable_if_t<is_tensor<TENSOR> >
        align_to(const TENSOR& src, point3f* dest) const {
            if (src.size() % 3 != 0)
                throw std::invalid_argument("tensor has incorrect size");
            static_assert(sizeof(point3f) == 3*sizeof(float));
            align_to(reinterpret_cast<const point3f*>(src.host()),
                     src.size()/3, dest);
        }

        void align_to(const point3f* src, unsigned n, point2f* dest) const {
            for ( ; 0 < n; --n, ++src, ++dest) {
                auto x = translation.x + dot(rotation.rows[0], *src);
                auto y = translation.y + dot(rotation.rows[1], *src);
                *dest = {
                    (x - 1.0f) / chip_width,
                    1.0f - y / chip_height
                };
            }
        }
        template <typename TENSOR>
        std::enable_if_t<is_tensor<TENSOR> >
        align_to(const TENSOR& src, point2f* dest) const {
            if (src.size() % 3 != 0)
                throw std::invalid_argument("tensor has incorrect size");
            static_assert(sizeof(point3f) == 3*sizeof(float));
            align_to(reinterpret_cast<const point3f*>(src.host()),
                     src.size()/3, dest);
        }

        template <typename TENSOR>
        std::enable_if_t<is_tensor<TENSOR>, std::vector<point2f> >
        align2d(const TENSOR& src) const {
            std::vector<point2f> vec(src.size()/3);
            align_to(src, vec.data());
            return vec;
        }
    };
}
