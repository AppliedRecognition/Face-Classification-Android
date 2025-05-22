#pragma once

#include "point2.hpp"
#include "core.hpp"

namespace dlib {
    struct chip_details;
}

namespace raw_image {

    /** \brief Semantically the same as dlib::chip_details but with a
     * distinct implemention.
     *
     * Conversions to and from dlib::chip_details are provided but use
     * template methods to avoid requiring dlib in cases where dlib is
     * not otherwise needed.
     */
    class scaled_chip : public rotated_box {
    public:
        unsigned out_width, out_height;

        scaled_chip() : rotated_box{}, out_width(0), out_height(0) {}
        scaled_chip(const rotated_box& rbox,
                    unsigned out_width = 0, unsigned out_height = 0)
            : rotated_box{rbox}, out_width(out_width), out_height(out_height) {}

        scaled_chip(const scaled_chip&) = default;
        scaled_chip& operator=(const scaled_chip&) = default;

        template <typename T, typename = std::enable_if_t<std::is_same_v<T, dlib::chip_details> > >
        scaled_chip(const T& cd)
            : rotated_box {
                    point2f {
                        float(cd.rect.left() + cd.rect.right()) / 2,
                        float(cd.rect.top() + cd.rect.bottom()) / 2
                    },
                    float(cd.rect.width()),
                    float(cd.rect.height()),
                    float(cd.angle)
                },
              out_width(unsigned(cd.cols)),
              out_height(unsigned(cd.rows)) {
        }
        
        template <typename T, typename = std::enable_if_t<std::is_same_v<T, dlib::chip_details> > >
        operator T() const {
            T cd;
            cd.rect = {
                (2*center.x - width + 1) / 2,
                (2*center.y - height + 1) / 2,
                (2*center.x + width - 1) / 2,
                (2*center.y + height - 1) / 2
            };
            cd.angle = angle;
            cd.rows = out_height;
            cd.cols = out_width;
            return cd;
        }

        friend inline const rotated_box&
        to_rotated_box(const scaled_chip& chip) {
            return chip;
        }
    };


    /** \brief Face alignment using RetinaFace landmarks.
     *
     * The landmarks are expected to either be the 5 RetinaFace landmarks
     * (two eyes, tip of nose and corners of mouth), or the dlib68 landmarks.
     * In the latter case, only the 5 landmarks from the 68 that coinside
     * with the RetinaFace landmarks are used.
     *
     * The y_offset value allows the center of the face to be adjusted up
     * or down by a fraction of the distance between the eyes and mouth.
     *
     * If scale_factor is 1 and y_offset is 0, then the resulting box will be
     * such that the eyes are at the top edge and the mouth is at the
     * bottom edge.  Provide a larger value for the scale_factor to zoom out.
     */
    rotated_box
    retina_align(stdx::span<const raw_image::point2f> landmarks,
                 float scale_factor = 1, float y_offset = 0);


    /** \brief Like dlib::extract_image_chip() but with
     * raw_image and scaled_chip.
     */
    plane_ptr
    extract_image_chip(const multi_plane_arg& image,
                       const scaled_chip& cd,
                       pixel_layout layout);
}
