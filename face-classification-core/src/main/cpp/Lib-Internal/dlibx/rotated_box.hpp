#pragma once

#include <dlib/image_transforms/interpolation.h>
#include <raw_image/point2.hpp>
#include <stdext/span.hpp>

namespace dlib {
    inline auto to_rotated_box(const chip_details& chip) {
        const auto cx = chip.rect.left() + chip.rect.right();
        const auto cy = chip.rect.top() + chip.rect.bottom();
        return raw_image::rotated_box {
            { float(cx/2), float(cy/2) },
            float(chip.rect.width()),
            float(chip.rect.height()),
            float(chip.angle)
        };
    }
}

namespace raw_image {
    inline auto to_chip_details(const rotated_box& rbox) {
        dlib::chip_details chip;
        chip.rect = {
            (2*rbox.center.x - rbox.width + 1) / 2,
            (2*rbox.center.y - rbox.height + 1) / 2,
            (2*rbox.center.x + rbox.width - 1) / 2,
            (2*rbox.center.y + rbox.height - 1) / 2
        };
        chip.angle = rbox.angle;
        return chip;
    }
}

namespace dlibx {
    using fpoint = dlib::vector<float,2>;

    /** \brief Face alignment using RetinaFace landmarks.
     *
     * The landmarks are expected to either be the 5 RetinaFace landmarks
     * (two eyes, tip of nose and corners of mouth), or the dlib68 landmarks.
     * In the latter case, only the 5 landmarks produced from the 68 are used.
     *
     * The y_offset value allows the center of the face to be adjusted up
     * or down by a fraction of the distance between the eyes and mouth.
     *
     * If scale_factor is 1 and y_offset is 0, then the resulting box will be
     * such that the eyes are at the top edge and the mouth is at the
     * bottom edge.  Provide a larger value for the scale_factor to zoom out.
     */
    raw_image::rotated_box
    retina_align(stdx::span<const fpoint> landmarks,
                 float scale_factor = 1, float y_offset = 0);
}
