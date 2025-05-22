#pragma once

#include <det/types.hpp>

namespace core {
    class context;
    struct context_data;
}

namespace render {
    using raw_image::image_size;
    using det::coordinate_type;
    using det::detected_coordinates;
    using det::face_coordinates;
    struct diagnostics;

    struct face_alignment : det::face_pose_type {
        float tx, ty, tz;
        unsigned focal_length;    // pixels
        coordinate_type image_center;
    };
}
