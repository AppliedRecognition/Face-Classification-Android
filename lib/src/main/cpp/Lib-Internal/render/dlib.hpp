#pragma once

#include "types.hpp"
#include "settings.hpp"

namespace render {
    struct diagnostics;
    
    namespace internal {
        raw_image::plane_ptr render_dlib(
            const raw_image::plane& raw_image,
            const detected_coordinates& pos,
            const render_settings& rsettings,
            const output_settings& osettings,
            diagnostics* = nullptr);

        void in_place_correct_lighting_dlib(
            const raw_image::plane& image,
            const raw_image::plane& visibility,
            const detected_coordinates& pos,
            const render_settings& rsettings,
            diagnostics* = nullptr);
    }
}
