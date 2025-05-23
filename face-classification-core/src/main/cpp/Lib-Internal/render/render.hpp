#pragma once

#include "types.hpp"
#include "settings.hpp"
#include <stdext/arg.hpp>


namespace render {
    /** \brief Render detected face.
     *
     * May return nullptr or throw exception in case of failure.
     */
    raw_image::plane_ptr
    render_face(stdx::arg<core::context_data> data,
                stdx::arg<const raw_image::plane> raw_image,
                const face_coordinates& detected_face,
                const render_settings& rsettings,
                const output_settings& osettings,
                diagnostics* = nullptr);

    /** \brief Histogram equalization.
     *
     * Image must be GRAY8 or YUV.
     *
     * The equalization is based only on the pixels within
     * the largest inscribed ellipse, but it is applied to all pixels.
     */
    void in_place_equalize_histogram(stdx::arg<const raw_image::plane> img);
}
