#pragma once

#include "types.hpp"
#include "settings.hpp"
#include <stdext/arg.hpp>
#include <array>

namespace render {

    /** \brief Align face model with detected landmarks.
     *
     * If focal_length == 0, then it is estimated as max(width,height).
     */
    face_alignment align_model(stdx::arg<core::context_data> data,
                               const face_coordinates& detected_face,
                               const image_size& size,
                               unsigned focal_length = 0);
    face_alignment align_model(stdx::arg<core::context_data> data,
                               const face_coordinates& detected_face,
                               stdx::arg<const raw_image::plane> raw_image,
                               unsigned focal_length = 0);

    
    /** \brief Estimate distance from camera focal point to tip of nose.
     *
     * This method is only accurate if the focal_length provided to
     * align_model() is correct.
     * An average interpupillary distance of 63mm is assumed.
     *
     * \return distance in meters
     */
    float estimate_distance(const face_alignment& alignment);

    
    /** \brief Render model in subject orientation.
     *
     * The returned coordinate is the offset which needs to be applied
     * to the subject landmarks to match the model.
     */
    std::pair<raw_image::plane_ptr,coordinate_type>
    render_model(stdx::arg<core::context_data> data,
                 const face_alignment& alignment);
    
    /** \brief Render frontalized face.
     *
     * The first of the pair is the frontalized image while 
     * the second is a GRAY8 image marking visibility.
     */
    std::pair<raw_image::plane_ptr,raw_image::plane_ptr>
    render_frontal(stdx::arg<core::context_data> data,
                   const face_coordinates& detected_face,
                   stdx::arg<const raw_image::plane> raw_image,
                   const face_alignment& alignment,
                   const render_settings& rsettings,
                   const output_settings& osettings,
                   diagnostics* = nullptr);

    
    /** \brief Mask out invisible regions.
     *
     * The visibility image must be GRAY8.
     * If pixel in visibility has value < threshold,
     * then corresponding pixel in image is set to color (little-endian).
     */
    void mask_visibility(raw_image::plane& image,
                         stdx::arg<const raw_image::plane> visibility,
                         unsigned threshold = 255,
                         std::array<unsigned char,4> color = {});
}
