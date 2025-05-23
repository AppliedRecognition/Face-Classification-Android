#pragma once

#include "input_extractor.hpp"
#include "chip_details.hpp"
#include "raw_image.hpp"
#include <raw_image/point_rounding.hpp>

namespace dlibx {
    /** \brief Extract region of image centered on point between the eyes.
     *
     * This extractor does not rotate or scale the cropped region.
     */
    struct eyecrop_extractor final : input_extractor {
        eyecrop_extractor(std::string name, unsigned width, unsigned height,
                          raw_image::pixel_layout layout)
            : input_extractor(move(name), width, height, layout) {
        }

        raw_image::scaled_chip
        chip_from_pts(const std::vector<raw_image::point2f>& pts) const override;

        raw_image::plane_ptr
        extract_from_chip(const raw_image::multi_plane_arg& image,
                          const raw_image::scaled_chip& cd) const override;
    };

    /** \brief Decode extractor description string.
     *
     * Format is "eyecropWWWxHHHpixel" where
     *   WWW is width,
     *   HHH is height,
     *   pixel is one of "rgb", "yuv" or "gray".
     */
    std::tuple<unsigned, unsigned, raw_image::pixel_layout>
    eyecrop_decode(std::string_view name);

    std::unique_ptr<const input_extractor>
    eyecrop_factory(const std::string_view& name);
}
