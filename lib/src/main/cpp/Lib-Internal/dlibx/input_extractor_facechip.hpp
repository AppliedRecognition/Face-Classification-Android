#pragma once

#include "input_extractor.hpp"
#include "chip_details.hpp"
#include "raw_image.hpp"
#include <raw_image/point_rounding.hpp>

namespace dlibx {
    /** \brief Extractor using dlib::get_face_chip_details() for alignment.
     */
    struct facechip_extractor final : input_extractor {
        const float padding;

        facechip_extractor(std::string name, unsigned size, float padding,
                           raw_image::pixel_layout layout)
            : input_extractor(move(name), size, size, layout),
              padding(padding) {
        }

        raw_image::scaled_chip
        chip_from_pts(const std::vector<raw_image::point2f>& pts) const override;
    };

    /** \brief Decode extractor description string.
     *
     * Format is "facechipDIM[+-]PADpixel" where
     *   DIM is the integer dimension (both width and height),
     *   PAD is the floating point padding parameter (*see below), and
     *   pixel is one of "rgb", "yuv" or "gray".
     *
     * If PAD starts with a "0", then it is interpretted as "0.".
     * For example, "025" is actually "0.25".
     */
    std::tuple<unsigned, float, raw_image::pixel_layout>
    facechip_decode(std::string_view name);

    std::unique_ptr<const input_extractor>
    facechip_factory(const std::string_view& name);


    /** \brief Similar to facechip but draws landmarks on the image.
     *
     * This extractor only works with 68 landmark inputs and is suitable
     * for classifiers that assess landmark accuracy.
     *
     * This extractor also supports rgba output.  In this case, the
     * landmarks are drawn on the alpha channel plane with the rgb data
     * kept intact (same as what facechip would produce).
     */
    struct lm68chip_extractor final : input_extractor {
        const float padding;

        lm68chip_extractor(std::string name, unsigned size, float padding,
                           raw_image::pixel_layout layout)
            : input_extractor(move(name), size, size, layout),
              padding(padding) {
        }

        raw_image::scaled_chip
        chip_from_pts(const std::vector<raw_image::point2f>& pts) const override;

        raw_image::plane_ptr
        extract_from_pts(const raw_image::multi_plane_arg& image,
                         const std::vector<raw_image::point2f>& pts) const override;

        // throws an exception as the landmarks are required for extraction
        raw_image::plane_ptr
        extract_from_chip(const raw_image::multi_plane_arg& image,
                          const raw_image::scaled_chip& cd) const override;
    };

    /** \brief Decode extractor description string.
     *
     * Format is "lm68chipDIM[+-]PADpixel" where
     *   DIM is the integer dimension (both width and height),
     *   PAD is the floating point padding parameter (*see below), and
     *   pixel is one of "rgb", "yuv" or "gray".
     *
     * If PAD starts with a "0", then it is interpretted as "0.".
     * For example, "025" is actually "0.25".
     */
    std::tuple<unsigned, float, raw_image::pixel_layout>
    lm68chip_decode(std::string_view name);

    std::unique_ptr<const input_extractor>
    lm68chip_factory(const std::string_view& name);


    /** \brief Similar to facechip but suitable for extracting from depth map.
     *
     * Input must be pixel::a16_le.
     * Output is pixel::gray8.
     */
    struct facedepth_extractor final : input_extractor {
        const float padding;

        facedepth_extractor(std::string name, unsigned size, float padding)
            : input_extractor(move(name), size, size, raw_image::pixel::a8),
              padding(padding) {
        }

        raw_image::scaled_chip
        chip_from_pts(const std::vector<raw_image::point2f>& pts) const override;

        /// first step of depth extraction (returns pixel::a16_le image)
        raw_image::plane_ptr
        extract_depth_chip(const raw_image::multi_plane_arg& image,
                           const raw_image::scaled_chip& cd) const;

        // converts from pixel::a16_le to gray8 using
        // the formula: 200 + threshold - value
        // where threshold is the 1st percentile of the values
        // objects close to camera have high value
        // objects far away and holes have zero value
        static void normalize_depth(raw_image::plane& img);

        raw_image::plane_ptr
        extract_from_chip(const raw_image::multi_plane_arg& image,
                          const raw_image::scaled_chip& cd) const override;
    };

    /** \brief Decode extractor description string.
     *
     * Format is "facedepthDIM[+-]PAD" where
     *   DIM is the integer dimension (both width and height), and
     *   PAD is the floating point padding parameter (*see below).
     *
     * If PAD starts with a "0", then it is interpretted as "0.".
     * For example, "025" is actually "0.25".
     */
    std::tuple<unsigned, float>
    facedepth_decode(std::string_view name);

    std::unique_ptr<const input_extractor>
    facedepth_factory(const std::string_view& name);
}
