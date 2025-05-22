#pragma once

#include "input_extractor.hpp"

namespace raw_image {
    /** \brief Extractor using the 5 landmarks detected by RetinaFace.
     */
    struct retina_extractor final : input_extractor {
        const float scale;
        const float yoffset;

        retina_extractor(std::string name, unsigned size,
                         float scale, float yoffset,
                         pixel_layout layout)
            : input_extractor(move(name), size, size, layout),
              scale(scale), yoffset(yoffset) {
        }

        /** \brief Chip details from either RetinaFace or dlib68 landmarks.
         *
         * The input may be either the 7 landmarks from RetinaFace or
         * the dlib68 set of landmarks.
         * Note that the RetinaFace detector (v7) returns the 7 landmarks:
         * eyes, nose, mouth and bounding box corners.
         * Even though only the first 5 are used, they must all be provided
         * to distinguish them from the the dlib5 landmarks (which cannot
         * be used here).
         */
        scaled_chip
        chip_from_pts(const std::vector<point2f>& pts) const override;

        // converts from pixel::a16_le to pixel::a8 using
        // the formula: 200 + threshold - value
        // where threshold is the 1st percentile of the values
        // objects close to camera have high value
        // objects far away and holes have zero value
        static void normalize_depth(plane& img);

        /** \brief Handle extraction including depth for rgbd output.
         *
         * For rgb, yuv or gray output this method is the same as
         * the base version.
         *
         * For rgbd output, the input multi-plane image must include
         * a depth channel in pixel::a16_le format.
         * The actual output is pixel::rgba32 with depth data in
         * the alpha channel.
         */
        plane_ptr
        extract_from_chip(const multi_plane_arg& image,
                          const scaled_chip& cd) const override;
    };

    /** \brief Decode extractor description string.
     *
     * Format is "retinaDIM*SCALE+YOFSpixel" where
     *   DIM is the integer dimension (both width and height),
     *   SCALE is the multiple of the eye to mouth distance,
     *   YOFS is the vertical center of face offset, and
     *   pixel is one of "gray", "yuv", "rgb", "rgbd" or "depth".
     *
     * For 3d output, specify "rgbd" output and include a pixel::a16_le
     * input plane. The actual output is pixel::rgba32.
     * Alternatively, if output is "depth" and assuming a pixel::a16_le
     * input plane, then only the depth channel is output with layout
     * pixel::a8;
     *
     * E.g. "retina112*2.95+0.35rgb"
     */
    std::tuple<unsigned, float, float, pixel_layout>
    retina_decode(std::string_view name);

    std::unique_ptr<const input_extractor>
    retina_factory(const std::string_view& name);
}
