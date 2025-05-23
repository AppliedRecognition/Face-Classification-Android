#pragma once

#include "types.hpp"
#include <raw_image/drawing.hpp>
#include <raw_image/point_rounding.hpp>

namespace det {
    /** \brief Organize landmarks into several lines to be drawn on image.
     *
     * The result is a list of lines, where each line is a list of points.
     * The order of the lines and the details of what each line depicts
     * is an implementation detail and subject to change.
     * The purpose of this method is to allow visual verification that the
     * landmark detection process is working as expected.
     *
     * In the case where landmark detection has not been preformed
     * (dc.landmarks is empty), then a single line with 2 points 
     * connecting the estimated eye coordinates is returned.
     *
     * For the overload which accepts detected_coordinates,
     * extra landmarks are tolerated but not returned as long as the type
     * is correct.
     * For the overload that accepts a vector of landmarks,
     * the number of landmarks must be exactly 5, 68 or 77.
     */
    std::vector<std::vector<coordinate_type> >
    to_lines(const detected_coordinates& dc);
    std::vector<std::vector<coordinate_type> >
    to_lines(const stdx::span<const coordinate_type>& landmarks);
    template <typename ITER>
    inline auto to_lines(ITER first, ITER last) {
        std::vector<coordinate_type> vec;
        vec.reserve(std::distance(first,last));
        for ( ; first != last; ++first)
            vec.emplace_back(raw_image::round_from(*first));
        return to_lines(vec);
    }


    /** \brief Draw landmark lines on image.
     */
    void draw_lines(
        raw_image::single_plane_arg dest,
        const std::vector<std::vector<coordinate_type> >& lines,
        int line_size = 1,
        raw_image::pixel_color line_color = raw_image::color_white,
        int circle_size = 1,
        raw_image::pixel_color circle_color = raw_image::color_black);

    template <typename LM, typename... Args>
    inline void draw_lines(raw_image::single_plane_arg dest,
                           const LM& detected, Args&&... args) {
        draw_lines(dest, to_lines(detected), std::forward<Args>(args)...);
    }
                    
}
