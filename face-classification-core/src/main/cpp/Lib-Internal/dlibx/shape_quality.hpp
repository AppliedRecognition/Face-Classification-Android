#pragma once

#include <dlib/image_processing/full_object_detection.h>
#include "pixel_intensity.hpp"
#include <raw_image/point_rounding.hpp>

namespace dlibx {

    using fpoint = dlib::vector<float,2>;

    
    /** \brief Extract feature pixels for landmark quality assessment.
     */
    raw_image::plane_ptr shape_extract_pixels(
        const pixel_intensity_base<unsigned char>& pi,
        const std::vector<fpoint>& pts);

    /** \brief Extract feature pixels for landmark quality assessment.
     */
    template <typename image_type>
    raw_image::plane_ptr shape_extract_pixels(
        const image_type& image,
        const dlib::full_object_detection& obj) {

        auto n = obj.num_parts();
        std::vector<fpoint> pts;
        pts.reserve(n);
        for (decltype(n) i = 0; i < n; ++i)
            pts.push_back(raw_image::round_from(obj.part(i)));
        const pixel_intensity_helper<unsigned char, image_type> pi(image);
        return shape_extract_pixels(pi, pts);
    }

    /** \brief Extract feature pixels for landmark quality assessment.
     */
    template <typename image_type, typename ITER>
    raw_image::plane_ptr shape_extract_pixels(
        const image_type& image,
        ITER shape_point_first, ITER shape_point_last) {

        std::vector<fpoint> pts;
        pts.reserve(68);
        for ( ; shape_point_first != shape_point_last; ++shape_point_first)
            pts.push_back(raw_image::round_from(*shape_point_first));
        const pixel_intensity_helper<unsigned char, image_type> pi(image);
        return shape_extract_pixels(pi, pts);
    }
    

    /** \brief Landmark quality assessment from feature pixels.
     */
    float shape_quality(const raw_image::plane& feature_pixels);

    /** \brief Landmark quality assessment.
     */
    template <typename image_type, typename ITER>
    float shape_quality(const image_type& image,
                        ITER shape_point_first, ITER shape_point_last) {
        return shape_quality(
            *shape_extract_pixels(image, shape_point_first, shape_point_last));
    }
}
