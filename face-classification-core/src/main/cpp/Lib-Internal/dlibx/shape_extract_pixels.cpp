
#include <dlib/geometry/drectangle.h>
#include <dlib/geometry/point_transforms.h>

#include "shape_quality.hpp"
#include "shape_extract_pixels.ipp"


raw_image::plane_ptr
dlibx::shape_extract_pixels(
    const pixel_intensity_base<unsigned char>& pi,
    const std::vector<fpoint>& pts) {
    
    assert(quality_deltas.size() == quality_height * quality_width);

    if (pts.size() != quality_shape.size())
        throw std::invalid_argument("shape_extract_pixels() requires dlib68 coordinates");

    // tform is scale and rotation only (no translation)
    const auto tform =
        dlib::matrix<float,2,2>(
            dlib::matrix_cast<float>(
                find_similarity_transform(quality_shape, pts).get_m()));
    
    auto r = raw_image::create(
        quality_width, quality_height, raw_image::pixel::gray8);
    assert(r->bytes_per_line == quality_width);
    auto dest = r->data;
    for (auto& d : quality_deltas) {
        const auto p = dlib::point(pts[d.first] + tform*d.second);
        *dest++ = pi(p.y(), p.x(), 128);
    }
    return r;
}

