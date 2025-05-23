
#include "raw_image.hpp"
#include <raw_image/transform.hpp>
#include <dlib/image_transforms/interpolation.h>

raw_image::ptr
raw_image::extract_image_chip(const multi_plane_arg& image,
                         const dlib::chip_details& cd,
                         pixel_layout layout) {
    const auto cx = float(1 + cd.rect.left() + cd.rect.right()) / 2;
    const auto cy = float(1 + cd.rect.top() + cd.rect.bottom()) / 2;
    const auto w = float(cd.rect.width() - 0.5);
    const auto h = float(cd.rect.height() - 0.5);
    const auto deg = float(cd.angle*(180/M_PI));
    return extract_region(image, cx, cy, w, h, deg,
                          unsigned(cd.cols), unsigned(cd.rows),
                          layout);
}
