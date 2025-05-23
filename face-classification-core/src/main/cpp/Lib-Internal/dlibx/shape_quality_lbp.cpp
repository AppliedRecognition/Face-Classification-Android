
#include "shape_quality.hpp"

#include "lbp_generic.hpp"
#include "lbpcascade_dlibqual0.ipp"
#include "lbpcascade_dlibqual1.ipp"

#include <array>
#include <iterator>
#include <numeric>


using cascade0 = lbpcascade_dlibqual0<lbp_generic>;
using cascade1 = lbpcascade_dlibqual1<lbp_generic>;
static_assert(cascade0::window_width == cascade1::window_width &&
              cascade0::window_height == cascade1::window_height,
              "cascades must have same window size");

static constexpr float calib[] = {
    0.281507f, 0.310954f, 0.296852f, 0.285098f, 0.297807f, 0.306687f, 0.34251f,
    0.354695f, 0.321443f, 0.341068f, 0.473705f, 0.505089f, 0.495814f, 0.413547f,
    0.278722f, 0.303953f, 0.3031f, 0.288141f, 0.290726f, 0.308585f, 0.341968f,
    0.351998f, 0.31673f, 0.335766f, 0.449708f, 0.480021f, 0.510259f, 0.413547f
};
static_assert(std::size(calib) == cascade0::num_stages + cascade1::num_stages,
    "calibration does not match cascades");

float dlibx::shape_quality(const raw_image::plane& features) {
    
    if (bytes_per_pixel(features.layout) != 1 ||
        features.width != cascade0::window_width ||
        features.height != cascade0::window_height)
        throw std::invalid_argument("invalid feature pixel object");

    integral_image<int> ii;
    const auto good = ii.set_image(features, features.height);
    assert(good);

    std::array<bool, cascade0::num_stages + cascade1::num_stages> r;
    cascade0::test_all(r.begin(), ii.sum.data(), ii.stride_table.data());
    cascade1::test_all(r.begin() + cascade0::num_stages,
                       ii.sum.data(), ii.stride_table.data());

    assert(r.size() == std::size(calib));
    return std::inner_product(r.begin(), r.end(), calib, 0.0f);
}
