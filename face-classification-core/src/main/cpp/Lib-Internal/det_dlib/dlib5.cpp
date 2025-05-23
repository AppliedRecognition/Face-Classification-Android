
#include <dlibx/shape_predictor.hpp>
#include <dlibx/shape_quality.hpp>
#include <dlibx/landmarks.hpp>

#include <raw_image/points.hpp>

#define __FBLIB_DET_PRIVATE_INTERNAL_USE_ONLY__
#include "internal.hpp"

#include <applog/core.hpp>


using namespace det;


/* Dlib 5-point landmarks are:
 *
 *    left    right
 *    (2 3)   (1 0)  <- eye corners
 *
 *         (4)       <- base of nose
 */

namespace {
    struct dlib5_model : internal::dlib_object<dlibx::shape_predictor> {
        dlib5_model(core::context_data& data)
            : dlib_object(data, models::type::landmark_detector,
                          models::landmark_detector::dlib5) {
        }
    };
}

static detected_coordinates
dlib5_detection(const eye_coordinates& eyes,
                const raw_image::plane& raw,
                core::thread_data& td,
                unsigned contrast_correction) {

    const auto known = [&] {
        auto el = to_image_point(
            raw_image::round_to<dlib::point>(eyes.eye_left), raw);
        auto er = to_image_point(
            raw_image::round_to<dlib::point>(eyes.eye_right), raw);
        if (raw.rotate & 4)
            std::swap(el,er);
        const auto d = (er - el) / 8;
        return std::vector<std::pair<unsigned,dlib::point> > {
            {2, el - d},  // outside
            {3, el + d},  // inside
            {1, er - d},  // inside
            {0, er + d},  // outside
        };
    }();

    auto& sp = *core::get<const dlib5_model>(td.context,td);

    const auto target_contrast = [&]() -> std::pair<float,float> {
        if (contrast_correction > 0)
            return { 30, 75 };
        else
            return { -1, -1 };
    }();

    // detect landmarks and quality
    const auto pts = sp(raw, known, target_contrast);

    // coodinates on given image
    detected_coordinates result(dt::dlib5);
    result.landmarks.reserve(pts.size());
    for (const auto& p : pts)
        result.landmarks.emplace_back(raw_image::round_to<coordinate_type>(p));

    // quality assessment
    //result.confidence = dlibx::shape_quality(raw, pts.begin(), pts.end());
    result.confidence = 10;  // not implemented
    // int todo_dlib5_confidence;

    // coordinates on "original" image
    result.landmarks.clear();
    for (const auto& p : pts) {
        const auto r = raw_image::round_to<coordinate_type>(p);
        result.landmarks.emplace_back(to_original_point(r,raw));
    }

    if (raw.rotate & 4)
        dlibx::symmetry_swap_dlib5(result.landmarks);

    result.set_eye_coordinates_from_landmarks();

    return result;
}

template <>
internal::landmarks_factory_function
internal::dlib_factory<lm::dlib5>(core::context_data&) {
    struct lmdet : landmarks_base {
        detected_coordinates
        operator()(const detected_coordinates& eyes,
                   const raw_image::plane& image,
                   core::thread_data& td,
                   unsigned contrast_correction) const override {
            return dlib5_detection(eyes, image, td, contrast_correction);
        }
    };
    return [](auto& data, const auto&) {
        core::get<const dlib5_model>(data.context,data);
        return std::make_unique<lmdet>();
    };
}
