
#include <dlibx/shape_predictor.hpp>
#include <dlibx/shape_quality.hpp>
#include <dlibx/landmarks.hpp>

#include <raw_image/points.hpp>

#define __FBLIB_DET_PRIVATE_INTERNAL_USE_ONLY__
#include "internal.hpp"

#include <applog/core.hpp>


using namespace det;



namespace {
    struct dlib68_model : internal::dlib_object<dlibx::shape_predictor> {
        dlib68_model(core::context_data& data)
            : dlib_object(data, models::type::landmark_detector,
                          models::landmark_detector::dlib68) {
        }
    };
}

static detected_coordinates
dlib68_detection(const detected_coordinates& dc,
                 const raw_image::plane& raw,
                 core::thread_data& td,
                 unsigned contrast_correction) {

    const auto known = [&] {
        using return_type = std::vector<std::pair<unsigned,dlib::point> >;
        if (dc.type == dt::dlib5) {
            auto el = to_image_point(
                raw_image::round_to<dlib::point>(dc.landmarks[2]), raw);
            auto er = to_image_point(
                raw_image::round_to<dlib::point>(dc.landmarks[0]), raw);
            auto nose = to_image_point(
                raw_image::round_to<dlib::point>(dc.landmarks[4]), raw);
            if (raw.rotate & 4)
                std::swap(el,er);
            return return_type {
                {36, el},
                {45, er},
                {33, nose}
            };
        }
        else {
            auto el = to_image_point(
                raw_image::round_to<dlib::point>(dc.eye_left), raw);
            auto er = to_image_point(
                raw_image::round_to<dlib::point>(dc.eye_right), raw);
            if (raw.rotate & 4)
                std::swap(el,er);
            const auto d = (er - el) / 5;
            return return_type {
                {36, el - d},
                {39, el + d},
                {42, er - d},
                {45, er + d}
            };
        }
    }();

    auto& sp = *core::get<const dlib68_model>(td.context,td);

    const auto target_contrast = [&]() -> std::pair<float,float> {
        if (contrast_correction > 0)
            return { 28, 40 };
        else
            return { -1, -1 };
    }();

    // detect landmarks and quality
    const auto pts = sp(raw, known, target_contrast);

    // coodinates on given image
    detected_coordinates result(dt::dlib68);
    result.landmarks.reserve(pts.size());
    for (const auto& p : pts)
        result.landmarks.emplace_back(raw_image::round_to<coordinate_type>(p));

    // quality assessment
    result.confidence = dlibx::shape_quality(raw, pts.begin(), pts.end());

    // coordinates on "original" image
    result.landmarks.clear();
    for (const auto& p : pts) {
        const auto r = raw_image::round_to<coordinate_type>(p);
        result.landmarks.emplace_back(to_original_point(r,raw));
    }

    if (raw.rotate & 4)
        dlibx::symmetry_swap_dlib68(result.landmarks);

    result.set_eye_coordinates_from_landmarks();

    return result;
}

template <>
internal::landmarks_factory_function
internal::dlib_factory<lm::dlib68>(core::context_data&) {
    struct lmdet : landmarks_base {
        detected_coordinates
        operator()(const detected_coordinates& dc,
                   const raw_image::plane& image,
                   core::thread_data& td,
                   unsigned contrast_correction) const override {
            return dlib68_detection(dc, image, td, contrast_correction);
        }
    };
    return [](auto& data, const auto&) {
        core::get<const dlib68_model>(data.context,data);
        return std::make_unique<lmdet>();
    };
}
