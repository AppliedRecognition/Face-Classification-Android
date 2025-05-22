
#include "chip_details.hpp"
#include "rotated_box.hpp"
#include <raw_image/point_rounding.hpp>
#include <raw_image/scaled_chip.hpp>
#include <raw_image/face_landmarks.hpp>
#include <cmath>

using dlib::dpoint;

static inline auto rot90cw(const dpoint& p) {
    return dpoint( -p.y(), p.x() );
}

dlib::chip_details
dlibx::get_face_chip_details(dpoint eye_left, dpoint eye_right,
                             unsigned long size, double padding) {
    const auto vec = eye_right - eye_left;
    const auto center = 0.5 * (eye_left + eye_right) + 0.524 * rot90cw(vec);
    const auto ofs = 0.963365 * (1+2*padding) * vec.length();
    dlib::chip_details chip;
    chip.rows = chip.cols = size;
    chip.angle = std::atan2(vec.y(), vec.x());
    chip.rect.left()   = center.x() - ofs + 1;
    chip.rect.right()  = center.x() + ofs;
    chip.rect.top()    = center.y() - ofs + 1;
    chip.rect.bottom() = center.y() + ofs;
    return chip;
}

// note: don't know why, but x_left + x_right = 0.98 (not 1.0).
static const dpoint retinaface_mean_landmark[] = {
    { 0.226, 0.217 }, // eye_left
    { 0.754, 0.217 }, // eye_right
    { 0.490, 0.516 }, // nose_tip
    { 0.254, 0.780 }, // mouth_left
    { 0.726, 0.780 }  // mouth_right
};
static const dpoint blazeface_mean_landmark[] = {
    { 0.226, 0.217 }, // eye_left
    { 0.754, 0.217 }, // eye_right
    { 0.490, 0.516 }, // nose_tip
    { 0.490, 0.780 }, // mouth_center
};

dlib::chip_details dlibx::get_face_chip_details(
    const std::vector<dpoint>& pts,
    unsigned long size, double padding) {

    DLIB_CASSERT(padding >= 0 && size > 0,
                 "\t chip_details get_face_chip_details()"
                 << "\n\t Invalid inputs were given to this function."
                 << "\n\t padding: " << padding
                 << "\n\t size:    " << size
        );

    switch(pts.size()) {
    case 2: // eyes only
        return get_face_chip_details(pts[0], pts[1], size, padding);

    case 7: { // RetinaFace 5 landmarks + bounding box
        std::vector<dpoint> from_points;
        from_points.reserve(std::size(retinaface_mean_landmark));
        for (auto&& pt : retinaface_mean_landmark)
            from_points.push_back(size * (padding+pt) / (2*padding+1));
        const auto to_points = std::vector<dpoint>(
            pts.begin(), pts.begin() + std::size(retinaface_mean_landmark));
        return dlib::chip_details(
            from_points, to_points, dlib::chip_dims(size,size));
    }

    case 8: { // BlazeFace 6 landmarks + bounding box
        std::vector<dpoint> from_points;
        from_points.reserve(std::size(blazeface_mean_landmark));
        for (auto&& pt : blazeface_mean_landmark)
            from_points.push_back(size * (padding+pt) / (2*padding+1));
        const auto to_points = std::vector<dpoint>(
            pts.begin(), pts.begin() + std::size(blazeface_mean_landmark));
        return dlib::chip_details(
            from_points, to_points, dlib::chip_dims(size,size));
    }

    case 478: { // Mediapipe Mesh478 landmarks detector
        using dt = raw_image::detection_type;
        const auto sub = landmark_subset(dt::mesh478, dt::dlib68);
        std::vector<dlib::point> parts;
        parts.reserve(sub.size());
        for (auto& idx : sub)
            parts.push_back(raw_image::round_from(pts[idx]));
        dlib::rectangle rect(parts.front());
        for (const auto& p : parts)
            rect += p;
        const auto shape = dlib::full_object_detection(rect, parts);
        return dlib::get_face_chip_details(shape, size, padding);
    }

    case 5:    // dlib5
    case 68: { // dlib68
        std::vector<dlib::point> parts;
        parts.reserve(pts.size());
        for (auto& p : pts)
            parts.push_back(raw_image::round_from(p));
        dlib::rectangle rect(parts.front());
        for (const auto& p : parts)
            rect += p;
        const auto shape = dlib::full_object_detection(rect, parts);
        return dlib::get_face_chip_details(shape, size, padding);
    }

    default:
        throw std::invalid_argument(
            "incorrect number of landmarks for get_face_chip_details()");
    }
}

raw_image::rotated_box
dlibx::retina_align(stdx::span<const fpoint> landmarks,
                    float scale_factor, float y_offset) {
    static_assert(sizeof(raw_image::point2f) == sizeof(fpoint));
    return raw_image::retina_align( {
            reinterpret_cast<const raw_image::point2f*>(landmarks.data()),
            landmarks.size() }, scale_factor, y_offset);
}
