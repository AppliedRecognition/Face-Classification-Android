
#include "frontalize.hpp"

#include <core/context.hpp>
#include <raw_image/core.hpp>
#include <raw_image/point_rounding.hpp>

#include <applog/core.hpp>

#include <opencv2/calib3d/calib3d.hpp>

#include <cmath>

#include "frontalize_model.ipp"


using namespace render;


static const cv::Mat distortion = cv::Mat::zeros(4,1,CV_32F);

static constexpr unsigned fit_select_pts[] = {
    17, 21, 22, 26,  // eye brows
    36, 39, 42, 45,  // eyes
    27, 28, 29, 30,  // nose (upper)
    31, 33, 35,       // nose (lower)
    48, 51, 54, 57,  // mouth
    8 // chin
};

face_alignment
render::align_model(stdx::arg<core::context_data> context,
                    const face_coordinates& detected_face,
                    const image_size& size,
                    unsigned focal_length) {
    if (!context)
        throw std::invalid_argument("invalid context object");

    if (size.width <= 0 || size.height <= 0)
        throw std::invalid_argument("invalid image size");
    if (focal_length <= 0)
        focal_length = std::max(size.width, size.height);

    const det::detected_coordinates* shape = nullptr;
    for (auto& s : detected_face)
        if (s.type == det::dt::dlib68) {
            shape = &s;
            break;
        }
    if (!shape || shape->landmarks.size() != 68)
        throw std::invalid_argument("dlib landmarks required");
 
    face_alignment result;
    result.focal_length = focal_length;
    result.image_center = {
        stdx::round_from(size.width / 2.0),
        stdx::round_from(size.height / 2.0)
    };
        
    const cv::Mat camera =
        (cv::Mat_<float>(3,3) <<
         focal_length, 0, result.image_center.x,
         0, focal_length, result.image_center.y,
         0, 0, 1);

    std::vector<cv::Point3f> pts3d;
    pts3d.reserve(68);
    std::vector<cv::Point2f> pts2d;
    pts2d.reserve(68);
    for (auto i : fit_select_pts) {
        pts3d.push_back(landmark_3d[i]);
        pts2d.push_back(raw_image::round_from(shape->landmarks[i]));
    }
    
    // initial estimate from select landmarks
    cv::Mat rv, tr;
    cv::solvePnP(pts3d, pts2d, camera, distortion,
                 rv, tr, false, cv::SOLVEPNP_EPNP);
    pts3d.clear();
    pts2d.clear();
    
    // improve estimate
    assert(rv.type() == CV_64F);
    const auto yaw = rv.at<double>(1);
    if (yaw < -10*M_PI/180) {
        // ignore right side jaw line (landmarks [9,16])
        pts3d.assign(landmark_3d, landmark_3d + 9);
        pts3d.insert(pts3d.end(), landmark_3d + 17, landmark_3d + 68);
        for (unsigned i = 0; i < 9; ++i)
            pts2d.push_back(raw_image::round_from(shape->landmarks[i]));
        for (unsigned i = 17; i < 68; ++i)
            pts2d.push_back(raw_image::round_from(shape->landmarks[i]));
        cv::solvePnP(pts3d, pts2d, camera, distortion,
                     rv, tr, true, cv::SOLVEPNP_ITERATIVE);
    }
    else if (yaw > 10*M_PI/180) {
        // ignore left side jaw line (landmarks [0,7])
        pts3d.assign(landmark_3d + 8, landmark_3d + 68);
        for (unsigned i = 8; i < 68; ++i)
            pts2d.push_back(raw_image::round_from(shape->landmarks[i]));
        cv::solvePnP(pts3d, pts2d, camera, distortion,
                     rv, tr, true, cv::SOLVEPNP_ITERATIVE);
    }
    else {  // no significant yaw -- use all landmarks
        pts3d.assign(landmark_3d, landmark_3d + 68);
        for (const auto& p : shape->landmarks)
            pts2d.push_back(raw_image::round_from(p));
        cv::solvePnP(pts3d, pts2d, camera, distortion,
                     rv, tr, true, cv::SOLVEPNP_ITERATIVE);
    }

    assert(tr.type() == CV_64F && tr.cols == 1 && tr.rows == 3);
    result.tx = float(tr.at<double>(0));
    result.ty = float(tr.at<double>(1));
    result.tz = float(tr.at<double>(2));

    assert(rv.type() == CV_64F && rv.cols == 1 && rv.rows == 3);
    result.pitch = float(rv.at<double>(0) * (180/M_PI));
    result.yaw   = float(rv.at<double>(1) * (180/M_PI));
    result.roll  = float(rv.at<double>(2) * (180/M_PI));
    
    return result;
}

face_alignment
render::align_model(stdx::arg<core::context_data> data,
                    const face_coordinates& detected_face,
                    stdx::arg<const raw_image::plane> image,
                    unsigned focal_length) {
    if (!image)
        throw std::invalid_argument("invalid image object");
    return align_model(data, detected_face, dimensions(image),
                       focal_length);
}

//static const auto eye_left  = 0.5f * (landmark_3d[36] + landmark_3d[39]);
static const auto eye_right = 0.5f * (landmark_3d[42] + landmark_3d[45]);
static const auto d_factor = 0.063f / eye_right.x / 2;

float render::estimate_distance(const face_alignment& alignment) {
    return d_factor * std::sqrt(alignment.tx*alignment.tx +
                                alignment.ty*alignment.ty +
                                alignment.tz*alignment.tz);
}
