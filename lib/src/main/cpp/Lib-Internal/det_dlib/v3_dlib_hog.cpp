
#include <dlib/image_processing/scan_fhog_pyramid.h>
#include <dlib/image_processing/object_detector.h>
#include <dlibx/raw_image.hpp>
#include <raw_image/transform.hpp>

#define __FBLIB_DET_PRIVATE_INTERNAL_USE_ONLY__
#include "internal.hpp"

#include <applog/core.hpp>


using namespace det;
using namespace det::internal;


namespace {
    using frontal_face_detector =
        dlib::object_detector<dlib::scan_fhog_pyramid<dlib::pyramid_down<6> > >;

    struct master_detector : dlib_object<frontal_face_detector> {
        master_detector(core::context_data& data)
            : dlib_object(data, models::type::face_detector,
                          models::face_detector::fhog) {
        }
    };

    struct dlib_face_detector {
        const frontal_face_detector& master;
        dlib_face_detector(core::thread_data& td)
            : master(*core::get<const master_detector>(td.context,td)) {
            const auto n = master.num_detectors();
            if (!(5 <= n && n <= 6)) {
                FILE_LOG(logERROR) << "invalid dlib frontal face detector";
                throw std::runtime_error("invalid face detector model");
            }
        }

        frontal_face_detector detector[4];

        /** \brief Construct modified detector.
         *
         *  Master contains 5 or 6 detectors:
         *   0 - frontal
         *   1 - profile (yaw)
         *   2 - profile (yaw)
         *   3 - frontal rolled
         *   4 - frontal rolled
         *   5 - frontal with face mask (optional)
         *
         * This gives 4 possible detectors:
         *   frontal only
         *   frontal + roll
         *   frontal + yaw
         *   frontal + roll + yaw
         */
        inline frontal_face_detector&
        operator[](const detection_settings& s) {
            const auto y = (s.v3_limit_pose&1) == 0;
            const auto r = (s.v3_limit_pose&2) == 0;
            const auto i = (y?2u:0) + (r?1u:0);
            auto& d = detector[i];
            if (d.num_detectors() <= 0) {
                FILE_LOG(logINFO) << "fhog face detector: frontal"
                                  << (y?" + yaw":"") << (r?" + roll":"");
                auto& scanner = master.get_scanner();
                auto& ot = master.get_overlap_tester();
                std::vector<frontal_face_detector::feature_vector_type> w;
                w.reserve(6);
                w.emplace_back(master.get_w(0));
                if (y) {
                    w.emplace_back(master.get_w(1));
                    w.emplace_back(master.get_w(2));
                }
                if (r) {
                    w.emplace_back(master.get_w(3));
                    w.emplace_back(master.get_w(4));
                }
                if (master.num_detectors() > 5)
                    w.emplace_back(master.get_w(5));
                d = frontal_face_detector(scanner, ot, w);
            }
            return d;
        }
    };

    dlib::rectangle mirror(dlib::rectangle r, long width) {
        const auto x = --width - r.right();
        r.set_right(width - r.left());
        r.set_left(x);
        return r;
    }
}

template<>
detector_factory_function
internal::dlib_factory<3>(core::context_data&) {

    struct v3 : detector_base {
        void prepare_thread(core::job_context& jc,
                            const detection_settings&,
                            unsigned) override {
            core::get<dlib_face_detector>(jc.data.thread,jc.data);
        }

        std::function<detection_result(core::job_context&)>
        detection_job(const detection_input& input,
                      json::value* diag) const override {
            return dlib_job<3>{input,diag};
        }
    };

    return [](auto& data, const auto&) {
        core::get<const master_detector>(data.context,data);
        return std::make_unique<v3>();
    };
}

template <>
detection_result dlib_job<3>::operator()(core::job_context& jc) {
    FILE_LOG(logDETAIL) << "job: [" << jc.job.order() << "] dlib";

    auto& image = input.image;
    detection_result result;

    const auto desired_pix = 340 * 1000 * input.settings.size_range;
    if (desired_pix < 10) {
        FILE_LOG(logWARNING) << "detection.size_range too small"
            " -- not doing face detection";
        return result;
    }

    const auto color_space = [&]() {
        if (bytes_per_pixel(image.layout) == 1)
            return image.layout;
        switch (image.layout) {
        case raw_image::pixel::yuv24_jpeg:
            return raw_image::pixel::y8_jpeg;
        case raw_image::pixel::yuv24_nv21:
            return raw_image::pixel::y8_nv21;
        default:
            return raw_image::pixel::gray8;
        }
    }();

    auto dimg = image; // for detection
    raw_image::plane_ptr dimg_buf;
    auto scale = float(image.width) * float(image.height) / desired_pix;
    if (scale > 1) {
        scale = std::sqrt(scale);
        const auto dw = stdx::round_to<unsigned>(float(image.width) / scale);
        const auto dh = stdx::round_to<unsigned>(float(image.height) / scale);
        if (dw < image.width && dh < image.height) {
            if (dw < 10 || dh < 10) {
                FILE_LOG(logWARNING) << "detection.size_range too small"
                    " -- not doing face detection";
                return result;
            }
            FILE_LOG(logDETAIL) << "scaling image from "
                                << dimg.width << 'x' << dimg.height
                                << " to " << dw << 'x' << dh;
            const auto it = input.settings.fast_scaling ?
                raw_image::inter::nearest : raw_image::inter::bilinear;
            dimg_buf = copy_resize(dimg, dw, dh, color_space, it);
            dimg = *dimg_buf;
        }
        else
            scale = 1.0f;
    }
    else
        scale = 1.0f;

    if (bytes_per_pixel(dimg.layout) != 1) {
        FILE_LOG(logDETAIL) << "convert to grayscale "
                            << dimg.width << 'x' << dimg.height;
        dimg_buf = copy(dimg, color_space);
        dimg = *dimg_buf;
    }

    std::vector<std::pair<double, dlib::rectangle> > dets;
    auto& detector =
        core::get<dlib_face_detector>(jc.data.thread,jc.data)[input.settings];
    {
        raw_image::fixed_dlib_image<unsigned char> fdimg(dimg);
        detector(fdimg, dets, input.settings.confidence_threshold);
        FILE_LOG(logDETAIL) << "dlib faces detected: " << dets.size();
        dimg_buf = nullptr;
    }

    if (image.scale > 0)
        scale *= float(1 << image.scale);
    else if (image.scale < 0)
        scale /= float(1 << -image.scale);

    std::vector<face_coordinates> faces;
    faces.reserve(dets.size());
    for (const auto& d : dets) {
        const auto r = (image.rotate&4) ?
            mirror(d.second,long(image.width)) : d.second;

        auto& dc = faces.emplace_back().emplace_back(dt::v3_dlib);
        dc.confidence = stdx::round_from(d.first);
        dc.landmarks.push_back( {
                scale * (float(r.left()) - 0.75f),
                scale * (float(r.top()) - 0.25f)
            });
        dc.landmarks.push_back( {
                scale * (float(r.right()) + 0.75f),
                scale * (float(r.bottom()) + 0.25f)
            });
        dc.set_eye_coordinates_from_landmarks();

        if (0) {
            // compare to previous calculation
            const auto cx = scale * 0.5f * float(r.left()+r.right());
            const auto cy = scale * 0.5f * float(r.top()+r.bottom());
            const auto delta = scale * 0.1f * float(r.width()+r.height());

            detected_coordinates dc2(dt::v3_dlib);
            // eye_distance is 0.4 * width or 0.2 * (width+height)
            dc2.eye_left.x = cx - delta;
            // eyes are at 0.5 * eye_distance above center
            dc2.eye_left.y = cy - 1.0f * delta;
            dc2.eye_right.x = cx + delta;
            dc2.eye_right.y = dc2.eye_left.y;

            auto sqr = [](auto x) { return x*x; };
            auto ex = sqr(dc.eye_left.x - dc2.eye_left.x) +
                sqr(dc.eye_right.x - dc2.eye_right.x);
            auto ey = sqr(dc.eye_left.y - dc2.eye_left.y) +
                sqr(dc.eye_right.y - dc2.eye_right.y);
            //FILE_LOG(logERROR) << dc2.eye_left.x << ' ' << dc.eye_left.x << ' ' << dc.eye_right.x << ' ' << dc2.eye_right.x;
            //FILE_LOG(logERROR) << ex << ' ' << ey;
            assert(ex+ey < 1e-5);
        }
    }
    
    return internal::landmark_detection(jc, input, move(faces));
}
