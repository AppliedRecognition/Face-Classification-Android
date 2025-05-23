
#include <dlibx/net_vector.hpp>

#define __FBLIB_DET_PRIVATE_INTERNAL_USE_ONLY__
#include "internal.hpp"
#include <det/rfb320_common.hpp>

#include <raw_image/transform.hpp>

#include <applog/core.hpp>


using namespace det;
using namespace det::internal;
using namespace det::rfb320;


namespace {

    struct rfb320_master : dlib_object<dlibx::net::vector> {
        rfb320_master(core::context_data& data)
            : dlib_object(data, models::type::face_detector,
                          models::face_detector::rfb320) {
        }
    };

    struct rfb320_net {
        const rfb320_master& master;
        dlibx::net::vector net;

        rfb320_net(core::thread_data& td)
            : master(core::get<const rfb320_master>(td.context,td)),
              net(*master) {
        }

        auto operator()(raw_image::plane image,
                        float size_range,
                        float score_threshold,
                        float iou_threshold,
                        raw_image::inter it,
                        json::array* diag) {

            if (empty(image))
                throw std::invalid_argument("image is empty");

            // dimensions of image before resize
            const auto fw = float(image.width);
            const auto fh = float(image.height);

            // input image
            raw_image::plane_ptr resized;
            const auto image_pixels = float(image.width)*float(image.height);
            const auto target_pixels = std::max(8192.f, 768*768*size_range);
            static constexpr auto block = 64;
            if (target_pixels < image_pixels ||
                (image.width&(block-1)) != 0 || (image.height&(block-1)) != 0) {
                // scale image with both width and height multiples of block
                const auto scale = (target_pixels < image_pixels ? std::sqrt(target_pixels / image_pixels) : 1.0f) / block;
                const auto w = block*std::max(1u,stdx::round_to<unsigned>(float(image.width) * scale));
                const auto h = block*std::max(1u,stdx::round_to<unsigned>(float(image.height) * scale));
                FILE_LOG(logDETAIL) << "image scaled from "
                                    << image.width << 'x' << image.height
                                    << " to " << w << 'x' << h;
                resized = copy_resize(image, w, h, raw_image::pixel::rgb24, it);
                image = *resized;
            }
            else if (image.layout != raw_image::pixel::rgb24) {
                FILE_LOG(logDETAIL) << "image converted from "
                                    << raw_image::diag(image);
                resized = copy(image, raw_image::pixel::rgb24);
                image = *resized;
            }
            // else image is ready

            // do detection
            std::vector<float> dets;
            net(image, dets, diag);

            // generate boxes
            const auto priors = rfb320::priors(image.width, image.height);
            auto bboxes = priors(dets, score_threshold);
            for (auto& box : bboxes) {
                box.x1 *= fw;
                box.y1 *= fh;
                box.x2 *= fw;
                box.y2 *= fh;
            }

            // merge overlapping boxes
            return nms(bboxes, iou_threshold);
        }
    };
}

template<>
detector_factory_function
internal::dlib_factory<6>(core::context_data&) {

    struct v6 : detector_base {
        void prepare_thread(core::job_context& jc,
                            const detection_settings&,
                            unsigned) override {
            core::get<rfb320_net>(jc.data.thread,jc.data);
        }

        std::function<detection_result(core::job_context&)>
        detection_job(const detection_input& input,
                      json::value* diag) const override {
            return dlib_job<6>{input,diag};
        }
    };

    return [](auto& data, const auto&) {
        core::get<const rfb320_master>(data.context,data);
        return std::make_unique<v6>();
    };
}

/*  Score vs. Confidence
 *   1.0         1.5
 *   0.9         1.0
 *   0.8         0.5
 *   0.7         0.0  <= recommended default
 *   0.6        -0.5
 *   0.5        -1.0
 *
 * So score = (3.5+conf)/5 and
 *     conf = score*5 - 3.5
 */

template<>
detection_result dlib_job<6>::operator()(core::job_context& jc) {
    FILE_LOG(logDETAIL) << "job: [" << jc.job.order() << "] rfb320 (dlib)";

    json::array* arr = nullptr;
    if (diag) {
        if (!json::is_type<json::array>(*diag))
            *diag = json::array{};
        arr = &get_array(*diag);
    }

    auto& detector = core::get<rfb320_net>(jc.data.thread, jc.data);
    const auto size_range = input.settings.size_range;
    const auto score_threshold = (input.settings.confidence_threshold+3.5f)/5;
    static constexpr auto iou_threshold = 0.3f;
    const auto it = input.settings.fast_scaling ?
        raw_image::inter::nearest : raw_image::inter::bilinear;

    auto dets = detector(input.image, size_range, score_threshold, iou_threshold, it, arr);
    FILE_LOG(logDETAIL) << "rfb320 faces detected: " << dets.size();

    std::vector<face_coordinates> faces;
    faces.reserve(dets.size());
    for (auto& d : dets) {
        if (input.image.rotate&4) d.mirror(input.image.width);
        auto& dc = faces.emplace_back().emplace_back(dt::v6_rfb320);
        dc.confidence = stdx::round_from(d.score*5 - 3.5f);
        dc.landmarks.push_back({d.x1,d.y1});
        dc.landmarks.push_back({d.x2,d.y2});
        dc.set_eye_coordinates_from_landmarks();
    }

    return internal::landmark_detection(jc, input, move(faces));
}
