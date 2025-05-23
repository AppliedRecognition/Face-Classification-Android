
#include "ncnn_common.hpp"

#define __FBLIB_DET_PRIVATE_INTERNAL_USE_ONLY__
#include "internal.hpp"
#include <det/rfb320_common.hpp>

#include <raw_image/transform.hpp>

#include <applog/core.hpp>


using namespace det;
using namespace det::internal;
using namespace det::rfb320;


namespace {

    constexpr float mean_vals[3] = { 127, 127, 127 };
    constexpr float norm_vals[3] = { 1.0f/128, 1.0f/128, 1.0f/128 };

    struct rfb320_net {
        ncnn::Net net;

        rfb320_net(core::context_data& data) {
            load_model(data,
                       models::type::face_detector,
                       models::face_detector::rfb320,
                       net);
        }

        auto operator()(raw_image::plane image,
                        float size_range,
                        float score_threshold,
                        float iou_threshold,
                        raw_image::inter it,
                        int num_threads = 1) const {

            if (empty(image))
                throw std::invalid_argument("image is empty");

            // input image
            ncnn::Mat in;
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
                const auto resized =
                    copy_resize(image, w, h, raw_image::pixel::rgb24, it);
                in = to_ncnn_rgb(resized);
            }
            else if (image.layout != raw_image::pixel::rgb24 &&
                     image.layout != raw_image::pixel::rgba32 &&
                     image.layout != raw_image::pixel::bgr24 &&
                     image.layout != raw_image::pixel::bgra32) {
                FILE_LOG(logDETAIL) << "image converted from " << diag(image);
                const auto conv = copy(image, raw_image::pixel::rgb24);
                in = to_ncnn_rgb(conv);
            }
            else // image used as is (or converted by ncnn)
                in = to_ncnn_rgb(image);

            // extractor setup
            auto ex = net.create_extractor();
            ex.set_num_threads(num_threads);
            in.substract_mean_normalize(mean_vals, norm_vals);
            ex.input("input", in);

            // do detection
            ncnn::Mat scores, boxes;
            ex.extract("scores", scores);
            ex.extract("boxes", boxes);

            // generate boxes
            const auto priors = rfb320::priors(unsigned(in.w), unsigned(in.h));
            auto bboxes =
                priors(boxes.channel(0), scores.channel(0), score_threshold);
            const auto fw = float(image.width);
            const auto fh = float(image.height);
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
internal::ncnn_factory<6>(core::context_data&) {

    struct v6 : detector_base {
        void prepare_thread(core::job_context&,
                            const detection_settings&,
                            unsigned) override {
        }

        std::function<detection_result(core::job_context&)>
        detection_job(const detection_input& input,
                      json::value* diag) const override {
            return internal::detection_job<6>{input,diag};
        }
    };

    return [](auto& data, const auto&) {
        core::get<const rfb320_net>(data.context,data);
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
detection_result detection_job<6>::operator()(core::job_context& jc) {
    FILE_LOG(logDETAIL) << "job: [" << jc.job.order() << "] rfb320 (ncnn)";

    auto& detector = core::get<const rfb320_net>(jc.data.context, jc.data);
    const auto size_range = input.settings.size_range;
    const auto score_threshold = (input.settings.confidence_threshold+3.5f)/5;
    static constexpr auto iou_threshold = 0.3f;
    const auto it = input.settings.fast_scaling ?
        raw_image::inter::nearest : raw_image::inter::bilinear;

    auto dets = detector(input.image, size_range,
                         score_threshold, iou_threshold,
                         it, jc.num_threads() > 0 ? 2 : 1);
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

        if (0) {
            // compare to previous calculation
            detected_coordinates dc2(dt::v6_rfb320);
            dc2.eye_left.y = dc2.eye_right.y = float(0.6465*d.y1 + 0.3535*d.y2);
            const auto cx = 0.5f * (d.x1 + d.x2);
            const auto delta = 0.2338f * (d.x2 - d.x1);
            dc2.eye_left.x = cx - delta;
            dc2.eye_right.x = cx + delta;

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
