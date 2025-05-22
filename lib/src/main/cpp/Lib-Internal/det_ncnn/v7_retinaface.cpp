
#include "ncnn_common.hpp"

#define __FBLIB_DET_PRIVATE_INTERNAL_USE_ONLY__
#include "internal.hpp"
#include <det/retina_common.hpp>

#include <raw_image/transform.hpp>

#include <applog/core.hpp>

using namespace det;
using namespace det::internal;
using namespace det::retina;


static inline void
generate_proposals(const anchors& a,
                   const ncnn::Mat& score_blob,
                   const ncnn::Mat& bbox_blob,
                   const ncnn::Mat& landmark_blob,
                   float score_threshold,
                   std::vector<FaceObject>& faceobjects) {
    const auto channel_size = score_blob.total() / unsigned(score_blob.c);
    const float* scores = score_blob;
    a.proposals(unsigned(score_blob.w), unsigned(score_blob.h), channel_size,
                scores + 2*channel_size, bbox_blob, landmark_blob,
                score_threshold, faceobjects);
}


namespace {
    struct retina_net {
        const anchors anchors32, anchors16, anchors8;
        ncnn::Net net;

        retina_net(core::context_data& data)
            : anchors32(32,32),
              anchors16(16,8),
              anchors8(8,2) {
            load_model(data,
                       models::type::face_detector,
                       models::face_detector::retina,
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
            float sw = 1, sh = 1; // scale factors
            const auto image_pixels = float(image.width)*float(image.height);
            const auto target_pixels = std::max(2048.f, 768*768*size_range);
            if (target_pixels < image_pixels) {
                // scale down image
                const auto scale = std::sqrt(target_pixels / image_pixels);
                const auto resized = copy_resize(
                    image,
                    stdx::round_from(float(image.width) * scale),
                    stdx::round_from(float(image.height) * scale),
                    raw_image::pixel::rgb24, it);
                FILE_LOG(logDETAIL) << "image scaled from "
                                    << image.width << 'x' << image.height
                                    << " to "
                                    << resized->width << 'x' << resized->height;
                sw = float(image.width) / float(resized->width);
                sh = float(image.height) / float(resized->height);
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
            ex.input("data", in);

            std::vector<FaceObject> faceproposals;

            // stride 32
            if (32 <= in.w && 32 <= in.h) {
                ncnn::Mat score_blob, bbox_blob, landmark_blob;
                ex.extract("face_rpn_cls_prob_reshape_stride32", score_blob);
                ex.extract("face_rpn_bbox_pred_stride32", bbox_blob);
                ex.extract("face_rpn_landmark_pred_stride32", landmark_blob);
                generate_proposals(anchors32,
                                   score_blob, bbox_blob, landmark_blob,
                                   score_threshold, faceproposals);
            }

            // stride 16
            if (16 <= in.w && 16 <= in.h) {
                ncnn::Mat score_blob, bbox_blob, landmark_blob;
                ex.extract("face_rpn_cls_prob_reshape_stride16", score_blob);
                ex.extract("face_rpn_bbox_pred_stride16", bbox_blob);
                ex.extract("face_rpn_landmark_pred_stride16", landmark_blob);
                generate_proposals(anchors16,
                                   score_blob, bbox_blob, landmark_blob,
                                   score_threshold, faceproposals);
            }

            // stride 8
            if (8 <= in.w && 8 <= in.h) {
                ncnn::Mat score_blob, bbox_blob, landmark_blob;
                ex.extract("face_rpn_cls_prob_reshape_stride8", score_blob);
                ex.extract("face_rpn_bbox_pred_stride8", bbox_blob);
                ex.extract("face_rpn_landmark_pred_stride8", landmark_blob);
                generate_proposals(anchors8,
                                   score_blob, bbox_blob, landmark_blob,
                                   score_threshold, faceproposals);
            }

            // sort all proposals by score from highest to lowest
            sort(faceproposals.begin(), faceproposals.end(),
                 [](const auto& a, const auto& b) {
                     return a.score > b.score;
                 });

            // apply nms with iou_threshold
            std::vector<unsigned> picked;
            nms_sorted_bboxes(faceproposals, picked, iou_threshold);

            // copy out chosen faces
            std::vector<FaceObject> faceobjects;
            faceobjects.reserve(picked.size());
            for (auto j : picked)
                faceobjects.push_back(faceproposals[j]);

            if (1 < sw || 1 < sh) {
                for (auto& face : faceobjects) {
                    face.tl.x *= sw;
                    face.tl.y *= sh;
                    face.br.x *= sw;
                    face.br.y *= sh;
                    for (auto& p : face.landmark) {
                        p.x *= sw;
                        p.y *= sh;
                    }
                }
            }

            return faceobjects;
        }
    };
}

template<>
detector_factory_function
internal::ncnn_factory<7>(core::context_data&) {

    struct v7 : detector_base {
        void prepare_thread(core::job_context&,
                            const detection_settings&,
                            unsigned) override {
        }

        std::function<detection_result(core::job_context&)>
        detection_job(const detection_input& input,
                      json::value* diag) const override {
            return internal::detection_job<7>{input,diag};
        }
    };

    return [](auto& data, const auto&) {
        core::get<const retina_net>(data.context,data);
        return std::make_unique<v7>();
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
detection_result detection_job<7>::operator()(core::job_context& jc) {
    FILE_LOG(logDETAIL) << "job: [" << jc.job.order() << "] retina (ncnn)";

    auto& detector = core::get<const retina_net>(jc.data.context, jc.data);
    const auto size_range = input.settings.size_range;
    const auto score_threshold = (input.settings.confidence_threshold+3.5f)/5;
    static constexpr auto iou_threshold = 0.4f;
    const auto it = input.settings.fast_scaling ?
        raw_image::inter::nearest : raw_image::inter::bilinear;

    auto dets = detector(input.image, size_range,
                         score_threshold, iou_threshold,
                         it, jc.num_threads() > 0 ? 2 : 1);
    FILE_LOG(logDETAIL) << "retina faces detected: " << dets.size();

    std::vector<face_coordinates> faces;
    faces.reserve(dets.size());
    for (auto& d : dets) {
        if (input.image.rotate&4) d.mirror(input.image.width);
        auto& dc = faces.emplace_back().emplace_back(dt::v7_retina);
        dc.confidence = stdx::round_from(d.score*5 - 3.5f);
        // landmarks are:
        //   eye_left, eye_right, nose_tip, mouth_left, mouth_right
        //   top_left, bottom_right
        dc.landmarks.reserve(2 + std::size(d.landmark));
        for (auto& p : d.landmark)
            dc.landmarks.push_back(p);
        dc.landmarks.emplace_back(d.tl);
        dc.landmarks.emplace_back(d.br);
        dc.set_eye_coordinates_from_landmarks();
    }

    return internal::landmark_detection(jc, input, move(faces));
}
