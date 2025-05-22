
#include <dlibx/net_vector.hpp>
#include <dlibx/tensor.hpp>

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
                   const dlib::tensor& score_blob,
                   const dlib::tensor& bbox_blob,
                   const dlib::tensor& landmark_blob,
                   float score_threshold,
                   std::vector<FaceObject>& faceobjects) {
    a.proposals(unsigned(score_blob.nc()), unsigned(score_blob.nr()),
                score_blob.host(), bbox_blob.host(), landmark_blob.host(),
                score_threshold, faceobjects);
}


namespace {
    struct retina_master : dlib_object<dlibx::net::vector> {
        const anchors anchors32, anchors16, anchors8;
        retina_master(core::context_data& data)
            : dlib_object(data, models::type::face_detector,
                          models::face_detector::retina),
              anchors32(32,32),
              anchors16(16,8),
              anchors8(8,2) {
        }
    };

    struct retina_net {
        const retina_master& master;
        dlibx::net::vector net;

        retina_net(core::thread_data& td)
            : master(core::get<const retina_master>(td.context,td)),
              net(*master) {
        }

        auto operator()(raw_image::plane image,
                        float size_range,
                        float score_threshold,
                        float iou_threshold,
                        raw_image::inter it,
                        json::array*) {

            if (empty(image))
                throw std::invalid_argument("image is empty");

            // input image
            raw_image::plane_ptr resized;
            float sw = 1, sh = 1; // scale factors
            const auto image_pixels = float(image.width)*float(image.height);
            const auto target_pixels = std::max(2048.f, 768*768*size_range);
            if (target_pixels < image_pixels) {
                // scale down image
                const auto scale = std::sqrt(target_pixels / image_pixels);
                resized = copy_resize(
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
                image = *resized;
            }
            else if (image.layout != raw_image::pixel::rgb24) {
                resized = copy(image, raw_image::pixel::rgb24);
                image = *resized;
            }
            // else image is ready

            // do detection
            std::array<dlib::resizable_tensor,9> dets;
            net({&image,1}, dets);

            // tally results
            std::vector<FaceObject> faceproposals;
            generate_proposals(master.anchors32, dets[0], dets[1], dets[2],
                               score_threshold, faceproposals);
            generate_proposals(master.anchors16, dets[3], dets[4], dets[5],
                               score_threshold, faceproposals);
            generate_proposals(master.anchors8, dets[6], dets[7], dets[8],
                               score_threshold, faceproposals);

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
internal::dlib_factory<7>(core::context_data&) {

    struct v7 : detector_base {
        void prepare_thread(core::job_context& jc,
                            const detection_settings&,
                            unsigned) override {
            core::get<retina_net>(jc.data.thread,jc.data);
        }

        std::function<detection_result(core::job_context&)>
        detection_job(const detection_input& input,
                      json::value* diag) const override {
            return dlib_job<7>{input,diag};
        }
    };

    return [](auto& data, const auto&) {
        core::get<const retina_master>(data.context,data);
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
detection_result dlib_job<7>::operator()(core::job_context& jc) {
    FILE_LOG(logDETAIL) << "job: [" << jc.job.order() << "] retina (dlib)";

    json::array* arr = nullptr;
    if (diag) {
        if (!json::is_type<json::array>(*diag))
            *diag = json::array{};
        arr = &get_array(*diag);
    }

    auto& detector = core::get<retina_net>(jc.data.thread, jc.data);
    const auto size_range = input.settings.size_range;
    const auto score_threshold = (input.settings.confidence_threshold+3.5f)/5;
    static constexpr auto iou_threshold = 0.4f;
    const auto it = input.settings.fast_scaling ?
        raw_image::inter::nearest : raw_image::inter::bilinear;

    auto dets = detector(input.image, size_range, score_threshold, iou_threshold, it, arr);
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
