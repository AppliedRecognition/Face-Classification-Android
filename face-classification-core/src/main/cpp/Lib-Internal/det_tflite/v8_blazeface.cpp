
#define __FBLIB_DET_PRIVATE_INTERNAL_USE_ONLY__
#include "internal.hpp"
#include <det/nms.hpp>

#include <raw_image/reader.hpp>
#include <raw_image/pixels.hpp>

#include <tensorflow/lite/kernels/register.h>

#include <applog/core.hpp>

using namespace det;
using namespace det::internal;

namespace {
    struct AnchorParams {
        int inputWidth   = 128;
        int inputHeight  = 128;

        float minScale = 0.1484375f;
        float maxScale = 0.75f;

        float offsetX = 0.5f;
        float offsetY = 0.5f;

        std::vector<std::pair<int,unsigned> > stride_counts = {
            {  8, 2 },
            { 16, 6 }
        };
    };

    // anchors are only center point x,y
    auto generate_anchors(const AnchorParams& param = {}) {
        const auto width  = float(param.inputWidth);
        const auto height = float(param.inputHeight);
        std::vector<coordinate_type> anchors;
        for (auto& pr : param.stride_counts) {
            const auto count = pr.second;
            const auto stride = pr.first;
            const auto rows = 1 + (param.inputHeight - 1) / stride;
            const auto cols = 1 + (param.inputWidth  - 1) / stride;
            for (int y = 0; y < rows; ++y) {
                auto center = coordinate_type {
                    0,
                    height * (float(y) + param.offsetY) / float(rows)
                };
                for (int x = 0; x < cols; ++x) {
                    center.x = width * (float(x) + param.offsetX) / float(cols);
                    anchors.insert(anchors.end(), count, center);
                }
            }
        }
        return anchors;
    }

    struct blaze_master : tflite_model {
        std::vector<coordinate_type> anchors;
        blaze_master(core::context_data& data)
            : tflite_model(data, models::type::face_detector,
                           models::face_detector::blaze128),
              anchors(generate_anchors()) {
        }
    };

    struct blaze_net {
        const blaze_master& master;
        tflite::ops::builtin::BuiltinOpResolver resolver;
        std::unique_ptr<tflite::Interpreter> interpreter;
        raw_image::plane_ptr input_rgb;

        blaze_net(core::thread_data& td)
            : master(core::get<const blaze_master>(td.context,td)) {

            // build interpreter
            tflite::InterpreterBuilder(*master.model, resolver)(&interpreter);
            
            // resize input tensors
            interpreter->AllocateTensors();

            {
                // verify expected input size
                auto& inputs = interpreter->inputs();
                assert(inputs.size() == 1);
                const auto input = inputs[0];
                auto const* dims = interpreter->tensor(input)->dims;
                assert(dims && dims->size == 4);
                FILE_LOG(logINFO) << "input dims: " << dims->data[0] << ' '
                                  << dims->data[1] << 'x'
                                  << dims->data[2] << ' '
                                  << dims->data[3];
                assert(dims->data[0] == 1 &&
                       0 < dims->data[1] &&
                       0 < dims->data[2] &&
                       dims->data[3] == 3); // channels
                const auto height = unsigned(dims->data[1]);
                const auto width  = unsigned(dims->data[2]);
                assert(width == 128 && height == 128);
                input_rgb = create(width, height, raw_image::pixel::rgb24);
            }

            {
                // verify output size
                auto& outputs = interpreter->outputs();
                assert(outputs.size() == 2);
                const auto out0 = outputs[0];
                const auto out1 = outputs[1];
                auto const* dims0 = interpreter->tensor(out0)->dims;
                assert(dims0 && dims0->size == 3);
                assert(dims0->data[0] == 1 && dims0->data[2] == 16);
                auto const* dims1 = interpreter->tensor(out1)->dims;
                assert(dims1 && dims1->size == 3);
                assert(dims1->data[0] == 1 && dims1->data[2] == 1);
                assert(0 < dims0->data[1] && dims0->data[1] == dims1->data[1]);
                //assert(dims0->data[1] == 896);  == 2*16*16 + 6*8*8
                assert(unsigned(dims0->data[1]) == master.anchors.size());
            }
        }

        // returns { offset, scale }
        auto scale_input(const raw_image::plane& image, bool fast_scale) {
            memset(input_rgb->data, 128,
                   input_rgb->height * input_rgb->bytes_per_line);

            std::pair<coordinate_type, coordinate_type> r;

            // input image -- scale to letterbox
            auto reader = raw_image::reader::construct(image);
            raw_image::plane roi; ///< destination
            if (image.height * input_rgb->width <= image.width * input_rgb->height) {
                // borders top and bottom
                const auto h =
                    1 + (image.height * input_rgb->width - 1) / image.width;
                assert(h <= input_rgb->height);
                const auto y_ofs = (input_rgb->height - h) / 2;
                roi = crop(input_rgb, 0, y_ofs, input_rgb->width, h);
                std::get<0>(r) = { 0, float(y_ofs) };
            }
            else {
                // borders left and right
                const auto w =
                    1 + (image.width * input_rgb->height - 1) / image.height;
                assert(w <= input_rgb->width);
                const auto x_ofs = (input_rgb->width - w) / 2;
                roi = crop(input_rgb, x_ofs, 0, w, input_rgb->height);
                std::get<0>(r) = { float(x_ofs), 0 };
            }
            std::get<1>(r).x = float(image.width) / float(roi.width);
            std::get<1>(r).y = float(image.height) / float(roi.height);

            if (fast_scale)
                reader = scale_nearest(move(reader), roi.width, roi.height);
            else if (input_rgb->width <= image.width ||
                     input_rgb->height <= image.height)
                reader = scale_area(move(reader), roi.width, roi.height);
            else
                reader = scale_interpolate(move(reader), roi.width, roi.height);
            convert(move(reader),roi.layout)->copy_to(roi);

            float* dest = interpreter->typed_input_tensor<float>(0);
            for (auto&& line : raw_image::pixels_bpp<3>(input_rgb))
                for (auto& px : line) {
                    *dest++ = float(px[0]-128)/128;
                    *dest++ = float(px[1]-128)/128;
                    *dest++ = float(px[2]-128)/128;
                }

            return r;
        }

        auto operator()(raw_image::plane image,
                        float score_threshold,
                        float iou_threshold,
                        bool fast_scale) {

            if (empty(image))
                throw std::invalid_argument("image is empty");

            const auto t = scale_input(image, fast_scale);

            interpreter->Invoke();

            const auto output_count = master.anchors.size();

            // 1 x output_count x 16
            float const* out0 = interpreter->typed_output_tensor<float>(0);
            // 1 x output_count x 1
            float const* out1 = interpreter->typed_output_tensor<float>(1);

            using namespace det::nms;
            std::vector<blaze_landmarks> candidates;
            candidates.reserve(output_count);

            for (unsigned i = 0; i < output_count; ++i, out0 += 16, out1++) {
                const auto conf = *out1;
                if (score_threshold <= conf) {
                    auto& bbox = candidates.emplace_back();
                    bbox.score = *out1;
                    const auto center = master.anchors[i]
                        + coordinate_type { out0[0], out0[1] };
                    bbox.tl = { center.x - out0[2]/2, center.y - out0[3]/2 };
                    bbox.br = { center.x + out0[2]/2, center.y + out0[3]/2 };
                    for (unsigned i = 0; i < 6; ++i) {
                        bbox.landmarks[i] = { out0[4+2*i], out0[5+2*i] };
                        bbox.landmarks[i] += center;
                    }
                }
            }
            sort_decreasing_score(candidates);
            candidates = blend_from_sorted(move(candidates), iou_threshold);

            // offset and scale to image
            const auto ofs = std::get<0>(t);
            const auto scale = std::get<1>(t);
            for (auto& bbox : candidates) {
                bbox.tl -= ofs;
                bbox.tl.x *= scale.x;
                bbox.tl.y *= scale.y;
                bbox.br -= ofs;
                bbox.br.x *= scale.x;
                bbox.br.y *= scale.y;
                for (auto& lm : bbox.landmarks) {
                    lm -= ofs;
                    lm.x *= scale.x;
                    lm.y *= scale.y;
                }
            }

            return candidates;
        }
    };
}

template<>
detector_factory_function
internal::tflite_factory<8>(core::context_data&) {

    struct v8 : detector_base {
        void prepare_thread(core::job_context& jc,
                            const detection_settings&,
                            unsigned) override {
            core::get<blaze_net>(jc.data.thread,jc.data);
        }

        std::function<detection_result(core::job_context&)>
        detection_job(const detection_input& input,
                      json::value* diag) const override {
            return internal::detection_job<8>{input,diag};
        }
    };

    return [](auto& data, const auto&) {
        core::get<const blaze_master>(data.context,data);
        return std::make_unique<v8>();
    };
}

// Note: score is offset by 0.5 to better line up with other detectors.
// So external score_threshold 0.0 is actually threshold 0.5 internally.

template<>
detection_result detection_job<8>::operator()(core::job_context& jc) {
    FILE_LOG(logDETAIL) << "job: [" << jc.job.order() << "] blaze (tflite)";

    auto& detector = core::get<blaze_net>(jc.data.thread, jc.data);
    const auto score_threshold = input.settings.confidence_threshold + 0.5f;
    static constexpr auto iou_threshold = 0.3f;
    const bool fast_scale = input.settings.fast_scaling;

    auto dets =
        detector(input.image, score_threshold, iou_threshold, fast_scale);
    FILE_LOG(logDETAIL) << "blaze faces detected: " << dets.size();

    std::vector<face_coordinates> faces;
    faces.reserve(dets.size());
    for (auto& d : dets) {
        if (input.image.rotate&4) d.mirror(input.image.width);
        auto& dc = faces.emplace_back().emplace_back(dt::v8_blaze);
        dc.confidence = d.score - 0.5f;
        // landmarks are:
        //   eye_left, eye_right, nose_tip, mouth, tragion_left, tragion_right
        //   top_left, bottom_right
        dc.landmarks.reserve(2 + std::size(d.landmarks));
        for (auto& p : d.landmarks)
            dc.landmarks.push_back(p);
        dc.landmarks.emplace_back(d.tl);
        dc.landmarks.emplace_back(d.br);
        dc.set_eye_coordinates_from_landmarks();
    }

    return internal::landmark_detection(jc, input, move(faces));
}
