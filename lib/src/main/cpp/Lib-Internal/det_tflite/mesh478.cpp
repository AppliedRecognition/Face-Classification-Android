
#define __FBLIB_DET_PRIVATE_INTERNAL_USE_ONLY__
#include "internal.hpp"

#include <raw_image/transform.hpp>
#include <raw_image/pixels.hpp>
#include <raw_image/face_landmarks.hpp>

#include <tensorflow/lite/kernels/register.h>

#include <applog/core.hpp>

using namespace det;

template <typename T>
static void symmetry_swap_mesh478(std::vector<T>& lm) {
    if (lm.size() != 478)
        throw std::logic_error(
            "incorrect number of landmarks for symmetry_swap_mesh478");
    auto map = mirrored_pairs(det::dt::mesh478);
    for (unsigned i = 0; i < map.size(); ++i) {
        const auto j = map[i];
        if (i < j)
            std::swap(lm[i],lm[j]);
    }
}

namespace {
    struct mesh478_master : internal::tflite_model {
        mesh478_master(core::context_data& data)
            : tflite_model(data, models::type::landmark_detector,
                           models::landmark_detector::mesh478) {
        }
    };

    struct mesh478_net {
        const mesh478_master& master;
        tflite::ops::builtin::BuiltinOpResolver resolver;
        std::unique_ptr<tflite::Interpreter> interpreter;
        unsigned width, height;

        static constexpr auto lm_count = 478;

        mesh478_net(core::thread_data& td)
            : master(core::get<const mesh478_master>(td.context,td)) {

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
                height = unsigned(dims->data[1]);
                width  = unsigned(dims->data[2]);
                assert(width == 256 && height == 256);
            }

            {
                // verify output size
                auto& outputs = interpreter->outputs();
                assert(outputs.size() == 3);
                const auto out0 = outputs[0];
                const auto out1 = outputs[1];
                const auto out2 = outputs[2];

                auto const* dims0 = interpreter->tensor(out0)->dims;
                assert(dims0 && dims0->size == 4);
                assert(dims0->data[0] == 1 && dims0->data[1] == 1 &&
                       dims0->data[2] == 1 && dims0->data[3] == 3*lm_count);

                auto const* dims1 = interpreter->tensor(out1)->dims;
                assert(dims1 && dims1->size == 4);
                assert(dims1->data[0] == 1 && dims1->data[1] == 1 &&
                       dims1->data[2] == 1 && dims1->data[3] == 1);

                auto const* dims2 = interpreter->tensor(out2)->dims;
                assert(dims2 && dims2->size == 2);
                assert(dims2->data[0] == 1 && dims2->data[1] == 1);
            }
        }

        detected_coordinates
        operator()(const detected_coordinates& dc,
                   const raw_image::plane& raw,
                   unsigned) {

            raw_image::rotated_box rbox;
            rbox.center = (dc.eye_left + dc.eye_right) * 0.5f;
            const auto ed = dc.eye_distance();
            rbox.angle = std::atan2(dc.eye_right.y - dc.eye_left.y,
                                    dc.eye_right.x - dc.eye_left.x);
            const auto right = raw_image::point2f {
                std::cos(rbox.angle), std::sin(rbox.angle)
            };
            const auto down = raw_image::point2f { -right.y, right.x };
            rbox.center += down * (ed * 0.3f);
            rbox.width = rbox.height = 3.4f*ed;

            const auto rgb =
                extract_region(raw, rbox.center.x, rbox.center.y,
                               rbox.width, rbox.height,
                               rbox.angle * float(180/M_PI),
                               width, height, raw_image::pixel::rgb24);

            {
                float* dest = interpreter->typed_input_tensor<float>(0);
                for (auto&& line : raw_image::pixels_bpp<3>(rgb))
                    for (auto& px : line) {
                        *dest++ = float(px[0]-128)/128;
                        *dest++ = float(px[1]-128)/128;
                        *dest++ = float(px[2]-128)/128;
                    }
            }

            interpreter->Invoke();

            // 1 x 1 x 1 x 3*lm_count
            float const* out0 = interpreter->typed_output_tensor<float>(0);
            // 1 x 1 x 1 x 1 Presence score of the face (presence)
            float const* out1 = interpreter->typed_output_tensor<float>(1);
            // 1 x 1
            // note: can't figure out what this output is
            // metadata: Score of whether tongue is out of mouth (tongue_out)
            // model card say both that and cheekPuff should be present
            // but only one value comes out here
            //float const* out2 = interpreter->typed_output_tensor<float>(2);

            // coodinates on given image
            detected_coordinates result(dt::mesh478);
            result.landmarks.reserve(lm_count);

            for (unsigned i = 0; i < lm_count; ++i, out0 += 3) {
                auto pt = raw_image::point2f {
                    out0[0] - float(width)/2, out0[1] - float(height)/2
                };
                pt *= rbox.width / float(width);
                pt = pt.x * right + pt.y * down;
                pt += rbox.center;
                result.landmarks.emplace_back(pt);
            }

            // quality assessment
            // try to match 0 to 10 range of dlib68 quality assessment
            result.confidence = (*out1 + 12)/3;
            if (result.confidence > 10)
                result.confidence = 10;
            else if (result.confidence < 0)
                result.confidence = 0;

            if (raw.rotate & 4)
                symmetry_swap_mesh478(result.landmarks);

            result.set_eye_coordinates_from_landmarks();

            return result;
        }
    };
}

template <>
internal::landmarks_factory_function
internal::tflite_factory<lm::mesh478>(core::context_data&) {
    struct lmdet : landmarks_base {
        detected_coordinates
        operator()(const detected_coordinates& dc,
                   const raw_image::plane& image,
                   core::thread_data& td,
                   unsigned contrast_correction) const override {
            return core::get<mesh478_net>(td.thread,td)
                (dc, image, contrast_correction);
        }
    };
    return [](auto& data, const auto&) {
        core::get<const mesh478_master>(data.context,data);
        return std::make_unique<lmdet>();
    };
}
