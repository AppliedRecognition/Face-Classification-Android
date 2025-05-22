
#include "ncnn_common.hpp"

#define __FBLIB_DET_PRIVATE_INTERNAL_USE_ONLY__
#include "internal.hpp"

#include <raw_image/input_extractor.hpp>
#include <raw_image/face_landmarks.hpp>
#include <raw_image/transform.hpp>

#include <applog/core.hpp>

using namespace det;

template <unsigned N, typename T>
static void symmetry_swap_mesh(std::vector<T>& lm) {
    static_assert(N == 68 || N == 478);
    if (lm.size() != N)
        throw std::logic_error(
            "incorrect number of landmarks for symmetry_swap_mesh68");
    auto map = mirrored_pairs(N == 68 ? det::dt::mesh68 : det::dt::mesh478);
    for (unsigned i = 0; i < map.size(); ++i) {
        const auto j = map[i];
        if (i < j)
            std::swap(lm[i],lm[j]);
    }
}

namespace {
    template <unsigned lm_count>
    struct facemesh_net {
        ncnn::Net net;

        facemesh_net(core::context_data& data) {
            load_model(data,
                       models::type::landmark_detector,
                       lm_count == 68 ? models::landmark_detector::mesh68
                       : models::landmark_detector::mesh478,
                       net);
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

            const auto width = 256;
            const auto height = 256;
            const auto rgb =
                extract_region(raw, rbox.center.x, rbox.center.y,
                               rbox.width, rbox.height,
                               rbox.angle * float(180/M_PI),
                               width, height, raw_image::pixel::rgb24);
            auto in = to_ncnn_rgb(rgb);

            // extractor setup
            auto ex = net.create_extractor();
            //ex.set_num_threads(num_threads);
            ex.input("input", in);

            // output
            ncnn::Mat score_blob, landmarks_blob;
            ex.extract("score", score_blob);
            ex.extract("landmarks", landmarks_blob);

            if (score_blob.w * score_blob.h * score_blob.d * score_blob.c  != 1)
                throw std::runtime_error(
                    "facemesh net produced score with incorrect size");
            if (landmarks_blob.w * landmarks_blob.h * landmarks_blob.d * landmarks_blob.c != 2*lm_count)
                throw std::runtime_error(
                    "facemesh net produced landmarks with incorrect size");

            // coodinates on given image
            detected_coordinates result(
                lm_count == 68 ? dt::mesh68 : dt::mesh478);
            result.landmarks.reserve(lm_count);

            // quality assessment
            // try to match 0 to 10 range of dlib68 quality assessment
            result.confidence = (score_blob[0] + 12)/3;
            if (result.confidence > 10)
                result.confidence = 10;
            else if (result.confidence < 0)
                result.confidence = 0;

            float const* lmt = landmarks_blob;
            const auto lmstride =
                landmarks_blob.total() / (2*lm_count);
            for (unsigned i = 0; i < lm_count; ++i, lmt += 2*lmstride) {
                auto pt = raw_image::point2f {
                    lmt[0] - float(width)/2, lmt[lmstride] - float(height)/2
                };
                pt *= rbox.width / float(width);
                pt = pt.x * right + pt.y * down;
                pt += rbox.center;
                result.landmarks.emplace_back(pt);
            }

            if (raw.rotate & 4)
                symmetry_swap_mesh<lm_count>(result.landmarks);

            result.set_eye_coordinates_from_landmarks();

            return result;
        }
    };
}

template <>
internal::landmarks_factory_function
internal::ncnn_factory<lm::mesh68>(core::context_data&) {
    struct lmdet : landmarks_base {
        detected_coordinates
        operator()(const detected_coordinates& dc,
                   const raw_image::plane& image,
                   core::thread_data& td,
                   unsigned contrast_correction) const override {
            return core::get<facemesh_net<68> >(td.context,td)
                (dc, image, contrast_correction);
        }
    };
    return [](auto& data, const auto&) {
        core::get<const facemesh_net<68> >(data.context,data);
        return std::make_unique<lmdet>();
    };
}

template <>
internal::landmarks_factory_function
internal::ncnn_factory<lm::mesh478>(core::context_data&) {
    struct lmdet : landmarks_base {
        detected_coordinates
        operator()(const detected_coordinates& dc,
                   const raw_image::plane& image,
                   core::thread_data& td,
                   unsigned contrast_correction) const override {
            return core::get<facemesh_net<478> >(td.context,td)
                (dc, image, contrast_correction);
        }
    };
    return [](auto& data, const auto&) {
        core::get<const facemesh_net<478> >(data.context,data);
        return std::make_unique<lmdet>();
    };
}
