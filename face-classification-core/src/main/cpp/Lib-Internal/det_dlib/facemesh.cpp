
#include <dlibx/net_vector.hpp>
#include <dlibx/tensor.hpp>

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
    struct facemesh_master : internal::dlib_object<dlibx::net::vector> {
        facemesh_master(core::context_data& data)
            : dlib_object(data, models::type::landmark_detector,
                          lm_count == 68 ? models::landmark_detector::mesh68
                          : models::landmark_detector::mesh478) {
        }
    };

    template <unsigned lm_count>
    struct facemesh_net {
        const facemesh_master<lm_count>& master;
        dlibx::net::vector net;

        facemesh_net(core::thread_data& td)
            : master(core::get<const facemesh_master<lm_count> >(td.context,td)),
              net(*master) {
            if (!net.input_extractor)
                throw std::runtime_error(
                    "facemesh net does not have input extractor");
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

            const auto width = net.input_extractor->width;
            const auto height = net.input_extractor->height;
            const auto rgb =
                extract_region(raw, rbox.center.x, rbox.center.y,
                               rbox.width, rbox.height,
                               rbox.angle * float(180/M_PI),
                               width, height, raw_image::pixel::rgb24);

            std::array<dlib::resizable_tensor,2> out;
            if (net(rgb, out) != 2)
                throw std::runtime_error(
                    "facemesh net produced the wrong number of outputs");

            // coodinates on given image
            detected_coordinates result(
                lm_count == 68 ? dt::mesh68 : dt::mesh478);
            result.landmarks.reserve(lm_count);
            dlib::tensor const* lmt;

            // quality assessment
            // try to match 0 to 10 range of dlib68 quality assessment
            if (out.front().size() == 1) {
                result.confidence = (*out.front().host() + 12)/3;
                lmt = &out.back();
            }
            else if (out.back().size() == 1) {
                result.confidence = (*out.back().host() + 12)/3;
                lmt = &out.front();
            }
            if (result.confidence > 10)
                result.confidence = 10;
            else if (result.confidence < 0)
                result.confidence = 0;

            if (lmt->size() != 2*lm_count)
                throw std::runtime_error(
                    "facemesh net produced incorrect number of landmarks");

            float const* out0 = lmt->host();
            for (unsigned i = 0; i < lm_count; ++i, out0 += 2) {
                auto pt = raw_image::point2f {
                    out0[0] - float(width)/2, out0[1] - float(height)/2
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
internal::dlib_factory<lm::mesh68>(core::context_data&) {
    struct lmdet : landmarks_base {
        detected_coordinates
        operator()(const detected_coordinates& dc,
                   const raw_image::plane& image,
                   core::thread_data& td,
                   unsigned contrast_correction) const override {
            return core::get<facemesh_net<68> >(td.thread,td)
                (dc, image, contrast_correction);
        }
    };
    return [](auto& data, const auto&) {
        core::get<const facemesh_master<68> >(data.context,data);
        return std::make_unique<lmdet>();
    };
}

template <>
internal::landmarks_factory_function
internal::dlib_factory<lm::mesh478>(core::context_data&) {
    struct lmdet : landmarks_base {
        detected_coordinates
        operator()(const detected_coordinates& dc,
                   const raw_image::plane& image,
                   core::thread_data& td,
                   unsigned contrast_correction) const override {
            return core::get<facemesh_net<478> >(td.thread,td)
                (dc, image, contrast_correction);
        }
    };
    return [](auto& data, const auto&) {
        core::get<const facemesh_master<478> >(data.context,data);
        return std::make_unique<lmdet>();
    };
}
