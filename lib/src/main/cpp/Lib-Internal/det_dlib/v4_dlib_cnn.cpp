
#include <dlibx/dnn_lmcon.hpp>
#include <dlib/dnn.h>
#include <dlibx/raw_image.hpp>
#include <raw_image/transform.hpp>

#define __FBLIB_DET_PRIVATE_INTERNAL_USE_ONLY__
#include "internal.hpp"

#include <applog/core.hpp>


namespace {
    using namespace dlib;
    using namespace dlibx;
    
    template <long num_filters, typename SUBNET>
    using con5d = lmcon<num_filters,5,5,2,2,SUBNET>;
    template <long num_filters, typename SUBNET>
    using con5 = lmcon<num_filters,5,5,1,1,SUBNET>;

    template <typename SUBNET>
    using downsampler = relu<affine<con5d<32, relu<affine<con5d<32, relu<affine<con5d<16,SUBNET>>>>>>>>>;
    template <typename SUBNET>
    using rcon5 = relu<affine<con5<45,SUBNET>>>;

    using net_type = loss_mmod<lmcon<1,9,9,1,1,rcon5<rcon5<rcon5<downsampler<input_rgb_image_pyramid<pyramid_down<6>>>>>>>>;
}

using namespace det;
using namespace det::internal;

namespace {
    struct cnn_master_detector : dlib_object<net_type> {
        cnn_master_detector(core::context_data& data)
            : dlib_object(data, models::type::face_detector,
                          models::face_detector::cnn) {
        }
    };

    struct cnn_face_detector {
        net_type net;
        inline net_type& operator*() { return net; }
        cnn_face_detector(core::thread_data& td)
            : net(*core::get<const cnn_master_detector>(td.context,td)) {
        }
    };

    dlib::rectangle mirror(dlib::rectangle r, long width) {
        const auto x = --width - r.right();
        r.set_right(width - r.left());
        r.set_left(x);
        return r;
    }

    template <typename T>
    struct single_object_view_t {
        T* ptr;
        inline const T* begin() const { return ptr; }
        inline const T* end()   const { return ptr + 1; }
    };
    template <typename T>
    inline auto single_object_view(T& obj) {
        return single_object_view_t<T>{&obj};
    }
}

template<>
detector_factory_function
internal::dlib_factory<4>(core::context_data&) {

    struct v4 : det::internal::detector_base {
        void prepare_thread(core::job_context& jc,
                            const detection_settings&,
                            unsigned) override {
            core::get<cnn_face_detector>(jc.data.thread,jc.data);
        }

        std::function<detection_result(core::job_context&)>
        detection_job(const detection_input& input,
                      json::value* diag) const override {
            return dlib_job<4>{input,diag};
        }
    };

    return [](auto& data, const auto&) {
        core::get<const cnn_master_detector>(data.context,data);
        return std::make_unique<v4>();
    };
}

static inline bool good_color_space(const raw_image::plane& img) {
    if (bytes_per_pixel(img.layout) == 1)
        return true;
    switch (img.layout) {
    case raw_image::pixel::rgb24:
    case raw_image::pixel::bgr24:
        return true;
    default: break;
    }
    return false;
}

template <>
detection_result dlib_job<4>::operator()(core::job_context& jc) {
    FILE_LOG(logDETAIL) << "job: [" << jc.job.order() << "] dlib";

    auto& image = input.image;
    detection_result result;

    const auto desired_pix = 500 * 1000 * input.settings.size_range;
    if (desired_pix < 10) {
        FILE_LOG(logWARNING) << "detection.size_range too small"
            " -- not doing face detection";
        return result;
    }
    
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
                                << image.width << 'x' << image.height
                                << " to " << dw << 'x' << dh;
            const auto it = input.settings.fast_scaling ?
                raw_image::inter::nearest : raw_image::inter::bilinear;
            if (good_color_space(dimg))
                dimg_buf = copy_resize(dimg, dw, dh, it);
            else
                dimg_buf = copy_resize(dimg, dw, dh,
                                       raw_image::pixel::rgb24, it);
            dimg = *dimg_buf;
        }
        else
            scale = 1.0f;
    }
    else
        scale = 1.0f;

    if (!good_color_space(dimg)) {
        FILE_LOG(logDETAIL) << "copying image "
                            << image.width << 'x' << image.height
                            << " to change color space";
        dimg_buf = copy(dimg, raw_image::pixel::rgb24);
        dimg = *dimg_buf;
    }

    const auto threshold = input.settings.confidence_threshold;
    std::vector<std::vector<dlib::mmod_rect> > vdets;
    auto& detector = *core::get<cnn_face_detector>(jc.data.thread, jc.data);
    if (bytes_per_pixel(dimg.layout) == 1) {
        raw_image::fixed_dlib_image<raw_image::rgb_from_gray8> fdimg(dimg);
        vdets = detector.process_batch(
            single_object_view(fdimg), 1, threshold);
    }
    else switch (dimg.layout) {
        case raw_image::pixel::rgb24: {
            raw_image::fixed_dlib_image<dlib::rgb_pixel> fdimg(dimg);
            vdets = detector.process_batch(
                single_object_view(fdimg), 1, threshold);
            break;
        }
        case raw_image::pixel::bgr24: {
            raw_image::fixed_dlib_image<dlib::bgr_pixel> fdimg(dimg);
            vdets = detector.process_batch(
                single_object_view(fdimg), 1, threshold);
            break;
        }
        default:
            throw std::invalid_argument("unsupported color_space");
        }
    assert(vdets.size() == 1);
    auto& dets = vdets.front();
    FILE_LOG(logDETAIL) << "dlib faces detected: " << dets.size();
    dimg_buf = nullptr;
    
    if (image.scale > 0)
        scale *= float(1 << image.scale);
    else if (image.scale < 0)
        scale /= float(1 << -image.scale);
    
    std::vector<face_coordinates> faces;
    faces.reserve(dets.size());
    for (const auto& d : dets) {
        const auto r = (image.rotate&4) ?
            mirror(d.rect,long(image.width)) : d.rect;

        auto& dc = faces.emplace_back().emplace_back(dt::v4_dlib);
        dc.confidence = stdx::round_from(d.detection_confidence);
        dc.landmarks.push_back( {
                scale * (float(r.left()) - 0.75f),
                scale * (float(r.top()) - 0.25f)
            });
        dc.landmarks.push_back( {
                scale * (float(r.right()) + 0.75f),
                scale * (float(r.bottom()) + 0.25f)
            });
        dc.set_eye_coordinates_from_landmarks();
    }

    return internal::landmark_detection(jc, input, move(faces));
}
