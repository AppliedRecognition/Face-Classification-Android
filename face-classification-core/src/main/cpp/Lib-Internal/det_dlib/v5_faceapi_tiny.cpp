
#define __FBLIB_DET_PRIVATE_INTERNAL_USE_ONLY__
#include "yolonet.hpp"
#include "internal.hpp"

#include <dlibx/raw_image.hpp>
#include <raw_image/transform.hpp>

#include <applog/core.hpp>


using namespace det;
using namespace det::internal;


namespace {
    inline auto sig(float x) {
        return 1.0f / (1 + std::exp(-x));
    }

    using net_type = yolo::tiny_face_detector<yolo::lmcon>;

    struct master_detector : dlib_object<net_type> {
        master_detector(core::context_data& data)
            : dlib_object(data, models::type::face_detector,
                          models::face_detector::tiny) {
        }
    };

    struct face_detector {
        net_type net;
        inline net_type& operator*() { return net; }
        face_detector(core::thread_data& td)
            : net(*core::get<const master_detector>(td.context,td)) {
        }
    };

}

template<>
detector_factory_function
internal::dlib_factory<5>(core::context_data&) {

    struct v5 : detector_base {
        void prepare_thread(core::job_context& jc,
                            const detection_settings&,
                            unsigned) override {
            core::get<face_detector>(jc.data.thread,jc.data);
        }

        std::function<detection_result(core::job_context&)>
        detection_job(const detection_input& input,
                      json::value* diag) const override {
            return dlib_job<5>{input,diag};
        }
    };

    return [](auto& data, const auto&) {
        core::get<const master_detector>(data.context,data);
        return std::make_unique<v5>();
    };
}

template<>
detection_result dlib_job<5>::operator()(core::job_context& jc) {
    FILE_LOG(logDETAIL) << "job: [" << jc.job.order() << "] fapi_tiny";

    auto& image = input.image;
    detection_result result;

    if (image.width < 10 || image.height < 10) {
        FILE_LOG(logWARNING) << "image too small"
            " -- not doing face detection";
        return result;
    }

    // output tensor size based on size_range and image aspect ratio
    const auto fw = float(image.width);
    const auto fh = float(image.height);
    const auto scale = std::sqrt(150*input.settings.size_range / (fw*fh));
    const auto rows = std::max(2, int(std::lround(scale * fh)));
    const auto cols = std::max(2, int(std::lround(scale * fw)));

    // todo: maybe we should have a maximum up-scaling here ?

    // resize image for input tensor
    const auto in_width = unsigned(cols-1)*32;
    const auto in_height = unsigned(rows-1)*32;
    FILE_LOG(logDETAIL) << "scaling image from "
                        << image.width << 'x' << image.height
                        << " to " << in_width << 'x' << in_height;
    const auto it = input.settings.fast_scaling ?
        raw_image::inter::nearest : raw_image::inter::bilinear;
    const auto scaled =
        copy_resize(image, in_width, in_height,
                    bytes_per_pixel(image.layout) == 1 ?
                    image.layout : raw_image::pixel::rgb24, it);

    // create input tensor
    auto& detector = *core::get<face_detector>(jc.data.thread, jc.data);
    dlib::resizable_tensor in;
    if (bytes_per_pixel(scaled->layout) == 1) {
        raw_image::fixed_dlib_image<raw_image::rgb_from_gray8> img(*scaled);
        input_layer(detector).to_tensor(&img, &img+1, in);
    }
    else {
        raw_image::fixed_dlib_image<dlib::rgb_pixel> img(*scaled);
        input_layer(detector).to_tensor(&img, &img+1, in);
    }

    // run detector to get output tensor
    auto& out = detector.forward(in);
    FILE_LOG(logDETAIL) << "detection: "
                        << in.nc() << 'x' << in.nr() << 'x' << in.k()
                        << " -> "
                        << out.nc() << 'x' << out.nr() << 'x' << out.k();
    assert(out.nr() == rows && out.nc() == cols &&
           out.k() == 5 * std::size(yolo::tiny_face_detector_boxes) &&
           out.num_samples() == 1);
    const auto image_size = rows * cols;

    const auto threshold = input.settings.confidence_threshold;
    const auto sw = fw / float(cols-1);
    const auto sh = fh / float(rows-1);

    // scan output tensor for detections
    std::vector<face_coordinates> faces;
    const float* src = out.host() + 4*image_size;
    for (auto& box : yolo::tiny_face_detector_boxes) {
        for (int j = 0; j < rows; ++j)
            for (int i = 0; i < cols; ++i, ++src) {
                auto p = src;
                const auto conf = *p;
                if (conf > threshold) {
                    p -= image_size;
                    const auto h = std::exp(*p) * box[1] * sh;
                    p -= image_size;
                    const auto w = std::exp(*p) * box[0] * sw;
                    p -= image_size;
                    const auto cy = (sig(*p) + float(j-1)) * sh;
                    p -= image_size;
                    auto cx = (sig(*p) + float(i-1)) * sw;
                    if (image.rotate & 4)
                        cx = fw - cx;

                    auto& dc = faces.emplace_back().emplace_back(dt::v5_fapi);
                    dc.confidence = conf;
                    dc.landmarks.push_back( { cx - w/2, cy - h/2 } );
                    dc.landmarks.push_back( { cx + w/2, cy + h/2 } );
                    dc.set_eye_coordinates_from_landmarks();
                }
            }
        src += 4*image_size;
    }

    return internal::landmark_detection(jc, input, move(faces));
}

