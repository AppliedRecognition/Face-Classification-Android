
#include <raw_image/input_extractor.hpp>

#include <mat.h>

#include "models.hpp"
#include "extract.hpp"

#include <rec/internal_prototype_1.hpp>
#include <rec/fpvc.hpp>

#include <det/types.hpp>
#include <core/thread_data.hpp>

#include <raw_image/point_rounding.hpp>
#include <raw_image/ncnn.hpp>

#include <applog/core.hpp>

#include <numeric>
#include <sstream>


using rec::version_type;


static auto get_chip_details(const det::face_coordinates& coordinates,
                             const raw_image::input_extractor& extractor) {
    // extract face chip
    const det::detected_coordinates* dcp = nullptr;
    for (const auto& dc : coordinates)
        if (dc.landmarks.size() > 2)
            dcp = &dc;
    if (!dcp) {
        std::stringstream ss;
        ss << "template extraction requires landmarks";
        for (const auto& dc : coordinates) {
            const auto x = stdx::round((dc.eye_left.x+dc.eye_right.x)/2);
            const auto y = stdx::round((dc.eye_left.y+dc.eye_right.y)/2);
            ss << " (" << int(dc.type)
               << ',' << dc.landmarks.size()
               << ',' << x << ',' << y << ')';
        }
        FILE_LOG(logERROR) << ss.str();
        throw std::logic_error(
            "template extraction requires landmarks");
    }
    std::vector<raw_image::point2f> pts;
    pts.reserve(dcp->landmarks.size());
    for (auto&& p : dcp->landmarks)
        pts.push_back(raw_image::round_from(p));
    return std::make_pair(extractor(pts), extractor.layout);
}

rec::rotated_box
rec::ncnn::bounding_box(const det::face_coordinates& coordinates,
                        version_type ver,
                        const core::context_data& cd) {
    const auto model_ptr = core::get<context_map>(cd.context).load(ver, &load_shared, ver, cd).first;
    if (!model_ptr || !model_ptr->extractor)
        throw std::runtime_error("failed to load model");
    return to_rotated_box(get_chip_details(coordinates, *model_ptr->extractor).first);
}

static bool warn_no_color = false;

rec::prototype_ptr
rec::ncnn::extract(const raw_image::multi_plane_arg& image,
                   const rotated_box& rbox,
                   version_type ver,
                   core::thread_data& td) {

    if (image.size() == 1 &&
        bytes_per_pixel(image.front().layout) == 1 &&
        !warn_no_color) {
        warn_no_color = true;
        FILE_LOG(logWARNING) << "rec: grayscale image used to extract template";
    }

    auto&& net_pair = core::get<context_map>(td.context)
        .load(ver, &load_shared, ver, td);
    auto& model_ptr = net_pair.first;
    if (!model_ptr || !model_ptr->extractor)
        throw std::runtime_error("failed to load model");
    auto& extractor = *model_ptr->extractor;

    // create chip_details object from rotated_box
    auto chip = raw_image::scaled_chip(rbox, extractor.width, extractor.height);

    // extract face chip
    auto face_chip = extract_image_chip(image, chip, extractor.layout);
    auto in = to_ncnn_rgb(face_chip);

    // extractor setup
    auto& net = model_ptr->net;
    auto ex = net.create_extractor();
    //ex.set_num_threads(num_threads);
    ex.input("data", in);

    // neural net: face chip -> vector
    ::ncnn::Mat desc;
    ex.extract("fc1", desc);
    const float* first = desc, *last = first + desc.total();

    auto v8 = internal::fpvc_vector_compress(first, last);
    auto proto = internal::prototype_1::make_shared(move(net_pair.second), move(v8));
    proto->thumb = move(face_chip);
    return proto;
}

rec::prototype_ptr
rec::ncnn::extract(const raw_image::multi_plane_arg& image,
                   const det::face_coordinates& coordinates,
                   version_type ver,
                   core::thread_data& td) {

    if (image.size() == 1 &&
        bytes_per_pixel(image.front().layout) == 1 &&
        !warn_no_color) {
        warn_no_color = true;
        FILE_LOG(logWARNING) << "rec: grayscale image used to extract template";
    }

    auto&& net_pair = core::get<context_map>(td.context)
        .load(ver, &load_shared, ver, td);
    auto& model_ptr = net_pair.first;
    if (!model_ptr || !model_ptr->extractor)
        throw std::runtime_error("failed to load model");
    auto& extractor = *model_ptr->extractor;

    // extract face chip
    const auto cdp = get_chip_details(coordinates, extractor);
    auto face_chip = extract_image_chip(image, cdp.first, cdp.second);
    auto in = to_ncnn_rgb(face_chip);

    // extractor setup
    auto& net = model_ptr->net;
    auto ex = net.create_extractor();
    //ex.set_num_threads(num_threads);
    ex.input("data", in);

    // neural net: face chip -> vector
    ::ncnn::Mat desc;
    ex.extract("fc1", desc);
    const float* first = desc, *last = first + desc.total();

    auto v8 = internal::fpvc_vector_compress(first, last);
    auto proto = internal::prototype_1::make_shared(move(net_pair.second), move(v8));
    proto->thumb = move(face_chip);
    return proto;
}

rec::prototype_ptr
rec::ncnn::from_face_chip(raw_image::plane_ptr face_chip,
                          version_type ver,
                          core::thread_data& td) {

    auto&& net_pair = core::get<context_map>(td.context)
        .load(ver, &load_shared, ver, td);
    auto& model_ptr = net_pair.first;
    if (!model_ptr)
        throw std::runtime_error("failed to load model");

    // extractor setup
    auto ex = model_ptr->net.create_extractor();
    //ex.set_num_threads(num_threads);
    auto in = to_ncnn_rgb(face_chip);
    ex.input("data", in);

    // neural net: face chip -> vector
    ::ncnn::Mat desc;
    ex.extract("fc1", desc);
    const float* first = desc, *last = first + desc.total();

    auto v8 = internal::fpvc_vector_compress(first, last);
    auto proto = internal::prototype_1::make_shared(move(net_pair.second), move(v8));
    proto->thumb = move(face_chip);
    return proto;
}
