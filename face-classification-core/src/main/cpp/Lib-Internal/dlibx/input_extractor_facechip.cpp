
#include "input_extractor_facechip.hpp"

#include <raw_image/pixels.hpp>
#include <raw_image/drawing.hpp>

#include <applog/core.hpp>
#include <stdext/bit.hpp>


using namespace dlibx;


static inline bool is_positive(const std::string& s) {
    for (auto c : s)
        if (c < '0' || '9' < c)
            return false;
    return !s.empty() && '0' < s.front();
}

template <typename PT>
static auto to_dpoint(const std::vector<PT>& pts) {
    std::vector<dlib::dpoint> dpts;
    dpts.reserve(pts.size());
    for (auto& p : pts)
        dpts.emplace_back(raw_image::round_from(p));
    return dpts;
}


/****************  facechip_extractor  ****************/

raw_image::scaled_chip
facechip_extractor::chip_from_pts(const std::vector<raw_image::point2f>& pts) const {
    return get_face_chip_details(to_dpoint(pts), width, padding);
}

std::tuple<unsigned, float, raw_image::pixel_layout>
dlibx::facechip_decode(std::string_view name) {
    auto t = std::tuple(0u, 0.0f, raw_image::pixel::none);

    if (name.size() < 14 || name.compare(0,8,"facechip") != 0)
        return t;
    name.remove_prefix(8);

    if (name.compare(name.size()-3,3,"rgb") == 0) {
        std::get<2>(t) = raw_image::pixel::rgb24;
        name.remove_suffix(3);
    }
    else if (name.compare(name.size()-3,3,"yuv") == 0) {
        std::get<2>(t) = raw_image::pixel::yuv;
        name.remove_suffix(3);
    }
    else if (name.compare(name.size()-4,4,"gray") == 0) {
        std::get<2>(t) = raw_image::pixel::gray8;
        name.remove_suffix(4);
    }
    else
        return t;

    const auto sep = name.find_first_of("+-");
    if (sep <= 0 || name.size()-1 <= sep)
        return t;

    const auto sdim = std::string(name, 0, sep);
    if (!is_positive(sdim))
        return t;

    name.remove_prefix(sep);
    auto spad = std::string(name);
    for (std::size_t i = 1; i < spad.size(); ++i)
        if ((spad[i] < '0' || '9' < spad[i]) && spad[i] != '.')
            return t;
    if (spad[1] == '0' && spad[2] != '.')
        spad.insert(2,1,'.');

    std::get<0>(t) = unsigned(atol(sdim.c_str()));
    std::get<1>(t) = float(atof(spad.c_str()));
    return t;
}

std::unique_ptr<const input_extractor>
dlibx::facechip_factory(const std::string_view& name) {
    const auto fc = facechip_decode(name);
    if (std::get<0>(fc) <= 0)
        return nullptr;
    return std::make_unique<facechip_extractor>(
        std::string(name), std::get<0>(fc), std::get<1>(fc), std::get<2>(fc));
}


/****************  lm68chip_extractor  ****************/

static const std::initializer_list<unsigned> dlib68_lines[] = {
    { 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16 },  // outline

    { 17,18,19,20,21 },  // left eyebrow
    { 22,23,24,25,26 },  // right eyebrow

    { 27,28,29,30,31,32,33,34,35 },  // nose

    { 36,37,38,39,40,41 },  // left eye
    { 42,43,44,45,46,47 },  // right eye

    { 48,49,50,51,52,53,54,55,56,57,58,59,48 },  // outer mouth
    { 60,61,62,63,64,65,66,67,60 }  // inner mouth
};

raw_image::scaled_chip
lm68chip_extractor::chip_from_pts(const std::vector<raw_image::point2f>& pts) const {
    return get_face_chip_details(to_dpoint(pts), width, padding);
}

raw_image::plane_ptr
lm68chip_extractor::extract_from_pts(const raw_image::multi_plane_arg& image,
                                     const std::vector<raw_image::point2f>& pts) const {
    const auto dpts = to_dpoint(pts);
    const auto cd = get_face_chip_details(dpts, width, padding);

    std::vector<raw_image::point2i> ipts;
    ipts.reserve(pts.size());
    const auto map = get_mapping_to_chip(cd); // point_transform_affine
    for (auto& p : dpts)
        ipts.emplace_back(raw_image::round_from(map(p)));
    const auto draw_lines = [&](auto&& img) {
        for (auto&& idx : dlib68_lines) {
            auto it = std::begin(idx);
            auto i0 = *it;
            while (++it != std::end(idx)) {
                const auto i1 = *it;
                line(img, ipts[i0], ipts[i1], raw_image::color_white, 1);
                i0 = i1;
            }
        }
        for (auto& p : ipts) // single black pixel per landmark
            circle(img, p, raw_image::color_black, 0);
    };

    auto img = extract_image_chip(image, cd, layout);
    if (layout != raw_image::pixel::rgba32)
        draw_lines(img);
    else {
        const auto alpha = create(width, height, raw_image::pixel::gray8);
        fill(alpha, raw_image::pixel_color{0x808080});
        draw_lines(alpha);
        copy_channel(alpha, raw_image::channel::ch0,
                     img, raw_image::channel::alpha);
    }
    return img;
}

raw_image::plane_ptr
lm68chip_extractor::extract_from_chip(
    const raw_image::multi_plane_arg&, const raw_image::scaled_chip&) const {
    throw std::logic_error("lm68chip extractor requires landmarks");
}

std::tuple<unsigned, float, raw_image::pixel_layout>
dlibx::lm68chip_decode(std::string_view name) {
    auto t = std::tuple(0u, 0.0f, raw_image::pixel::none);

    if (name.size() < 14 || name.compare(0,8,"lm68chip") != 0)
        return t;
    name.remove_prefix(8);

    if (name.compare(name.size()-3,3,"yuv") == 0) {
        std::get<2>(t) = raw_image::pixel::yuv;
        name.remove_suffix(3);
    }
    else if (name.compare(name.size()-3,3,"rgb") == 0) {
        std::get<2>(t) = raw_image::pixel::rgb24;
        name.remove_suffix(3);
    }
    else if (name.compare(name.size()-4,4,"rgba") == 0) {
        std::get<2>(t) = raw_image::pixel::rgba32;
        name.remove_suffix(4);
    }
    else if (name.compare(name.size()-4,4,"gray") == 0) {
        std::get<2>(t) = raw_image::pixel::gray8;
        name.remove_suffix(4);
    }
    else
        return t;

    const auto sep = name.find_first_of("+-");
    if (sep <= 0 || name.size()-1 <= sep)
        return t;

    const auto sdim = std::string(name, 0, sep);
    if (!is_positive(sdim))
        return t;

    name.remove_prefix(sep);
    auto spad = std::string(name);
    for (std::size_t i = 1; i < spad.size(); ++i)
        if ((spad[i] < '0' || '9' < spad[i]) && spad[i] != '.')
            return t;
    if (spad[1] == '0' && spad[2] != '.')
        spad.insert(2,1,'.');

    std::get<0>(t) = unsigned(atol(sdim.c_str()));
    std::get<1>(t) = float(atof(spad.c_str()));
    return t;
}

std::unique_ptr<const input_extractor>
dlibx::lm68chip_factory(const std::string_view& name) {
    const auto fc = lm68chip_decode(name);
    if (std::get<0>(fc) <= 0)
        return nullptr;
    return std::make_unique<lm68chip_extractor>(
        std::string(name), std::get<0>(fc), std::get<1>(fc), std::get<2>(fc));
}


/****************  facedepth_extractor  ****************/

void facedepth_extractor::normalize_depth(raw_image::plane& img) {
    if (img.bytes_per_line != 2 * img.width)
        throw std::runtime_error(
            "facedepth_extractor: image is not packed as expected");

    const auto u16data =
        stdx::span<uint16_t>(reinterpret_cast<uint16_t*>(img.data),
                             img.width * img.height);

    // change holes from z = 0 to z = 65535
    std::vector<uint16_t> vec;
    vec.reserve(u16data.size());
    for (auto& z : u16data) {
        if (z == 0) z = 65535; // holes become max depth
        vec.push_back(z);
    }

    // find 1st percentile of distances
    const auto n = vec.size() / 100;
    std::nth_element(vec.begin(), vec.begin()+long(n), vec.end());
    int thres = vec[n];
    FILE_LOG(logDETAIL) << "facedepth_extractor: min distance " << thres;
    thres += 200;

    // remap distances so that z' = thres - z = 200 + min_dist - z
    img.layout = raw_image::pixel::gray8;
    auto src_iter = u16data.cbegin();
    for (auto&& line : raw_image::pixels<uint8_t>(img))
        for (auto& z : line)
            z = stdx::round_from(thres - int(*src_iter++));
}

raw_image::plane_ptr
facedepth_extractor::extract_depth_chip(const raw_image::multi_plane_arg& image,
                                        const raw_image::scaled_chip& cd) const {
    if (image.size() != 1 || image.front().layout != raw_image::pixel::a16_le)
        throw std::invalid_argument("depth image has invalid pixel layout");
    static_assert(stdx::endian::native == stdx::endian::little);
    return extract_image_chip(image, cd, raw_image::pixel::a16_le);
}

raw_image::plane_ptr
facedepth_extractor::extract_from_chip(const raw_image::multi_plane_arg& image,
                                       const raw_image::scaled_chip& cd) const {
    auto img = extract_depth_chip(image,cd);
    normalize_depth(*img);
    return img;
}

raw_image::scaled_chip
facedepth_extractor::chip_from_pts(const std::vector<raw_image::point2f>& pts) const {
    return get_face_chip_details(to_dpoint(pts), width, padding);
}

std::tuple<unsigned, float>
dlibx::facedepth_decode(std::string_view name) {
    auto t = std::tuple(0u, 0.0f);

    if (name.size() < 10 || name.compare(0,9,"facedepth") != 0)
        return t;
    name.remove_prefix(9);

    const auto sep = name.find_first_of("+-");
    if (sep <= 0 || name.size()-1 <= sep)
        return t;

    const auto sdim = std::string(name, 0, sep);
    if (!is_positive(sdim))
        return t;

    name.remove_prefix(sep);
    auto spad = std::string(name);
    for (std::size_t i = 1; i < spad.size(); ++i)
        if ((spad[i] < '0' || '9' < spad[i]) && spad[i] != '.')
            return t;
    if (spad[1] == '0' && spad[2] != '.')
        spad.insert(2,1,'.');

    std::get<0>(t) = unsigned(atol(sdim.c_str()));
    std::get<1>(t) = float(atof(spad.c_str()));
    return t;
}

std::unique_ptr<const input_extractor>
dlibx::facedepth_factory(const std::string_view& name) {
    const auto fc = facedepth_decode(name);
    if (std::get<0>(fc) <= 0)
        return nullptr;
    return std::make_unique<facedepth_extractor>(
        std::string(name), std::get<0>(fc), std::get<1>(fc));
}
