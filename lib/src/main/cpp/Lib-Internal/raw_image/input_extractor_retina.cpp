
#include "input_extractor_retina.hpp"
//#include "rotated_box.hpp"

#include "pixels.hpp"
//#include <raw_image/drawing.hpp>
#include "point_rounding.hpp"

#include <applog/core.hpp>
#include <stdext/bit.hpp>

#include <algorithm>

using namespace raw_image;


static inline bool is_positive(const std::string& s) {
    for (auto c : s)
        if (c < '0' || '9' < c)
            return false;
    return !s.empty() && '0' < s.front();
}


/****************  retina_extractor  ****************/

void retina_extractor::normalize_depth(plane& img) {
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
    std::nth_element(vec.begin(), next(vec.begin(),long(n)), vec.end());
    int thres = vec[n];
    FILE_LOG(logDETAIL) << "facedepth_extractor: min distance " << thres;
    thres += 200;

    // remap distances so that z' = thres - z = 200 + min_dist - z
    img.layout = pixel::a8;
    auto src_iter = u16data.cbegin();
    for (auto&& line : pixels<uint8_t>(img))
        for (auto& z : line)
            z = stdx::round_from(thres - int(*src_iter++));
}

plane_ptr
retina_extractor::extract_from_chip(const multi_plane_arg& image,
                                    const scaled_chip& cd) const {
    if (image.size() <= 1 && layout != pixel::a8)
        return extract_image_chip(image, cd, layout);

    // find depth channel
    const plane* dplane = nullptr;
    std::vector<plane> color_only;
    color_only.reserve(image.size());
    for (auto&& p : image) {
        if (p.layout != pixel::a16_le)
            color_only.emplace_back(p);
        else if (!dplane)
            dplane = &p;
        // else there are multiple depth planes ?
    }
    if (layout != pixel::rgba32 &&
        layout != pixel::a8)
        return extract_image_chip(color_only, cd, layout);
    if (!dplane)
        throw std::invalid_argument("depth image has invalid pixel layout");

    // extract depth chip
    static_assert(stdx::endian::native == stdx::endian::little);
    auto depth = extract_image_chip(*dplane, cd, pixel::a16_le);
    normalize_depth(*depth);
    if (layout != pixel::rgba32)
        return depth;

    // combine color chip with depth chip
    auto r = extract_image_chip(color_only, cd, layout);
    copy_channel(depth, channel::ch0, r, channel::alpha);
    return r;
}

scaled_chip
retina_extractor::chip_from_pts(const std::vector<point2f>& pts) const {
    auto lm = stdx::span<const point2f>(pts);
    if (2 <= lm.size() && lm.size() <= 8)
        lm = lm.first(lm.size() - 2); // remove bounding box corners
    // note: final lm must be 5 or 6 landmarks (from v7 or v8 detector)
    const auto rbox = retina_align(lm, scale, yoffset);
    return { rbox, width, height };
}

std::tuple<unsigned, float, float, pixel_layout>
raw_image::retina_decode(std::string_view name) {
    auto t = std::tuple(0u, 0.0f, 0.0f, pixel::none);

    if (name.size() < 12 || name.compare(0,6,"retina") != 0)
        return t;
    name.remove_prefix(6);

    // assert(name.size() >= 5);
    if (name.compare(name.size()-3,3,"rgb") == 0) {
        std::get<3>(t) = pixel::rgb24;
        name.remove_suffix(3);
    }
    else if (name.compare(name.size()-3,3,"yuv") == 0) {
        std::get<3>(t) = pixel::yuv;
        name.remove_suffix(3);
    }
    else if (name.compare(name.size()-4,4,"gray") == 0) {
        std::get<3>(t) = pixel::gray8;
        name.remove_suffix(4);
    }
    else if (name.compare(name.size()-4,4,"rgbd") == 0) {
        std::get<3>(t) = pixel::rgba32;
        name.remove_suffix(4);
    }
    else if (name.compare(name.size()-5,5,"depth") == 0) {
        std::get<3>(t) = pixel::a8;
        name.remove_suffix(5);
    }
    else
        return t;

    const auto sep0 = name.find('*');
    const auto sep1 = name.find_first_of("+-");
    if (!(0 < sep0 && sep0 + 1 < sep1 && sep1 < name.size() - 1))
        return t;

    const auto sdim = std::string(name, 0, sep0);
    if (!is_positive(sdim))
        return t;

    const auto sscale  = std::string(name, sep0 + 1, sep1 - sep0 - 1);
    const auto soffset = std::string(name, sep1, name.size() - sep1);

    std::get<0>(t) = unsigned(atol(sdim.c_str()));
    std::get<1>(t) = float(atof(sscale.c_str()));
    std::get<2>(t) = float(atof(soffset.c_str()));
    return t;
}

std::unique_ptr<const input_extractor>
raw_image::retina_factory(const std::string_view& name) {
    const auto fc = retina_decode(name);
    if (std::get<0>(fc) <= 0)
        return nullptr;
    return std::make_unique<retina_extractor>(
        std::string(name), std::get<0>(fc),
        std::get<1>(fc), std::get<2>(fc),
        std::get<3>(fc));
}
