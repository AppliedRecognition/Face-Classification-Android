
#include "input_extractor_facechip.hpp"
#include "input_extractor_license.hpp"
#include "input_extractor_eyecrop.hpp"
#include "input_extractor_box.hpp"

/*
#include "rotated_box.hpp"

#include <applog/core.hpp>

#include <shared_mutex>
#include <unordered_map>
*/

using namespace dlibx;


static inline bool is_positive(const std::string& s) {
    for (auto c : s)
        if (c < '0' || '9' < c)
            return false;
    return !s.empty() && '0' < s.front();
}


/****************  class eyecrop_extractor  ****************/

raw_image::scaled_chip
eyecrop_extractor::chip_from_pts(const std::vector<raw_image::point2f>& pts) const {
    raw_image::point2f c;
    switch (pts.size()) {
    case 2: // eyes only
    case 7: // retina7
        c = 0.5f * (pts[0] + pts[1]);
        break;

    case 5: // dlib5
        c = 0.25f * (pts[0] + pts[1] + pts[2] + pts[3]);
        break;

    case 68: // dlib68
        c = {0,0};
        for (unsigned i = 36; i < 48; ++i)
            c += pts[i];
        c *= 1.0f / 12;
        break;

    default:
        throw std::invalid_argument("landmarks vector has invalid size");
    }

    dlib::chip_details cd;
    cd.rect = {
        (2*c.x - float(width)  - 0.5f) / 2, // left
        (2*c.y - float(height) - 0.5f) / 2, // top
        (2*c.x + float(width)  - 1.5f) / 2, // right
        (2*c.y + float(height) - 1.5f) / 2, // bottom
    };
    cd.angle = 0;
    cd.rows = height;
    cd.cols = width;
    return cd;
}

raw_image::plane_ptr
eyecrop_extractor::extract_from_chip(const raw_image::multi_plane_arg& image,
                                     const raw_image::scaled_chip& chip) const {
    const dlib::chip_details cd = chip;
    auto cx = float(std::round((1 + cd.rect.left() + cd.rect.right())/2));
    if (width&1) cx += 0.5f; // should this be minus?
    auto cy = float(std::round((1 + cd.rect.top() + cd.rect.bottom())/2));
    if (height&1) cy += 0.5f;
    return extract_region(image, cx, cy, float(width), float(height), 0,
                          width, height, layout);
}

std::tuple<unsigned, unsigned, raw_image::pixel_layout>
dlibx::eyecrop_decode(std::string_view name) {
    auto t = std::tuple(0u, 0u, raw_image::pixel::none);
    if (name.size() < 13 || name.compare(0,7,"eyecrop") != 0)
        return t;
    name.remove_prefix(7);

    auto end = name.size() - 3;
    if (name.compare(end, 3, "rgb") == 0)
        std::get<2>(t) = raw_image::pixel::rgb24;
    else if (name.compare(end, 3, "yuv") == 0)
        std::get<2>(t) = raw_image::pixel::yuv;
    else if (name.compare(--end,4,"gray") == 0)
        std::get<2>(t) = raw_image::pixel::gray8;
    else
        return t;
    name = name.substr(0,end);

    const auto sep = name.find('x');
    if (!(0 < sep && sep < name.size() - 1))
        return t;

    const auto w = std::string(name, 0, sep);
    if (!is_positive(w))
        return t;
    const auto h = std::string(name, sep+1, name.size()-sep-1);
    if (!is_positive(h))
        return t;

    std::get<0>(t) = unsigned(atol(w.c_str()));
    std::get<1>(t) = unsigned(atol(h.c_str()));
    return t;
}

std::unique_ptr<const input_extractor>
dlibx::eyecrop_factory(const std::string_view& name) {
    const auto ld = eyecrop_decode(name);
    if (std::get<0>(ld) <= 0 || std::get<1>(ld) <= 0)
        return nullptr;
    return std::make_unique<eyecrop_extractor>(
        std::string(name),
        std::get<0>(ld), std::get<1>(ld), std::get<2>(ld));
}


/****************  class license_extractor  ****************/

std::tuple<unsigned, unsigned, int, raw_image::pixel_layout, bool>
dlibx::license_decode(const std::string_view& name) {
    auto t = std::tuple(0u, 0u, -1, raw_image::pixel::none, false);

    if (name.size() < 13 || name.compare(0,7,"license") != 0)
        return t;

    auto end = name.size() - 3;
    if (name.compare(end,3,"rgb") == 0)
        std::get<3>(t) = raw_image::pixel::rgb24;
    else if (name.compare(--end,4,"rgbn") == 0) {
        std::get<3>(t) = raw_image::pixel::rgb24;
        std::get<4>(t) = true;
    }
    else if (name.compare(end,4,"gray") == 0)
        std::get<3>(t) = raw_image::pixel::gray8;
    else
        return t;

    const auto rad = name.find('r');
    if (7 < rad && rad < end - 1) {
        const auto r = std::string(name, rad+1, end-rad-1);
        if ((std::get<2>(t) = atoi(r.c_str())) < 0)
            return t;
        end = rad;
    }

    const auto sep = name.find('x');
    if (!(7 < sep && sep < end - 1))
        return t;

    const auto w = std::string(name, 7, sep-7);
    if (!is_positive(w))
        return t;
    const auto h = std::string(name, sep+1, end-sep-1);
    if (!is_positive(h))
        return t;

    std::get<0>(t) = unsigned(atol(w.c_str()));
    std::get<1>(t) = unsigned(atol(h.c_str()));
    return t;
}

std::unique_ptr<const input_extractor>
dlibx::license_factory(const std::string_view& name) {
    const auto ld = license_decode(name);
    if (std::get<0>(ld) <= 0 || std::get<1>(ld) <= 0)
        return nullptr;
    switch (std::get<3>(ld)) {
    case raw_image::pixel::rgb24:
        return std::make_unique<license_extractor<dlib::rgb_pixel> >(
            std::string(name), std::get<0>(ld), std::get<1>(ld),
            std::get<2>(ld), std::get<4>(ld));
    case raw_image::pixel::gray8:
        return std::make_unique<license_extractor<unsigned char> >(
            std::string(name), std::get<0>(ld), std::get<1>(ld),
            std::get<2>(ld), std::get<4>(ld));
    default: return nullptr;
    }
}


/****************  class box_extractor  ****************/

std::tuple<unsigned, unsigned, raw_image::pixel_layout, bool>
dlibx::box_decode(const std::string_view& name) {
    auto t = std::tuple(0u, 0u, raw_image::pixel::none, false);
    if (name.size() < 9 || name.compare(0,3,"box") != 0)
        return t;

    auto end = name.size() - 1;
    if (name[end] == 'n')
        --end, std::get<3>(t) = true;
    if (name.compare(end -= 2, 3, "rgb") == 0)
        std::get<2>(t) = raw_image::pixel::rgb24;
    else if (name.compare(end, 3, "yuv") == 0)
        std::get<2>(t) = raw_image::pixel::yuv;
    else if (name.compare(--end,4,"gray") == 0)
        std::get<2>(t) = raw_image::pixel::gray8;
    else
        return t;

    const auto sep = name.find('x',3);
    if (!(3 < sep && sep < end - 1))
        return t;

    const auto w = std::string(name, 3, sep-3);
    if (!is_positive(w))
        return t;
    const auto h = std::string(name, sep+1, end-sep-1);
    if (!is_positive(h))
        return t;

    std::get<0>(t) = unsigned(atol(w.c_str()));
    std::get<1>(t) = unsigned(atol(h.c_str()));
    return t;
}

std::unique_ptr<const input_extractor>
dlibx::box_factory(const std::string_view& name) {
    const auto ld = box_decode(name);
    if (std::get<0>(ld) <= 0 || std::get<1>(ld) <= 0)
        return nullptr;
    return std::make_unique<box_extractor>(
        std::string(name),
        std::get<0>(ld), std::get<1>(ld),
        std::get<2>(ld), std::get<3>(ld));
}
