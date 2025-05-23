
#include "render.hpp"
#include "dlib.hpp"

#include <det/math.hpp>
#include <core/context.hpp>
#include <raw_image/core.hpp>
#include <stdext/rounding.hpp>

#include <applog/core.hpp>

#include <algorithm>
#include <cassert>


using namespace render;
using det::sqr;


void render::in_place_equalize_histogram(
    stdx::arg<const raw_image::plane> img) {

    if (!img)
        throw std::invalid_argument("image is nullptr");
    const auto N = bytes_per_pixel(img);
    if (N != 1 && !same_channel_order(img->layout, raw_image::pixel::yuv))
        throw std::invalid_argument("equalize_histogram requires GRAY8 or YUV");

    std::vector<unsigned char> vals;
    vals.reserve(img->width*img->height);
    const auto w2 = sqr(img->width);
    const auto h2 = sqr(img->height);
    const unsigned char* cline = img->data;
    for (unsigned y = 0; y < img->height; ++y, cline += img->bytes_per_line) {
        // sqr(2*x-w+1) / sqr(w) + sqr(2*y-h+1) / sqr(h) = 1
        const auto a = std::sqrt(w2 - w2 * sqr(2*y-img->height+1) / h2);
        assert(a <= img->width);
        const auto xofs = stdx::round_to<unsigned>((img->width-a)/2);
        auto first = cline + N*xofs;
        auto last = cline + N*(img->width - xofs);
        assert(first <= last);
        for ( ; first < last; first += N)
            vals.push_back(*first);
    }
    std::sort(vals.begin(), vals.end());

    std::vector<unsigned char> mapping;
    mapping.reserve(256);
    for (unsigned i = 0; i < vals.size(); ++i) {
        while (vals[i] >= mapping.size())
            mapping.push_back((256*i/vals.size())&0xff);
        if (mapping.back() >= 128)
            mapping.back() = (256*i/vals.size()) & 0xff;
    }
    assert(mapping.size() == 1u + vals.back());
    assert(mapping.back() == 255);
    mapping.resize(256, 255);

    unsigned char* line = img->data;
    for (unsigned y = 0; y < img->height; ++y, line += img->bytes_per_line) {
        auto p = line;
        for (unsigned x = 0; x < img->width; ++x, p += N)
            *p = mapping[*p];
    }
}

raw_image::plane_ptr
render::render_face(stdx::arg<core::context_data> context,
                    stdx::arg<const raw_image::plane> image,
                    const face_coordinates& pos,
                    const render_settings& rsettings,
                    const output_settings& osettings,
                    diagnostics* d) {

    if (!context)
        throw std::invalid_argument("invalid context object");
    if (!image)
        throw std::invalid_argument("invalid image object");
    
    const det::detected_coordinates* shape = nullptr;
    for (auto& s : pos)
        if (s.type == det::dt::dlib68) {
            shape = &s;
            break;
        }
    if (shape)
        return internal::render_dlib(*image, *shape, rsettings, osettings, d);
    else {
        FILE_LOG(logWARNING) << "dlib landmarks required";
        return nullptr;
    }
}

