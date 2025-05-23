
#include "drawing.hpp"
#include "points.hpp"
#include "point2.hpp"
#include <algorithm>
#include <cstring>

using namespace raw_image;

template <typename T>
static constexpr auto sqr(T x) { return x*x; }

void raw_image::fill(single_plane_arg dest, pixel_color color) {
    if (empty(dest)) return;
    auto line = dest->data;
    const auto bpp = bytes_per_pixel(dest->layout);
    const auto cpx = to_layout(dest->layout, color);
    bool single_value = true;
    for (unsigned i = 1; i < bpp; ++i)
        if (cpx[i] != cpx[0]) {
            single_value = false;
            break;
        }
    if (single_value) {
        // grayscale or all channels have same value
        for (auto j = dest->height; j > 0; --j, line += dest->bytes_per_line)
            memset(line, cpx[0], bpp * dest->width);
    }
    else {
        for (auto j = dest->height; j > 0; --j, line += dest->bytes_per_line) {
            auto px = line;
            for (auto i = dest->width; i > 0; --i, px += bpp)
                memcpy(px, cpx.data(), bpp);
        }
    }
}

namespace {
    struct flood_fill {
        const raw_image::plane& img;
        const unsigned bpp;
        const std::array<uint8_t,4> target;
        const std::array<uint8_t,4> replacement;

        auto* line_ptr(unsigned y) const {
            return img.data + y*img.bytes_per_line;
        }

        auto get_target(unsigned x, unsigned y) const {
            std::array<uint8_t,4> target{0,0,0,0};
            std::copy_n(line_ptr(y) + x*bpp, bpp, target.data());
            return target;
        }

        bool is_target(const uint8_t* other) const {
            auto* tp = target.data();
            for (auto n = bpp; n > 0; --n, ++other, ++tp)
                if (*tp != *other) return false;
            return true;
        }

        flood_fill(const raw_image::plane& img, unsigned x, unsigned y,
                   std::array<uint8_t,4> replacement)
            : img(img),
              bpp(bytes_per_pixel(img.layout)),
              target(get_target(x,y)),
              replacement(replacement) {
        }

        unsigned operator()(unsigned x, unsigned y) const {
            auto line = line_ptr(y), end = line + img.width*bpp;

            // fill continuous line
            auto first = line + x*bpp, last = first + bpp;
            //assert(is_target(first));
            unsigned nfilled = 1;
            std::copy_n(replacement.data(), bpp, first);
            while (last < end && is_target(last)) {
                std::copy_n(replacement.data(), bpp, last);
                ++nfilled;
                last += bpp;
            }
            while (line < first) {
                auto p = first - bpp;
                if (!is_target(p))
                    break;
                std::copy_n(replacement.data(), bpp, first = p);
                ++nfilled;
            }

            if (0 < y) { // check above
                unsigned y0 = y - 1, x0 = unsigned(first - line) / bpp;
                for (auto p = first - img.bytes_per_line,
                         end = last - img.bytes_per_line; p != end; p += bpp, ++x0)
                    if (is_target(p)) nfilled += operator()(x0,y0);
            }

            if (++y < img.height) { // check below
                unsigned x0 = unsigned(first - line) / bpp;
                for (auto p = first + img.bytes_per_line,
                         end = last + img.bytes_per_line; p != end; p += bpp, ++x0)
                    if (is_target(p)) nfilled += operator()(x0,y);
            }

            return nfilled;
        }
    };
}

unsigned raw_image::fill(single_plane_arg dest, unsigned x, unsigned y,
                         std::array<uint8_t,4> color) {
    if (empty(dest) || dest->width <= x || dest->height <= y)
        throw std::invalid_argument("start position for flood fill is outside image");
    const flood_fill ff{*dest, x, y, color};
    if (ff.is_target(ff.replacement.data()))
        return 0;
    return ff(x,y);
}

unsigned raw_image::fill(single_plane_arg dest, unsigned x, unsigned y,
                         pixel_color color) {
    if (empty(dest) || dest->width <= x || dest->height <= y)
        throw std::invalid_argument("start position for flood fill is outside image");
    const flood_fill ff{*dest, x, y, to_layout(dest->layout, color)};
    if (ff.is_target(ff.replacement.data()))
        return 0;
    return ff(x,y);
}

void raw_image::circle(single_plane_arg dest,
                       double x, double y, pixel_color color, int radius) {
    if (empty(dest)) return;
    const auto p = to_image_point(point2f{float(x),float(y)},*dest);
    const auto bpp = bytes_per_pixel(dest->layout);
    const auto cpx = to_layout(dest->layout, color);

    if (radius == 0) {
        // draw a single pixel
        if (0 <= p.x && 0 <= p.y) {
            const unsigned x = stdx::round_from(p.x);
            const unsigned y = stdx::round_from(p.y);
            if (x < dest->width && y < dest->height) {
                auto px = dest->data + y * dest->bytes_per_line + x * bpp;
                memcpy(px, cpx.data(), bpp);
            }
        }
        return;
    }

    // radius is not zero
    const auto fr = float(std::abs(radius));
    int lx = stdx::round_from(std::floor(p.x - (fr+1)));
    if (lx < 0) lx = 0;
    int ty = stdx::round_from(std::floor(p.y - (fr+1)));
    if (ty < 0) ty = 0;
    int rx = stdx::round_from(std::ceil(p.x + (fr+1)));
    rx = std::min(rx,int(dest->width)-1);
    int by = stdx::round_from(std::ceil(p.y + (fr+1)));
    by = std::min(by,int(dest->height)-1);

    if (!(lx <= rx && ty <= by))
        return;  // circle is completely outside image

    const auto interpolate =
        [&](uint8_t* px, float fg) {
            auto src = cpx.data();
            for (auto n = bpp; n > 0; --n, ++px, ++src)
                *px = stdx::round_from((1-fg)**px + fg**src);
        };

    const float r0 = sqr(fr-0.75f);
    const float r1 = sqr(fr-0.25f);
    const float r2 = sqr(fr+0.25f);
    const float r3 = sqr(fr+0.75f);
    const auto d0 = r1-r0;
    const auto d3 = r3-r2;

    auto line = dest->data 
        + unsigned(ty) * dest->bytes_per_line + unsigned(lx) * bpp;
    for ( ; ty <= by; ++ty, line += dest->bytes_per_line) {
        const auto dy = sqr(p.y - float(ty));
        auto px = line;
        for (auto x = lx; x <= rx; ++x, px += bpp) {
            const auto d = dy + sqr(p.x - float(x));
            if (d < r2) {
                if (radius < 0) // solid circle
                    memcpy(px, cpx.data(), bpp);
                else if (r1 < d)
                    memcpy(px, cpx.data(), bpp);
                else if (r0 < d)
                    interpolate(px,(d-r0)/d0);
            }
            else if (d < r3)
                interpolate(px,(r3-d)/d3);
        }
    }
}

void raw_image::line(single_plane_arg dest,
                     double x0, double y0, double x1, double y1,
                     pixel_color color, unsigned width) {
    if (empty(dest)) return;
    const auto p0 = to_image_point(point2f{float(x0),float(y0)},*dest);
    const auto p1 = to_image_point(point2f{float(x1),float(y1)},*dest);
    const auto bpp = bytes_per_pixel(dest->layout);
    const auto cpx = to_layout(dest->layout, color);

    if (std::abs(x0-x1) < 1 && std::abs(y0-y1) < 1) {
        // single point (draw a solid circle instead)
        circle(dest,(x0+x1)/2,(y0+y1)/2,color,-int(width+1)/2);
        return;
    }

    int ty = stdx::round_from(std::floor(std::min(p0.y,p1.y)-0.5f));
    if (ty < 0) ty = 0;
    int by = stdx::round_from(std::ceil(std::max(p0.y,p1.y)+0.5f));
    by = std::min(by,int(dest->height));
    if (by <= ty)
        return;  // line is completely above or below the image

    int lx = stdx::round_from(std::floor(std::min(p0.x,p1.x)-0.5f));
    if (lx < 0) lx = 0;
    int rx = stdx::round_from(std::ceil(std::max(p0.x,p1.x)+0.5f));
    rx = std::min(rx,int(dest->width));
    if (rx <= lx)
        return;  // line is completely left or right of the image

    // a*x0 + b*y0 + c = 0
    // a*x1 + b*y1 + c = 0
    const float a = float(y1 - y0);
    const float b = float(x0 - x1);
    const float c = float(x1*y0 - x0*y1);
    const float denom = std::sqrt(sqr(a)+sqr(b));

    const auto w0 = float(width)/2;
    const auto w1 = w0 + 0.5f;

    const auto dw =
        std::abs(a) < 1 ? float(rx-lx) : float(width+2)*(1+std::abs(b/a))/2;

    const auto interpolate =
        [&](uint8_t* px, float fg) {
            auto src = cpx.data();
            for (auto n = bpp; n > 0; --n, ++px, ++src)
                *px = stdx::round_from((1-fg)**px + fg**src);
        };

    auto line = dest->data
        + unsigned(ty) * dest->bytes_per_line + unsigned(lx) * bpp;
    for ( ; ty < by; ++ty, line += dest->bytes_per_line) {
        const auto dy = c + b*float(ty);
        auto px = line;
        if (std::abs(a) < 1) {
            // line is horizontal
            for (auto x = lx; x < rx; ++x, px += bpp) {
                auto d = std::abs(a*float(x) + dy) / denom;
                if (d <= w0)
                    memcpy(px, cpx.data(), bpp);
                else if (d < w1)
                    interpolate(px,2*(w1-d));
            }
        }
        else {
            const auto x = -(b*float(ty)+c) / a;
            auto x0 = std::max(lx,stdx::round_to<int>(x-dw));
            auto x1 = std::min(rx,stdx::round_to<int>(x+dw));
            px += (x0-lx)*int(bpp);
            for ( ; x0 < x1; ++x0, px += bpp) {
                auto d = std::abs(a*float(x0) + dy) / denom;
                if (d <= w0)
                    memcpy(px, cpx.data(), bpp);
                else if (d < w1)
                    interpolate(px,2*(w1-d));
            }
        }
    }
}

/* stb_easy_font copied from:
 * https://github.com/nothings/stb/blob/master/stb_easy_font.h
 */

namespace {
    struct stb_ef_info_struct {
        unsigned char advance, h_seg, v_seg;
    };
}

static constexpr stb_ef_info_struct stb_ef_charinfo[96] = {
    {  6,  0,  0 },  {  3,  0,  0 },  {  5,  1,  1 },  {  7,  1,  4 },
    {  7,  3,  7 },  {  7,  6, 12 },  {  7,  8, 19 },  {  4, 16, 21 },
    {  4, 17, 22 },  {  4, 19, 23 },  { 23, 21, 24 },  { 23, 22, 31 },
    { 20, 23, 34 },  { 22, 23, 36 },  { 19, 24, 36 },  { 21, 25, 36 },
    {  6, 25, 39 },  {  6, 27, 43 },  {  6, 28, 45 },  {  6, 30, 49 },
    {  6, 33, 53 },  {  6, 34, 57 },  {  6, 40, 58 },  {  6, 46, 59 },
    {  6, 47, 62 },  {  6, 55, 64 },  { 19, 57, 68 },  { 20, 59, 68 },
    { 21, 61, 69 },  { 22, 66, 69 },  { 21, 68, 69 },  {  7, 73, 69 },
    {  9, 75, 74 },  {  6, 78, 81 },  {  6, 80, 85 },  {  6, 83, 90 },
    {  6, 85, 91 },  {  6, 87, 95 },  {  6, 90, 96 },  {  7, 92, 97 },
    {  6, 96,102 },  {  5, 97,106 },  {  6, 99,107 },  {  6,100,110 },
    {  6,100,115 },  {  7,101,116 },  {  6,101,121 },  {  6,101,125 },
    {  6,102,129 },  {  7,103,133 },  {  6,104,140 },  {  6,105,145 },
    {  7,107,149 },  {  6,108,151 },  {  7,109,155 },  {  7,109,160 },
    {  7,109,165 },  {  7,118,167 },  {  6,118,172 },  {  4,120,176 },
    {  6,122,177 },  {  4,122,181 },  { 23,124,182 },  { 22,129,182 },
    {  4,130,182 },  { 22,131,183 },  {  6,133,187 },  { 22,135,191 },
    {  6,137,192 },  { 22,139,196 },  {  6,144,197 },  { 22,147,198 },
    {  6,150,202 },  { 19,151,206 },  { 21,152,207 },  {  6,155,209 },
    {  3,160,210 },  { 23,160,211 },  { 22,164,216 },  { 22,165,220 },
    { 22,167,224 },  { 22,169,228 },  { 21,171,232 },  { 21,173,233 },
    {  5,178,233 },  { 22,179,234 },  { 23,180,238 },  { 23,180,243 },
    { 23,180,248 },  { 22,189,248 },  { 22,191,252 },  {  5,196,252 },
    {  3,203,252 },  {  5,203,253 },  { 22,210,253 },  {  0,214,253 },
};

static constexpr unsigned char stb_ef_hseg[214] = {
    97,37,69,84,28,51,2,18,10,49,98,41,65,25,81,105,33,9,97,1,97,37,37,36,
    81,10,98,107,3,100,3,99,58,51,4,99,58,8,73,81,10,50,98,8,73,81,4,10,50,
    98,8,25,33,65,81,10,50,17,65,97,25,33,25,49,9,65,20,68,1,65,25,49,41,
    11,105,13,101,76,10,50,10,50,98,11,99,10,98,11,50,99,11,50,11,99,8,57,
    58,3,99,99,107,10,10,11,10,99,11,5,100,41,65,57,41,65,9,17,81,97,3,107,
    9,97,1,97,33,25,9,25,41,100,41,26,82,42,98,27,83,42,98,26,51,82,8,41,
    35,8,10,26,82,114,42,1,114,8,9,73,57,81,41,97,18,8,8,25,26,26,82,26,82,
    26,82,41,25,33,82,26,49,73,35,90,17,81,41,65,57,41,65,25,81,90,114,20,
    84,73,57,41,49,25,33,65,81,9,97,1,97,25,33,65,81,57,33,25,41,25,
};

static constexpr unsigned char stb_ef_vseg[253] = {
    4,2,8,10,15,8,15,33,8,15,8,73,82,73,57,41,82,10,82,18,66,10,21,29,1,65,
    27,8,27,9,65,8,10,50,97,74,66,42,10,21,57,41,29,25,14,81,73,57,26,8,8,
    26,66,3,8,8,15,19,21,90,58,26,18,66,18,105,89,28,74,17,8,73,57,26,21,
    8,42,41,42,8,28,22,8,8,30,7,8,8,26,66,21,7,8,8,29,7,7,21,8,8,8,59,7,8,
    8,15,29,8,8,14,7,57,43,10,82,7,7,25,42,25,15,7,25,41,15,21,105,105,29,
    7,57,57,26,21,105,73,97,89,28,97,7,57,58,26,82,18,57,57,74,8,30,6,8,8,
    14,3,58,90,58,11,7,74,43,74,15,2,82,2,42,75,42,10,67,57,41,10,7,2,42,
    74,106,15,2,35,8,8,29,7,8,8,59,35,51,8,8,15,35,30,35,8,8,30,7,8,8,60,
    36,8,45,7,7,36,8,43,8,44,21,8,8,44,35,8,8,43,23,8,8,43,35,8,8,31,21,15,
    20,8,8,28,18,58,89,58,26,21,89,73,89,29,20,8,8,30,7,
};

easy_font::easy_font(const char* text, unsigned scale,
                     int extra_char_spacing, int extra_line_spacing) {
    if (scale <= 0 || !text)
        return; // cannot render at zero scale

    const auto draw_segs =
        [&](auto x, auto y, auto const* segs, auto count, auto vertical) {
            for ( ; count > 0; --count, ++segs) {
                const auto seg = *segs;
                const auto len = scale * (seg&7);
                x += scale * ((seg>>3)&1);
                const auto y0 = y + scale * (seg>>4);
                if (vertical)
                    rects.push_back({x,y0,scale,len});
                else
                    rects.push_back({x,y0,len,scale});
            }
        };

    const auto line_spacing = unsigned(extra_line_spacing + int(scale * 12));
    for (unsigned x = 0, y = 0; *text; ++text) {
        if (*text == '\n') {
            y += line_spacing;
            x = 0;
        }
        else if (32 <= *text && *text < 127) {
            const auto idx = *text - 32;
            const auto advance = stb_ef_charinfo[idx].advance;
            const auto y_ch = (advance & 16) ? y + scale : y;
            const auto h_seg = stb_ef_charinfo[idx  ].h_seg;
            const auto h_num = stb_ef_charinfo[idx+1].h_seg - h_seg;
            const auto v_seg = stb_ef_charinfo[idx  ].v_seg;
            const auto v_num = stb_ef_charinfo[idx+1].v_seg - v_seg;
            draw_segs(x, y_ch, stb_ef_hseg + h_seg, h_num, 0);
            draw_segs(x, y_ch, stb_ef_vseg + v_seg, v_num, 1);
            x += unsigned(extra_char_spacing + int(scale * (advance & 15)));
        }
        else // non-ascii character
            x += scale;
    }
}

void easy_font::render(const plane& dest, pixel_color c) const {
    for (auto& r : rects)
        if (r.x + r.w <= dest.width && r.y + r.h <= dest.height)
            fill(crop(dest,r.x,r.y,r.w,r.h),c);
}
