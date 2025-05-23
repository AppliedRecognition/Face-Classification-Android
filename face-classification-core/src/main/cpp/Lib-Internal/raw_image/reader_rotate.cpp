
#include "reader.hpp"
#include "point2.hpp"
#include <stdext/rounding.hpp>

#include <cassert>
#include <cstring>

using namespace raw_image;


// 0 to 50 gradians (inclusive)
// { sin_numerator, cos_numerator, denominator }
// invariant: sin^2 + cos^2 = denom^2
static constexpr short table[][3] = {
    {0,1,1},            {127,8064,8065},
    {508,16125,16133},  {340,7221,7229},
    {795,12628,12653},  {204,2597,2605},
    {1060,11211,11261}, {1308,11845,11917},
    {1287,10184,10265}, {1808,12705,12833},
    {2540,16029,16229}, {1397,8004,8125},
    {3048,15985,16273}, {1651,7980,8149},
    {3556,15933,16325}, {1368,5695,5857},
    {2280,8881,9169},   {603,2204,2285},
    {7,24,25},          {3519,11440,11969},
    {3232,9945,10457},  {12,35,37},
    {693,1924,2045},    {2415,6392,6833},
    {5709,14420,15509}, {5808,14065,15217},
    {5915,13668,14893}, {1820,4029,4421},
    {1615,3432,3793},   {5285,10788,12013},
    {300,589,661},      {6123,11564,13085},
    {2812,5115,5837},   {3652,6405,7373},
    {6148,10395,12077}, {429,700,821},
    {7956,12533,14845}, {6848,10425,12473},
    {104,153,185},      {4329,6160,7529},
    {8007,11024,13625}, {3,4,5},
    {8436,10877,13765}, {6204,7747,9925},
    {225,272,353},      {7828,9165,12053},
    {765,868,1157},     {7261,7980,10789},
    {8791,9360,12841},  {5056,5217,7265},
    {4060,4059,5741}
};
static_assert(std::size(table) == 51);
#ifndef __WIN32__
static_assert([](){
    for (auto&& p : table)
        if (p[0]*p[0] + p[1]*p[1] != p[2]*p[2])
            return false;
    return true;
}());
#endif

static auto lookup_right(int angle_gradians) {
    angle_gradians = angle_gradians % 400;
    if (angle_gradians < 0)
        angle_gradians += 400;
    assert(0 <= angle_gradians && angle_gradians < 400);

    std::pair<point2i, int> r;
    const auto i = angle_gradians % 100;
    if (i < 50) {
        assert(0 <= i);
        r.second  = table[i][2];
        r.first.x = table[i][1];
        r.first.y = table[i][0];
    }
    else {
        assert(i < 100);
        r.second  = table[100-i][2];
        r.first.x = table[100-i][0];
        r.first.y = table[100-i][1];
    }

    const auto j = angle_gradians / 100;
    if (j&1) {
        std::swap(r.first.x, r.first.y);
        r.first.x = -r.first.x;
    }
    if (j&2) {
        r.first.x = -r.first.x;
        r.first.y = -r.first.y;
    }

    return r;
}

namespace {
    struct rotate_reader : reader {
        const plane source_image;
        const uint32_t padding_value;
        const uint32_t base_value;
        using int_type = point2i::value_type;
        int_type denom;
        point2i right, down, line;

        static auto cs(pixel_layout cs, bool expand) {
            if (raw_image::bytes_per_pixel(cs) >= 4 || !expand)
                return cs;
            if (cs == pixel::rgb24)
                return pixel::rgba32;
            if (cs == pixel::bgr24)
                return pixel::bgra32;
            return pixel_layout(0x4ff); // undefined 4 byte pixel
        }

        rotate_reader(const plane& src,
                      int angle_gradians,
                      float center_x, float center_y,
                      unsigned width, unsigned height,
                      uint32_t padding_value = 0,
                      bool expand = false,
                      uint32_t base_value = 0)
            : reader(width, height, cs(src.layout, expand)),
              source_image(src),
              padding_value(padding_value),
              base_value(base_value) {

            switch (src.rotate&3) {
            case 1:
                std::swap(center_x,center_y);
                center_x = float(src.width) - center_x;
                angle_gradians += 100;
                break;
            case 2:
                center_x = float(src.width) - center_x;
                center_y = float(src.height) - center_y;
                angle_gradians += 200;
                break;
            case 3:
                std::swap(center_x,center_y);
                center_y = float(src.height) - center_y;
                angle_gradians += 300;
                break;
            case 0:
            default:
                break;
            }
            if (src.rotate&4) {
                center_x = float(src.width) - center_x;
                angle_gradians = 400 - angle_gradians;
            }

            // note: right and down are half-pixel steps in source_image
            // this is accomplished by denom = 2*r.second
            const auto r = lookup_right(angle_gradians);
            denom = 2 * r.second;
            right = r.first;
            down.x = -right.y;
            down.y = right.x;

            if (src.rotate&4) {
                right.x = -right.x;
                right.y = -right.y;
            }

            line.x = stdx::round_to<int_type>(2 * float(denom) * center_x);
            line.y = stdx::round_to<int_type>(2 * float(denom) * center_y);
            line += right * -stdx::round_to<int_type>(width-1);
            line += down  * -stdx::round_to<int_type>(height-1);
            line /= 2;
        }

        void line_next() override {
            line += down;
        }
    };

    /// When off the image, copy nearest edge pixel.
    template <unsigned BPP>
    struct rotate_replicate final : rotate_reader {
        const std::size_t sbpl;
        
        rotate_replicate(const plane& src,
                        int angle_gradians,
                        float cx, float cy,
                        unsigned width, unsigned height)
            : rotate_reader(src, angle_gradians, cx, cy, width, height),
              sbpl(src.bytes_per_line) {
        }

        void line_copy(void* _dest) override {
            auto dest = static_cast<unsigned char*>(_dest);
            auto pt = line;
            for (auto n = width(); n > 0; --n, pt += right, dest += BPP) {
                unsigned x = 0, y = 0;
                if (pt.x > 0) {
                    x = unsigned(pt.x / denom);
                    if (x >= source_image.width)
                        x = source_image.width - 1;
                }
                if (pt.y > 0) {
                    y = unsigned(pt.y / denom);
                    if (y >= source_image.height)
                        y = source_image.height - 1;
                }
                auto src = source_image.data + y*sbpl + x*BPP;
                memcpy(dest, src, BPP);
            }
        }
    };

    /// When off the image, use the padding_value.
    template <unsigned BPP>
    struct rotate_padded final : rotate_reader {
        static_assert(0 < BPP && BPP < 4);
        const std::size_t sbpl;

        rotate_padded(const plane& src,
                      int angle_gradians,
                      float center_x, float center_y,
                      unsigned width, unsigned height,
                      uint32_t padding_value)
            : rotate_reader(src, angle_gradians, center_x, center_y,
                            width, height, padding_value),
              sbpl(src.bytes_per_line) {
        }

        void line_copy(void* _dest) override {
            auto dest = static_cast<unsigned char*>(_dest);
            auto pt = line;
            for (auto n = width(); n > 0; --n, pt += right, dest += BPP) {
                if (pt.x >= 0 && pt.y >= 0) {
                    const auto y = unsigned(pt.y / denom);
                    if (y < source_image.height) {
                        const auto x = unsigned(pt.x / denom);
                        if (x < source_image.width) {
                            auto const* src =
                                source_image.data + y*sbpl + x*BPP;
                            memcpy(dest, src, BPP);
                            continue;
                        }
                    }
                }
                memcpy(dest, &padding_value, BPP);
            }
        }
    };

    /// When off the image, use the padding_value.
    template <>
    struct rotate_padded<4> final : rotate_reader {
        const std::size_t sipl;  // source integers per line
        uint32_t const* const src;
        
        rotate_padded(const plane& _src,
                      int angle_gradians,
                      float center_x, float center_y,
                      unsigned width, unsigned height,
                      uint32_t padding_value)
            : rotate_reader(_src, angle_gradians, center_x, center_y,
                            width, height, padding_value),
              sipl(_src.bytes_per_line/4),
              src(reinterpret_cast<const uint32_t*>(_src.data)) {
            if ((_src.bytes_per_line&3) ||
                (reinterpret_cast<std::uintptr_t>(_src.data)&3))
                throw std::runtime_error("image pixels must be aligned");
        }

        void line_copy(void* _dest) override {
            auto dest = static_cast<uint32_t*>(_dest);
            auto pt = line;
            for (auto n = width(); n > 0; --n, pt += right, ++dest) {
                if (pt.x >= 0 && pt.y >= 0) {
                    const auto y = unsigned(pt.y / denom);
                    if (y < source_image.height) {
                        const auto x = unsigned(pt.x / denom);
                        if (x < source_image.width) {
                            *dest = src[y*sipl+x];
                            continue;
                        }
                    }
                }
                *dest = padding_value;
            }
        }
    };

    /// When off the image, use the padding_value.
    /// Expand from SBPP to 4 bytes per pixel.
    /// base_value is copied first, then source pixel bytes.
    template <unsigned SBPP>
    struct rotate_expand final : rotate_reader {
        static_assert(0 < SBPP && SBPP < 4);
        const std::size_t sbpl;
        
        rotate_expand(const plane& src,
                      int angle_gradians,
                      float center_x, float center_y,
                      unsigned width, unsigned height,
                      uint32_t padding_value,
                      uint32_t base_value)
            : rotate_reader(src, angle_gradians, center_x, center_y,
                            width, height, padding_value, true, base_value),
              sbpl(src.bytes_per_line) {
            assert(bytes_per_pixel() == 4);
        }

        void line_copy(void* _dest) override {
            auto dest = static_cast<uint32_t*>(_dest);
            auto pt = line;
            for (auto n = width(); n > 0; --n, pt += right, ++dest) {
                if (pt.x >= 0 && pt.y >= 0) {
                    const auto y = unsigned(pt.y / denom);
                    if (y < source_image.height) {
                        const auto x = unsigned(pt.x / denom);
                        if (x < source_image.width) {
                            *dest = base_value;
                            auto const* src =
                                source_image.data + y*sbpl + x*SBPP;
                            memcpy(dest, src, SBPP);
                            continue;
                        }
                    }
                }
                *dest = padding_value;
            }
        }
    };
}

std::unique_ptr<reader>
raw_image::rotate_gradians(const plane& src,
                      int angle_gradians,
                      float center_x, float center_y,
                      unsigned width, unsigned height) {
    switch (bytes_per_pixel(src)) {
    case 1: return std::make_unique<rotate_replicate<1> >(
        src, angle_gradians, center_x, center_y, width, height);
    case 2: return std::make_unique<rotate_replicate<2> >(
        src, angle_gradians, center_x, center_y, width, height);
    case 3: return std::make_unique<rotate_replicate<3> >(
        src, angle_gradians, center_x, center_y, width, height);
    case 4: return std::make_unique<rotate_replicate<4> >(
        src, angle_gradians, center_x, center_y, width, height);
    }
    throw std::invalid_argument("source image has unknown pixel layout");
}

std::unique_ptr<reader>
raw_image::rotate_gradians(const plane& src,
                      int angle_gradians,
                      float center_x, float center_y,
                      unsigned width, unsigned height,
                      uint32_t padding_value) {
    switch (bytes_per_pixel(src)) {
    case 1: return std::make_unique<rotate_padded<1> >(
        src, angle_gradians, center_x, center_y, width, height, padding_value);
    case 2: return std::make_unique<rotate_padded<2> >(
        src, angle_gradians, center_x, center_y, width, height, padding_value);
    case 3: return std::make_unique<rotate_padded<3> >(
        src, angle_gradians, center_x, center_y, width, height, padding_value);
    case 4: return std::make_unique<rotate_padded<4> >(
        src, angle_gradians, center_x, center_y, width, height, padding_value);
    }
    throw std::invalid_argument("source image has unknown pixel layout");
}

std::unique_ptr<reader>
raw_image::rotate_gradians(const plane& src,
                      int angle_gradians,
                      float center_x, float center_y,
                      unsigned width, unsigned height,
                      uint32_t padding_value,
                      uint32_t base_value) {
    switch (bytes_per_pixel(src)) {
    case 1: return std::make_unique<rotate_expand<1> >(
        src, angle_gradians, center_x, center_y, width, height,
        padding_value, base_value);
    case 2: return std::make_unique<rotate_expand<2> >(
        src, angle_gradians, center_x, center_y, width, height,
        padding_value, base_value);
    case 3: return std::make_unique<rotate_expand<3> >(
        src, angle_gradians, center_x, center_y, width, height,
        padding_value, base_value);
    case 4: return std::make_unique<rotate_padded<4> >(
        src, angle_gradians, center_x, center_y, width, height,
        padding_value);
    }
    throw std::invalid_argument("source image has unknown pixel layout");
}
