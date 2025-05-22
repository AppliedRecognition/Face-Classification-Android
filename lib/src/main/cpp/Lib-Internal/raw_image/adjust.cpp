
#include "adjust.hpp"
#include <stdext/rounding.hpp>

#include <cassert>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <vector>

using namespace raw_image;

float raw_image::measure_brightness(single_plane_arg image) {
    if (!image) throw std::invalid_argument("empty image");
    uint64_t sum = 0;
    if (bytes_per_pixel(image) == 1) {
        for (auto&& line : read_lines_bpp<1>(image))
            for (auto&& pixel : line)
                sum += pixel;
    }
    else {
        const auto pl = to_color_class(image->layout) == cc::yuv_nv21 ?
            pixel::yuv24_nv21 : pixel::yuv24_jpeg;
        for (auto&& line : read_lines_bpp<3>(image, pl))
            for (auto&& pixel : line)
                sum += pixel[0];
    }
    const auto mean = double(sum) / image->width / image->height;
    return float(mean);
}

// figure out how many pixels to use from each line to get desired area
// returns count of pixels selected
static unsigned
fill_widths(std::vector<unsigned>& lines, unsigned width, float area) {
    // we work with double width,height and x,y so
    // width|height odd  -> e.g. x,y runs -4 -2 0 +2 +4
    // width|height even -> e.g. x,y runs -5 -3 -1 +1 +3 +5
    const auto height = unsigned(lines.size());
    const auto w2 = (width-1) * (width-1);
    const auto h2 = (height-1) * (height-1);
    const auto thres =
        stdx::round_to<uint64_t>(double(w2)*double(h2)*area*4/M_PI);
    unsigned count = 0;
    int y = -int(height - 1);
    for (auto& z : lines) {
        const auto y2 = uint64_t(y*y)*w2;
        if (y2 <= thres) {
            auto w = unsigned(std::floor(std::sqrt((thres - y2) / h2)));
            if (((w^width)&1) != 0)
                w += 1; // either w and width even, or w and width odd
            z = w <= width ? w : width;
            count += z;
        }
        else // this shouldn't happen if ellipse is inscribed or larger
            z = 0;
        y += 2;
    }
    return count;
}

bc_result
raw_image::measure_brightness_contrast(single_plane_arg image, float area) {
    if (!image) throw std::invalid_argument("empty image");
    if (!(0 < area)) throw std::invalid_argument("area must be positive");

    uint64_t sum = 0, s2 = 0, count = 0;

    if (1 <= area) { // entire image
        if (bytes_per_pixel(image) == 1) {
            for (auto&& line : read_lines_bpp<1>(image))
                for (auto&& pixel : line)
                    sum += pixel, s2 += pixel*unsigned(pixel), ++count;
        }
        else {
            const auto pl = to_color_class(image->layout) == cc::yuv_nv21 ?
                pixel::yuv24_nv21 : pixel::yuv24_jpeg;
            for (auto&& line : read_lines_bpp<3>(image, pl))
                for (auto&& pixel : line)
                    sum += pixel[0], s2 += pixel[0]*unsigned(pixel[0]), ++count;
        }
    }

    else { // partial image (area < 1)
        auto roi = *image;
        if (area < M_PI/4) { // crop image
            const auto frac = 1.0 - std::sqrt(area/(M_PI/4));
            unsigned padx = stdx::round_from(double(roi.width)*frac/2);
            if (roi.width <= 2*padx)
                padx = (roi.width-1) / 2;
            unsigned pady = stdx::round_from(double(roi.height)*frac/2);
            if (roi.height <= 2*pady)
                pady = (roi.height-1) / 2;
            if (0 < padx || 0 < pady) {
                const auto before = double(roi.width) * double(roi.height);
                roi = crop(roi, padx, pady,
                           roi.width  - 2*padx,
                           roi.height - 2*pady);
                const auto after = double(roi.width) * double(roi.height);
                area *= float(before / after);
            }
        }

        std::vector<unsigned> lines(roi.height);
        const auto target = area * double(roi.width) * double(roi.height);
        unsigned actual;
        float scale = area;
        for (auto n = 5; n > 0; --n) {
            // iterative approach to get correct count
            actual = fill_widths(lines, roi.width, scale);
            scale *= float(target / double(actual));
        }

        auto witer = lines.begin();
        if (bytes_per_pixel(image) == 1) {
            for (auto&& line : read_lines_bpp<1>(roi)) {
                if (0 < *witer) {
                    int x = -int(roi.width - *witer)/2;
                    for (auto&& pixel : line) {
                        if (x < 0)
                            ++x;
                        else {
                            sum += pixel, s2 += pixel*unsigned(pixel), ++count;
                            if (*witer <= unsigned(++x))
                                break;
                        }
                    }
                }
                ++witer;
            }
        }
        else {
            const auto pl = to_color_class(roi.layout) == cc::yuv_nv21 ?
                pixel::yuv24_nv21 : pixel::yuv24_jpeg;
            for (auto&& line : read_lines_bpp<3>(roi, pl)) {
                if (0 < *witer) {
                    int x = -int(roi.width - *witer)/2;
                    for (auto&& pixel : line) {
                        if (x < 0)
                            ++x;
                        else {
                            sum += pixel[0],
                                s2 += pixel[0]*unsigned(pixel[0]), ++count;
                            if (*witer <= unsigned(++x))
                                break;
                        }
                    }
                }
                ++witer;
            }
        }
        assert(count == actual);
    }

    const auto mean = double(sum) / double(count);
    const auto ms2 = double(s2) / double(count);
    return bc_result {
        float(mean),
        float(std::sqrt(ms2 - mean*mean)),
        stdx::round_from(count) // will clamp at maximum 32-bit unsigned
    };
}

namespace {
    template <unsigned BPP>
    struct linear_quad {
        int alpha, beta;
        void operator()(uint8_t* dest, const uint8_t* src,
                        unsigned nquads) const {
            for ( ; nquads > 0; --nquads)
                for (auto n = 4; n > 0; --n, ++dest, ++src) {
                    *dest = stdx::round_from((beta + alpha**src) / 256);
                    for (auto x = BPP-1; x > 0; --x)
                        *++dest = *++src;
                }
        }
    };
    template <typename T>
    struct linear_quad_t {
        float alpha, beta;
        void operator()(uint8_t* _dest, const uint8_t* _src,
                        unsigned nquads) const {
            auto* dest = reinterpret_cast<T*>(_dest);
            auto* src = reinterpret_cast<T const*>(_src);
            for ( ; nquads > 0; --nquads)
                for (auto n = 4; n > 0; --n, ++dest, ++src)
                    *dest = stdx::round_from(beta + alpha**src);
        }
    };
}

std::unique_ptr<reader>
raw_image::linear_adjust(std::unique_ptr<reader> src, float alpha, float beta,
                         pixel_layout output_layout) {
    if (!src) throw std::invalid_argument("empty image");
    const auto ai = stdx::round_to<int>(256 * alpha);
    const auto bi = stdx::round_to<int>(256 * beta + 128);
    const auto layout = src->layout();
    if (bytes_per_pixel(layout) == 1)
        src = transform_quads(move(src), layout, linear_quad<1>{ai,bi});
    else if (layout == pixel::a16_le)
        src = transform_quads(
            move(src), layout, linear_quad_t<uint16_t>{alpha,beta});
    else if (layout == pixel::f32)
        src = transform_quads(
            move(src), layout, linear_quad_t<float>{alpha,beta});
    else {
        const auto pl = to_color_class(layout) == cc::yuv_nv21 ?
            pixel::yuv24_nv21 : pixel::yuv24_jpeg;
        src = transform_quads(
            convert(move(src), pl), pl, linear_quad<3>{ai,bi});
    }
    return convert(move(src), output_layout);
}

std::unique_ptr<reader>
raw_image::linear_adjust(std::unique_ptr<reader> src, float alpha, float beta) {
    const auto layout = src->layout();
    return linear_adjust(move(src), alpha, beta, layout);
}

void
raw_image::in_place_linear_adjust(const plane& image, float alpha, float beta) {
    auto reader = linear_adjust(reader::construct(image), alpha, beta);
    switch (image.layout) {
    case pixel::rgba32:
    case pixel::bgra32:
        reader->map_to(image, {0,1,2,4});
        return;

    case pixel::argb32:
    case pixel::abgr32:
        reader->map_to(image, {4,1,2,3});
        return;

    default:
        reader->copy_to(image);
    }
}

namespace {
    template <typename T, unsigned els_per_pixel>
    struct blender final : reader {
        const std::unique_ptr<reader> src1;
        const std::unique_ptr<reader> src2;
        const float alpha1, alpha2, beta;

        blender(std::unique_ptr<reader> src1,
                std::unique_ptr<reader> src2,
                float alpha1, float alpha2, float beta)
            : reader(std::min(src1->width(), src2->width()),
                     std::min(src1->height(), src2->height()),
                     src1->layout()),
              src1(move(src1)),
              src2(move(src2)),
              alpha1(alpha1),
              alpha2(alpha2),
              beta(beta) {
            assert(raw_image::bytes_per_pixel(this->src1->layout()) == els_per_pixel*sizeof(T));
        }

        void line_next() override {
            if (!src1->next_line() || !src2->next_line())
                throw std::logic_error("unexpected end of image");
        }

        void line_copy(void* _dest) override {
            auto* dest = static_cast<T*>(_dest);
            auto const* s1 = reinterpret_cast<const T*>(src1->get_line());
            auto const* s2 = reinterpret_cast<const T*>(src2->get_line());
            for (auto n = width(); n > 0; --n)
                for (auto j = els_per_pixel; j > 0; --j, ++dest, ++s1, ++s2)
                    *dest = stdx::round_from(
                        alpha1*float(*s1) + alpha2*float(*s2) + beta);
        }
    };
}

std::unique_ptr<reader>
raw_image::blend(std::unique_ptr<reader> src1, float alpha1,
                 std::unique_ptr<reader> src2, float alpha2, float beta) {
    if (!src1 || !src2 || src1->layout() != src2->layout())
        throw std::invalid_argument("image layouts must match for blend");

    // non-uint8 channel types
    switch (src1->layout()) {
    case pixel::a16_le:
        return std::make_unique<blender<uint16_t,1> >(
            move(src1),move(src2),alpha1,alpha2,beta);
    case pixel::f32:
        return std::make_unique<blender<float,1> >(
            move(src1),move(src2),alpha1,alpha2,beta);
    default: break;
    }

    // uint8 channel types
    switch (bytes_per_pixel(src1->layout())) {
    case 1:
        return std::make_unique<blender<uint8_t,1> >(
            move(src1),move(src2),alpha1,alpha2,beta);
    case 2:
        return std::make_unique<blender<uint8_t,2> >(
            move(src1),move(src2),alpha1,alpha2,beta);
    case 3:
        return std::make_unique<blender<uint8_t,3> >(
            move(src1),move(src2),alpha1,alpha2,beta);
    case 4:
        return std::make_unique<blender<uint8_t,4> >(
            move(src1),move(src2),alpha1,alpha2,beta);
    }
    throw std::runtime_error("invalid pixel layout for blend");
}

namespace {
    template <unsigned BPP>
    struct rotate_quad_yuv {
        float s, c;
        void operator()(uint8_t* dest, const uint8_t* src,
                        unsigned nquads) const {
            for ( ; nquads > 0; --nquads)
                for (auto n = 4; n > 0; --n, dest += BPP, src += BPP) {
                    if (BPP >= 3)
                        dest[0] = src[0];
                    const auto u = float(int(src[BPP-2]) - 128);
                    const auto v = float(int(src[BPP-1]) - 128);
                    dest[BPP-2] = stdx::round_from(128 + c*u - s*v);
                    dest[BPP-1] = stdx::round_from(128 + s*u + c*v);
                }
        }
    };
}

std::unique_ptr<reader>
raw_image::rotate_yuv(std::unique_ptr<reader> src, float angle) {
    if (!src) throw std::invalid_argument("empty image");
    if (src->bytes_per_pixel() <= 1)
        throw std::invalid_argument("rotate_yuv requires a color image");

    const auto layout = src->layout();
    const auto ccls = to_color_class(layout);
    if (ccls != cc::yuv_jpeg && ccls != cc::yuv_nv21)
        src = convert(move(src), raw_image::pixel::yuv);
    // src now has layout yuv, uv or vu

    const auto s = std::sin(angle);
    const auto c = std::cos(angle);
    switch (src->bytes_per_pixel()) {
    case 2:
        return transform_quads(move(src), layout, rotate_quad_yuv<2>{s,c});
    case 3:
        return transform_quads(move(src), layout, rotate_quad_yuv<3>{s,c});
    default:
        throw std::logic_error("unexpected bytes per pixel");
    }
}

std::unique_ptr<reader>
raw_image::matrix_multiply(std::unique_ptr<reader> src1, const plane& src2) {
    if (!src1 || src1->layout() != pixel::f32 ||
        src2.layout != pixel::f32)
        throw std::invalid_argument(
            "matrix multiply only works with float pixels");
    if (src1->width() != src2.width)
        throw std::invalid_argument(
            "matrix multiply requires both images to have same width");
    if (src2.bytes_per_line < 4*src2.width || src2.bytes_per_line % 4 != 0)
        throw std::runtime_error("image has incorrect bytes per line");

    struct mmult final : reader {
        const std::unique_ptr<reader> src1;
        const plane& src2;
        const unsigned float_per_line;

        mmult(std::unique_ptr<reader> src1,
                const plane& src2)
            : reader(src2.height, src1->height(), src1->layout()),
              src1(move(src1)),
              src2(src2),
              float_per_line(src2.bytes_per_line / 4) {
        }

        void line_next() override {
            if (!src1->next_line())
                throw std::logic_error("unexpected end of image");
        }

        void line_copy(void* _dest) override {
            auto* dest = static_cast<float*>(_dest);
            auto const* s1 = reinterpret_cast<const float*>(src1->get_line());
            auto const* s2 = reinterpret_cast<const float*>(src2.data);
            for (auto n = src2.height; n > 0; --n, s2 += float_per_line, ++dest)
                *dest = std::inner_product(s1, s1 + src2.width, s2, 0.0f);
        }
    };

    return std::make_unique<mmult>(move(src1), src2);
}
