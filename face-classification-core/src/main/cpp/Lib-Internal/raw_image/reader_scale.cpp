
#include "reader.hpp"
#include <stdext/rounding.hpp>
#include <cstring>
#include <vector>


using namespace raw_image;
using stdx::round_to;


template <std::size_t N, typename T>
static auto make_filled_array(T val) {
    std::array<T,N> r;
    r.fill(val);
    return r;
}

template <typename T, std::size_t N, typename U>
static auto& operator+=(std::array<T,N>& dest, const std::array<U,N>& src) {
    for (std::size_t i = 0; i < N; ++i)
        dest[i] += src[i];
    return dest;
}

template <typename T, typename U, std::size_t N>
static auto operator*(T a, const std::array<U,N>& b) {
    std::array<decltype(a*b.front()),N> dest;
    for (std::size_t i = 0; i < N; ++i)
        dest[i] = a * b[i];
    return dest;
}

/// interpolate single "mid" pixel to N pixels
template <unsigned N>
static auto interpolate(uint8_t mid, uint8_t right);
template <unsigned N>
static auto interpolate(uint8_t left, uint8_t mid, uint8_t right);

template <>
inline auto interpolate<2>(uint8_t mid, uint8_t right) {
    const auto d = (mid <= right ? right - mid + 1 : right - mid - 1) / 4;
    std::array<uint8_t,2> r;
    r[0] = round_to<uint8_t>(mid - d);
    r[1] = uint8_t(2*mid - r[0]);
    return r;
}
template <>
inline auto interpolate<2>(uint8_t left, uint8_t mid, uint8_t right) {
    const auto d0 = mid - left;
    const auto d1 = right - mid;
    const auto lo = std::min(d0,d1);
    const auto hi = std::max(d0,d1);
    std::array<uint8_t,2> r;
    if (0 < lo) {       // left < mid < right
        const auto d = std::min((hi+1)/4, lo/2);
        r[0] = uint8_t(mid - d);
        r[1] = uint8_t(mid + d);
    }
    else if (hi < 0) {  // left > mid > right
        const auto d = std::max((lo-1)/4, hi/2);
        r[0] = uint8_t(mid - d);
        r[1] = uint8_t(mid + d);
    }
    else
        r[0] = r[1] = mid;
    return r;
}

template <>
inline auto interpolate<3>(uint8_t mid, uint8_t right) {
    const auto d = (mid <= right ? right - mid + 1 : right - mid - 1) / 3;
    std::array<uint8_t,3> r;
    r[0] = round_to<uint8_t>(mid - d);
    r[1] = round_to<uint8_t>(2*mid - r[0] - d);
    r[2] = uint8_t(3*mid - r[0] - r[1]);
    return r;
}
template <>
inline auto interpolate<3>(uint8_t left, uint8_t mid, uint8_t right) {
    std::array<uint8_t,3> r;
    const auto a = 52*mid + left + right + 27;  // the 27 is for rounding
    const auto b = 9 * (right-left);
    r[0] = round_to<uint8_t>((a-b)/54);
    r[2] = round_to<uint8_t>((a+b)/54);
    r[1] = round_to<uint8_t>(3*mid - r[0] - r[2]);
    return r;
}


namespace {
    /// upscale height by an integer factor using interpolation
    template <unsigned SCALE>
    class interpolate_vert final : public reader {
    public:
        interpolate_vert(std::unique_ptr<reader> src)
            : reader(src->width(), src->height() * SCALE, src->layout()),
              src(move(src)),
              bytes_to_copy(width()*bytes_per_pixel()) {
        }

    private:
        const std::unique_ptr<reader> src;
        const unsigned bytes_to_copy;  // width * bytes_per_pixel

        // buffer contains SCALE lines of output followed by 2 lines of input
        std::unique_ptr<unsigned char[]> buffer;
        unsigned bpl = 0;        // bytes_per_line (in buffer)
        unsigned output_pos = 0; // [0,SCALE)
        unsigned input_pos = 0;  // 0 or 1 -- location of top line

        void init() {
            bpl = bytes_per_line();
            buffer = std::make_unique<unsigned char[]>((2+SCALE)*bpl);
            auto dest = buffer.get();
            auto src0 = dest + SCALE*bpl;
            src->copy_to(src0);
            if (src->next_line()) {
                // interpolate top 2 lines
                auto src1 = src0 + bpl;
                src->copy_to(src1);
                for (auto n = bytes_to_copy; n > 0; --n,
                         ++src0, ++src1, ++dest) {
                    auto d = dest;
                    for (auto x : interpolate<SCALE>(*src0, *src1)) {
                        *d = x;
                        d += bpl;
                    }
                }
            }
            else // input height is only 1 -- simply replicate the line
                for (auto n = SCALE; n > 0; --n, dest += bpl)
                    memcpy(dest, src0, bytes_to_copy);
        }

        void line_next() override {
            if (++output_pos >= SCALE) {
                auto dest = buffer.get();
                auto src0 = dest + (SCALE+input_pos)*bpl;
                input_pos ^= 1;
                auto src1 = dest + (SCALE+input_pos)*bpl;
                if (src->next_line()) {
                    auto src2 = **src;
                    for (auto n = bytes_to_copy; n > 0; --n,
                             ++src0, ++src1, ++src2, ++dest) {
                        auto d = dest;
                        for (auto x : interpolate<SCALE>(*src0, *src1, *src2)) {
                            *d = x;
                            d += bpl;
                        }
                        *src0 = *src2;
                    }
                }
                else {  // last line -- interpolate bottom 2 lines
                    // note: src and dest in reverse order for last 2 lines
                    dest += (SCALE-1)*bpl; // last output line
                    for (auto n = bytes_to_copy; n > 0; --n,
                             ++src0, ++src1, ++dest) {
                        auto d = dest;
                        for (auto x : interpolate<SCALE>(*src1, *src0)) {
                            *d = x;
                            d -= bpl;
                        }
                    }
                }
                output_pos = 0;
            }
        }

        void line_copy(void* dest) override {
            if (!bpl) init();
            memcpy(dest, buffer.get() + output_pos*bpl, bytes_to_copy);
        }

        const unsigned char* line_direct() override {
            if (!bpl) init();
            return bytes_per_line() <= bpl ?
                buffer.get() + output_pos*bpl : nullptr;
        }

        bool buffered_internally() const override {
            return true;
        }
    };

    /// scale height by either dropping rows or replicating rows
    class nearest_vert final : public reader {
    public:
        nearest_vert(std::unique_ptr<reader> src, unsigned height)
            : reader(src->width(), height, src->layout(),
                     src->pixels_per_line()),
              src(move(src)) {
            // when downscaling we might skip some lines immediately
            const auto target = (2*height-1) * this->src->height();
            while (target < 2 * height * (this->src->lines_remaining()-1))
                this->src->next_line();
        }

    private:
        const std::unique_ptr<reader> src;
        bool new_line = true;

        void line_next() override {
            const auto target = (2*lines_remaining()-1) * src->height();
            while (target < 2 * height() * (src->lines_remaining()-1)) {
                src->next_line();
                new_line = true;
            }
        }

        void line_copy(void* dest) override {
            if (!is_buffer(dest))
                src->copy_to(dest, bytes_per_line());
            else if (new_line) {
                src->copy_to(dest, bytes_per_line());
                new_line = false;
            }
        }

        const unsigned char* line_direct() override {
            return pixels_per_line() <= src->pixels_per_line() ?
                **src : nullptr;
        }

        bool buffered_internally() const override {
            src->force_buffer(bytes_per_line());
            return true;
        }
    };

    /// scale height to arbitrary value using area method
    class scale_vert final : public reader {
    public:
        static auto min_pixels_per_line(const reader& src) {
            const auto bpp = src.bytes_per_pixel();
            const auto nbytes = unsigned(src.width()*bpp + 3) & ~3u;
            return (nbytes + bpp - 1) / bpp;
        }

        scale_vert(std::unique_ptr<reader> src, unsigned height)
            : reader(src->width(), height, src->layout(),
                               min_pixels_per_line(*src)),
              src(move(src)),
              ofs(this->src->height()/2),
              dh(this->src->height()),
              sh(height),
              line((this->src->width() * this->src->bytes_per_pixel() + 3) / 4,
                   make_filled_array<4>(uint32_t(ofs))) {
            this->src->set_pixels_per_line(pixels_per_line());
        }

    private:
        const std::unique_ptr<reader> src;
        const unsigned ofs;  ///< for rounding
        unsigned dh, sh;
        std::vector<std::array<uint32_t,4> > line;
        bool line_done = false;

        void line_next() override {
            if (!line_done)
                throw std::runtime_error("line not copied");
            line_done = false;
            if (!*src)
                throw std::logic_error("unexpected end of image");
        }

        void line_copy(void* destv) override {
            if (line_done)
                throw std::runtime_error("attempt to copy same line twice");
            line_done = true;

            while (sh < dh) {
                auto sp = src->as_bpp<4>();
                for (auto& z : line)
                    z += sh * *sp++;
                dh -= sh;
                sh = height();
                if (!src->next_line())
                    throw std::logic_error("unexpected end of image");
            }
            auto dest = static_cast<uint8_t*>(destv);
            if (sh < dh + src->height()) {
                const auto h_next = sh - dh;
                auto sp = src->as_bpp<4>();
                for (auto& z : line) {
                    z += dh * *sp;
                    for (auto zi : z)
                        *dest++ = (zi / src->height()) & 0xff;
                    z.fill(ofs);
                    z += h_next * *sp++;
                }
                dh = src->height() - h_next;
                sh = m_height;
                src->next_line();
            }
            else { // sh >= dh + src.height
                auto sp = src->as_bpp<4>();
                for (auto& z : line) {
                    z += dh * *sp++;
                    for (auto zi : z)
                        *dest++ = (zi / src->height()) & 0xff;
                    z.fill(ofs);
                }
                sh -= dh;
                dh = src->height();
            }
        }

        bool buffered_internally() const override {
            src->force_buffer();
            return true;
        }
    };


    /// upscale width by an integer factor using interpolation
    template <unsigned BPP, unsigned SCALE>
    class interpolate_horz final : public reader {
    public:
        interpolate_horz(std::unique_ptr<reader> src)
            : reader(src->width() * SCALE, src->height(),
                               src->layout()),
              src(move(src)) {
            if (BPP != this->src->bytes_per_pixel())
                throw std::invalid_argument("bytes_per_pixel mismatch");
        }

    private:
        const std::unique_ptr<reader> src;

        void line_next() override {
            if (!src->next_line())
                throw std::logic_error("unexpected end of image");
        }

        void line_copy(void* dest) override {
            auto dp = static_cast<uint8_t*>(dest);
            auto sp = **src;

            if (src->width() <= 1) {
                // make single pixel into a SCALE x SCALE block
                if (src->width() == 1)
                    for (unsigned j = SCALE; j > 0; --j, dp += BPP)
                        memcpy(dp, sp, BPP);
                return;
            }

            // left-most pixel
            for (unsigned i = 0; i < BPP; ++i, ++dp) {
                auto dest = dp;
                for (auto x : interpolate<SCALE>(sp[i],sp[i+BPP])) {
                    *dest = x;
                    dest += BPP;
                }
            }
            dp += (SCALE-1)*BPP;

            // middle pixels
            for (auto j = src->width()-2; j > 0; --j) {
                for (auto i = BPP; i > 0; --i, ++sp, ++dp) {
                    auto dest = dp;
                    for (auto x : interpolate<SCALE>(*sp,sp[BPP],sp[2*BPP])) {
                        *dest = x;
                        dest += BPP;
                    }
                }
                dp += (SCALE-1)*BPP;
            }

            // right-most pixel
            dp += (SCALE-1)*BPP; // right-most subpixel
            for (auto i = BPP; i > 0; --i, ++sp, ++dp) {
                auto dest = dp;
                for (auto x : interpolate<SCALE>(sp[BPP],*sp)) {
                    *dest = x;
                    dest -= BPP; // reverse order
                }
            }
        }

        bool buffered_internally() const override {
            src->force_buffer();
            return true;
        }
    };

    template <unsigned SCALE>
    std::unique_ptr<reader> make_interpolate_horz(
        std::unique_ptr<reader> src) {
        switch (src->bytes_per_pixel()) {
        case 1:
            return std::make_unique<interpolate_horz<1,SCALE>>(move(src));
        case 2:
            return std::make_unique<interpolate_horz<2,SCALE>>(move(src));
        case 3:
            return std::make_unique<interpolate_horz<3,SCALE>>(move(src));
        case 4:
            return std::make_unique<interpolate_horz<4,SCALE>>(move(src));
        default:
            throw std::invalid_argument("unsupported bytes per pixel");
        }
    }


    /// scale width by either dropping pixels or replicating pixels
    template <unsigned BPP>
    class nearest_horz final : public reader {
    public:
        nearest_horz(std::unique_ptr<reader> src, unsigned width)
            : reader(width, src->height(), src->layout()),
              src(move(src)) {
            if (BPP != this->src->bytes_per_pixel())
                throw std::invalid_argument("bytes_per_pixel mismatch");
        }

    private:
        const std::unique_ptr<reader> src;

        void line_next() override {
            if (!src->next_line())
                throw std::logic_error("unexpected end of image");
        }

        void line_copy(void* dest) override {
            auto sp = **src;
            unsigned src_remain = src->width();

            // when downscaling we might start by skipping a few source pixels
            const auto target = (2*width()-1) * src->width();
            while (target < 2 * width() * (src_remain-1)) {
                sp += BPP;
                --src_remain;
            }

            auto dp = static_cast<uint8_t*>(dest);
            for (unsigned dest_remain = width(); ; ) {
                memcpy(dp, sp, BPP);
                dp += BPP;
                if (--dest_remain == 0)
                    break;
                const auto target = (2*dest_remain-1) * src->width();
                while (target < 2 * width() * (src_remain-1)) {
                    sp += BPP;
                    --src_remain;
                }
            }
        }

        bool buffered_internally() const override {
            src->force_buffer();
            return true;
        }
    };

    std::unique_ptr<reader> make_nearest_horz(
        std::unique_ptr<reader> src, unsigned width) {
        switch (src->bytes_per_pixel()) {
        case 1:
            return std::make_unique<nearest_horz<1>>(move(src), width);
        case 2:
            return std::make_unique<nearest_horz<2>>(move(src), width);
        case 3:
            return std::make_unique<nearest_horz<3>>(move(src), width);
        case 4:
            return std::make_unique<nearest_horz<4>>(move(src), width);
        default:
            throw std::invalid_argument("unsupported bytes per pixel");
        }
    }

    /// scale width to arbitrary value using area method
    template <unsigned BPP>
    class scale_horz final : public reader {
    public:
        scale_horz(std::unique_ptr<reader> src, unsigned width)
            : reader(width, src->height(), src->layout()),
              src(move(src)),
              ofs(this->src->width()/2) {
            if (BPP != this->src->bytes_per_pixel())
                throw std::invalid_argument("bytes_per_pixel mismatch");
        }

    private:
        const std::unique_ptr<reader> src;
        const unsigned ofs;  ///< for rounding

        void line_next() override {
            if (!src->next_line())
                throw std::logic_error("unexpected end of image");
        }

        void line_copy(void* destv) override {
            auto dest = static_cast<uint8_t*>(destv);
            auto sp = src->as_bpp<BPP>();
            auto sw = width();
            for (auto j = m_width; j > 0; --j) {
                std::array<uint32_t,BPP> z;
                z.fill(ofs);
                auto dw = src->width();
                while (sw <= dw) {
                    z += sw * *sp;
                    dw -= sw;
                    sw = m_width;
                    ++sp;
                }
                if (0 < dw) { // && dw < sw
                    z += dw * *sp;
                    sw -= dw;
                }
                for (auto zi : z)
                    *dest++ = (zi / src->width()) & 0xff;
            }
        }

        bool buffered_internally() const override {
            src->force_buffer();
            return true;
        }
    };

    std::unique_ptr<reader> make_scale_horz(
        std::unique_ptr<reader> src, unsigned width) {
        switch (src->bytes_per_pixel()) {
        case 1:
            return std::make_unique<scale_horz<1>>(move(src),width);
        case 2:
            return std::make_unique<scale_horz<2>>(move(src),width);
        case 3:
            return std::make_unique<scale_horz<3>>(move(src),width);
        case 4:
            return std::make_unique<scale_horz<4>>(move(src),width);
        default:
            throw std::invalid_argument("unsupported bytes per pixel");
        }
    }
}

std::unique_ptr<reader>
raw_image::scale_nearest(std::unique_ptr<reader> src,
                         unsigned width, unsigned height) {
    if (!src)
        throw std::invalid_argument("input reader is nullptr");
    if (height < src->height())
        src = std::make_unique<nearest_vert>(move(src), height);
    if (width != src->width())
        src = make_nearest_horz(move(src), width);
    if (height > src->height())
        src = std::make_unique<nearest_vert>(move(src), height);
    return src;
}

std::unique_ptr<reader>
raw_image::scale_area(std::unique_ptr<reader> src,
                      unsigned width, unsigned height) {
    if (!src)
        throw std::invalid_argument("input reader is nullptr");
    if (width < src->width())
        src = make_scale_horz(move(src), width);
    if (height < src->height())
        src = std::make_unique<scale_vert>(move(src), height);
    else if (height > src->height()) {
        const auto factor = height / src->height();
        if (height == factor * src->height())
            src = std::make_unique<nearest_vert>(move(src), height);
        else
            src = std::make_unique<scale_vert>(move(src), height);
    }
    if (width > src->width()) {
        const auto factor = width / src->width();
        if (width == factor * src->width())
            src = make_nearest_horz(move(src), width);
        else
            src = make_scale_horz(move(src), width);
    }
    return src;
}

std::unique_ptr<reader>
raw_image::scale_interpolate(std::unique_ptr<reader> src,
                             unsigned width, unsigned height) {
    if (!src)
        throw std::invalid_argument("input reader is nullptr");

    while (width != src->width() || height != src->height()) {

        if (width < src->width())
            src = make_scale_horz(move(src), width);

        const auto halfh = src->height()/2;
        if (height <= src->height() + halfh) {
            if (height != src->height())
                src = std::make_unique<scale_vert>(move(src), height);
        }
        else if (height < 2*src->height() + halfh)
            src = std::make_unique<interpolate_vert<2> >(move(src));
        else
            src = std::make_unique<interpolate_vert<3> >(move(src));

        const auto halfw = src->width()/2;
        if (width <= src->width() + halfw) {
            if (width != src->width())
                src = make_scale_horz(move(src), width);
        }
        else if (width < 2*src->width() + halfw)
            src = make_interpolate_horz<2>(move(src));
        else
            src = make_interpolate_horz<3>(move(src));
    }
    return src;
}
