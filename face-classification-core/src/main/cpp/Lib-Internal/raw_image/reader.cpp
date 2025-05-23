
#include "reader.hpp"
#include "color_convert.hpp"

#include <applog/core.hpp>

#include <algorithm>
#include <cassert>
#include <cstring>
#include <vector>


using namespace raw_image;


const uint8_t* reader::get_line() {
    if (!m_buf) {
        if (auto p = line_direct()) return p;
        m_buf = std::make_unique<uint8_t[]>(m_bytes_per_line);
    }
    if (!m_line_copied_to_buf) {
        line_copy(m_buf.get());
        m_line_copied_to_buf = true;
    }
    return m_buf.get();
}

void* reader::copy_to(void* dest, unsigned dest_bytes) {
    if (dest_bytes >= m_bytes_per_line && !m_buf)
        line_copy(dest);
    else {
        if (!m_line_copied_to_buf) {
            if (!m_buf)
                m_buf = std::make_unique<uint8_t[]>(m_bytes_per_line);
            line_copy(m_buf.get());
            m_line_copied_to_buf = true;
        }
        memcpy(dest, m_buf.get(), dest_bytes);
    }
    return dest;
}

void reader::copy_to(const plane& dest, unsigned per_line) {
    if (per_line > dest.bytes_per_line)
        per_line = dest.bytes_per_line;
    auto n = std::min(dest.height, lines_remaining());
    for (auto d = dest.data; n > 0; --n, next_line(), d += dest.bytes_per_line)
        copy_to(d, per_line);
}

void* reader::map_to(void* dest_, unsigned dest_bpp,
                     const std::array<unsigned,4>& dest_idx) {
    if (!m_line_copied_to_buf) {
        if (!m_buf)
            m_buf = std::make_unique<uint8_t[]>(m_bytes_per_line);
        line_copy(m_buf.get());
        m_line_copied_to_buf = true;
    }
    auto const* src = m_buf.get();
    auto dest = static_cast<uint8_t*>(dest_);
    for (auto n = m_pixels_per_line; n > 0; --n,
             src += m_bpp, dest += dest_bpp)
        for (unsigned i = 0; i < m_bpp; ++i)
            if (dest_idx[i] < dest_bpp)
                dest[dest_idx[i]] = src[i];
    return dest_;
}

void reader::map_to(const plane& dest, const std::array<unsigned,4>& dest_idx) {
    const auto dest_bpp = raw_image::bytes_per_pixel(dest.layout);
    auto n = std::min(dest.height, lines_remaining());
    for (auto d = dest.data; n > 0; --n, next_line(), d += dest.bytes_per_line)
        map_to(d, dest_bpp, dest_idx);
}

void reader::rotate_to(const plane& dest, unsigned rotate) {
    const auto bpp = bytes_per_pixel();
    if (bpp != raw_image::bytes_per_pixel(dest.layout))
        throw std::invalid_argument("rotate_to: source reader and destination image have different bytes per pixel");
    auto n = std::min(lines_remaining(),
                      rotate&1 ? dest.width : dest.height);
    auto line = dest.data;
    int per_pixel, per_line;
    switch (rotate & 7) {
    case 0: {
        const auto bpl = manages_pixel_buffer(single_plane_arg(dest)) ?
            dest.bytes_per_line : dest.width*bpp;
        copy_to(dest, bpl);
        return;
    }
    case 6: { // flip
        const auto bpl = manages_pixel_buffer(single_plane_arg(dest)) ?
            dest.bytes_per_line : dest.width*bpp;
        line += dest.height * dest.bytes_per_line;
        for ( ; n > 0; --n, next_line())
            copy_to(line -= dest.bytes_per_line, bpl);
        return;
    }
    case 2: // flip and mirror
        per_pixel = -int(bpp);
        per_line = -int(dest.bytes_per_line);
        line += (dest.height-1) * dest.bytes_per_line
            + (dest.width-1) * bpp;
        break;
    case 4: // mirror
        per_pixel = -int(bpp);
        per_line = int(dest.bytes_per_line);
        line += (dest.width-1) * bpp;
        break;
    case 1:
        per_pixel = -int(dest.bytes_per_line);
        per_line = int(bpp);
        line += (dest.height-1) * dest.bytes_per_line;
        break;
    case 3:
        per_pixel = int(dest.bytes_per_line);
        per_line = -int(bpp);
        line += (dest.width-1) * bpp;
        break;
    case 5: // transpose
        per_pixel = int(dest.bytes_per_line);
        per_line = int(bpp);
        break;
    case 7:
        per_pixel = -int(dest.bytes_per_line);
        per_line = -int(bpp);
        line += (dest.height-1) * dest.bytes_per_line
            + (dest.width-1) * bpp;
        break;
    }
    for ( ; n > 0; --n, next_line(), line += per_line) {
        auto s = get_line();
        auto d = line;
        for (auto j = width(); j > 0; --j, d += per_pixel, s += bpp)
            memcpy(d, s, bpp);
    }
}

namespace {
    /** \brief Optimized reader for single plane rotate 0 and 6 (flip).
     *
     * The layout may be overriden, but bytes_per_pixel must match the image.
     * The only meaningful overrides are GRAY8 -> R8, G8 or B8.
     */
    class memcpy_reader final : public reader {
    public:
        memcpy_reader(const plane& img, pixel_layout cs, bool flip)
            : reader(img.width, img.height, cs),
              bytes_to_copy(img.width * raw_image::bytes_per_pixel(cs)),
              incr(flip ? -int(img.bytes_per_line) : int(img.bytes_per_line)),
              line(flip ? img.data+std::size_t(img.height-1)*img.bytes_per_line : img.data) {
            if (raw_image::bytes_per_pixel(cs) != raw_image::bytes_per_pixel(img.layout))
                throw std::invalid_argument("layout has incompatible bytes_per_pixel");
        }

    private:
        const unsigned bytes_to_copy;
        const int incr;
        const uint8_t* line;

        void line_next() override {
            line += incr;
        }
        void line_copy(void* dest) override {
            memcpy(dest, line, bytes_to_copy);
        }
        const uint8_t* line_direct() override {
            return bytes_per_line() <= bytes_to_copy ? line : nullptr;
        }
    };


    /** \brief Optimized reader for single plane images with no change
     * in pixel layout.
     *
     * For rotate == 0 or rotate == 6, memcpy_reader above is a better choice.
     *
     * The layout may be overriden, but bytes_per_pixel must match the image.
     * The only meaningful overrides are GRAY8 -> R8, G8 or B8.
     */
    class pixel_reader final : public reader {
    public:
        pixel_reader(const plane& img, pixel_layout cs, unsigned rot)
            : reader(rot&1 ? img.height : img.width,
                     rot&1 ? img.width : img.height, cs),
              line(img.data),
              line_incr(int(img.bytes_per_line)),
              px_incr(int(m_bpp)) {
            if (raw_image::bytes_per_pixel(cs) != raw_image::bytes_per_pixel(img.layout))
                throw std::invalid_argument("layout has incompatible bytes_per_pixel");
            if (rot&1)
                std::swap(line_incr, px_incr);
            switch (rot&7) {
            case 2:
            case 7: line_incr = -line_incr; px_incr = -px_incr; break;
            case 3:
            case 4: px_incr = -px_incr; break;
            case 1:
            case 6: line_incr = -line_incr; break;
            }
            if (line_incr < 0)
                line += (m_height-1) * unsigned(-line_incr);
            if (px_incr < 0)
                line += (m_width-1) * unsigned(-px_incr);
        }

    private:
        const uint8_t* line;
        int line_incr;
        int px_incr;

        void line_next() override {
            line += line_incr;
        }
        void line_copy(void* vdest) override {
            auto px = line;
            uint8_t* dest = static_cast<uint8_t*>(vdest);
            for (auto j = m_width; j > 0; --j, dest += m_bpp, px += px_incr)
                memcpy(dest, px, m_bpp);
        }
    };


    // { pointer to channel value,
    //   increment to next pixel in line,
    //   increment to start of next line }
    // note: the increments may be negative
    using channel_record = std::tuple<const uint8_t*, int, int>;


    /** \brief General reader from multi-plane image in any orientation.
     */
    template <unsigned BPP>
    class channel_reader : public reader {
    public:
        channel_reader(unsigned width, unsigned height,
                       pixel_layout cs,
                       const channel_record* channels)
            : reader(width, height, cs) {
            if (bytes_per_pixel() != BPP)
                throw std::invalid_argument("layout has incompatible bytes_per_pixel");
            for (unsigned i = 0; i < BPP; ++i, ++channels) {
                line[i].first = std::get<0>(*channels);
                line[i].second = std::get<1>(*channels);
                incr[i] = std::get<2>(*channels);
            }
        }

    protected:
        std::array<std::pair<const uint8_t*, int>, BPP> line;
        std::array<int, BPP> incr;

        void line_next() override {
            auto it = incr.begin();
            for (auto& p : line)
                p.first += *it++;
        }

        void line_copy(void* destv) override {
            auto dest = static_cast<uint8_t*>(destv);
            auto line = this->line;
            for (auto i = width(); i > 0; --i)
                for (auto& p : line) {
                    *dest++ = *p.first;
                    p.first += p.second;
                }
        }
    };

    /** \brief Specialized reader to upsample by 2 all but first channel.
     */
    template <unsigned BPP>
    class channel_up2 final : public channel_reader<BPP> {
    public:
        using channel_reader<BPP>::channel_reader;

    private:
        void line_next() override {
            auto it = this->incr.begin();
            auto jt = this->line.begin();
            jt->first += *it;
            if ((this->lines_remaining()&1) == 0) {
                while (++jt != this->line.end())
                    jt->first += *++it;
            }
        }

        void line_copy(void* destv) override {
            auto dest = static_cast<uint8_t*>(destv);
            auto line = this->line;
            for (auto i = this->width()/2; i > 0; --i) {
                for (auto& p : line)
                    *dest++ = *p.first;
                line[0].first += line[0].second;
                for (auto& p : line) {
                    *dest++ = *p.first;
                    p.first += p.second;
                }
            }
        }
    };
}

static void image_params(const uint8_t*& first_pixel, int& bpp, int& bpl,
                         const plane& img, unsigned rotate) {
    first_pixel = img.data;
    bpp = int(bytes_per_pixel(img.layout));
    bpl = int(img.bytes_per_line);
    switch (rotate&7) {
    case 0:  // start at top left
    case 5:
        break;
    case 2:  // start at bottom right
    case 7:
        first_pixel += bpl * int(img.height-1);
        first_pixel += bpp * int(img.width-1);
        bpl = -bpl;
        bpp = -bpp;
        break;
    case 1:
    case 4:  // start at top right
        first_pixel += bpp * int(img.width-1);
        bpp = -bpp;
        break;
    case 3:
    case 6:  // start at bottom left
        first_pixel += bpl * int(img.height-1);
        bpl = -bpl;
        break;
    }
    if (rotate&1)
        std::swap(bpp, bpl);
}

static const uint8_t s_zero = 0;
static const uint8_t s_128 = 128;

namespace {
    struct channel_mapping {
        channel_record rgba[4] = {};
        channel_record yuv_jpeg[3] = {};
        channel_record yuv_nv21[3] = {};
        bool uv_half_res = false;
        bool split_res = false;
        unsigned w, h;

        // apply mappings for source image
        void update(pixel_layout cs,
                    const uint8_t* data, int bpp, int bpl) {
            switch (to_color_class(cs)) {

            case cc::gray:
                yuv_jpeg[0] = { data, bpp, bpl };
                rgba[0] = { data, bpp, bpl };
                rgba[1] = { data, bpp, bpl };
                rgba[2] = { data, bpp, bpl };
                break;

            case cc::yuv_jpeg: {
                const color_channels<cc::yuv_jpeg> channels(cs);
                if (channels.y_idx >= 0)
                    yuv_jpeg[0] = { data + channels.y_idx, bpp, bpl };
                if (channels.u_idx >= 0)
                    yuv_jpeg[1] = { data + channels.u_idx, bpp, bpl };
                if (channels.v_idx >= 0)
                    yuv_jpeg[2] = { data + channels.v_idx, bpp, bpl };
                break;
            }

            case cc::yuv_nv21: {
                const color_channels<cc::yuv_nv21> channels(cs);
                if (channels.y_idx >= 0)
                    yuv_nv21[0] = { data + channels.y_idx, bpp, bpl };
                if (channels.u_idx >= 0)
                    yuv_nv21[1] = { data + channels.u_idx, bpp, bpl };
                if (channels.v_idx >= 0)
                    yuv_nv21[2] = { data + channels.v_idx, bpp, bpl };
                break;
            }

            case cc::alpha:
                rgba[3] = { data, bpp, bpl };
                break;

            case cc::rgb: {
                const color_channels<cc::rgb> channels(cs);
                if (channels.red >= 0)
                    rgba[0] = { data + channels.red, bpp, bpl };
                if (channels.green >= 0)
                    rgba[1] = { data + channels.green, bpp, bpl };
                if (channels.blue >= 0)
                    rgba[2] = { data + channels.blue, bpp, bpl };
                if (channels.alpha >= 0)
                    rgba[3] = { data + channels.alpha, bpp, bpl };
                break;
            }
            default:
                FILE_LOG(logWARNING) << "raw_image::reader: unknown pixel layout class (for " << cs << ")";
                break;
            }
        }

        void fill_missing() {
            if (std::get<0>(yuv_nv21[0]) ||
                std::get<0>(yuv_nv21[1]) ||
                std::get<0>(yuv_nv21[2])) {
                for (auto& t : yuv_nv21)
                    if (!std::get<0>(t))
                        t = { &s_128, 0, 0 };
                for (auto& t : yuv_jpeg) t = {};
                for (auto& t : rgba) t = {};
            }
            else if (std::get<0>(yuv_jpeg[1]) ||
                     std::get<0>(yuv_jpeg[2])) {
                for (auto& t : yuv_jpeg)
                    if (!std::get<0>(t))
                        t = { &s_128, 0, 0 };
                for (auto& t : yuv_nv21) t = {};
                for (auto& t : rgba) t = {};
            }
            else if (std::get<0>(rgba[0]) ||
                     std::get<0>(rgba[1]) ||
                     std::get<0>(rgba[2])) {
                for (auto& t : rgba)
                    if (!std::get<0>(t))
                        t = { &s_zero, 0, 0 };
                if (std::get<0>(yuv_jpeg[0])) {  // y -> rgb or y -> yuv
                    yuv_jpeg[1] = { &s_128, 0, 0 };
                    yuv_jpeg[2] = { &s_128, 0, 0 };
                }
            }
        }

        std::vector<channel_record>
        to_layout(pixel_layout dest_cs) {
            std::vector<channel_record> channels;
            const auto bpp = raw_image::bytes_per_pixel(dest_cs);
            channels.reserve(bpp);

            switch (to_color_class(dest_cs)) {

            case cc::alpha:
                channels.emplace_back(rgba[3]);
                break;

            case cc::gray:
                channels.emplace_back(yuv_jpeg[0]);
                break;

            case cc::yuv_jpeg: {
                const color_channels<cc::yuv_jpeg> chan(dest_cs);
                std::pair<int,unsigned> ord[] = {
                    { chan.y_idx, 0 },
                    { chan.u_idx, 1 },
                    { chan.v_idx, 2 },
                };
                std::sort(std::begin(ord), std::end(ord));
                for (auto& p : ord)
                    if (p.first >= 0)
                        channels.emplace_back(yuv_jpeg[p.second]);
                if (bpp == 2 && uv_half_res) w/=2, h/=2;
                if (bpp == 3) split_res = uv_half_res;
                break;
            }
            case cc::yuv_nv21: {
                const color_channels<cc::yuv_nv21> chan(dest_cs);
                std::pair<int,unsigned> ord[] = {
                    { chan.y_idx, 0 },
                    { chan.u_idx, 1 },
                    { chan.v_idx, 2 },
                };
                std::sort(std::begin(ord), std::end(ord));
                for (auto& p : ord)
                    if (p.first >= 0)
                        channels.emplace_back(yuv_nv21[p.second]);
                if (bpp == 2 && uv_half_res) w/=2, h/=2;
                if (bpp == 3) split_res = uv_half_res;
                break;
            }

            case cc::rgb: {
                const color_channels<cc::rgb> chan(dest_cs);
                std::pair<int,unsigned> ord[] = {
                    { chan.red, 0 },
                    { chan.green, 1 },
                    { chan.blue, 2 },
                    { chan.alpha, 3 }
                };
                std::sort(std::begin(ord), std::end(ord));
                for (auto& p : ord)
                    if (p.first >= 0)
                        channels.emplace_back(rgba[p.second]);
                break;
            }

            default: break;
            }

            return channels;
        }
    };
}

static std::unique_ptr<reader> read_channels(
    const multi_plane_arg& multi_plane,
    unsigned rotate,
    pixel_layout dest_layout) {

    if (multi_plane.empty())
        return {};

    if (multi_plane.size() == 1) {
        auto& img = multi_plane.front();
        if (img.layout == dest_layout ||
            (img.layout == pixel::gray8 &&
             (dest_layout == pixel::r8 ||
              dest_layout == pixel::g8 ||
              dest_layout == pixel::b8))) {
            switch (rotate) {
            case 0:
                return std::make_unique<memcpy_reader>(
                    img, dest_layout, false);
            case 6:
                return std::make_unique<memcpy_reader>(
                    img, dest_layout, true);
            }
            return std::make_unique<pixel_reader>(img, dest_layout, rotate);
        }
    }

    channel_mapping map;
    map.w = multi_plane.front().width;
    map.h = multi_plane.front().height;

    for (auto& img : multi_plane) {
        if (same_channel_order(img.layout, pixel::uv16_jpeg) ||
            same_channel_order(img.layout, pixel::vu16_jpeg) ||
            same_channel_order(img.layout, pixel::u8_jpeg) ||
            same_channel_order(img.layout, pixel::v8_jpeg)) {
            if (img.width == map.w && img.height == map.h)
                map.uv_half_res = false;
            else if (img.width*2 == map.w && img.height*2 == map.h)
                map.uv_half_res = true;
            else
                throw std::invalid_argument("image plane dimension mismatch");
        }
        else if (img.width != map.w || img.height != map.h)
            throw std::invalid_argument("image plane dimension mismatch");

        const uint8_t* data;
        int bpp, bpl;
        image_params(data, bpp, bpl, img, rotate);

        map.update(img.layout, data, bpp, bpl);
    }

    map.fill_missing();

    if (rotate & 1)
        std::swap(map.w, map.h);

    const auto channels = map.to_layout(dest_layout);

    if (!channels.empty()) {
        assert(channels.size() == raw_image::bytes_per_pixel(dest_layout));
        for (auto& t : channels)
            if (!std::get<0>(t))
                return {};

        switch (channels.size()) {
        case 1:
            assert(!map.split_res);
            return std::make_unique<channel_reader<1> >(
                map.w,map.h,dest_layout,channels.data());

        case 2:
            assert(!map.split_res);
            return std::make_unique<channel_reader<2> >(
                map.w,map.h,dest_layout,channels.data());

        case 3:
            if (map.split_res)
                return std::make_unique<channel_up2<3> >(
                    map.w,map.h,dest_layout,channels.data());
            else
                return std::make_unique<channel_reader<3> >(
                    map.w,map.h,dest_layout,channels.data());

        case 4:
            assert(!map.split_res);
            return std::make_unique<channel_reader<4> >(
                map.w,map.h,dest_layout,channels.data());
        }
    }

    return {};
}

static auto yuv24_jpeg_to(pixel_layout dest_cs,
                          std::unique_ptr<reader> yuv) {
    static constexpr auto src_cs = pixel::yuv24_jpeg;
    using from = color_convert_from<color_class::yuv_jpeg>;
    switch (to_color_class(dest_cs)) {

    case color_class::rgb: {
        using conv = color_convert_to<color_class::rgb,from,4>;
        return transform_quads(move(yuv), dest_cs, conv{dest_cs, src_cs});
    }

    case color_class::r85g10b05: {
        using conv = color_convert_to<color_class::r85g10b05,from,4>;
        return transform_quads(move(yuv), dest_cs, conv{dest_cs, src_cs});
    }

    default: return std::unique_ptr<reader>{};
    }
}

static auto yuv24_nv21_to(pixel_layout dest_cs,
                          std::unique_ptr<reader> yuv) {
    static constexpr auto src_cs = pixel::yuv24_nv21;
    using from = color_convert_from<color_class::yuv_nv21>;
    switch (to_color_class(dest_cs)) {

    case color_class::rgb: {
        using conv = color_convert_to<color_class::rgb,from,4>;
        return transform_quads(move(yuv), dest_cs, conv{dest_cs, src_cs});
    }

    case color_class::r85g10b05: {
        using conv = color_convert_to<color_class::r85g10b05,from,4>;
        return transform_quads(move(yuv), dest_cs, conv{dest_cs, src_cs});
    }

    default: return std::unique_ptr<reader>{};
    }
}

static auto rgb24_to(pixel_layout dest_cs,
                     std::unique_ptr<reader> rgb) {
    static constexpr auto src_cs = pixel::rgb24;
    using from = color_convert_from<color_class::rgb>;
    switch (to_color_class(dest_cs)) {

    case color_class::gray: {
        using conv = color_convert_to<color_class::gray,from,4>;
        return transform_quads(move(rgb), dest_cs, conv{dest_cs, src_cs});
    }

    case color_class::yuv_jpeg: {
        using conv = color_convert_to<color_class::yuv_jpeg,from,4>;
        return transform_quads(move(rgb), dest_cs, conv{dest_cs, src_cs});
    }

    case color_class::yuv_nv21: {
        using conv = color_convert_to<color_class::yuv_nv21,from,4>;
        return transform_quads(move(rgb), dest_cs, conv{dest_cs, src_cs});
    }

    case color_class::r85g10b05: {
        using conv = color_convert_to<color_class::r85g10b05,from,4>;
        return transform_quads(move(rgb), dest_cs, conv{dest_cs, src_cs});
    }

    default: return std::unique_ptr<reader>{};
    }
}

std::unique_ptr<reader>
reader::construct_with_opts(
    const multi_plane_arg& src,
    const stdx::options_tuple<rotate,pixel_layout>& opts) {

    if (src.empty()) return {};

    auto dest_layout = std::get<pixel_layout>(opts);
    if (dest_layout == pixel::none)
        dest_layout = src.front().layout;
    const auto dest_cc = to_color_class(dest_layout);

    const auto rot = unsigned(std::get<rotate>(opts)) & 7;

    if (src.size() == 1 && src.front().layout == dest_layout)
        return read_channels(src, rot, dest_layout);

    bool src_rgb = false, src_jpeg = false, src_nv21 = false;
    for (auto& plane : src) {
        switch (to_color_class(plane.layout)) {
        case cc::rgb:      src_rgb = true;  break;
        case cc::yuv_jpeg: src_jpeg = true; break;
        case cc::yuv_nv21:
            if (plane.layout != pixel::y8_nv21)
                src_nv21 = true;
            break;
        default: break;
        }
    }

    if (src_jpeg) {
        if (dest_cc == cc::yuv_jpeg || dest_cc == cc::gray)
            return read_channels(src, rot, dest_layout);
        else
            return yuv24_jpeg_to(
                dest_layout, read_channels(src, rot, pixel::yuv24_jpeg));
    }

    if (src_nv21) {
        if (dest_cc == cc::yuv_nv21)
            return read_channels(src, rot, dest_layout);
        else
            return yuv24_nv21_to(
                dest_layout, read_channels(src, rot, pixel::yuv24_nv21));
    }

    if (src_rgb) {
        if (dest_cc == cc::rgb || dest_cc == cc::alpha)
            return read_channels(src, rot, dest_layout);
        else
            return rgb24_to(
                dest_layout, read_channels(src, rot, pixel::rgb24));
    }

    // only Y8 -> some RGB
    return read_channels(src, rot, dest_layout);
}

std::unique_ptr<reader>
raw_image::convert(std::unique_ptr<reader> src, pixel_layout dest_cs) {
    if (!src || src->layout() == dest_cs || dest_cs == pixel::none)
        return src;

    const auto src_cs = src->layout();
    const auto dest_cc = to_color_class(dest_cs);

    switch (to_color_class(src_cs)) {

    case cc::gray: {
        using from = color_convert_from<cc::gray>;
        switch (dest_cc) {
        case cc::yuv_jpeg: {
            using conv = color_convert_to<cc::yuv_jpeg,from,4>;
            return transform_quads(move(src), dest_cs, conv{dest_cs, src_cs});
        }
        case cc::rgb: {
            using conv = color_convert_to<cc::rgb,from,4>;
            return transform_quads(move(src), dest_cs, conv{dest_cs, src_cs});
        }
        case cc::r85g10b05: {
            using conv = color_convert_to<cc::r85g10b05,from,4>;
            return transform_quads(move(src), dest_cs, conv{dest_cs, src_cs});
        }
        default: break;
        }
        break;
    }

    case cc::yuv_jpeg: {
        using from = color_convert_from<cc::yuv_jpeg>;
        switch (dest_cc) {
        case cc::gray: {
            using conv = color_convert_to<cc::gray,from,4>;
            return transform_quads(move(src), dest_cs, conv{dest_cs, src_cs});
        }
        case cc::yuv_jpeg: {
            using conv = color_convert_to<cc::yuv_jpeg,from,4>;
            return transform_quads(move(src), dest_cs, conv{dest_cs, src_cs});
        }
        case cc::rgb: {
            using conv = color_convert_to<cc::rgb,from,4>;
            return transform_quads(move(src), dest_cs, conv{dest_cs, src_cs});
        }
        case cc::r85g10b05: {
            using conv = color_convert_to<cc::r85g10b05,from,4>;
            return transform_quads(move(src), dest_cs, conv{dest_cs, src_cs});
        }
        default: break;
        }
        break;
    }

    case cc::yuv_nv21: {
        using from = color_convert_from<cc::yuv_nv21>;
        switch (dest_cc) {
        case cc::yuv_nv21: {
            using conv = color_convert_to<cc::yuv_nv21,from,4>;
            return transform_quads(move(src), dest_cs, conv{dest_cs, src_cs});
        }
        case cc::rgb: {
            using conv = color_convert_to<cc::rgb,from,4>;
            return transform_quads(move(src), dest_cs, conv{dest_cs, src_cs});
        }
        case cc::r85g10b05: {
            using conv = color_convert_to<cc::r85g10b05,from,4>;
            return transform_quads(move(src), dest_cs, conv{dest_cs, src_cs});
        }
        default: break;
        }
        break;
    }

    case cc::rgb: {
        using from = color_convert_from<cc::rgb>;
        switch (dest_cc) {
        case cc::rgb: {
            using conv = color_convert_to<cc::rgb,from,4>;
            return transform_quads(move(src), dest_cs, conv{dest_cs, src_cs});
        }
        case cc::gray: {
            using conv = color_convert_to<cc::gray,from,4>;
            return transform_quads(move(src), dest_cs, conv{dest_cs, src_cs});
        }
        case cc::yuv_jpeg: {
            using conv = color_convert_to<cc::yuv_jpeg,from,4>;
            return transform_quads(move(src), dest_cs, conv{dest_cs, src_cs});
        }
        case cc::yuv_nv21: {
            using conv = color_convert_to<cc::yuv_nv21,from,4>;
            return transform_quads(move(src), dest_cs, conv{dest_cs, src_cs});
        }
        case cc::r85g10b05: {
            using conv = color_convert_to<cc::r85g10b05,from,4>;
            return transform_quads(move(src), dest_cs, conv{dest_cs, src_cs});
        }
        case cc::alpha: {
            using conv = color_convert_to<cc::alpha,from,4>;
            return transform_quads(move(src), dest_cs, conv{dest_cs, src_cs});
        }
        default: break;
        }
        break;
    }

    default: break;
    }

    FILE_LOG(logERROR) << "conversion from " << src_cs << " to " << dest_cs
                       << " not implemented";
    return {};
}

std::array<uint8_t,4>
raw_image::to_layout(pixel_layout dest_cs, pixel_color c) {
    static constexpr auto src_cs = pixel::none; // constant pixel color
    using from = color_convert_from<color_class::rgb>;
    switch (to_color_class(dest_cs)) {
    case color_class::rgb: {
        using conv = color_convert_to<color_class::rgb,from>;
        return conv{dest_cs,src_cs,c}(nullptr);
    }
    case color_class::gray: {
        using conv = color_convert_to<color_class::gray,from>;
        return conv{dest_cs,src_cs,c}(nullptr);
    }
    case color_class::yuv_jpeg: {
        using conv = color_convert_to<color_class::yuv_jpeg,from>;
        return conv{dest_cs,src_cs,c}(nullptr);
    }
    case color_class::yuv_nv21: {
        using conv = color_convert_to<color_class::yuv_nv21,from>;
        return conv{dest_cs,src_cs,c}(nullptr);
    }
    case color_class::r85g10b05: {
        using conv = color_convert_to<color_class::r85g10b05,from>;
        return conv{dest_cs,src_cs,c}(nullptr);
    }
    default:
        if (c == color_black)
            return {0,0,0,0};
        throw std::invalid_argument("unknown pixel layout");
    }
}
