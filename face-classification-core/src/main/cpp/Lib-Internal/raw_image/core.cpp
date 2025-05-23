
#include "core.hpp"
#include "reader.hpp"

#include <applog/core.hpp>

#include <algorithm>
#include <cassert>
#include <iomanip>
#include <stdexcept>
#include <sstream>


using namespace raw_image;


image_size raw_image::dimensions(const multi_plane_arg& mp) {
    if (mp.empty()) return {0,0};
    auto& image = mp.front();
    image_size r {
        (image.rotate&1) ? image.height : image.width,
        (image.rotate&1) ? image.width : image.height
    };
    if (image.scale > 0) {
        r.width  <<= image.scale;
        r.height <<= image.scale;
    }
    else if (image.scale < 0) {
        // note: rounded down -- is that right?
        r.width  >>= -image.scale;
        r.height >>= -image.scale;
    }
    return r;
}

std::string raw_image::to_string(pixel_layout cs) {
    switch (cs) {
    case pixel::r85g10b05: return "R85G10B05";

    case pixel::y8_jpeg: return "Y8_JPEG";
    case pixel::u8_jpeg: return "U8_JPEG";
    case pixel::v8_jpeg: return "V8_JPEG";
    case pixel::uv16_jpeg: return "UV16_JPEG";
    case pixel::vu16_jpeg: return "VU16_JPEG";
    case pixel::yuv24_jpeg: return "YUV24_JPEG";

    case pixel::y8_nv21: return "Y8_NV21";
    case pixel::u8_nv21: return "U8_NV21";
    case pixel::v8_nv21: return "V8_NV21";
    case pixel::uv16_nv21: return "UV16_NV21";
    case pixel::vu16_nv21: return "VU16_NV21";
    case pixel::yuv24_nv21: return "YUV24_NV21";

    case pixel::r8: return "R8";
    case pixel::g8: return "G8";
    case pixel::b8: return "B8";
    case pixel::a8: return "A8";
    case pixel::a16_le: return "A16_LE";

    case pixel::rgb24: return "RGB24";
    case pixel::bgr24: return "BGR24";

    case pixel::argb32: return "ARGB32";
    case pixel::abgr32: return "ABGR32";
    case pixel::rgba32: return "RGBA32";
    case pixel::bgra32: return "BGRA32";

    case pixel::f32: return "F32";

    case pixel::none: break; // LAYOUT(0)
    }
    return "LAYOUT(" + std::to_string(unsigned(cs)) + ')';
}

std::string raw_image::to_string(color_class _cc) {
    switch (_cc) {
    case cc::unknown:   return "unknown";
    case cc::alpha:     return "alpha";
    case cc::gray:      return "gray";
    case cc::yuv_jpeg:  return "yuv_jpeg";
    case cc::yuv_nv21:  return "yuv_nv21";
    case cc::rgb:       return "rgb";
    case cc::r85g10b05: return "r85g10b05";
    }
    return "color(" + std::to_string(unsigned(_cc)) + ')';
}

std::string_view raw_image::describe_error(const multi_plane_arg& mp) {
    if (mp.empty() || (mp.front().width <= 0 && mp.front().height <= 0))
        return {};  // empty image is ok

    for (auto& image : mp) {
        if (!image.data)
            return "image pixels is null pointer";

        const auto bpp = bytes_per_pixel(image);
        if (bpp <= 0 || 4 < bpp)
            return "image pixel layout is invalid (bytes per pixel)";

        if (image.width >= 1024*1024*1024)
            return "image width is too large";

        const auto bpl = bpp * image.width;
        if (image.bytes_per_line < bpl)
            return "image bytes_per_line insufficient for width";

        if (bpp == 4 &&
            ((image.bytes_per_line&3) != 0 ||
             (reinterpret_cast<std::size_t>(image.data)&3) != 0))
            return "image pixels are not aligned for 32-bits per pixel";
    }
    return {};
}

void raw_image::throw_if_invalid(
    const multi_plane_arg& image, std::string_view method) {
    const auto e = describe_error(image);
    if (!e.empty()) {
        std::string s;
        if (!method.empty()) {
            s += method;
            s += ": ";
        }
        s += e;
        FILE_LOG(logERROR) << s;
        throw std::invalid_argument(s);
    }
}

void raw_image::throw_if_invalid_or_empty(
    const multi_plane_arg& image, std::string_view method) {
    if (empty(image)) {
        std::string s;
        if (!method.empty()) {
            s += method;
            s += ": ";
        }
        s += "image is empty";
        throw std::invalid_argument(s);
    }
    throw_if_invalid(image,method);
}

std::string raw_image::diag(single_plane_arg image) {

    if (!image)
        return "0x0 (nullptr)";
    if (image->width <= 0 || image->height <= 0)
        return "0x0 (empty)";

    std::stringstream ss;
    ss << image->width << 'x' << image->height;

    const auto bpp = bytes_per_pixel(image);
    ss << 'x' << bpp << ' ' << to_string(image->layout);

    ss << " bpl=" << image->bytes_per_line;
    const auto bpl = bpp * image->width;
    if (image->bytes_per_line < bpl)
        ss << '<' << bpl << '!';
    else if (image->bytes_per_line > bpl)
        ss << '+';

    if (image->rotate&7)
        ss << " rotate=" << (image->rotate&7);
    if (image->scale)
        ss << " scale=" << (image->scale);

    if (image->data == nullptr) {
        ss << " data=nullptr";
        return ss.str();
    }

    const auto internal = reinterpret_cast<const unsigned char*>(image.get())
        + plane_struct_padded_size;
    ss << (image->data == internal ? " data=internal" : " data=external");

    if (bpp == 4 &&
        ((image->bytes_per_line&3) != 0 ||
         (reinterpret_cast<std::size_t>(image->data)&3u) != 0))
        ss << " BAD_ALIGNMENT";

    if (image->bytes_per_line >= bpl && 0 < bpp && bpp <= 4) {
        ss << std::setfill('0') << std::hex;
        ss << " [first=" << std::setw(2) << int(*image->data);
        auto last = image->data
            + (image->height-1)*image->bytes_per_line
            + image->width*bpp - 1;
        ss << " last=" << std::setw(2) << int(*last) << " byte]";
    }

    return ss.str();
}

plane_ptr
raw_image::create(unsigned width, unsigned height, pixel_layout layout) {
    // allocate lines with enough space for a multiple of 4 pixels
    const auto bytes_per_pixel = raw_image::bytes_per_pixel(layout);
    const auto bytes_per_line = bytes_per_pixel * ((width+3)&~3u);

    static constexpr auto ofs = plane_struct_padded_size;
    const auto nbytes = height * std::size_t(bytes_per_line) + ofs;
    const auto buf = operator new(nbytes);
    std::fill_n(static_cast<unsigned char*>(buf), ofs, 0);

    plane_ptr image(static_cast<plane*>(buf));
    image->data = static_cast<unsigned char*>(buf) + ofs;
    image->width = width;
    image->height = height;
    image->bytes_per_line = bytes_per_line;
    image->layout = layout;
    return image;
}

plane raw_image::crop(
    single_plane_arg image, unsigned x, unsigned y, unsigned w, unsigned h) {

    throw_if_invalid(image);
    if (x > image->width || y > image->height ||
        w > image->width || h > image->height ||
        x+w > image->width || y+h > image->height) {
        FILE_LOG(logERROR) << "invalid crop: "
                           << image->width << 'x' << image->height << " -> "
                           << w << 'x' << h << '+' << x << '+' << y;
        throw std::invalid_argument("attempt to crop beyond image border");
    }
    auto r = *image;
    r.width = w;
    r.height = h;
    r.data += x * bytes_per_pixel(image);
    r.data += y * std::size_t(image->bytes_per_line);
    return r;
}

static auto after_transpose(unsigned rot) {
    rot ^= 5 | ((rot<<1)^(rot>>1));
    return rot & 7;
}
static auto after_rotate(unsigned before, unsigned rot) {
    if (rot&1) {
        rot = after_transpose(rot);
        before = after_transpose(before);
    }
    return before ^ rot;
}

plane_ptr raw_image::copy_with_opts(
    const multi_plane_arg& src,
    const stdx::options_tuple<rotate,pixel_layout>& opts) {

    if (src.empty())
        throw std::invalid_argument("image has no planes");
    for (auto& x : src)
        throw_if_invalid(x);

    const auto r = reader::construct_with_opts(src, opts);
    if (!r) {
        auto cs = std::get<pixel_layout>(opts);
        if (cs == pixel::none) cs = src.front().layout;
        FILE_LOG(logERROR) << "layout conversion not implemented:"
                           << std::endl << "\tto:\t"
                           << to_string(cs)
                           << std::endl << "\tfrom:\t"
                           << diag(src.front());
        throw std::runtime_error("layout conversion not implemented");
    }
    auto dest = create(r->width(), r->height(), r->layout());
    dest->rotate = after_rotate(
        src.front().rotate, unsigned(std::get<rotate>(opts)) & 7);
    dest->scale = src.front().scale;
    if (std::max(src.front().width, src.front().height) == 2*std::max(dest->width, dest->height))
        ++dest->scale;  // extracting uv plane at half resolution
    r->copy_to(*dest, dest->bytes_per_line);
    return dest;
}

plane_ptr raw_image::copy(stdx::arg<reader> src, rotate rot) {
    if (!src) throw std::invalid_argument("reader is nullptr");
    plane_ptr dest;
    if (unsigned(rot)&1)
        dest = create(src->lines_remaining(), src->width(), src->layout());
    else
        dest = create(src->width(), src->lines_remaining(), src->layout());
    src->rotate_to(*dest, unsigned(rot));
    return dest;
}

void raw_image::copy_pixels(
    const multi_plane_arg& src, single_plane_arg dest, unsigned rot) {
    for (auto& img : src)
        throw_if_invalid(img);
    const auto r = reader::construct(src, rotate(rot), dest->layout);
    if (!r) {
        if (src.empty())
            throw std::invalid_argument("image has no planes");
        FILE_LOG(logERROR) << "layout conversion not implemented:"
                           << std::endl << "\tto:\t"
                           << diag(dest)
                           << std::endl << "\tfrom:\t"
                           << diag(src.front());
        throw std::runtime_error("layout conversion not implemented");
    }
    throw_if_invalid(dest);
    if (r->width() != dest->width || r->height() != dest->height)
        throw std::invalid_argument("attempt to copy images with different dimensions");
    r->copy_to(*dest);
}

plane_ptr raw_image::convert(plane& image, pixel_layout new_layout) {

    if (image.layout == new_layout)
        return {};  // nothing to do

    const auto old_bpp = bytes_per_pixel(image.layout);
    const auto new_bpp = bytes_per_pixel(new_layout);
    if (new_bpp <= 1 && old_bpp <= 1) {
        // note: converting from one single channel to another is shallow
        if (new_bpp == 1 && old_bpp == 1)
            image.layout = new_layout;
        return {};
    }

    // if the image was created by create(), then
    // we can use all of bytes_per_line if necessary
    const auto full_bpl = manages_pixel_buffer(single_plane_arg(image));

    if (old_bpp < new_bpp &&
        (image.bytes_per_line < image.width * new_bpp || !full_bpl))
        return copy(image, new_layout);  // copy to new image
        
    // convert in place
    throw_if_invalid(image);
    const auto r = reader::construct(image, new_layout);
    if (!r) {
        FILE_LOG(logERROR) << "layout conversion not implemented:"
                           << std::endl << "\tto:\t"
                           << to_string(new_layout)
                           << std::endl << "\tfrom:\t"
                           << diag(image);
        throw std::runtime_error("layout conversion not implemented");
    }

    const auto avail = full_bpl ? image.bytes_per_line : image.width * old_bpp;
    r->force_buffer(avail);
    r->copy_to(image, avail);

    image.layout = new_layout;
    return {};
}


static auto number_from_name(channel ch, pixel_layout cs) {
    if (int(ch) >= 0)
        return unsigned(ch);
    switch (cs) {
    case pixel::y8_jpeg:
    case pixel::y8_nv21:
    case pixel::r85g10b05:
        if (ch == channel::y) return 0u;
        break;
    case pixel::u8_jpeg:
    case pixel::u8_nv21:
        if (ch == channel::u) return 0u;
        break;
    case pixel::v8_jpeg:
    case pixel::v8_nv21:
        if (ch == channel::v) return 0u;
        break;

    case pixel::r8:
        if (ch == channel::r) return 0u;
        break;
    case pixel::g8:
        if (ch == channel::g) return 0u;
        break;
    case pixel::b8:
        if (ch == channel::b) return 0u;
        break;

    case pixel::a8:
        if (ch == channel::alpha) return 0u;
        break;

    case pixel::uv16_jpeg:
    case pixel::uv16_nv21:
        switch (ch) {
        case channel::u: return 0u;
        case channel::v: return 1u;
        default: break;
        }
        break;
    case pixel::vu16_jpeg:
    case pixel::vu16_nv21:
        switch (ch) {
        case channel::u: return 1u;
        case channel::v: return 0u;
        default: break;
        }
        break;
    case pixel::yuv24_jpeg:
    case pixel::yuv24_nv21:
        switch (ch) {
        case channel::y: return 0u;
        case channel::u: return 1u;
        case channel::v: return 2u;
        default: break;
        }
        break;

    case pixel::rgb24:
        switch (ch) {
        case channel::r: return 0u;
        case channel::g: return 1u;
        case channel::b: return 2u;
        default: break;
        }
        break;
    case pixel::bgr24:
        switch (ch) {
        case channel::r: return 2u;
        case channel::g: return 1u;
        case channel::b: return 0u;
        default: break;
        }
        break;

    case pixel::argb32:
        switch (ch) {
        case channel::r: return 1u;
        case channel::g: return 2u;
        case channel::b: return 3u;
        case channel::alpha: return 0u;
        default: break;
        }
        break;
    case pixel::abgr32:
        switch (ch) {
        case channel::r: return 3u;
        case channel::g: return 2u;
        case channel::b: return 1u;
        case channel::alpha: return 0u;
        default: break;
        }
        break;
    case pixel::rgba32:
        switch (ch) {
        case channel::r: return 0u;
        case channel::g: return 1u;
        case channel::b: return 2u;
        case channel::alpha: return 3u;
        default: break;
        }
        break;
    case pixel::bgra32:
        switch (ch) {
        case channel::r: return 2u;
        case channel::g: return 1u;
        case channel::b: return 0u;
        case channel::alpha: return 3u;
        default: break;
        }
        break;
    case pixel::a16_le:
    case pixel::f32:
    case pixel::none: break;
    }
    throw std::invalid_argument("channel not present in image");
}

void raw_image::copy_channel(
    single_plane_arg src,  channel src_ch,
    single_plane_arg dest, channel dest_ch) {
    throw_if_invalid(src);
    throw_if_invalid(dest);
    if (src->width != dest->width || src->height != dest->height)
        throw std::invalid_argument("attempt to copy images with different dimensions");
    const auto src_n = number_from_name(src_ch, src->layout);
    const auto dest_n = number_from_name(dest_ch, dest->layout);
    const auto src_bpp = bytes_per_pixel(src->layout);
    const auto dest_bpp = bytes_per_pixel(dest->layout);
    if (src_n >= src_bpp)
        throw std::invalid_argument("source channel out of range");
    if (dest_n >= dest_bpp)
        throw std::invalid_argument("destination channel out of range");
    const uint8_t* src_line = src->data + src_n;
    uint8_t* dest_line = dest->data + dest_n;
    for (auto i = src->height; i > 0; --i,
             src_line += src->bytes_per_line,
             dest_line += dest->bytes_per_line) {
        auto sp = src_line;
        auto dp = dest_line;
        for (auto j = src->width; j > 0; --j, sp += src_bpp, dp += dest_bpp)
            *dp = *sp;
    }
}

static auto gray8_from_first_byte(const uint8_t* px) {
    return *px;
}
template <int rofs, int dir>
static auto gray8_from_rgb(const uint8_t* px) {
    const auto r = unsigned(px[rofs]);
    const auto g = unsigned(px[rofs+dir]);
    const auto b = unsigned(px[rofs+2*dir]);
    const auto y = (19595*r + 38470*g + 7471*b + 32768) >> 16;
    return uint8_t(y);
}

auto raw_image::gray8_from_pixel(pixel_layout layout)
    -> unsigned char(*)(const unsigned char*) {

    switch (layout) {
    case pixel::y8_jpeg:
    case pixel::y8_nv21:
    case pixel::r85g10b05:
    case pixel::yuv24_jpeg:
    case pixel::yuv24_nv21:
        return &gray8_from_first_byte;

    case pixel::rgb24:
    case pixel::rgba32:
        return &gray8_from_rgb<0,1>;

    case pixel::bgr24:
    case pixel::bgra32:
        return &gray8_from_rgb<2,-1>;

    case pixel::argb32:
        return &gray8_from_rgb<1,1>;
    case pixel::abgr32:
        return &gray8_from_rgb<3,-1>;
        
    default:
        FILE_LOG(logERROR) << "image color space: " << int(layout);
        throw std::invalid_argument("invalid image layout");
    }
}
