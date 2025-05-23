
#include "io.hpp"
#include "jpeg.hpp"
#include "png.hpp"
#include "tiff.hpp"

#include <raw_image/transform.hpp>

#include <applog/core.hpp>

#include <algorithm>
#include <cctype>
#include <string_view>


using namespace raw_image;


template <typename... Args>
static auto tiff_load(pixel_layout layout, unsigned rot, Args&&... args) {
    auto img = tiff_load(std::forward<Args>(args)...);
    if (rot&1)
        img = copy(img, layout, rotate(rot));
    else {
        if (layout != pixel::none && layout != img->layout)
            if (auto p = convert(*img, layout))
                img = move(p);
        if (rot&7)
            in_place_rotate(*img, rot);
    }
    return img;
}

std::unique_ptr<plane>
internal::from_binary(
    const void* data, std::size_t size,
    const stdx::options_tuple<rotate,pixel_layout>& opts) {

    if (!data || size <= 0)
        throw std::invalid_argument("raw_image::from_binary() insufficient image data");
    FILE_LOG(logDETAIL) << "raw_image::from_binary() " << size << " bytes";

    const auto layout = std::get<pixel_layout>(opts);
    const auto rot = std::get<rotate>(opts);

    switch (*static_cast<const unsigned char*>(data)) {

    case 'I':   // tiff
    case 'M':
        return ::tiff_load(layout, unsigned(rot), data, size);

    case 0x89:  // png
        return copy(convert(png_load(data, size, layout), layout), rot);

    default:   // jpeg
        return copy(convert(jpeg_load(data, size, layout), layout), rot);
    }
}

std::unique_ptr<plane>
internal::load(
    stdx::file_ptr infile, const std::string& path,
    const stdx::options_tuple<rotate,pixel_layout>& opts) {

    if (!infile) {
        FILE_LOG(logERROR) << "failed to open: " << path;
        throw std::runtime_error("failed to open file");
    }
    const auto header = fgetc(infile.get());
    if (header == EOF) {
        FILE_LOG(logWARNING) << "empty file: " << path;
        throw std::runtime_error("failed to read file");
    }
    if (ungetc(header,infile.get()) != header) {
        FILE_LOG(logERROR) << "ungetc failed: " << path;
        throw std::runtime_error("unknown file error");
    }

    FILE_LOG(logDETAIL) << "raw_image::load: " << path;

    const auto layout = std::get<pixel_layout>(opts);
    const auto rot = std::get<rotate>(opts);

    switch (header) {

    case 'I':  // tiff
    case 'M':
        return ::tiff_load(layout, unsigned(rot), infile.get());

    case 0x89: // png
        return copy(convert(png_load(move(infile), layout), layout), rot);

    default:   // jpeg
        return copy(convert(jpeg_load(move(infile), layout), layout), rot);
    }
}

static bool iequals(std::string_view a, std::string_view b) {
    return a.size() == b.size() &&
        std::equal(a.begin(), a.end(), b.begin(),
                   [](char c1, char c2) {
                       return c1 == c2 ||
                           std::toupper(c1) == std::toupper(c2);
                   });
}

stdx::binary internal::to_binary(
    plane const* image,
    const stdx::options_tuple<jpeg_quality,png_option>& opts) {

    const auto& opt_png = std::get<png_option>(opts);
    const auto& opt_jpeg = std::get<jpeg_quality>(opts);

    if (opt_png && 0 < opt_jpeg.quality) {
        FILE_LOG(logERROR) << "raw_image::to_binary() called with both jpeg and png options";
        throw std::logic_error("raw_image::to_binary() called with both jpeg and png options");
    }

    return opt_png ? png_binary(image) : jpeg_binary(image, opt_jpeg.quality);
}

void internal::save(
    plane const* image, FILE* outfile, const std::string& filename,
    const stdx::options_tuple<jpeg_quality,png_option>& opts) {

    if (!outfile) {
        FILE_LOG(logERROR) << "failed to open output file for image \""
                           << filename << '"';
        throw std::runtime_error("failed to open output file for image");
    }

    const auto sv = std::string_view(filename);
    const auto dot_png =
        4 <= sv.size() && iequals(sv.substr(sv.size()-4,4),".png");
    const auto dot_jpeg =
        (4 <= sv.size() && iequals(sv.substr(sv.size()-4,4),".jpg")) ||
        (5 <= sv.size() && iequals(sv.substr(sv.size()-4,4),".jpeg"));
    const auto& opt_png = std::get<png_option>(opts);
    auto opt_jpeg = std::get<jpeg_quality>(opts);

    stdx::binary buf;
    if (opt_png && opt_jpeg.quality <= 0) {
        if (dot_jpeg)
            FILE_LOG(logWARNING) << "writing png image to jpeg file: \""
                                 << filename << '"';
        buf = internal::to_binary(image,opt_png);
    }
    else if (0 < opt_jpeg.quality && !opt_png) {
        if (dot_png)
            FILE_LOG(logWARNING) << "writing jpeg image to png file: \""
                                 << filename << '"';
        buf = internal::to_binary(image,opt_jpeg);
    }
    else if (dot_png)
        buf = internal::to_binary(image,png);
    else if (dot_jpeg) {
        if (opt_jpeg.quality <= 0)
            opt_jpeg.quality = 90;
        buf = internal::to_binary(image,opt_jpeg);
    }
    else {
        FILE_LOG(logWARNING) << "assuming jpeg for write to file: \""
                             << filename << "'";
        buf = internal::to_binary(image,jpeg);
    }

    if (fwrite(buf.data(), buf.size(), 1, outfile) != 1) {
        FILE_LOG(logINFO) << "error writing image to file \""
                          << filename << '"';
        throw std::runtime_error("error while writing image to file");
    }
}
