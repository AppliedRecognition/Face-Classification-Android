
#include "png.hpp"

#include <png.h>

#include <applog/core.hpp>
#include <stdext/bswap.hpp>

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <stdexcept>

using namespace raw_image;

namespace {
    struct png_decompress {
        png_structp png;
        png_infop info;
        image_size size;
        pixel_layout layout;
        std::vector<uint16_t> swab16;

        void read_header(pixel_layout desired) {
            png_read_info(png, info);

            png_uint_32 width;
            png_uint_32 height;
            int bit_depth;
            int color_type;
            int interlace_method;
            if (png_get_IHDR(png, info,
                             &width, &height,
                             &bit_depth, &color_type,
                             &interlace_method, nullptr, nullptr) != 1) {
                FILE_LOG(logERROR) << "png: failed to decode header";
                throw std::runtime_error("failed to decode png header");
            }
            if (interlace_method != PNG_INTERLACE_NONE) {
                FILE_LOG(logERROR) << "png: interlacing not supported ("
                                   << interlace_method << ")";
                throw std::runtime_error("interlaced png not supported");
            }
            size.width = width;
            size.height = height;

            // input transformations
            png_set_expand(png);   // 1,2,4 bit -> 8 bit, palette-> rgb

            if (desired != pixel::a16_le)
                png_set_strip_16(png); // 16 -> 8 bit

            if (desired == pixel::gray8 &&
                (color_type & PNG_COLOR_MASK_COLOR)) {
                png_set_rgb_to_gray(png,PNG_ERROR_ACTION_NONE,0.30,0.59);
                png_set_strip_alpha(png);
            }

            if ((color_type & PNG_COLOR_MASK_ALPHA) &&
                !(color_type & PNG_COLOR_MASK_COLOR)) {
                png_set_strip_alpha(png);
            }

            png_read_update_info(png, info);
            color_type = png_get_color_type(png,info);
            bit_depth = png_get_bit_depth(png,info);

            if (color_type == PNG_COLOR_TYPE_GRAY && bit_depth == 16 &&
                desired == pixel::a16_le) {
                layout = pixel::a16_le;
                if (png_get_channels(png,info) != 1) {
                    FILE_LOG(logERROR) << "png: numer of channels mismatch ("
                                       << png_get_channels(png,info)
                                       << " != 1)";
                    throw std::runtime_error("png numer of channels mismatch");
                }
                swab16.resize(width);
                return;
            }

            switch (color_type) {
            case PNG_COLOR_TYPE_GRAY: layout = pixel::gray8; break;
            case PNG_COLOR_TYPE_RGB:  layout = pixel::rgb24; break;
            case PNG_COLOR_TYPE_RGBA: layout = pixel::rgba32; break;
            default:
                FILE_LOG(logERROR) << "png: color type not supported ("
                                   << color_type << ")";
                throw std::runtime_error("png color type not supported");
            }
            if (bit_depth != 8) {
                FILE_LOG(logERROR) << "png: bit depth not supported ("
                                   << bit_depth << ")";
                throw std::runtime_error("png bit depth not supported");
            }
            if (png_get_channels(png,info) != bytes_per_pixel(layout)) {
                FILE_LOG(logERROR) << "png: numer of channels mismatch ("
                                   << png_get_channels(png,info) << " != "
                                   << bytes_per_pixel(layout) << ')';
                throw std::runtime_error("png numer of channels mismatch");
            }
        }

        static void PNGAPI error_fn(png_structp, png_const_charp msg) {
            FILE_LOG(logERROR) << "png: " << msg;
            throw std::runtime_error("error reading png");
        }
        static void PNGAPI warning_fn(png_structp, png_const_charp msg) {
            FILE_LOG(logWARNING) << "png: " << msg;
        }

        png_decompress()
            : png(png_create_read_struct(
                      PNG_LIBPNG_VER_STRING,
                      nullptr,
                      &error_fn,
                      &warning_fn)),
              info(png_create_info_struct(png)) {
            if (!png || !info) {
                if (png) png_destroy_read_struct(&png, nullptr, nullptr);
                FILE_LOG(logERROR) << "png: failed to initialize library";
                throw std::runtime_error("failed to initialize png library");
            }
        }
        virtual ~png_decompress() {
            png_destroy_read_struct(&png, &info, nullptr);
        }
        png_decompress(png_decompress&&) = delete;
        png_decompress(const png_decompress&) = delete;
        png_decompress& operator=(png_decompress&&) = delete;
        png_decompress& operator=(const png_decompress&) = delete;
    };

    struct png_file : png_decompress {
        stdx::file_ptr file;
        png_file(stdx::file_ptr file, pixel_layout desired)
            : file(move(file)) {
            png_init_io(png, this->file.get());
            read_header(desired);
        }
    };

    struct png_data : png_decompress {
        const unsigned char* data;
        std::size_t size;

        static void
        data_read_fn(png_structp png, png_bytep dest, png_size_t len) {
            auto& obj = *static_cast<png_data*>(png_get_io_ptr(png));
            if (len <= obj.size) {
                memcpy(dest, obj.data, len);
                obj.data += len;
                obj.size -= len;
            }
            else {
                memset(dest, 0, len);
                memcpy(dest, obj.data, obj.size);
                obj.size = 0;
            }
        }

        png_data(const void* data, std::size_t size, pixel_layout desired)
            : data(static_cast<const unsigned char*>(data)),
              size(size) {
            png_set_read_fn(png, this, &data_read_fn);
            read_header(desired);
        }
    };

    struct png_binary : png_data {
        stdx::binary data;
        png_binary(stdx::binary data, pixel_layout desired)
            : png_data(data.data(), data.size(), desired),
              data(move(data)) {
        }
    };

    struct png_reader final : reader {
        std::unique_ptr<png_decompress> png;

        png_reader(std::unique_ptr<png_decompress> _png)
            : reader(_png->size.width, _png->size.height, _png->layout),
              png(move(_png)) {
        }

        void line_next() override {
        }

        void line_copy(void* dest) override {
            if (png->swab16.empty())
                png_read_row(png->png, static_cast<png_bytep>(dest), nullptr);
            else {
                png_read_row(
                    png->png, reinterpret_cast<png_bytep>(png->swab16.data()),
                    nullptr);
                transform(png->swab16.begin(), png->swab16.end(),
                          static_cast<uint16_t*>(dest),
                          [](auto x) { return bswap_16(x); } );
            }
        }
    };
}

std::unique_ptr<reader>
raw_image::png_load(stdx::file_ptr file, pixel_layout desired) {
    auto png = std::make_unique<png_file>(move(file),desired);
    return std::make_unique<png_reader>(move(png));
}

std::unique_ptr<reader>
raw_image::png_load(const void* data, std::size_t size, pixel_layout desired) {
    auto png = std::make_unique<png_data>(data,size,desired);
    return std::make_unique<png_reader>(move(png));
}

std::unique_ptr<reader>
raw_image::png_load(stdx::binary data, pixel_layout desired) {
    auto png = std::make_unique<png_binary>(move(data),desired);
    return std::make_unique<png_reader>(move(png));
}


/****************  png encode  ****************/

namespace {
    struct png_encode {
        png_structp png;
        png_infop info;
        std::vector<png_byte> buf;

        static void PNGAPI error_fn(png_structp, png_const_charp msg) {
            FILE_LOG(logERROR) << "png: " << msg;
            throw std::runtime_error("error writing png");
        }
        static void PNGAPI warning_fn(png_structp, png_const_charp msg) {
            FILE_LOG(logWARNING) << "png: " << msg;
        }

        void write(png_bytep data, png_size_t size) {
            FILE_LOG(logTRACE) << "png_encode: " << size << " bytes";
            if (data && size > 0)
                buf.insert(buf.end(), data, data + size);
        }
        void flush() {
            FILE_LOG(logTRACE) << "png_encode: flush";
        }

        static void PNGAPI write_fn(
            png_structp png, png_bytep data, png_size_t size) {
            static_cast<png_encode*>(png_get_io_ptr(png))->write(data, size);
        }
        static void PNGAPI flush_fn(png_structp png) {
            static_cast<png_encode*>(png_get_io_ptr(png))->flush();
        }

        png_encode()
            : png(png_create_write_struct(PNG_LIBPNG_VER_STRING, nullptr,
                                          &error_fn, &warning_fn)) {
            if (!png)
                throw std::runtime_error("failed to create libpng write_struct");
            info = png_create_info_struct(png);
            if (!info) {
                png_destroy_write_struct(&png, nullptr); // is this kosher?
                throw std::runtime_error("failed to initialize libpng write_struct");
            }
            png_set_write_fn(png, this, &write_fn, &flush_fn);
        }

        ~png_encode() {
            png_destroy_write_struct(&png, &info);
        }
    };
}

stdx::binary internal::png_binary(plane const* image) {
    throw_if_invalid_or_empty(image);

    std::vector<uint16_t> swab16;

    int bit_depth = 8;
    int color_type;
    if (bytes_per_pixel(*image) == 1)
        color_type = PNG_COLOR_TYPE_GRAY;
    else if (image->layout == pixel::rgb24)
        color_type = PNG_COLOR_TYPE_RGB;
    else if (image->layout == pixel::rgba32)
        color_type = PNG_COLOR_TYPE_RGBA;
    else if (image->layout == pixel::a16_le) {
        bit_depth = 16;
        color_type = PNG_COLOR_TYPE_GRAY;
        swab16.resize(image->width);
    }
    else {
        FILE_LOG(logERROR) << "raw_image: image type "
                           << to_string(image->layout)
                           << " not support for png encode";
        throw std::runtime_error("unsupported image type for png encode");
    }

    png_encode enc{};
    png_set_IHDR(
        enc.png, enc.info,
        image->width, image->height, bit_depth, color_type,
        PNG_INTERLACE_NONE,
        PNG_COMPRESSION_TYPE_DEFAULT,
        PNG_FILTER_TYPE_DEFAULT);
    png_write_info(enc.png, enc.info);

    png_bytep src = image->data;
    if (swab16.empty())
        for (auto i = image->height; i > 0; --i, src += image->bytes_per_line)
            png_write_row(enc.png, src);
    else {
        // have to byte swap 16-bit samples
        for (auto i = image->height; i > 0; --i, src += image->bytes_per_line) {
            auto* s16 = reinterpret_cast<const uint16_t*>(src);
            transform(s16, s16 + image->width, swab16.begin(),
                      [](auto x) { return bswap_16(x); } );
            png_write_row(
                enc.png, reinterpret_cast<png_const_bytep>(swab16.data()));
        }
    }

    png_write_end(enc.png, nullptr);
    return stdx::binary(move(enc.buf));
}
