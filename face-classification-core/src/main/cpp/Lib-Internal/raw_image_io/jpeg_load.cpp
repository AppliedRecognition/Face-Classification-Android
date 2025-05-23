
#include "jpeg.hpp"

extern "C" {
#include <jpeglib.h>
}

#include <cstdio>
#include <stdexcept>

#include <applog/core.hpp>


using namespace raw_image;


static void mem_init_source(j_decompress_ptr) {
    // no work necessary here
}

static boolean mem_fill_input_buffer(j_decompress_ptr cinfo) {
    static JOCTET EOI[2];
    EOI[0] = JOCTET(0xFF);
    EOI[1] = JOCTET(JPEG_EOI);
    cinfo->src->next_input_byte = EOI;
    cinfo->src->bytes_in_buffer = 2;
    return TRUE;
}

static void mem_skip_input_data(j_decompress_ptr cinfo, long num_bytes) {
    if (num_bytes > 0) {
        auto n = std::size_t(num_bytes);
        if (n > cinfo->src->bytes_in_buffer)
            n = cinfo->src->bytes_in_buffer;
        cinfo->src->next_input_byte += n;
        cinfo->src->bytes_in_buffer -= n;
    }
}

static void mem_term_source(j_decompress_ptr) {
    // no work necessary here
}

static void jpeg_mem_src(j_decompress_ptr cinfo, const void* buf, size_t len) {
    if (!cinfo->src)    /* first time for this JPEG object? */
        cinfo->src = static_cast<jpeg_source_mgr*>(
            (*cinfo->mem->alloc_small)(reinterpret_cast<j_common_ptr>(cinfo),
                                       JPOOL_PERMANENT,
                                       sizeof(jpeg_source_mgr)));
    
    cinfo->src->bytes_in_buffer = len;
    cinfo->src->next_input_byte = static_cast<const JOCTET*>(buf);

    cinfo->src->init_source = mem_init_source;
    cinfo->src->fill_input_buffer = mem_fill_input_buffer;
    cinfo->src->skip_input_data = mem_skip_input_data;
    cinfo->src->resync_to_restart = jpeg_resync_to_restart;
    cinfo->src->term_source = mem_term_source;
}

namespace {
    struct jpeg_decompress {
        jpeg_decompress_struct dinfo;
        jpeg_error_mgr jerr;
        pixel_layout layout;
        image_size size, original_size;
        int scale = 0;

        void read_header(pixel_layout desired, int minpx) {
            jpeg_read_header(&dinfo, TRUE);
            original_size.width  = dinfo.output_width  = dinfo.image_width;
            original_size.height = dinfo.output_height = dinfo.image_height;

            if (dinfo.jpeg_color_space == JCS_CMYK ||
                dinfo.jpeg_color_space == JCS_YCCK)
                FILE_LOG(logWARNING) << "attempting to read CMYK or YCCK jpeg";

            if (dinfo.jpeg_color_space == JCS_GRAYSCALE ||
                desired == pixel::gray8) {
                dinfo.out_color_space = JCS_GRAYSCALE;
                layout = pixel::gray8;
            }
            else if (dinfo.jpeg_color_space == JCS_YCbCr &&
                     (desired == pixel::none ||
                      to_color_class(desired) == cc::yuv_jpeg)) {
                dinfo.out_color_space = JCS_YCbCr;
                layout = pixel::yuv24_jpeg;
            }
            else { // convert to rgb
                dinfo.out_color_space = JCS_RGB;
                layout = pixel::rgb24;
            }

            if (minpx >= 0) {
                auto px = std::size_t(original_size.width/2)
                    * std::size_t(original_size.height/2);
                while (unsigned(minpx) <= px) {
                    if (++scale >= 3) break;
                    px /= 4;
                }
            }
            dinfo.scale_num = 1;
            dinfo.scale_denom = dinfo.scale_num << scale;

            jpeg_start_decompress(&dinfo);
            size.width = dinfo.output_width;
            size.height = dinfo.output_height;
        }

        static void error_exit(j_common_ptr dinfo) {
            char buf[JMSG_LENGTH_MAX];
            (*dinfo->err->format_message)(dinfo,buf);
            FILE_LOG(logERROR) << "jpeg: " << buf;
            throw std::runtime_error("error decoding jpeg data");
        }

        jpeg_decompress() {
            dinfo.err = jpeg_std_error(&jerr);
            jerr.error_exit = &error_exit;
            jpeg_create_decompress(&dinfo);
        }
        virtual ~jpeg_decompress() {
            jpeg_destroy_decompress(&dinfo);
        }
        jpeg_decompress(jpeg_decompress&&) = delete;
        jpeg_decompress(const jpeg_decompress&) = delete;
        jpeg_decompress& operator=(jpeg_decompress&&) = delete;
        jpeg_decompress& operator=(const jpeg_decompress&) = delete;
    };

    struct jpeg_file : jpeg_decompress {
        stdx::file_ptr file;
        jpeg_file(stdx::file_ptr file, pixel_layout desired, int minpx)
            : file(move(file)) {
            jpeg_stdio_src(&dinfo, this->file.get());
            read_header(desired, minpx);
        }
    };

    struct jpeg_data : jpeg_decompress {
        jpeg_data(const void* data, std::size_t size,
                  pixel_layout desired, int minpx) {
            jpeg_mem_src(&dinfo, data, size);
            read_header(desired, minpx);
        }
    };

    struct jpeg_binary : jpeg_data {
        stdx::binary data;
        jpeg_binary(stdx::binary data, pixel_layout desired, int minpx)
            : jpeg_data(data.data(), data.size(), desired, minpx),
              data(move(data)) {
        }
    };

    struct jpeg_reader final : reader_ex {
        std::unique_ptr<jpeg_decompress> jpeg;

        jpeg_reader(std::unique_ptr<jpeg_decompress> _jpeg)
            : reader_ex(_jpeg->size.width, _jpeg->size.height, _jpeg->layout),
              jpeg(move(_jpeg)) {
            original_size = jpeg->original_size;
            scale = jpeg->scale;
        }

        void line_next() override {
        }

        void line_copy(void* dest) override {
            auto p = static_cast<JSAMPLE*>(dest);
            if (jpeg_read_scanlines(&jpeg->dinfo, &p, 1) != 1)
                throw std::runtime_error("failed to read scanline");
            if (jpeg->dinfo.output_scanline == jpeg->dinfo.output_height)
                jpeg_finish_decompress(&jpeg->dinfo);
        }
    };
}

std::unique_ptr<reader_ex>
raw_image::jpeg_load(
    stdx::file_ptr file, pixel_layout desired, int min_pixels) {
    auto jpeg = std::make_unique<jpeg_file>(move(file),desired,min_pixels);
    return std::make_unique<jpeg_reader>(move(jpeg));
}

std::unique_ptr<reader_ex>
raw_image::jpeg_load(
    const void* data, std::size_t size, pixel_layout desired, int min_pixels) {
    auto jpeg = std::make_unique<jpeg_data>(data,size,desired,min_pixels);
    return std::make_unique<jpeg_reader>(move(jpeg));
}

std::unique_ptr<reader_ex>
raw_image::jpeg_load(
    stdx::binary data, pixel_layout desired, int min_pixels) {
    auto jpeg = std::make_unique<jpeg_binary>(move(data),desired,min_pixels);
    return std::make_unique<jpeg_reader>(move(jpeg));
}
