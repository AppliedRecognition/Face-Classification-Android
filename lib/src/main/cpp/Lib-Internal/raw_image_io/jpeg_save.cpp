
#include "io.hpp"
#include "jpeg.hpp"

#include <applog/core.hpp>

extern "C" {
#include <jpeglib.h>
}

#include <setjmp.h>

#include <cstdio>  // needed by jpeglib.h
#include <cstdlib>
#include <memory>
#include <stdexcept>


using namespace raw_image;


namespace {
    struct my_error_mgr : public jpeg_error_mgr {
        jmp_buf setjmp_buffer;   /* for return to caller */
    };
    using my_error_ptr = my_error_mgr*;
}

static void error_exit(j_common_ptr cinfo) {
    /* cinfo->err really points to a my_error_mgr struct, so coerce pointer */
    my_error_ptr myerr = reinterpret_cast<my_error_ptr>(cinfo->err);
    
    /* Always display the message. */
    /* We could postpone this until after returning, if we chose. */
    (*cinfo->err->output_message) (cinfo);
    
    /* Return control to the setjmp point */
    longjmp(myerr->setjmp_buffer, 1);
}

static void jpeg_save_image(jpeg_compress_struct& cinfo,
                            const plane& image, int quality) {

    /* Step 3: set parameters for compression */
    
    /* First we supply a description of the input image.
     * Four fields of the cinfo struct must be filled in:
     */
    cinfo.image_width = image.width;
    cinfo.image_height = image.height;

    if (bytes_per_pixel(image.layout) == 1) {
        cinfo.input_components = 1;
        cinfo.in_color_space = JCS_GRAYSCALE;
    }
    else switch (image.layout) {
        case pixel::rgb24:
        case pixel::bgr24:
            // distinguished below when loading line buffer
            cinfo.input_components = 3;
            cinfo.in_color_space = JCS_RGB;
            break;

        case pixel::yuv24_jpeg:
            cinfo.input_components = 3;
            cinfo.in_color_space = JCS_YCbCr;
            break;

        default:
            FILE_LOG(logERROR) << "jpeg_save: pixel layout '"
                               << image.layout << "' not supported";
            throw std::runtime_error("pixel layout not supported");
        }

    /* Now use the library's routine to set default compression parameters.
     * (You must set at least cinfo.in_color_space before calling this,
     * since the defaults depend on the source pixel layout.)
     */
    jpeg_set_defaults(&cinfo);
    /* Now you can set any non-default parameters you wish to.
     * Here we just illustrate the use of quality (quantization table) scaling:
     */
    jpeg_set_quality(&cinfo, quality, TRUE /* limit to baseline-JPEG values */);
    /* Step 4: Start compressor */
    
    /* TRUE ensures that we will write a complete interchange-JPEG file.
     * Pass TRUE unless you are very sure of what you're doing.
     */
    jpeg_start_compress(&cinfo, TRUE);

    // save markers -cws
    // jpeg_write_marker(...)

    /* Step 5: while (scan lines remain to be written) */
    /*           jpeg_write_scanlines(...); */
    
    /* Here we use the library's state variable cinfo.next_scanline as the
     * loop counter, so that we don't have to keep track ourselves.
     * To keep things simple, we pass one scanline per call; you can pass
     * more if you wish, though.
     */
    assert(cinfo.input_components > 0);
    const auto row_stride = image.width * unsigned(cinfo.input_components); /* JSAMPLEs per row in image_buffer */

    /* Make a one-row-high sample array that will go away when done with image */
    JSAMPARRAY buffer;            /* Output row buffer */
    buffer = (*cinfo.mem->alloc_sarray)(reinterpret_cast<j_common_ptr>(&cinfo), JPOOL_IMAGE, row_stride, 1);

    const unsigned char* src_line = image.data;
    while (cinfo.next_scanline < cinfo.image_height) {
        /* jpeg_write_scanlines expects an array of pointers to scanlines.
         * Here the array is only one element long, but you could pass
         * more than one scanline at a time if that's more convenient.
         */
        if (image.layout == pixel::bgr24) {
            unsigned char* out = buffer[0];
            const unsigned char* in = src_line;
            for (unsigned j = 0; j < cinfo.image_width; ++j, in += 3) {
                *out++ = in[2];
                *out++ = in[1];
                *out++ = in[0];
            }
        }
        else
            memcpy(buffer[0],src_line,cinfo.image_width*unsigned(cinfo.input_components));

        jpeg_write_scanlines(&cinfo, buffer, 1);
        src_line += image.bytes_per_line;
    }
    
    /* Step 6: Finish compression */
    jpeg_finish_compress(&cinfo);

    /* Step 7: release JPEG compression object */
    
    /* This is an important step since it will release a good deal of memory. */
    jpeg_destroy_compress(&cinfo);
}

typedef struct {
    jpeg_destination_mgr pub; /* public fields */
    JOCTET* buffer;     /* start of buffer */
    std::vector<JOCTET>* dest;
} mem_destination_mgr;

using mem_dest_ptr = mem_destination_mgr*;

static constexpr auto OUTPUT_BUF_SIZE = 4096;

static void mem_init_destination(j_compress_ptr cinfo) {
    mem_dest_ptr dest = reinterpret_cast<mem_dest_ptr>(cinfo->dest);
    
    /* Allocate output buffer --- it will be released when done with image */
    dest->buffer = static_cast<JOCTET*>(
        (*cinfo->mem->alloc_small)(reinterpret_cast<j_common_ptr>(cinfo), 
                                   JPOOL_IMAGE,
                                   OUTPUT_BUF_SIZE * sizeof(JOCTET)));
    
    dest->pub.next_output_byte = dest->buffer;
    dest->pub.free_in_buffer = OUTPUT_BUF_SIZE;
}

static boolean mem_empty_output_buffer(j_compress_ptr cinfo) {
    mem_dest_ptr dest = reinterpret_cast<mem_dest_ptr>(cinfo->dest);
    dest->dest->insert(dest->dest->end(),
                       dest->buffer, dest->buffer + OUTPUT_BUF_SIZE);
    dest->pub.next_output_byte = dest->buffer;
    dest->pub.free_in_buffer = OUTPUT_BUF_SIZE;
    return TRUE;
}

static void mem_term_destination(j_compress_ptr cinfo) {
    mem_dest_ptr dest = reinterpret_cast<mem_dest_ptr>(cinfo->dest);
    const auto len = OUTPUT_BUF_SIZE - dest->pub.free_in_buffer;
    dest->dest->insert(dest->dest->end(),
                       dest->buffer, dest->buffer + len);
}

static void jpeg_mem_dest(j_compress_ptr cinfo, 
                          std::vector<JOCTET>* dest_buf) {
    mem_dest_ptr dest;
    
    /* The destination object is made permanent so that multiple JPEG images
     * can be written to the same file without re-executing jpeg_stdio_dest.
     * This makes it dangerous to use this manager and a different destination
     * manager serially with the same JPEG object, because their private object
     * sizes may be different.  Caveat programmer.
     */
    if (cinfo->dest == NULL) {  /* first time for this JPEG object? */
        cinfo->dest = static_cast<jpeg_destination_mgr*>(
            (*cinfo->mem->alloc_small)(reinterpret_cast<j_common_ptr>(cinfo), 
                                       JPOOL_PERMANENT,
                                       sizeof(mem_destination_mgr)));
    }
    
    dest = reinterpret_cast<mem_dest_ptr>(cinfo->dest);
    dest->pub.init_destination = mem_init_destination;
    dest->pub.empty_output_buffer = mem_empty_output_buffer;
    dest->pub.term_destination = mem_term_destination;
    dest->dest = dest_buf;
}

stdx::binary internal::jpeg_binary(plane const* image, unsigned q) {

    throw_if_invalid_or_empty(image);

    auto quality = q;
    if (quality <= 0) quality = 90;
    else if (100 < quality) quality = 100;

    jpeg_compress_struct cinfo;
    my_error_mgr jerr;
    
    /* Step 1: allocate and initialize JPEG compression object */
    cinfo.err = jpeg_std_error(&jerr);
    jerr.error_exit = error_exit;

    /* Establish the setjmp return context for save_error_exit to use. */
    if (setjmp(jerr.setjmp_buffer)) {
        // If we get here, the JPEG code has signaled an error.
        jpeg_destroy_compress(&cinfo);
        FILE_LOG(logWARNING) << "jpeg_data_from_raw_image: exception";
        throw std::runtime_error("jpeg compress failed");
    }

    /* Now we can initialize the JPEG compression object. */
    jpeg_create_compress(&cinfo);
    
    /* Step 2: specify data destination (eg, a file) */
    /* Note: steps 2 and 3 can be done in either order. */
    std::vector<JOCTET> jpeg;
    jpeg_mem_dest(&cinfo,&jpeg);
    
    jpeg_save_image(cinfo, *image, int(quality));
    return jpeg;
}
  
