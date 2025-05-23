#pragma once

#include <raw_image/reader.hpp>
#include <stdext/binary.hpp>
#include <stdext/stdio.hpp>

namespace raw_image {
    /** \brief Reader with extra fields for possible scaling.
     *
     * For jpeg images that are scaled down by 2, 4 or 8 when loading,
     * the original image size and the plane::scale parameter are provided.
     * Note the scale value will be 0, 1, 2, or 3 coresponding to down
     * scaling of 1, 2, 4, or 8.
     */
    struct reader_ex : reader {
        using reader::reader;
        image_size original_size;
        int scale;
    };


    /** \brief Decode jpeg image from file or data.
     *
     * The desired_layout is a hint for optimization purposes.
     * The returned layout may differ.
     *
     * If min_pixels >= 0, then the jpeg may be decoded faster by
     * down scaling during decode.
     * The image will only be down scaled if the result has the specified
     * minimum number of pixels.
     * In this case, the original_size and scale values in the returned
     * reader_ex are set appropriately.
     *
     * When reading from data/size buffer, the data must remain valid
     * for the duration that the reader is active.
     * The overload accepting stdx::binary keeps a copy of it within
     * the returned reader object to ensure the data remains valid.
     */
    std::unique_ptr<reader_ex>
    jpeg_load(stdx::file_ptr file,
              pixel_layout desired_layout = pixel::none,
              int min_pixels = -1);

    std::unique_ptr<reader_ex>
    jpeg_load(stdx::binary data,
              pixel_layout desired_layout = pixel::none,
              int min_pixels = -1);

    std::unique_ptr<reader_ex>
    jpeg_load(const void* data, std::size_t size,
              pixel_layout desired_layout = pixel::none,
              int min_pixels = -1);


    // internal method used by to_binary() and save()
    namespace internal {
        stdx::binary jpeg_binary(plane const* image, unsigned qual = 0);
    }
}
