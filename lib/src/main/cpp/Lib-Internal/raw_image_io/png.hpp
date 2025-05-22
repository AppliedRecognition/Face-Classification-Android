#pragma once

#include <raw_image/reader.hpp>
#include <stdext/stdio.hpp>
#include <stdext/binary.hpp>

namespace raw_image {
    /** \brief Decode png image from file or data.
     *
     * The desired_layout is a hint for optimization purposes.
     * The returned layout may differ.
     *
     * When reading from data/size buffer, the data must remain valid
     * for the duration that the reader is active.
     */
    std::unique_ptr<reader>
    png_load(stdx::file_ptr file, pixel_layout desired_layout = pixel::none);

    std::unique_ptr<reader>
    png_load(stdx::binary data, pixel_layout desired_layout = pixel::none);

    std::unique_ptr<reader>
    png_load(const void* data, std::size_t size,
             pixel_layout desired_layout = pixel::none);


    // internal method used by to_binary() and save()
    namespace internal {
        stdx::binary png_binary(plane const* image);
    }
}
