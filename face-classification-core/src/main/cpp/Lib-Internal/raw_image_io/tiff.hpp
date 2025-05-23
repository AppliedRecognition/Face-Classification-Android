#pragma once

#include <raw_image/types.hpp>

namespace raw_image {
    /** \brief Decode tiff image from file or data.
     *
     * This method always returns a packed rgba32 image.
     */
    plane_ptr tiff_load(FILE* file);
    plane_ptr tiff_load(const void* data, std::size_t size);
}
