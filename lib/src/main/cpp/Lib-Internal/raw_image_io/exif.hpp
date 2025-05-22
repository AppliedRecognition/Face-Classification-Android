#pragma once

#include <libexif/exif-data.h>
#include <stdext/rounding.hpp>
#include <cctype>
#include <cstring>

namespace raw_image {

    template <typename T, typename = void>
    struct is_pointer_to_bytes : std::false_type {};
    template <typename T>
    struct is_pointer_to_bytes<
        T, std::enable_if_t<std::is_pointer<T>::value &&
                            std::is_void<std::remove_pointer_t<T> >::value> >
        : std::true_type {};
    template <typename T>
    struct is_pointer_to_bytes<
        T, std::enable_if_t<std::is_pointer<T>::value &&
                            sizeof(std::remove_pointer_t<T>) == 1> >
        : std::true_type {};
    
    
    /** \brief Get rotate value needed to turn image upright.
     *
     * The container must have a data() method returning const void* or
     * const CHAR* where sizeof(CHAR) == 1.
     * It must also have a size() method.
     * Note that stdx::binary satisfies these requirements.
     *
     * The value returned is compatible with raw_image::rotate parameter
     * and the raw image methods accepting this parameter.
     * Returns 0 if the exif data could not be read or is invalid.
     *
     * If exif_orientation is not null, then the raw exif orientation value,
     * nominally between 1 and 8 inclusive, is returned.  
     * This value is set to 0 if the exif data could not be read.
     *
     * This method is a template so that libexif need not be linked in if
     * this method is not being used.
     */
    template <typename CONTAINER>
    unsigned rotate_from_exif_data(const CONTAINER& image,
                                   int* exif_orientation = nullptr) {
        auto* const data_ = image.data();
        static_assert(is_pointer_to_bytes<decltype(data_)>::value);
        auto* const data = reinterpret_cast<const unsigned char*>(data_);
        if (exif_orientation) (*exif_orientation) = 0;
        unsigned result = 0;
        if (auto* exif = exif_data_new_from_data(
                data, stdx::round_from(image.size()))) {
            if (auto* entry = exif_data_get_entry(exif, EXIF_TAG_ORIENTATION)) {
                const auto endian = exif_data_get_byte_order(exif);
                const auto o = exif_get_short(entry->data, endian);
                static constexpr unsigned table[] = { 0, 4, 2, 6, 5, 3, 7, 1 };
                if (0 < o && o <= 8)
                    result = table[o-1];
                if (exif_orientation) (*exif_orientation) = o;
            }
            exif_data_free(exif);
        }
        return result;
    }

    /*
    #include "binary_file_mapping.hpp"
    template <typename PATH>
    unsigned rotate_from_exif_file(const PATH& path,
                                   int* exif_orientation = nullptr) {
        return rotate_from_exif_data(
            boostx::binary_file_mapping(path), exif_orientation);
    }
    */
}
