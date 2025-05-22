#pragma once

#include <raw_image/core.hpp>
#include <stdext/path_traits.hpp>
#include <stdext/stdio.hpp>
#include <stdext/binary.hpp>

namespace raw_image {
    
    /** \brief Jpeg output selection with quality specification.
     */
    struct jpeg_quality {
        unsigned quality;
        jpeg_quality(unsigned quality = 0) : quality(quality) {}
        inline auto operator()(unsigned q) const { return jpeg_quality{q}; }
    };
    const auto jpeg = jpeg_quality(90);  // default quality setting


    /** \brief Png output selection.
     */
    struct png_tag;
    using png_option = stdx::option_bool<png_tag>;
    const png_option png{true};


    namespace internal {
        /* These are the internal implementation methods and not meant
         * to be called directly.  Use the versions outside the internal
         * namespace below.
         */
        plane_ptr from_binary(
            const void* data, std::size_t size,
            const stdx::options_tuple<rotate,pixel_layout>& opts);
        plane_ptr load(
            stdx::file_ptr file, const std::string& imagepath,
            const stdx::options_tuple<rotate,pixel_layout>& opts);
        stdx::binary to_binary(
            plane const* image,
            const stdx::options_tuple<jpeg_quality,png_option>& opts);
        void save(
            plane const* image, FILE* outfile, const std::string& imagepath,
            const stdx::options_tuple<jpeg_quality,png_option>& opts);
    }
    

    /** \brief Decode image from memory.
     *
     * This method will auto detect JPEG, TIFF and PNG images.
     *
     * If rotate > 0, rotate image by multiple of 90 degrees.
     * If rotate & 4, mirror image before rotation.
     */
    template <typename... Opts>
    inline auto
    from_binary(const void* data, std::size_t size, Opts&&... opts) {
        return internal::from_binary(
            data, size, { std::forward<Opts>(opts)... } );
    }

    template <typename... Opts>
    inline auto
    from_binary(const stdx::binary& data, Opts&&... opts) {
        return internal::from_binary(
            data.data(), data.size(), { std::forward<Opts>(opts)... } );
    }


    /** \brief Load image from file.
     *
     * This method will auto detect and load JPEG, TIFF and PNG images.
     *
     * If rotate > 0, rotate image by multiple of 90 degrees.
     * If rotate & 4, mirror image before rotation.
     */
    template <typename PATH, typename... Opts>
    inline std::enable_if_t<stdx::is_fopen_path_v<PATH>, plane_ptr>
    load(const PATH& imagepath, Opts&&... opts) {
        return internal::load(
            stdx::fopen_rb(imagepath), stdx::generic_string(imagepath),
            { std::forward<Opts>(opts)... } );
    }


    /** \brief Encode image to memory.
     *
     * Both jpeg and png are supported.
     * If neither is specified as an option then jpeg is selected.
     */
    template <typename... Opts>
    inline auto to_binary(single_plane_arg image, Opts&&... opts) {
        return internal::to_binary(
            image.get(), { std::forward<Opts>(opts)... });
    }


    /** \brief Write image to file.
     *
     * Both jpeg and png are supported.
     * If neither is specified as an option then the choice is made based on
     * the extension of the filename.
     */
    template <typename PATH, typename... Opts>
    inline std::enable_if_t<stdx::is_fopen_path_v<PATH> >
    save(single_plane_arg image, const PATH& imagepath, Opts&&... opts) {
        internal::save(
            image.get(), stdx::fopen_wb(imagepath).get(),
            stdx::generic_string(imagepath),
            { std::forward<Opts>(opts)... });
    }
}


