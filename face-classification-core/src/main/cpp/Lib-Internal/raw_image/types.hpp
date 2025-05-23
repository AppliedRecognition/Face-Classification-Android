#pragma once

#include <memory>

namespace raw_image {

    /** \brief Image pixel layouts.
     *
     * Each constant has structure 0x0NTS where
     *   N is the number of bytes per pixel,
     *   T is the pixel format (what channels in what order), and
     *   S is the specific color subclass (rgb, jpeg, nv21, etc).
     * This structure allows for multiple kinds of grayscale and YUV.
     *
     * Note that this structure is an implementation detail that is
     * subject to change.  It is best not to rely on it.
     * Use the methods bytes_per_pixel(), same_channel_order() and
     * to_color_class() below to reliably extract this metadata.
     */
    enum class pixel_layout : unsigned {
        none = 0,  ///< default or no value assigned

        gray8 = 0x102,

        y8_jpeg = 0x102,
        y8_nv21 = 0x103,
        u8_jpeg = 0x112,
        u8_nv21 = 0x113,
        v8_jpeg = 0x122,
        v8_nv21 = 0x123,

        r8 = 0x131,
        g8 = 0x141,
        b8 = 0x151,
        a8 = 0x1f0,

        r85g10b05 = 0x104,  ///< grayscale used for face detection

        uv16_jpeg = 0x202,
        uv16_nv21 = 0x203,
        vu16_jpeg = 0x212,
        vu16_nv21 = 0x213,

        a16_le = 0x2f0,  ///< 16 bit unsigned in little endian byte order

        rgb24 = 0x301,
        bgr24 = 0x311,

        yuv = 0x322, ///< full range [0-255] as in JPEG standard
        yuv24_jpeg = 0x322,
        yuv24_nv21 = 0x323,
        
        argb32 = 0x401,
        abgr32 = 0x411,
        rgba32 = 0x421,
        bgra32 = 0x431,

        f32 = 0x4fe  ///< 32 bit float in host endian byte order
    };
    using pixel = pixel_layout;


    /** \brief Number of bytes needed per pixel.
     */
    constexpr unsigned bytes_per_pixel(pixel_layout cs) {
        return unsigned(cs)>>8;
    }


    /** \brief Test if two pixel layouts represent the same channels
     * in the same order.
     *
     * The pixel layout is how many channels, what channels and in what order.
     * For example, use this method if you need a YUV24, but you don't care
     * if it's YUV24_JPEG or YUV24_NV21.
     */
    constexpr bool same_channel_order(pixel_layout a, pixel_layout b) {
        return ((unsigned(a) ^ unsigned(b)) & ~15u) == 0;
    }


    /** \brief Color class values.
     *
     * Note that Y8_JPEG (aka GRAY8) has class gray, not yuv_jpeg.
     * However,  Y8_NV21 has class yuv_nv21.
     */
    enum class color_class {
        unknown = 0, alpha, gray, yuv_jpeg, yuv_nv21, rgb, r85g10b05
    };
    using cc = color_class;


    /** \brief Color class from pixel layout.
     */
    constexpr auto to_color_class(pixel_layout cs) {
        const auto bpp = bytes_per_pixel(cs);
        if (bpp < 1 || 4 < bpp) return cc::unknown;
        switch (unsigned(cs)&15) {
        case 0: return cs == pixel::a8 ? cc::alpha : cc::unknown;
        case 1: return cc::rgb;
        case 2: return cs == pixel::gray8 ? cc::gray : cc::yuv_jpeg;
        case 3: return cc::yuv_nv21;
        case 4: return cs == pixel::r85g10b05 ? cc::r85g10b05 : cc::unknown;
        }
        return cc::unknown;
    }


    /** \brief Single plane pixel buffer metadata.
     */
    struct plane {
        unsigned char* data = nullptr;
        unsigned width = 0;
        unsigned height = 0;
        unsigned bytes_per_line = 0;
        pixel_layout layout = pixel::gray8;

        /** \brief Rotation by multiple of 90 degrees required to make
         * image upright.
         *
         * If rotate & 4, then mirror image before rotation.
         */
        unsigned rotate = 0;

        /** \brief Power of 2 scaling required to make original image.
         *
         * If scale > 0, then the stored image has been downsampled.
         * It must be scaled up by 2^scale (width and height) to restore
         * to original.
         *
         * Scale < 0 indicates upsampling.
         */
        int scale = 0;
    };

    using plane_ptr = std::unique_ptr<plane>;
    using ptr = plane_ptr;
}
