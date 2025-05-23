#pragma once

#include "color.hpp"
#include <stdext/rounding.hpp>
#include <utility>

namespace raw_image {

    /** \brief List of channel indices and write() method to write
     * channel values in correct order.
     */
    template <color_class cc, typename = void>
    struct color_channels;

    /// rgba specialization
    template <color_class cc>
    struct color_channels<
        cc, std::enable_if_t<cc == color_class::rgb ||
                             cc == color_class::alpha> > {

        int red = -1, green = -1, blue = -1, alpha = -1;

        constexpr color_channels(pixel_layout cs) {
            switch (cs) {
            case pixel::a8: alpha = 0; break;
            case pixel::r8: red = 0; break;
            case pixel::g8: green = 0; break;
            case pixel::b8: blue = 0; break;
            case pixel::rgb24: red=0, green=1, blue=2; break;
            case pixel::bgr24: red=2, green=1, blue=0; break;
            case pixel::rgba32: red=0, green=1, blue=2, alpha=3; break;
            case pixel::bgra32: red=2, green=1, blue=0, alpha=3; break;
            case pixel::argb32: red=1, green=2, blue=3, alpha=0; break;
            case pixel::abgr32: red=3, green=2, blue=1, alpha=0; break;
            case pixel::none:
            default: break; // constant pixel color
            }
        }

        constexpr inline void write(
            uint8_t* dest, uint8_t r, uint8_t g, uint8_t b, uint8_t a) const {
            if (red >= 0) dest[red] = r;
            if (green >= 0) dest[green] = g;
            if (blue >= 0) dest[blue] = b;
            if (alpha >= 0) dest[alpha] = a;
        }
    };

    /// yuv specialization
    template <color_class cc>
    struct color_channels<
        cc, std::enable_if_t<cc == color_class::gray ||
                             cc == color_class::yuv_jpeg ||
                             cc == color_class::yuv_nv21> > {

        int y_idx = -1, u_idx = -1, v_idx = -1, alpha = -1;

        constexpr color_channels(pixel_layout cs) {
            switch (cs) {
            case pixel::y8_jpeg: y_idx=0; break;
            case pixel::u8_jpeg: u_idx=0; break;
            case pixel::v8_jpeg: v_idx=0; break;
            case pixel::uv16_jpeg: u_idx=0, v_idx=1; break;
            case pixel::vu16_jpeg: u_idx=1, v_idx=0; break;
            case pixel::yuv24_jpeg: y_idx=0, u_idx=1, v_idx=2; break;
            case pixel::y8_nv21: y_idx=0; break;
            case pixel::u8_nv21: u_idx=0; break;
            case pixel::v8_nv21: v_idx=0; break;
            case pixel::uv16_nv21: u_idx=0, v_idx=1; break;
            case pixel::vu16_nv21: u_idx=1, v_idx=0; break;
            case pixel::yuv24_nv21: y_idx=0, u_idx=1, v_idx=2; break;
            default: break; // every channel default
            }
        }

        constexpr inline void write(
            uint8_t* dest, uint8_t y, uint8_t u, uint8_t v, uint8_t a) const {
            if (y_idx >= 0) dest[y_idx] = y;
            if (u_idx >= 0) dest[u_idx] = u;
            if (v_idx >= 0) dest[v_idx] = v;
            if (alpha >= 0) dest[alpha] = a;
        }
    };

    /// r85g10b05 specialization
    template <>
    struct color_channels<color_class::r85g10b05> {
        constexpr color_channels(pixel_layout) {}
        constexpr inline void write(
            uint8_t* dest, uint8_t g, uint8_t, uint8_t, uint8_t) const {
            *dest = g;
        }
    };


    /** \brief Read pixels with specific pixel layout from memory and
     * provide color conversion methods.
     */
    template <color_class, typename = void>
    struct color_convert_from;

    /// from rgb
    template <color_class cc>
    struct color_convert_from<
        cc, std::enable_if_t<cc == color_class::rgb ||
                             cc == color_class::alpha> > {

        const color_channels<color_class::rgb> channels;

        uint8_t const* red_ptr = nullptr;
        uint8_t const* green_ptr = nullptr;
        uint8_t const* blue_ptr = nullptr;
        uint8_t const* alpha_ptr = nullptr;

        int red_incr = 0, green_incr = 0, blue_incr = 0, alpha_incr = 0;

        uint8_t red_def, green_def, blue_def, alpha_def;


        /** \brief Construct from constant values.
         *
         * The arguments may be pixel_color, channel_value (constant_*)
         * or an integer.  An integer will apply to all channels.
         * For each channel, the first matching type (left to right) applies.
         */
        template <typename... Args>
        constexpr color_convert_from(
            pixel_layout cs, Args&&... args)
            : channels(cs),
              red_def(constant_red(args...)),
              green_def(constant_green(args...)),
              blue_def(constant_blue(args...)),
              alpha_def(constant_alpha(args...)) {
            const auto bpp = int(bytes_per_pixel(cs));
            if (channels.red >= 0)   red_incr = bpp;
            if (channels.green >= 0) green_incr = bpp;
            if (channels.blue >= 0)  blue_incr = bpp;
            if (channels.alpha >= 0) alpha_incr = bpp;
        }

        constexpr auto begin_line(const uint8_t* src) {
            red_ptr = channels.red >= 0 ? src + channels.red : &red_def;
            green_ptr = channels.green >= 0 ? src + channels.green : &green_def;
            blue_ptr = channels.blue >= 0 ? src + channels.blue : &blue_def;
            alpha_ptr = channels.alpha >= 0 ? src + channels.alpha : &alpha_def;
        }
        constexpr auto& operator++() {
            red_ptr += red_incr;
            green_ptr += green_incr;
            blue_ptr += blue_incr;
            alpha_ptr += alpha_incr;
            return *this;
        }

        constexpr auto alpha() const {
            return *alpha_ptr;
        }
        
        constexpr inline auto rgb() const {
            return std::array<uint8_t,3> { *red_ptr, *green_ptr, *blue_ptr };
        }

        static constexpr inline uint8_t lsb(int x) {
            return unsigned(x) & 0xff;
        }

        constexpr inline auto yuv_jpeg() const {
            const auto [r,g,b] = rgb();
            const auto y = (19595*r + 38470*g + 7471*b + 32768) >> 16;
            return std::array<uint8_t,3> {
                lsb(y),
                // cb = 128 + 0.564*(b-y)
                lsb((1155*(b-y) + 257*1024) >> 11),
                // cr = 128 + 0.713*(r-y)
                lsb((5841*(r-y) + (257*4096-1060)) >> 13)
            };
        }

        constexpr inline auto yuv_nv21() const {
            // for y: output range is [16, 236)
            // for u and v: range is no more than [16, 240)
            const auto [r,g,b] = rgb();
            return std::array<uint8_t,3> {
                lsb((66*r + 129*g + 25*b + 4224) >> 8),
                lsb((-38*r -74*g +112*b + 32768) >> 8),
                lsb((112*r -94*g  -18*b + 32768) >> 8)
            };
        }

        constexpr inline auto r85g10b05() const {
            const auto [r,g,b] = rgb();
            return lsb((218*r + 25*g + 13*b) >> 8);
        }
    };

    /// from yuv_jpeg
    template <>
    struct color_convert_from<color_class::yuv_jpeg> {

        const color_channels<color_class::yuv_jpeg> channels;

        uint8_t const* y_ptr = nullptr;
        uint8_t const* u_ptr = nullptr;
        uint8_t const* v_ptr = nullptr;
        uint8_t const* alpha_ptr = nullptr;

        int y_incr = 0, u_incr = 0, v_incr = 0, alpha_incr = 0;

        uint8_t y_def, u_def, v_def, alpha_def;


        /** \brief Construct from constant values.
         *
         * The arguments may be channel_value (constant_*) or an integer.
         * An integer will apply to all channels.
         * For each channel, the first matching type (left to right) applies.
         */
        template <typename... Args>
        constexpr color_convert_from(
            pixel_layout cs, Args&&... args)
            : channels(cs),
              y_def(constant_gray(args...)),
              u_def(constant_u(args...)),
              v_def(constant_v(args...)),
              alpha_def(constant_alpha(args...)) {
            const auto bpp = int(bytes_per_pixel(cs));
            if (channels.y_idx >= 0) y_incr = bpp;
            if (channels.u_idx >= 0) u_incr = bpp;
            if (channels.v_idx >= 0) v_incr = bpp;
            if (channels.alpha >= 0) alpha_incr = bpp;
        }

        constexpr auto begin_line(const uint8_t* src) {
            y_ptr = channels.y_idx >= 0 ? src + channels.y_idx : &y_def;
            u_ptr = channels.u_idx >= 0 ? src + channels.u_idx : &u_def;
            v_ptr = channels.v_idx >= 0 ? src + channels.v_idx : &v_def;
            alpha_ptr = channels.alpha >= 0 ? src + channels.alpha : &alpha_def;
        }
        constexpr auto& operator++() {
            y_ptr += y_incr;
            u_ptr += u_incr;
            v_ptr += v_incr;
            alpha_ptr += alpha_incr;
            return *this;
        }

        constexpr auto alpha() const {
            return *alpha_ptr;
        }
        
        constexpr inline auto yuv_jpeg() const {
            return std::array<uint8_t,3> { *y_ptr, *u_ptr, *v_ptr };
        }

        constexpr inline auto rgb() const {
            using stdx::round_to;
            const auto [y,u,v] = yuv_jpeg();
            const auto r =
                round_to<uint8_t>((5841*y + ((v-128)<<13) + 2920) / 5841);
            const auto b =
                round_to<uint8_t>((1155*y + ((u-128)<<11) + 575) / 1155);
            const auto g =
                round_to<uint8_t>(((y<<16) - 19595*r - 7471*b + 19235) / 38470);
            return std::array<uint8_t,3> { r, g, b };
        }

        constexpr inline auto r85g10b05() const {
            const auto [y,u,v] = yuv_jpeg();
            return stdx::round_to<uint8_t>(
                (1024*y + 1213*(v-128) + 87*(u-128)) >> 10);
        }
    };

    /** \brief Special case: from gray8 (y8_jpeg).
     *
     * In conversion to rgb or r85g10b05, the gray8 value is simply copied.
     * That is to say, a non-zero default u or v value is ignored.
     */
    template <>
    struct color_convert_from<color_class::gray> {
        uint8_t const* y_ptr = nullptr;
        uint8_t def_u, def_v, def_alpha;

        template <typename... Args>
        constexpr color_convert_from(
            pixel_layout, Args&&... args)
            : def_u(constant_u(args...)),
              def_v(constant_v(args...)),
              def_alpha(constant_alpha(args...)) {
        }

        constexpr auto begin_line(const uint8_t* src) {
            y_ptr = src;
        }
        constexpr auto& operator++() {
            ++y_ptr;
            return *this;
        }

        constexpr auto alpha() const {
            return def_alpha;
        }
        
        constexpr inline auto yuv_jpeg() const {
            return std::array<uint8_t,3> { *y_ptr, def_u, def_v };
        }

        constexpr inline auto rgb() const {
            return std::array<uint8_t,3> { *y_ptr, *y_ptr, *y_ptr };
        }

        constexpr inline auto r85g10b05() const {
            return *y_ptr;
        }
    };

    /// from yuv_nv21
    template <>
    struct color_convert_from<color_class::yuv_nv21> {

        const color_channels<color_class::yuv_nv21> channels;

        uint8_t const* y_ptr = nullptr;
        uint8_t const* u_ptr = nullptr;
        uint8_t const* v_ptr = nullptr;
        uint8_t const* alpha_ptr = nullptr;

        int y_incr = 0, u_incr = 0, v_incr = 0, alpha_incr = 0;

        uint8_t y_def, u_def, v_def, alpha_def;


        /** \brief Construct from constant values.
         *
         * The arguments may be channel_value (constant_*) or an integer.
         * An integer will apply to all channels.
         * For each channel, the first matching type (left to right) applies.
         */
        template <typename... Args>
        constexpr color_convert_from(
            pixel_layout cs, Args&&... args)
            : channels(cs),
              y_def(constant_gray(args...)),
              u_def(constant_u(args...)),
              v_def(constant_v(args...)),
              alpha_def(constant_alpha(args...)) {
            const auto bpp = int(bytes_per_pixel(cs));
            if (channels.y_idx >= 0) y_incr = bpp;
            if (channels.u_idx >= 0) u_incr = bpp;
            if (channels.v_idx >= 0) v_incr = bpp;
            if (channels.alpha >= 0) alpha_incr = bpp;
        }

        constexpr auto begin_line(const uint8_t* src) {
            y_ptr = channels.y_idx >= 0 ? src + channels.y_idx : &y_def;
            u_ptr = channels.u_idx >= 0 ? src + channels.u_idx : &u_def;
            v_ptr = channels.v_idx >= 0 ? src + channels.v_idx : &v_def;
            alpha_ptr = channels.alpha >= 0 ? src + channels.alpha : &alpha_def;
        }
        constexpr auto& operator++() {
            y_ptr += y_incr;
            u_ptr += u_incr;
            v_ptr += v_incr;
            alpha_ptr += alpha_incr;
            return *this;
        }

        constexpr auto alpha() const {
            return *alpha_ptr;
        }

        constexpr inline auto yuv_nv21() const {
            return std::array<uint8_t,3> { *y_ptr, *u_ptr, *v_ptr };
        }

        /// rgb from yuv (jpeg or nv21) or from constant value
        constexpr inline auto rgb() const {
            const auto yuv = yuv_nv21();
            const auto y = 1192*(yuv[0]-16);
            const auto u = yuv[1] - 128;
            const auto v = yuv[2] - 128;
            return std::array<uint8_t,3> {
                stdx::round_from((y + 1634*v) >> 10),
                stdx::round_from((y - 833*v - 400*u) >> 10),
                stdx::round_from((y + 2066*u) >> 10)
            };
        }

        constexpr inline auto r85g10b05() const {
            const auto [y,u,v] = yuv_nv21();
            return stdx::round_to<uint8_t>(
                (1192*(y-16) + 1306*(v-128) + 63*(u-128)) >> 10);
        }
    };


    /** \brief Operator to call the appropriate conversion for the specified
     * destination color class.
     *
     * This is a helper class for color_convert_to below.
     */
    template <color_class dest_cc, typename = void>
    struct color_convert_to_cc_helper;

    template <>
    struct color_convert_to_cc_helper<color_class::rgb> {
        template <typename SRC_T>
        constexpr auto operator()(const SRC_T& src) const {
            return src.rgb();
        }
    };

    template <color_class dest_cc>
    struct color_convert_to_cc_helper<dest_cc, std::enable_if_t<dest_cc == color_class::gray || dest_cc == color_class::yuv_jpeg> > {
        template <typename SRC_T>
        constexpr auto operator()(const SRC_T& src) const {
            return src.yuv_jpeg();
        }
    };

    template <>
    struct color_convert_to_cc_helper<color_class::yuv_nv21> {
        template <typename SRC_T>
        constexpr auto operator()(const SRC_T& src) const {
            return src.yuv_nv21();
        }
    };

    template <>
    struct color_convert_to_cc_helper<color_class::r85g10b05> {
        template <typename SRC_T>
        constexpr auto operator()(const SRC_T& src) const {
            return std::array<uint8_t,3> { src.r85g10b05(), 0, 0 };
        }
    };

    template <>
    struct color_convert_to_cc_helper<color_class::alpha> {
        template <typename SRC_T>
        constexpr auto operator()(const SRC_T& src) const {
            return std::array<uint8_t,3> { src.alpha(), 0, 0 };
        }
    };


    /** Methods for conversion from some layout to another layout.
     */
    template <color_class dest_cc,
              typename SRC_T,
              unsigned default_per_group = 1>
    struct color_convert_to : SRC_T {

        const color_convert_to_cc_helper<dest_cc> from{};
        const color_channels<dest_cc> dest;
        const unsigned dest_bpp;

        template <typename... Args>
        constexpr color_convert_to(
            pixel_layout dest_cs, Args&&... src_args)
            : SRC_T(std::forward<Args>(src_args)...),
              dest(dest_cs),
              dest_bpp(bytes_per_pixel(dest_cs)) {
        }

        /** \brief Single pixel conversion.
         */
        constexpr auto operator()(const uint8_t* src_px) {
            this->begin_line(src_px);
            const auto [c0,c1,c2] = from(static_cast<const SRC_T&>(*this));
            std::array<uint8_t,4> r{};
            dest.write(r.data(), c0, c1, c2, this->alpha());
            return r;
        }

        /** \brief Convert a line in groups of per_group pixels.
         *
         * Larger values of per_group (2, 4, 8, etc.) allow for SIMD
         * instructions to be used, but the number of pixels per line
         * must be a multiple of this value.
         */
        template <unsigned per_group = 1>
        void convert_line(uint8_t* dest_line,
                          const uint8_t* src_line,
                          unsigned ngroups) {
            this->begin_line(src_line);
            for ( ; ngroups > 0; --ngroups)
                for (auto i = per_group; i > 0; --i,
                         dest_line += dest_bpp, ++*this) {
                    const auto [c0,c1,c2] =
                        from(static_cast<const SRC_T&>(*this));
                    dest.write(dest_line, c0, c1, c2, this->alpha());
                }
        }

        inline void operator()(uint8_t* dest_line, const uint8_t* src_line,
                               unsigned ngroups) {
            convert_line<default_per_group>(dest_line, src_line, ngroups);
        }
    };


    /** \brief Convert single pixel.
     *
     * \returns std::array<uint8_t,4>
     */
    template <pixel_layout dest_cs>
    constexpr auto to_layout(pixel_color c) {
        using from = color_convert_from<color_class::rgb>;
        using conv = color_convert_to<to_color_class(dest_cs),from>;
        return conv{dest_cs,pixel::none,c}(nullptr);
    }
}
