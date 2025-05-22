#pragma once

#include "types.hpp"
#include <array>
#include <cstdint>

namespace raw_image {

    /** \brief Pixel colors in RGB format.
     *
     * Blue is the least significant byte so as a hex literal 
     * the format is 0x00RRGGBB.
     */
    enum class pixel_color : unsigned {};
    const auto color_black   = pixel_color{0x000000};
    const auto color_blue    = pixel_color{0x0000ff};
    const auto color_cyan    = pixel_color{0x00ffff};
    const auto color_green   = pixel_color{0x00ff00};  // bright green
    const auto color_yellow  = pixel_color{0xffff00};
    const auto color_red     = pixel_color{0xff0000};
    const auto color_magenta = pixel_color{0xff00ff};
    const auto color_white   = pixel_color{0xffffff};


    /** \brief Convert pixel_color to bytes in specified pixel layout.
     *
     * This method always returns 4 bytes even if fewer are required.
     * The unused latter bytes will be zero.
     *
     * Note: implementation is in reader.cpp
     */
    std::array<uint8_t,4>
    to_layout(pixel_layout cs, pixel_color c);


    /** \brief Constant channel values.
     */
    template <pixel_layout cs, uint8_t default_value = 0>
    struct channel_value {
        using value_type = uint8_t;
        uint8_t value;

        constexpr operator value_type() const { return value; }
        constexpr value_type operator()() const { return value; }

        template <typename T>
        static constexpr auto from_arg_or(const T& v0, value_type v1) {
        }
        static constexpr auto from_args() {
            return default_value;
        }
        template <typename Arg0, typename... Args>
        static constexpr value_type
        from_args(const Arg0& a0, const Args&... args) {
            if (std::is_same<channel_value,Arg0>::value ||
                std::is_integral<Arg0>::value)
                return value_type(a0);
            if (std::is_same<pixel_color,Arg0>::value)
                switch (cs) {
                case pixel::r8: return (unsigned(a0) >> 16) & 0xff;
                case pixel::g8: return (unsigned(a0) >> 8) & 0xff;
                case pixel::b8: return unsigned(a0) & 0xff;
                default: break;
                }
            return from_args(args...);
        }
        template <typename... Args>
        constexpr channel_value(Args&&... args)
            : value(from_args(args...)) {}
    };
    using constant_alpha = channel_value<pixel::a8>;
    using constant_red = channel_value<pixel::r8>;
    using constant_green = channel_value<pixel::g8>;
    using constant_blue = channel_value<pixel::b8>;

    // these yuv constants are used for both jpeg and nv21
    using constant_gray = channel_value<pixel::gray8>;
    using constant_u = channel_value<pixel::u8_jpeg,128>;
    using constant_v = channel_value<pixel::v8_jpeg,128>;
}
