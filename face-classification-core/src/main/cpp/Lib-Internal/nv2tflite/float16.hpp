#pragma once

#include <cstdint>

enum class float16_t : uint16_t {};

inline float to_float(float16_t f16) {
    // MSB -> LSB
    // float16 is 1bit:sign, 5bit:exponent, 10bit:fraction
    // float32 is 1bit:sign, 8bit:exponent, 23bit:fraction
    // for normal exponent(1 to 0x1e): value=2**(exponent-15)*(1.fraction)
    // for denormalized exponent(0): value=2**-14*(0.fraction)
    const auto x16 = uint16_t(f16);
    union { uint32_t x32; float f32; };
    x32 = uint32_t(x16 & 0x8000) << 16; // sign
    unsigned exponent = (x16 >> 10) & 0x1Fu;
    unsigned fraction = x16 & 0x3FFu;
    if (exponent == 0x1F)   // Inf or NaN
        x32 |= (0xFFu << 23) | (fraction << 13);
    else if (0 < exponent)  // ordinary number
        x32 |= ((exponent + (127-15)) << 23) | (fraction << 13);
    else if (0 < fraction) {
        // can be represented as ordinary value in float32
        // 2 ** -14 * 0.0101
        // => 2 ** -16 * 1.0100
        exponent = 127 - 14;
        while ((fraction & (1 << 10)) == 0) {
            --exponent;
            fraction <<= 1;
        }
        fraction &= 0x3FF;
        x32 |= (exponent << 23) | (fraction << 13);
    }
    // else zero (with sign bit)
    return f32;
}
