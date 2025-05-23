#pragma once

#include "bit.hpp"

#if defined(__APPLE_)
#define bswap_16 OSSwapInt16
#define bswap_32 OSSwapInt32
#define bswap_64 OSSwapInt64

#elif defined(_MSC_VER)
#include <stdlib.h>
#define bswap_16 _byteswap_ushort
#define bswap_32 _byteswap_ulong
#define bswap_64 _byteswap_uint64

#else  // standard platforms
#include <byteswap.h>
#endif

#include <algorithm>


namespace stdx {
    template <bool enable, std::size_t SIZE> struct bswap_N;

    template <bool enable>
    struct bswap_N<enable,1> {
        template <typename T>
        static inline T bswap(T x) { 
            static_assert(sizeof(T) == 1, "unexpected argument size");
            return x;
        }
    };

    template <bool enable>
    struct bswap_N<enable,2> {
        template <typename T>
        static inline T bswap(T x) { 
            static_assert(sizeof(T) == 2, "unexpected argument size");
            return enable ? bswap_16(x) : x;
        }
    };

    template <bool enable>
    struct bswap_N<enable,4> {
        template <typename T>
        static inline T bswap(T x) { 
            static_assert(sizeof(T) == 4, "unexpected argument size");
            return enable ? bswap_32(x) : x;
        }
    };

    template <bool enable>
    struct bswap_N<enable,8> {
        template <typename T>
        static inline T bswap(T x) { 
            static_assert(sizeof(T) == 8, "unexpected argument size");
            return enable ? bswap_64(x) : x;
        }
    };

    // convert between little-endian and machine byte order
    template <typename T>
    inline T bswap_le(T x) {
#if __BYTE_ORDER == __LITTLE_ENDIAN
        return bswap_N<false,sizeof(T)>::bswap(x);
#elif __BYTE_ORDER == __BIG_ENDIAN
        return bswap_N<true,sizeof(T)>::bswap(x);
#else
#error unsupported endianess
#endif
    }

    // like std::copy but reverse bytes if machine is not big-endian
    template <typename InputIt, typename OutputIt>
    inline OutputIt copy_be(InputIt first, InputIt last, OutputIt d_first) {
#if __BYTE_ORDER == __LITTLE_ENDIAN
        while (first != last)
            *d_first++ = *--last;
        return d_first;

#elif __BYTE_ORDER == __BIG_ENDIAN
        return std::copy(first, last, d_first);
#else
#error unsupported endianess
#endif
    }
}
