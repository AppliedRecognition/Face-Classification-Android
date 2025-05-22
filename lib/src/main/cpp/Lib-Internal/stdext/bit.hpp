#pragma once

#if __has_include(<endian.h>)
#include <endian.h>
#endif

#if __has_include(<libkern/OSByteOrder.h>)
#include <libkern/OSByteOrder.h>
#endif

#if !defined(__BYTE_ORDER) && defined(BYTE_ORDER)  // Apple
#define __BYTE_ORDER    BYTE_ORDER
#define __LITTLE_ENDIAN LITTLE_ENDIAN
#define __BIG_ENDIAN    BIG_ENDIAN
#endif

#if !defined(__BYTE_ORDER) && defined(_MSC_VER) // Windows
#define __BYTE_ORDER    4321
#define __LITTLE_ENDIAN 4321
#define __BIG_ENDIAN    1234
#endif

#include <type_traits>
#include <cstring>

/** \brief Some of what is in the C++20 <bit> header.
 */
namespace stdx {

    /** \brief C++20 endian enum.
     */
    enum class endian : decltype(__BYTE_ORDER) {
        little = __LITTLE_ENDIAN,
        big    = __BIG_ENDIAN,
        native = __BYTE_ORDER
    };
    static_assert(endian::little != endian::big);

    /** \brief C++20 bit_cast.
     *
     * For constexpr compiler support is required.
     */
    template <class To, class From>
    inline typename std::enable_if_t<
        sizeof(To) == sizeof(From) &&
        std::is_trivially_copyable_v<From> &&
        std::is_trivially_copyable_v<To>,
        To>
    bit_cast(const From& src) noexcept {
        static_assert(std::is_trivially_constructible_v<To>);
        To dst;
        std::memcpy(&dst, &src, sizeof(To));
        return dst;
    }
}
