#pragma once

#include <vector>
#include <cstdint>
#include <istream>
#include <stdexcept>
#include <stdext/bswap.hpp>
#include <stdext/binary.hpp>
#include <stdext/forward_iterator.hpp>

namespace rec {
    namespace internal {
        template <std::size_t N> struct uint_of_size;
        template <> struct uint_of_size<1> { using type = uint8_t ; };
        template <> struct uint_of_size<2> { using type = uint16_t; };
        template <> struct uint_of_size<4> { using type = uint32_t; };
        template <> struct uint_of_size<8> { using type = uint64_t; };

        using serialize_buffer_type = std::vector<unsigned char>;

        template <typename T, typename BUF>
        void serialize_value(BUF& buf, T x) {
            using uint = typename uint_of_size<sizeof(T)>::type;
            using byte = typename BUF::value_type;
            static_assert(sizeof(byte) == 1,
                          "buffer must contain byte sized values");
            union { T x; uint i; byte b[sizeof(T)]; } v = { x };
            v.i = stdx::bswap_le(v.i);
            buf.insert(buf.end(), v.b, v.b + sizeof(T));
        }

        template <typename T>
        T deserialize_value(const void* src) {
            using uint = typename uint_of_size<sizeof(T)>::type;
            const union { uint i; T x; } v = { 
                stdx::bswap_le(*static_cast<const uint*>(src)) 
            };
            return v.x;
        }

        template <typename T>
        T deserialize_value(std::istream& in) {
            using uint = typename uint_of_size<sizeof(T)>::type;
            union { char b[sizeof(T)]; uint i; T x; } v;
            in.read(v.b,sizeof(T));
            if (std::size_t(in.gcount()) != sizeof(T))
                throw std::runtime_error("read from stream failed");
            v.i = stdx::bswap_le(v.i);
            return v.x;
        }

        template <typename T, typename OUT>
        void deserialize_sequence(OUT out, const void* src, std::size_t len) {
            const uint8_t* p = static_cast<const uint8_t*>(src);
            for ( ; len > 0; --len, p += sizeof(T)) {
                *out = deserialize_value<T>(p);
                ++out;
            }
        }

        template <typename T, typename OUT>
        void deserialize_sequence(OUT out, std::istream& in, std::size_t len) {
            for ( ; len > 0; --len) {
                *out = deserialize_value<T>(in);
                ++out;
            }
        }


        /** \brief Check for and remove compression.
         */
        bool is_compressed(const void* src, std::size_t len);
        stdx::binary remove_compression(const void* src, std::size_t len);


        /* Prototype serialization for version 3 and up.
         *
         *   byte 0: version number
         *   byte 1: element count (or 0 if count is per-vector, below)
         *   byte 2: element type (fpvc 0x10=8bit, 0x11=12bit, 0x12=16bit)
         *   byte 3: number of feature vectors (1 or 2)
         *
         *   for each feature vector (fpvc):
         *     4 bytes: element count only if not in header (little endian)
         *     4 bytes: (float) coefficient
         *     elements: integer 8, 12 or 16 bits each
         *     padding: if necessary to multiple of 4 bytes
         */
        bool is_prototype(const void* data, std::size_t size);

        // see fpvc.hpp for prototype serialize / deserialize methods


        /* Multiple prototype serialization format:
         *
         *   byte 0: zero (to distinquish from single prototype)
         *   byte 1: version
         *   byte 2: reserved (zero)
         *   byte 3: 1 = multi-prototype (note: 0 = pca)
         *
         *   for each prototype:
         *     4-byte length
         *     serialized prototype (padded to multiple of 4)
         *
         *   footer: 4-byte zero
         *
         * Note: this format is deprecated in favor of a
         * json::array of binaries (raw serialized prototypes).
         * Only deserialize_multiple() remains for backward compatibility.
         */
        std::vector<std::pair<const void*, std::size_t> >
        deserialize_multiple(const void* src, std::size_t len);
    }
}
