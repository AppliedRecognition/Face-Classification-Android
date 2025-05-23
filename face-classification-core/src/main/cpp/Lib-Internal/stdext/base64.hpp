#pragma once

#include "binary.hpp"
#include <stdexcept>
#include <string_view>

namespace stdx {

    /// runtime error for invalid base64 encoding
    struct invalid_base64 : public std::runtime_error {
        invalid_base64(const char* msg) : runtime_error(msg) {}
    };

    constexpr inline unsigned char base64_decode_char(char c) {
        using uchar = unsigned char;
        if ('A' <= c && c <= 'Z')
            return uchar(c-'A');
        if ('a' <= c && c <= 'z')
            return uchar(c-'a'+26);
        if ('0' <= c && c <= '9')
            return uchar(c-'0'+52);
        if (c == '+')
            return 62;
        if (c == '/')
            return 63;
        throw invalid_base64("invalid base64 character");
    }

    // decode 4 characters from base 64 as 1 to 3 bytes
    // returns number of bytes written to dest
    constexpr inline unsigned base64_decode3(void* dest, const char* src) {
        using uchar = unsigned char;
        auto d = static_cast<uchar*>(dest);
        auto c0 = base64_decode_char(*src++);
        auto c1 = base64_decode_char(*src++);
        *d++ = uchar((c0<<2) + (c1>>4));
        if (*src=='=')
            return 1;
        auto c2 = base64_decode_char(*src++);
        *d++ = uchar((c1<<4) + (c2>>2));
        if (*src=='=')
            return 2;
        auto c3 = base64_decode_char(*src++);
        *d++ = uchar((c2<<6) + c3);
        return 3;
    }

    inline binary base64_decode(std::string_view src) {
        std::vector<unsigned char> result;
        result.reserve(3*((src.length()+3)/4));
        unsigned char buf_dest[3];
        unsigned buf_valid = 0;
        char buf_src[4];
        buf_src[2] = '=';
        buf_src[3] = '=';
        for (std::size_t i = 0; i < src.length(); ++i) {
            if (!isspace(src[i]) && !iscntrl(src[i])) {
                buf_src[buf_valid++] = src[i];
                if (buf_valid == 4) {
                    auto len = base64_decode3(buf_dest,buf_src);
                    result.insert(result.end(),buf_dest,buf_dest+len);
                    buf_valid = 0;
                    buf_src[2] = '=';
                    buf_src[3] = '=';
                }
            }
        }
        if (buf_valid == 1)
            throw invalid_base64("invalid base64 string");
        else if (buf_valid > 1) {
            auto len = base64_decode3(buf_dest,buf_src);
            result.insert(result.end(),buf_dest,buf_dest+len);
        }
        return result;
    }

    // encode 1 to 3 bytes as exactly 4 base64 characters (w/o null terminator)
    // returns remaining bytes in source if len > 3, otherwise 0
    inline std::size_t
    base64_encode3(char* dest, const void* src, std::size_t len) {
        static constexpr auto charset =
            "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        const auto* s = static_cast<const unsigned char*>(src);
        *dest++ = charset[s[0]>>2];  // top 6 bits of s[0]
        if (len >= 3) {
            // bottom 2 bits of s[0] ++ top 4 bits of s[1]
            *dest++ = charset[((s[0]&3)<<4) | (s[1]>>4)];
            // bottom 4 bits of s[1] ++ top 2 bits of s[2]
            *dest++ = charset[((s[1]&15))<<2 | (s[2]>>6)];
            // bottom 6 bits of s[2]
            *dest++ = charset[s[2]&0x3f];
            return len - 3;
        }
        if (len == 2) {
            // bottom 2 bits of s[0] ++ top 4 bits of s[1]
            *dest++ = charset[((s[0]&3)<<4) | (s[1]>>4)];
            // bottom 4 bits of s[1]
            *dest++ = charset[(s[1]&15)<<2];
            *dest++ = '=';
        }
        else if (len == 1) {
            // bottom 2 bits of s[0]
            *dest++ = charset[(s[0]&3)<<4];
            *dest++ = '=';
            *dest++ = '=';
        }
        return 0;
    }

    inline std::string base64_encode(const void* src, std::size_t len) {
        const auto* s = static_cast<const unsigned char*>(src);
        char buf[4];
        std::string out;
        out.reserve(4*((len+2)/3));
        while (len > 0) {
            len = base64_encode3(buf,s,len);
            out.append(buf,4);
            s += 3;
        }
        return out;
    }
    inline auto base64_encode(const stdx::binary& bin) {
        return base64_encode(bin.data(), bin.size());
    }
}
