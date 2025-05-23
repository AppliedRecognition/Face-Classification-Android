#pragma once

#include "types.hpp"

#include <applog/assert.hpp>
#include <stdext/bswap.hpp>

#include <optional>
#include <iterator>

namespace {
    struct enc_history {
        using string_map_type = std::map<json::string,unsigned>;
        string_map_type string_map;
        unsigned num_strings;
        bool base_traits_sent;
        std::optional<unsigned> stream_string_ref;
        std::optional<unsigned> stream_binary_ref;
        std::optional<unsigned> stream_array_ref;
        enc_history() noexcept : num_strings(0), base_traits_sent(false) {}
        enc_history(enc_history&&) = delete;
        enc_history(const enc_history&) = delete;
        enc_history& operator=(enc_history&&) = delete;
        enc_history& operator=(const enc_history&) = delete;
    };

    // max output is 4 bytes
    template <typename T>
    void encode_unsigned(std::string& dest, T i) {
        if (i < (1<<7)) {
            AR_CHECK(i >= 0);
            dest += char(i);
        }
        else if (i < (1<<14)) {
            dest += char(0x80|(i>>7));
            dest += char(i&0x7f);
        }
        else if (i < (1<<21)) {
            dest += char(0x80|(i>>14));
            dest += char(0x80|((i>>7)&0x7f));
            dest += char(i&0x7f);
        }
        else {
            AR_CHECK(i < (1<<29));
            dest += char(0x80|(i>>22));
            dest += char(0x80|((i>>15)&0x7f));
            dest += char(0x80|((i>>8)&0x7f));
            dest += char(i);
        }
    }

    // output is 9 bytes
    void encode_double(std::string& dest, double d) {
        AR_CHECK(sizeof(double) == 8);
        dest += char(5);  // double-marker
        union { double d; char buf[8]; } v = { d };
        stdx::copy_be(v.buf, v.buf+8, std::back_inserter(dest));
    }
}
