
#include "hash32.hpp"
#include <dlib/general_hash/murmur_hash3.h>
#include <cstdint>
#include <limits>

std::string dlibx::hash32(std::string_view s, unsigned id) {
    static constexpr auto B32 = "abcdefghijklmnopqrstuvwxyz234567";
    static constexpr auto imax = std::numeric_limits<int>::max();
    const auto n = s.size() <= imax ? int(s.size()) : imax;
    const uint32_t h = dlib::murmur_hash3(s.data(), n, id);
    return {
        B32[((h>>30)+(id<<2))&31],
        B32[(h>>25)&31],
        B32[(h>>20)&31],
        B32[(h>>15)&31],
        B32[(h>>10)&31],
        B32[(h>>5)&31],
        B32[h&31]
    };
}
