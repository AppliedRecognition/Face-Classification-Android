#pragma once

#include <array>
#include <dlibx/tensor.hpp>
#include <flatbuffers/vector.h>

namespace conv {

    using shape_type = std::array<unsigned,4>;

    inline auto shape_size(const std::array<unsigned,4>& s) {
        std::size_t n = 1;
        for (auto x : s)
            n *= x;
        return n;
    }

    inline std::ostream& operator<<(std::ostream& out, const shape_type& s) {
        return out << s[0] << 'x' << s[1] << 'x' << s[2] << 'x' << s[3];
    }

    inline auto to_shape(const dlib::tensor& t) {
        return shape_type {
            unsigned(t.num_samples()),
            unsigned(t.k()),
            unsigned(t.nr()),
            unsigned(t.nc())
        };
    }
    
    inline auto to_shape(const flatbuffers::Vector<int32_t>& vec) {
        if (4 < vec.size())
            throw std::runtime_error("invalid tensor shape (bad size)");
        for (auto i : vec)
            if (i <= 0)
                throw std::runtime_error("invalid tensor shape (bad element)");
        shape_type s = {1,1,1,1};
        std::copy(vec.begin(), vec.end(), s.begin());
        return s;
    }

    // returns dest + src_rows * src_cols
    inline float*
    transpose(const float* src, unsigned src_rows, unsigned src_cols,
              float* dest) {
        for (auto end = src + src_cols; src != end; ++src)
            for (auto p = src, end = src + src_rows * src_cols;
                 p != end; p += src_cols, ++dest)
                *dest = *p;
        return dest;
    }

    // switch from RCK to KRC
    // do this twice for inverse
    inline auto rotate(const dlib::tensor& src) {
        dlib::resizable_tensor dest(
            src.num_samples(), src.nc(), src.k(), src.nr());
        float const* sp = src.host();
        float* dp = dest.host_write_only();
        for (auto n = src.num_samples(); n > 0; --n) {
            dp = transpose(
                sp, unsigned(src.k()*src.nr()), unsigned(src.nc()), dp);
            sp += src.k() * src.nr() * src.nc();
        }
        return dest;
    }


}
