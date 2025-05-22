#pragma once

#include <type_traits>
#include <vector>
#include <raw_image/core.hpp>

namespace {
    // note: dest must be src_width+1 by src_height+1
    template <typename U, typename V>
    void integral(U* dest, unsigned dest_els_per_line,
                  const V* src, unsigned src_els_per_line,
                  unsigned src_width, unsigned src_height) {
        for (auto p = dest, end = dest + src_width; p <= end; ++p)
            *p = U();
        for (auto j = src_height; j > 0; --j) {
            auto dest0 = dest;
            auto dest1 = dest += dest_els_per_line;
            *dest1 = *dest0;
            auto src0 = src;
            src += src_els_per_line;
            U x = U();
            for (auto i = src_width; i > 0; --i) {
                x += *src0;
                ++src0;
                ++dest0;
                ++dest1;
                *dest1 = *dest0 + x;
            }
        }
    }

    template <typename T>
    struct integral_image {
        std::vector<T> sum;
        unsigned rows = 0, cols = 0;
        std::vector<unsigned> stride_table;

        bool set_image(const raw_image::plane& image,
                       unsigned window_height) {
            if (image.width <= 0 || image.height <= 0 ||
                bytes_per_pixel(image.layout) != 1) {
                cols = rows = 0;
                return false;
            }
            rows = image.height + 1;
            cols = image.width + 1;
            sum.resize(size_t(rows)*size_t(cols)+4);
            integral(sum.data(), cols,
                     image.data, image.bytes_per_line,
                     image.width, image.height);
            stride_table.clear();
            for (unsigned i = 1; i <= window_height; ++i)
                stride_table.push_back(i*cols);
            return true;
        }
    };
    
    struct lbp_generic {
        struct empty {};

        static inline empty init() { return empty(); }

        template <typename T>
        struct inter {
            const T* pos;
            unsigned stride;
            unsigned width;
        };

        struct index_pair {
            unsigned i,j;
        };

        template <unsigned WIDTH, typename T>
        static inline inter<T> load(const T* pos, unsigned stride) {
            return {pos,stride,WIDTH};
        }
        template <typename T>
        static inline
        inter<T> load(const T* pos, unsigned stride, unsigned width) {
            return {pos,stride,width};
        }

        template <typename T>
        static inline index_pair calc(inter<T>& in, empty) {
            // always do calc in 32-bits and truncate to 16-bit later if needed
            using U = typename std::conditional<std::is_signed<T>::value,int,unsigned>::type;
            U cval;
            U x[8];

            // integral (p):
            //  0 1 2 3
            //  4 5 6 7
            //  8 9 a b
            //  c d e f
            // x array:
            //   0 1 2
            //   7 c 3
            //   6 5 4
            // final bits in reverse order of x array

            // p[0]
            auto line = in.pos;
            x[0] = *line;

            // p[1]
            line += in.width;
            x[0] -= *line;
            x[1] = *line;

            // p[2]
            line += in.width;
            x[1] -= *line;
            x[2] = *line;

            // p[3]
            line += in.width;
            x[2] -= *line;

            // p[4]
            line = in.pos += in.stride;
            x[0] -= *line;
            x[7] = *line;

            // p[5]
            line += in.width;
            cval = *line;
            x[0] += cval;
            x[1] -= cval;
            x[7] -= cval;

            // p[6]
            line += in.width;
            x[3] = *line;
            cval -= x[3];
            x[1] += x[3];
            x[2] -= x[3];

            // p[7]
            line += in.width;
            x[2] += *line;
            x[3] -= *line;

            // p[8]
            line = in.pos += in.stride;
            x[6] = *line;
            x[7] -= *line;

            // p[9]
            line += in.width;
            x[5] = *line;
            cval -= x[5];
            x[6] -= x[5];
            x[7] += x[5];

            // p[10]
            line += in.width;
            x[4] = *line;
            cval += x[4];
            x[3] -= x[4];
            x[5] -= x[4];

            // p[11]
            line += in.width;
            x[3] += *line;
            x[4] -= *line;

            // p[12]
            line = in.pos += in.stride;
            x[6] -= *line;
            
            // p[13]
            line += in.width;
            x[5] -= *line;
            x[6] += *line;

            // p[14]
            line += in.width;
            x[4] -= *line;
            x[5] += *line;

            // p[15]
            line += in.width;
            x[4] += *line;

            //unsigned r = 0;
            //for (unsigned i = 0; i < 8; ++i)
            //    r = (r<<1) | (x[i] >= cval ? 1 : 0);

            const T c = cval;
            unsigned i = T(x[0]) >= c ? 1 : 0;
            i = (i<<1) | (T(x[1]) >= c ? 1 : 0);
            i = (i<<1) | (T(x[2]) >= c ? 1 : 0);

            unsigned j = T(x[3]) >= c ? 1 : 0;
            j = (j<<1) | (T(x[4]) >= c ? 1 : 0);
            j = (j<<1) | (T(x[5]) >= c ? 1 : 0);
            j = (j<<1) | (T(x[6]) >= c ? 1 : 0);
            j = (j<<1) | (T(x[7]) >= c ? 1 : 0);

            return {i,j};
        }

        static inline
        unsigned get_top(const index_pair& p) {
            return p.i;
        }
        static inline
        unsigned get_bottom(const index_pair& p) {
            return p.j;
        }
    };
}
