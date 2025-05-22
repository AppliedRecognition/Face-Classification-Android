#pragma once

#include "tensor.hpp"

#include <cstring>

namespace dlibx {

    /** \brief Virtual base class for specific img2col object.
     */
    class img2col_base {
    public:
        const long filter_nr, filter_nc;
        const long window_nr, window_nc; // including dilation
        const int stride_y, stride_x;
        const int padding_y, padding_x;

        const long sample_nr, sample_nc, sample_k;
        const long sample_px;  // nc*nr

        const long out_nr, out_nc; // output tensor (after matrix multiply)
        const long mat_nr, mat_nc; // temporary matrix (before matrix multiply)

        virtual ~img2col_base() = default;

        virtual float
        operator()(const float* src, long row, float* dest) const = 0;

    protected:
        img2col_base(long sample_nr, long sample_nc, long sample_k,
                     long filter_nr, long filter_nc,
                     long window_nr, long window_nc,
                     int stride_y, int stride_x,
                     int padding_y, int padding_x)
            : filter_nr(filter_nr),
              filter_nc(filter_nc),
              window_nr(window_nr),
              window_nc(window_nc),
              stride_y(stride_y),
              stride_x(stride_x),
              padding_y(padding_y),
              padding_x(padding_x),
              sample_nr(sample_nr),
              sample_nc(sample_nc),
              sample_k(sample_k),
              sample_px(sample_nr*sample_nc),
              out_nr(1 + (sample_nr+(2*padding_y-window_nr))/stride_y),
              out_nc(1 + (sample_nc+(2*padding_x-window_nc))/stride_x),
              mat_nr(out_nr*out_nc),
              mat_nc(sample_k*filter_nr*filter_nc) {
            DLIB_CASSERT(0 < filter_nr && filter_nr <= window_nr &&
                         0 < filter_nc && filter_nc <= window_nc);
            DLIB_CASSERT(window_nr <= sample_nr + 2*padding_y &&
                         window_nc <= sample_nc + 2*padding_x,
                         "Filter window too large for padded image.");
            DLIB_CASSERT(mat_nr > 0 && mat_nc > 0);
            DLIB_CASSERT(out_nr > 0 && out_nc > 0);
            DLIB_CASSERT(stride_y > 0 && stride_x > 0);
        }
    };

    template <long nc, long stride>
    struct simple_copy {
        const float result = 0;
        inline float operator()(float x) const {
            return x;
        }
        inline void operator()(float* dest, const float* src) const {
            for (auto i = nc; i > 0; --i, ++dest, src += stride)
                *dest = *src;
        }
    };
    template <long nc>
    struct simple_copy<nc,1> {
        const float result = 0;
        inline float operator()(float x) const {
            return x;
        }
        inline void operator()(float* dest, const float* src) const {
            memcpy(dest, src, nc*sizeof(float));
        }
    };

    template <long nc, long stride>
    struct compute_maxabs {
        float result = 0;
        inline float operator()(float x) {
            result = std::max(result, std::abs(x));
            return x;
        }
        inline void operator()(float* dest, const float* src) {
            for (auto i = nc; i > 0; --i, ++dest, src += stride)
                result = std::max(result, std::abs(*dest = *src));
        }
    };

    template <long _filter_nr, long _filter_nc,
              int _dilate_y, int _dilate_x,
              int _padding_y, int _padding_x,
              template<long,long> class copy_method = simple_copy>
    class img2col : public img2col_base {

        static_assert(_filter_nr > 0 && _filter_nc > 0, "Invalid filter.");
        static_assert(_dilate_y > 0 && _dilate_x > 0, "Invalid dilation.");
        static_assert(_padding_y >= 0 && _padding_x >= 0, "Invalid padding.");

        static constexpr auto _window_nr = 1 + (_filter_nr-1) * _dilate_y;
        static constexpr auto _window_nc = 1 + (_filter_nc-1) * _dilate_x;

    public:
        img2col(int stride_y, int stride_x,
                const dlib::tensor& data, long data_k)
            : img2col_base(long(data.nr()), long(data.nc()), data_k,
                           _filter_nr, _filter_nc,
                           _window_nr, _window_nc,
                           stride_y, stride_x,
                           _padding_y, _padding_x) {
        }
        img2col(int stride_y, int stride_x,
                const dlib::tensor& data)
            : img2col_base(long(data.nr()), long(data.nc()), long(data.k()),
                           _filter_nr, _filter_nc,
                           _window_nr, _window_nc,
                           stride_y, stride_x,
                           _padding_y, _padding_x) {
        }

        float operator()(
            const float* src, long row, float* dest) const final override {

            DLIB_CASSERT(0 <= row && row < mat_nr);
            const auto last = dest + mat_nc;

            const auto r = (row / out_nc) * stride_y - _padding_y;
            const auto c = (row % out_nc) * stride_x - _padding_x;
            src += r*sample_nc + c;
            const auto row_incr = sample_nc * _dilate_y;

            copy_method<_filter_nc,_dilate_x> copy;

            if (_padding_y == 0 ||
                (0 <= r && r + _window_nr <= sample_nr)) {

                // no padding top or bottom
                if (_padding_x == 0 ||
                    (0 <= c && c + _window_nc <= sample_nc)) {
                    // no padding anywhere -- fast path
                    for (long k = 0; k < sample_k; ++k) {
                        auto row = src;
                        for (long y = 0; y < _filter_nr; ++y,
                                 row += row_incr, dest += _filter_nc)
                            copy(dest, row);
                        src += sample_px;
                    }
                }

                else {
                    // pad left or right
                    const auto unc = unsigned(sample_nc);
                    // note: if j < 0, then unsigned(j) > unc
                    for (long k = 0; k < sample_k; ++k) {
                        auto row = src;
                        for (long y = 0; y < _filter_nr; ++y,
                                 row += row_incr) {
                            auto px = row;
                            for (long x = 0, j = c; x < _filter_nc; ++x,
                                     ++j, px += _dilate_x, ++dest)
                                *dest = unsigned(j) < unc ? copy(*px) : 0;
                        }
                        src += sample_px;
                    }
                }
            }

            else {
                // pad top or bottom
                const auto unr = unsigned(sample_nr);
                // note: if i < 0, then unsigned(i) > unr
                if (_padding_x == 0 ||
                    (0 <= c && c + _window_nc <= sample_nc)) {
                    // no padding left or right
                    for (long k = 0; k < sample_k; ++k) {
                        auto row = src;
                        for (long y = 0, i = r; y < _filter_nr; ++y, ++i) {
                            if (unsigned(i) < unr)
                                copy(dest, row);
                            else
                                memset(dest, 0, _filter_nc*sizeof(float));
                            dest += _filter_nc;
                            row += row_incr;
                        }
                        src += sample_px;
                    }
                }

                else {
                    // pad both rows and columns
                    const auto unc = unsigned(sample_nc);
                    // note: if j < 0, then unsigned(j) > unc
                    for (long k = 0; k < sample_k; ++k) {
                        auto row = src;
                        for (long y = 0, i = r; y < _filter_nr; ++y, ++i) {
                            if (unsigned(i) < unr) {
                                auto px = row;
                                for (long x = 0, j = c; x < _filter_nc; ++x,
                                         ++j, px += _dilate_x, ++dest)
                                    *dest = unsigned(j) < unc ? copy(*px) : 0;
                            }
                            else  {
                                memset(dest, 0, _filter_nc*sizeof(float));
                                dest += _filter_nc;
                            }
                            row += row_incr;
                        }
                        src += sample_px;
                    }
                }
            }

            DLIB_CASSERT(last == dest);
            return copy.result;
        }
    };
}
