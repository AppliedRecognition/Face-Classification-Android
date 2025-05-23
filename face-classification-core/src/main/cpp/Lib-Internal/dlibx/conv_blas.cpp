
#include "conv.hpp"
#include "conv_blas.hpp"
#include "aligned_matrix.hpp"

#include <core/parallelize.hpp>

#include <applog/core.hpp>

using namespace dlibx;

const dlib::resizable_tensor dlibx::empty_tensor; // from tensor.hpp

const dlib::tensor&
dlibx::apply_padding(const dlib::tensor& input, dlib::resizable_tensor& output,
                     int top, int left, int bottom, int right) {
    if (top < 0 || left < 0 || bottom < 0 || right < 0)
        throw std::invalid_argument("invalid padding dimensions");
    output.set_size(input.num_samples(), input.k(),
                    input.nr() + (top+bottom),
                    input.nc() + (left+right));
    const auto output_buffer = output.host_write_only();
    auto dest = output_buffer; // current position
    auto next = dest + top*output.nc() + left; // next image data
    auto src = input.host();
    for (auto n = input.num_samples(); n > 0; --n) {
        for (auto k = input.k(); k > 0; --k,
                 next += (top+bottom)*output.nc()) {
            for (auto r = input.nr(); r > 0; --r,
                     next += output.nc(), src += input.nc()) {
                std::fill(dest, next, 0.0f);
                dest = std::copy_n(src, input.nc(), next);
            }
        }
    }
    std::fill(dest, output_buffer + output.size(), 0.0f);
    return output;
}

static thread_local std::shared_ptr<dlib::resizable_tensor> padded_tensor;

static std::shared_ptr<dlib::resizable_tensor>
make_padding(const dlib::tensor& input,
             int top, int left, int bottom, int right) {
    std::shared_ptr<dlib::resizable_tensor> t;
    if (padded_tensor.use_count() > 1)
        // padded_tensor is already in use so we have to create a temporary
        t = std::make_shared<dlib::resizable_tensor>();
    else {
        if (!padded_tensor)
            padded_tensor = std::make_shared<dlib::resizable_tensor>();
        t = padded_tensor; // copy the shared_ptr so use_count() > 1
    }
    apply_padding(input, *t, top, left, bottom, right);
    return t;
}

std::shared_ptr<const dlib::tensor>
dlibx::apply_padding(const dlib::tensor& input,
                     int top, int left, int bottom, int right) {
    return make_padding(input, top, left, bottom, right);
}

static bool warn_context_not_found = false;

template <typename PTR>
static auto num_threads(PTR const* context) {
    std::string s;
    if (context) {
        if (auto n = context->num_threads()) {
            s += " (";
            s += std::to_string(n+1);
            s += " threads)";
        }
    }
    return s;
}


/****************  class forward_conv  ****************/

struct forward_conv::internal {
    int nr, nc, dy, dx, sy, sx, py, px, wy, wx;
    int filter_size, in_channels, out_channels;
    float const* filter_data;

    internal(int nr, int nc, int dy, int dx, int sy, int sx, int py, int px,
             const dlib::tensor& filters)
        : nr(nr), nc(nc), dy(dy), dx(dx), sy(sy), sx(sx), py(py), px(px),
          wy(1 + (nr-1) * dy),
          wx(1 + (nc-1) * dx),
          filter_size(int(filters.k()*filters.nr()*filters.nc())),
          in_channels(filter_size/(nr*nc)),
          out_channels(int(filters.num_samples())),
          filter_data(filters.host()) {
        if (nr < 1 || nc < 1 || dy < 1 || dx < 1 || sy < 1 || sx < 1 ||
            py < 0 || wy <= py || px < 0 || wx <= px)
            throw std::invalid_argument("invalid convolution arguments");
        if (out_channels < 1 || in_channels < 1 ||
            filter_size != in_channels*nr*nc)
            throw std::invalid_argument("invalid filters for convolution");
    }

    void pointwise(const dlib::tensor& in, dlib::resizable_tensor& out) {
        if (in_channels != in.k()) {
            throw std::invalid_argument("tensor has incorrect number of channels for pointwise convolution");
        }
        out.set_size(in.num_samples(), out_channels, in.nr(), in.nc());
        if (in.num_samples() <= 0)
            return;

        const auto channel_px = long(in.nr() * in.nc());
        const auto src_size = channel_px * in_channels;
        const auto dest_size = channel_px * out_channels;

        const auto context = core::job_context::this_context();
        const auto nthreads = context ? context->num_threads() : 0;
        if (!context && !warn_context_not_found) {
            warn_context_not_found = true;
            FILE_LOG(logWARNING) << "conv: job_context not found -- using single thread/core";
        }

        if (nthreads <= 0) {
            // not parallized
            FILE_LOG(logTRACE) << "pointwise samples: " << in.num_samples();
            matrix_view A { filter_data,
                            out_channels, filter_size,
                            filter_size, 1 };
            matrix_view B { in.host(),
                            in_channels, channel_px,
                            channel_px, 1 };
            auto dest = out.host_write_only();
            for (long n = 0; n < in.num_samples(); ++n,
                     B.data += src_size, dest += dest_size)
                multiply(A, B, dest, channel_px);
        }

        else if (in.num_samples() > 1) {
            // parallize across samples
            FILE_LOG(logTRACE) << "pointwise samples: " << in.num_samples()
                               << num_threads(context);
            parallelize(
                [A = matrix_view { filter_data,
                                   out_channels, filter_size,
                                   filter_size, 1 },
                    src = in.host(), src_size, in_k=in_channels, channel_px,
                    dest = out.host_write_only(), dest_size](long i) {
                    matrix_view B {
                        src + i*src_size, in_k, channel_px, channel_px, 1 };
                    multiply(A, B, dest + i*dest_size, channel_px);
                }, in.num_samples(), *context, nthreads);
        }

        else {
            // parallize across output channels
            FILE_LOG(logTRACE) << "pointwise channels: " << out_channels
                               << num_threads(context);
            parallelize(
                [B = matrix_view { in.host(), in_channels, channel_px,
                                   channel_px, 1 },
                    dest = out.host_write_only(), channel_px,
                    fd=filter_data, fs=filter_size](long i) {
                    matrix_view A {
                        fd + i*fs, 1, fs, fs, 1 };
                    multiply(A, B, dest + i*channel_px, channel_px);
                }, out_channels, *context, nthreads);
        }
    }

    // parallelize per sample when num_samples > 1 and padding required
    template <int s_nr, int s_nc, int s_dy, int s_dx>
    void conv_per_sample(const dlib::tensor& input, dlib::tensor& output) {

        // parallelize per sample
        struct state {
            const dlib::tensor& input;
            dlib::tensor& output;
            std::atomic<unsigned> next{0};

            const int m_nr, m_nc, m_dy, m_dx, sy, sx, py, px;

            constexpr auto nr() const { return s_nr > 0 ? s_nr : m_nr; }
            constexpr auto nc() const { return s_nc > 0 ? s_nc : m_nc; }
            constexpr auto dy() const { return s_dy > 0 ? s_dy : m_dy; }
            constexpr auto dx() const { return s_dx > 0 ? s_dx : m_dx; }

            // sliding window size
            constexpr auto wy() const { return 1 + (nr()-1) * dy(); }
            constexpr auto wx() const { return 1 + (nc()-1) * dx(); }

            float* const output_buffer;
            const long output_channel;
            const long output_sample;

            const long padded_channel;
            const long padded_sample;
            const long padded_stride;  // between rows of stride
            const long padded_step;    // between rows of window

            const long filter_size;
            const matrix_view A; // the filters

            state(float const* filter_data,
                  const dlib::tensor& input,
                  dlib::tensor& output,
                  int nr, int nc,
                  int dy, int dx,
                  int sy, int sx,
                  int py, int px)
                : input(input),
                  output(output),
                  m_nr(s_nr > 0 ? s_nr : nr),
                  m_nc(s_nc > 0 ? s_nc : nc),
                  m_dy(s_dy > 0 ? s_dy : dy),
                  m_dx(s_dx > 0 ? s_dx : dx),
                  sy(sy), sx(sx), py(py), px(px),
                  output_buffer(output.host_write_only()),
                  output_channel(long(output.nr() * output.nc())),
                  output_sample(long(output.k()) * output_channel),
                  padded_channel(long((input.nr()+2*py) * (input.nc()+2*px))),
                  padded_sample(long(input.k()) * padded_channel),
                  padded_stride(sy * long(input.nc()+2*px)),
                  padded_step(this->dy() * long(input.nc()+2*px)),
                  filter_size(long(input.k()*(this->nr()*this->nc()))),
                  A{filter_data, long(output.k()), filter_size, filter_size, 1} {
            }

            void operator()() {
                dlibx::aligned_matrix<float,64> tmp(long(output.nc()),filter_size);
                const auto Bt = matrix_view {
                    &tmp(0,0), tmp.nc(), 1, tmp.nr(), tmp.elements_per_row()
                };

                // dlib::alias_tensor is not thread safe
                const auto unpadded_sample =
                    dlib::alias_tensor(1,input.k(),input.nr(),input.nc());

                std::shared_ptr<dlib::resizable_tensor> padded;

                for (const auto end = unsigned(output.num_samples()); ; ) {
                    const auto n = next.fetch_add(1,std::memory_order_relaxed);
                    if (n >= end) break;

                    auto dest = output_buffer + n*output_sample;

                    auto&& unpadded =
                        unpadded_sample(input, n*unpadded_sample.size());
                    if (!padded)
                        padded = make_padding(unpadded, py, px, py, px);
                    else
                        apply_padding(unpadded, *padded, py, px);
                    auto src_row = padded->host();

                    for (auto l = output.nr(); l > 0; --l, dest += output.nc(),
                             src_row += padded_stride) {
                        // img2col per output image row
                        auto src = src_row;
                        for (long r = 0; r < tmp.nr(); ++r, src += sx) {
                            auto sc = src;
                            auto bp = &tmp(r,0);
                            for (auto i = input.k(); i > 0; --i,
                                     sc += padded_channel) {
                                auto sr = sc;
                                for (auto j = nr(); j > 0; --j,
                                         sr += padded_step) {
                                    if (s_dx == 1)
                                        bp = std::copy_n(sr, nc(), bp);
                                    else {
                                        auto px = sr;
                                        for (auto k = nc(); k > 0; --k, ++bp,
                                                 px += dx())
                                            *bp = *px;
                                    }
                                }
                            }
                        }
                        multiply(A, Bt, dest, output_channel);
                    }
                }
            }
        };

        const auto context = core::job_context::this_context();
        if (!context && !warn_context_not_found) {
            warn_context_not_found = true;
            FILE_LOG(logWARNING) << "conv: job_context not found -- using single thread/core";
        }

        FILE_LOG(logTRACE) << "conv samples: " << output.num_samples()
                           << num_threads(context);
        parallelize(
            state(filter_data, input, output, nr, nc, dy, dx, sy, sx, py, px),
            context);
    }

    /// returns true if num_samples > 0
    bool allocate_output(const dlib::tensor& input,
                         dlib::resizable_tensor& output) const {
        const auto padded_width = input.nc()+2*px;
        const auto padded_height = input.nr()+2*py;
        if (padded_width < wx || padded_height < wy || input.k() != in_channels)
            throw std::invalid_argument("tensor has incorrect size for convolution");
        output.set_size(input.num_samples(),
                        out_channels,
                        1 + (padded_height - wy) / sy,
                        1 + (padded_width - wx) / sx);
        return output.num_samples() > 0;
    }

    template <int s_nr, int s_nc, int s_dy, int s_dx>
    void conv_(const dlib::tensor& input, dlib::resizable_tensor& output) {
        if (!allocate_output(input, output))
            return;
        if (output.num_samples() > 1 && (py > 0 || px > 0)) {
            conv_per_sample<s_nr,s_nc,s_dy,s_dx>(input,output);
            return;
        }

        // parallelize on output image rows
        struct state {
            const dlib::tensor& input;
            dlib::tensor& output;
            std::atomic<long> next{0};

            const int m_nr, m_nc, m_dy, m_dx, sy, sx;

            constexpr auto nr() const { return s_nr > 0 ? s_nr : m_nr; }
            constexpr auto nc() const { return s_nc > 0 ? s_nc : m_nc; }
            constexpr auto dy() const { return s_dy > 0 ? s_dy : m_dy; }
            constexpr auto dx() const { return s_dx > 0 ? s_dx : m_dx; }

            // sliding window size
            constexpr auto wy() const { return 1 + (nr()-1) * dy(); }
            constexpr auto wx() const { return 1 + (nc()-1) * dx(); }

            float* const output_buffer;
            const long output_channel;
            const long output_sample;

            float const* const input_buffer;
            const long input_channel;
            const long input_sample;
            const long input_stride;  // between rows of stride
            const long input_step;    // between rows of window

            const long filter_size;
            const matrix_view A; // the filters

            state(float const* filter_data,
                  const dlib::tensor& input,
                  dlib::tensor& output,
                  int nr, int nc,
                  int dy, int dx,
                  int sy, int sx)
                : input(input),
                  output(output),
                  m_nr(s_nr > 0 ? s_nr : nr),
                  m_nc(s_nc > 0 ? s_nc : nc),
                  m_dy(s_dy > 0 ? s_dy : dy),
                  m_dx(s_dx > 0 ? s_dx : dx),
                  sy(sy), sx(sx),
                  output_buffer(output.host_write_only()),
                  output_channel(long(output.nr() * output.nc())),
                  output_sample(long(output.k()) * output_channel),
                  input_buffer(input.host()),
                  input_channel(long(input.nr() * input.nc())),
                  input_sample(long(input.k()) * input_channel),
                  input_stride(sy * long(input.nc())),
                  input_step(this->dy() * long(input.nc())),
                  filter_size(long(input.k()*(this->nr()*this->nc()))),
                  A{filter_data, long(output.k()), filter_size, filter_size, 1} {
            }

            void operator()() {
                dlibx::aligned_matrix<float,64> tmp(long(output.nc()),filter_size);
                const auto Bt = matrix_view {
                    &tmp(0,0), tmp.nc(), 1, tmp.nr(), tmp.elements_per_row()
                };

                for (const auto end = output.num_samples() * output.nr(); ; ) {
                    const auto ni = next.fetch_add(1,std::memory_order_relaxed);
                    if (ni >= end) break;

                    const auto sample_idx = long(ni / output.nr());
                    const auto row_idx = long(ni % output.nr());

                    // img2col for output image row
                    auto src = input_buffer
                        + sample_idx*input_sample
                        + row_idx*input_stride;
                    for (long r = 0; r < tmp.nr(); ++r, src += sx) {
                        auto sc = src;
                        auto bp = &tmp(r,0);
                        for (auto i = input.k(); i > 0; --i,
                                 sc += input_channel) {
                            auto sr = sc;
                            for (auto j = nr(); j > 0; --j,
                                     sr += input_step) {
                                if (s_dx == 1)
                                    bp = std::copy_n(sr, nc(), bp);
                                else {
                                    auto px = sr;
                                    for (auto k = nc(); k > 0; --k, ++bp,
                                             px += dx())
                                        *bp = *px;
                                }
                            }
                        }
                    }

                    auto dest = output_buffer
                        + sample_idx*output_sample
                        + row_idx*output.nc();
                    multiply(A, Bt, dest, output_channel);
                }
            }
        };

        const auto context = core::job_context::this_context();
        if (!context && !warn_context_not_found) {
            warn_context_not_found = true;
            FILE_LOG(logWARNING) << "conv: job_context not found -- using single thread/core";
        }

        FILE_LOG(logTRACE) << "conv rows: " << output.num_samples()*output.nr()
                           << num_threads(context);
        if (py <= 0 && px <= 0)
            parallelize(
                state(filter_data, input, output,
                      nr, nc, dy, dx, sy, sx),
                context);
        else {
            auto padded = apply_padding(input, py, px);
            parallelize(
                state(filter_data, *padded, output,
                      nr, nc, dy, dx, sy, sx),
                context);
        }
    }
};

void forward_conv::setup(int nr, int nc, int dy, int dx,
                         int sy, int sx, int py, int px,
                         const dlib::tensor& filters) {
    state = std::make_unique<internal>(nr, nc, dy, dx, sy, sx, py, px, filters);

    if (nr == 1 && nc == 1 && sy == 1 && sx == 1) {
        m = &internal::pointwise;
        return;
    }

    if (dy == dx) {
        if (nr == nc) {
            if (dy == 1) {
                switch (nr) {
                case 3: m = &internal::conv_<3,3,1,1>; return;
                case 5: m = &internal::conv_<5,5,1,1>; return;
                case 7: m = &internal::conv_<7,7,1,1>; return;
                }
            }
            else if (nr == 3) {
                switch (dy) {
                case 2: m = &internal::conv_<3,3,2,2>; return;
                case 3: m = &internal::conv_<3,3,3,3>; return;
                case 5: m = &internal::conv_<3,3,5,5>; return;
                }
            }
        }
        else { // nr != nc
            /* Others:
               con_1_3_1_1
               con_3_1_1_1
               con_1_7_1_1
               con_7_1_1_1
            */
        }
    }

    // general case
    m = &internal::conv_<0,0,0,0>;
}

forward_conv::forward_conv() = default;
forward_conv::~forward_conv() = default;
forward_conv::forward_conv(const forward_conv&) {}
forward_conv& forward_conv::operator=(const forward_conv& other) {
    if (this != &other)
        state = nullptr;
    return *this;
}
void forward_conv::reset() {
    state = nullptr;
}



/****************  class forward_convdw  ****************/

struct forward_convdw::internal {
    int nr, nc, dy, dx, sy, sx, py, px, wy, wx;
    int filter_size, out_channels;
    float const* filter_data;

    internal(int nr, int nc, int dy, int dx, int sy, int sx, int py, int px,
             const dlib::tensor& filters)
        : nr(nr), nc(nc), dy(dy), dx(dx), sy(sy), sx(sx), py(py), px(px),
          wy(1 + (nr-1) * dy),
          wx(1 + (nc-1) * dx),
          filter_size(nr*nc),
          out_channels(int(filters.size()/unsigned(nr*nc))),
          filter_data(filters.host()) {
        if (nr < 1 || nc < 1 || dy < 1 || dx < 1 || sy < 1 || sx < 1)
            throw std::invalid_argument("invalid convolution arguments");
        if (out_channels < 1 ||
            filters.size() != std::size_t(out_channels)*unsigned(filter_size))
            throw std::invalid_argument("invalid filters for convolution");
    }

    /// returns true if num_samples > 0
    bool allocate_output(const dlib::tensor& input,
                         dlib::resizable_tensor& output) const {
        const auto mult = out_channels / int(input.k());
        if (mult < 1 || out_channels != mult * input.k())
            throw std::logic_error("tensor has wrong number of channels for convolution");
        const auto padded_width = input.nc()+2*px;
        const auto padded_height = input.nr()+2*py;
        if (padded_width < wx || padded_height < wy)
            throw std::invalid_argument("tensor has incorrect size for convolution");
        output.set_size(input.num_samples(),
                        out_channels,
                        1 + (padded_height - wy) / sy,
                        1 + (padded_width - wx) / sx);
        return input.num_samples() > 0;
    }

    template <int s_nr, int s_nc, int s_dy, int s_dx>
    void conv_(const dlib::tensor& input, dlib::resizable_tensor& output) {

        if (!allocate_output(input, output))
            return;

        // parallelize on input channels
        struct state {
            float const* const filter_data;
            const dlib::tensor& input;
            dlib::tensor& output;
            const int mult;
            std::atomic<long> next{0};

            const int m_nr, m_nc, m_dy, m_dx, sy, sx;

            constexpr auto nr() const { return s_nr > 0 ? s_nr : m_nr; }
            constexpr auto nc() const { return s_nc > 0 ? s_nc : m_nc; }
            constexpr auto dy() const { return s_dy > 0 ? s_dy : m_dy; }
            constexpr auto dx() const { return s_dx > 0 ? s_dx : m_dx; }

            // sliding window size
            constexpr auto wy() const { return 1 + (nr()-1) * dy(); }
            constexpr auto wx() const { return 1 + (nc()-1) * dx(); }

            float* const output_buffer;
            const long output_channel;
            const long output_sample;

            float const* const input_buffer;
            const long input_channel;
            const long input_sample;
            const long input_step;  // between rows of window
            const long end_of_row_delta;

            state(float const* filter_data,
                  const dlib::tensor& input,
                  dlib::tensor& output,
                  int nr, int nc,
                  int dy, int dx,
                  int sy, int sx)
                : filter_data(filter_data),
                  input(input),
                  output(output),
                  mult(int(output.k() / input.k())),
                  m_nr(s_nr > 0 ? s_nr : nr),
                  m_nc(s_nc > 0 ? s_nc : nc),
                  m_dy(s_dy > 0 ? s_dy : dy),
                  m_dx(s_dx > 0 ? s_dx : dx),
                  sy(sy), sx(sx),
                  output_buffer(output.host_write_only()),
                  output_channel(long(output.nr() * output.nc())),
                  output_sample(long(output.k()) * output_channel),
                  input_buffer(input.host()),
                  input_channel(long(input.nr() * input.nc())),
                  input_sample(long(input.k()) * input_channel),
                  input_step(this->dy() * long(input.nc())),
                  end_of_row_delta(long(sy*input.nc() - sx*output.nc())) {
                assert(output.k() == mult * input.k());
            }

            void operator()() {
                dlibx::aligned_matrix<float,64> tmp(output_channel, nr()*nc());
                const auto Bt = matrix_view {
                    &tmp(0,0), tmp.nc(), 1, tmp.nr(), tmp.elements_per_row()
                };
                auto A = matrix_view { nullptr, 1, nr()*nc(), nr()*nc(), 1 };

                for (const auto end = input.num_samples() * input.k(); ; ) {
                    const auto ni = next.fetch_add(1,std::memory_order_relaxed);
                    if (ni >= end) break;

                    // img2col on single input channel
                    auto src = input_buffer + ni*input_channel;
                    long out_col = 0;
                    for (long r = 0; r < tmp.nr(); ++r, src += sx) {
                        auto sc = src;
                        auto bp = &tmp(r,0);
                        for (auto j = nr(); j > 0; --j, sc += input_step) {
                            if (s_dx == 1)
                                bp = std::copy_n(sc, nc(), bp);
                            else {
                                auto px = sc;
                                for (auto k = nc(); k > 0; --k, ++bp, px += dx())
                                    *bp = *px;
                            }
                        }
                        if (++out_col >= output.nc()) {
                            src += end_of_row_delta;
                            out_col = 0;
                        }
                    }

                    const auto no = ni * mult;
                    A.data = filter_data + nr()*nc() * (no % long(output.k()));
                    auto dest = output_buffer + no*output_channel;
                    for (auto n = mult; n > 0; --n,
                             A.data += nr()*nc(), dest += output_channel)
                        multiply(A, Bt, dest, output_channel);
                }
            }
        };

        const auto context = core::job_context::this_context();
        if (!context && !warn_context_not_found) {
            warn_context_not_found = true;
            FILE_LOG(logWARNING) << "conv: job_context not found -- using single thread/core";
        }

        FILE_LOG(logTRACE) << "convdw channels: "
                           << input.num_samples()*input.k()
                           << num_threads(context);
        if (py <= 0 && px <= 0)
            parallelize(
                state(filter_data, input, output,
                      nr, nc, dy, dx, sy, sx),
                context);
        else {
            auto padded = apply_padding(input, py, px);
            parallelize(
                state(filter_data, *padded, output,
                      nr, nc, dy, dx, sy, sx),
                context);
        }
    }
};

void forward_convdw::setup(int nr, int nc, int dy, int dx,
                           int sy, int sx, int py, int px,
                           const dlib::tensor& filters) {

    if (nr < 1 || nc < 1 || dy < 1 || dx < 1 || sy < 1 || sx < 1 ||
        py < 0 || nr <= py || px < 0 || nc <= px)
        throw std::invalid_argument("invalid convolution arguments");
    if (filters.k()*filters.nr()*filters.nc() != nr*nc)
        throw std::logic_error("tensor has wrong kernel size for convolution");

    state = std::make_unique<internal>(nr, nc, dy, dx, sy, sx, py, px, filters);
    if (nr == 3 && nc == 3 && dy == 1 && dx == 1)
        m = &internal::conv_<3,3,1,1>;
    else
        m = &internal::conv_<0,0,0,0>; // general case
}

forward_convdw::forward_convdw() = default;
forward_convdw::~forward_convdw() = default;
forward_convdw::forward_convdw(const forward_convdw&) {}
forward_convdw& forward_convdw::operator=(const forward_convdw& other) {
    if (this != &other)
        state = nullptr;
    return *this;
}
void forward_convdw::reset() {
    state = nullptr;
}
