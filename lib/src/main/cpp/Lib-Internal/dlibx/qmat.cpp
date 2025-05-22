
#include "qmat.hpp"
#include "bfloat16.hpp"

#include <chrono>
#include <list>
#include <numeric>
#include <stdext/rounding.hpp>
#include <stdext/bit.hpp>

#include <core/context.hpp>
#include <core/thread_data.hpp>
#include <core/job_queue.hpp>

#include <applog/core.hpp>

#include "matrix_ops.hpp"
#include "library_init.hpp"
#include "atomic_counter.hpp"

using namespace dlibx;

namespace dlibx {
    namespace ops {
        const machine_detail machine = machine_detail::detect();
    }
}

qmat::qmat(const qmat& other)
    : data(other.data),
      nrows(other.nrows),
      ncols(other.ncols),
      row_coeff(&data(nrows,0)),
      m_rhs_limit(other.m_rhs_limit),
      bytes_per_value(other.bytes_per_value) {
}

qmat& qmat::operator=(const qmat& other) {
    if (this != &other) {
        data = other.data;
        nrows = other.nrows;
        ncols = other.ncols;
        row_coeff = &data(nrows,0);
        m_rhs_limit = other.m_rhs_limit;
        bytes_per_value = bytes_per_value;
    }
    return *this;
}

void qmat::set_size(long rows, long cols) {
    if (rows > 0 && cols > 0) {
        // number of float values to fit the requested integer values
        const auto float_cols =
            long(1 + (std::size_t(cols)*bytes_per_value - 1) / sizeof(float));
        // number of extra rows needed to fit the per-row coefficients
        const auto xrows = 1 + (rows - 1) / float_cols;
        data.set_size(rows + xrows, float_cols);
        nrows = rows;
        ncols = cols;
        row_coeff = &data(nrows,0);
        std::fill(row_coeff, &data(data.nr(),0), 0); // zero coefficients
    }
    else nrows = 0, ncols = 0, row_coeff = nullptr;
}

template <typename T>
int qmat_t<T>::calc_rhs_limit() const {
    // compute rhs_limit such that int32 overflow is not possible
    int limit = std::numeric_limits<value_type>::max();
    for (long r = 0; r < nr(); ++r) {
        int sum = 0;
        for (auto p = ptr(r), end = p + nc(); p != end; ++p)
            sum += std::abs(*p);
        if (0 < sum) {
            // want sum * rhs_limit <= i32max
            const auto z = std::numeric_limits<int32_t>::max() / sum;
            limit = std::min(limit, z);
        }
    }
    if (limit < 100)
        FILE_LOG(logWARNING) << "qmat: low rhs_limit " << limit;
    return limit;
}

template <typename T>
int qmat_t<T>::assign_lhs(const dlib::matrix<float>& lhs, int bits) {

    const auto nc = lhs.nc();
    if (nc < 2)
        throw std::invalid_argument(
            "qmat: single column matrix not supported");
    set_size(lhs.nr(), nc);

    if (bits < 4)
        bits = 4;
    else if (unsigned(bits) > 8*sizeof(value_type))
        bits = 8*sizeof(value_type);
    const auto lhs_max = value_type((1<<(bits-1))-1);
    const auto lhs_min = value_type(-lhs_max-1);

    // least +ve bfloat16
    const auto minbf16 = stdx::bit_cast<float>(0x00010000u);

    // error in parts per 1000
    int e1000_min = 0, e1000_max = 0;

    for (long r = 0; r < lhs.nr(); ++r) {
        float vmax = 0;
        double mag = 0;
        for (long c = 0; c < nc; ++c) {
            const auto x = lhs(r,c);
            vmax = std::max(vmax, std::abs(x));
            mag += double(x)*double(x);
        }
        auto c = vmax / lhs_max;
        truncate_to_bfloat16(&c, 1);
        if (c < minbf16)
            c = minbf16;
        coeff(r) = c;

        // target sum of squares of values
        const auto target = stdx::round_to<int64_t>(mag/c/c);
        if (target <= 0) {
            FILE_LOG(logWARNING) << "qmat::assign_lhs() all zero row";
            std::fill(ptr(r), ptr(r+1), 0);
        }
        else {
            const auto quantize =
                [&](float base) {
                    int64_t mag = 0;
                    for (long c = 0; c < nc; ++c) {
                        auto x = stdx::round_to<value_type>(lhs(r,c) / base);
                        if (x < lhs_min)
                            x = lhs_min;
                        else if (x > lhs_max)
                            x = lhs_max;
                        value(r,c) = x;
                        mag += x * x;
                    }
                    return mag;
                };
            unsigned i = 0;
            int64_t err = 0;
            for (float lo = c/2, hi = c*2; ; ++i) {
                const float mid = (lo + hi) / 2;
                if (lo < mid && mid < hi) {
                    err = quantize(mid) - target;
                    if (err < 0)
                        hi = mid;
                    else if (err > 0)
                        lo = mid;
                    else break;
                }
                else break;
            }
            const auto e1000 = int((1000*err + target/2) / target);
            e1000_min = std::min(e1000_min, e1000);
            e1000_max = std::max(e1000_max, e1000);
            std::fill(&value(r,nc), &value(r+1,0), 0);
        }
    }

    if (std::max(e1000_max,-e1000_min) > std::max(0, 10 - bits)) {
        if (e1000_min < -e1000_max)
            FILE_LOG(logINFO) << "qmat::quantize_lhs() error "
                              << e1000_min << "/1000";
        else if (e1000_max > -e1000_min)
            FILE_LOG(logINFO) << "qmat::quantize_lhs() error +"
                              << e1000_max << "/1000";
        else if (e1000_max > 0)  // e1000_min == -e1000_max
            FILE_LOG(logINFO) << "qmat::quantize_lhs() error "
                              << e1000_max << "/1000";
    }

    return m_rhs_limit = calc_rhs_limit();
}

template <typename T>
inline void qmat_t<T>::quantize_row(
    long r, const float* src, int limit, float vmax) {
    if (vmax > 0) {
        const auto c = vmax / float(limit);
        coeff(r) = c;
        ops::multiply_and_round(ptr(r), src, unsigned(nc()), 1/c);
    }
    else
        coeff(r) = 0;
}

template <typename T>
void qmat_t<T>::assign_rhs(const dlib::matrix<float>& mf, int limit) {
    DLIB_CASSERT(limit > 0);
    if (limit > std::numeric_limits<value_type>::max())
        limit = std::numeric_limits<value_type>::max();
    assert(mf.nr() > 0 && mf.nc() > 0);
    set_size(mf.nr(), mf.nc());
    m_rhs_limit = 0; // only lhs has this value
    dlibx::aligned_matrix<float,64> buf(1,mf.nc());
    std::fill(&buf(0,mf.nc()), &buf(1,0), 0);
    auto src = &mf(0,0);
    const auto src_stride = &mf(1,0) - &mf(0,0);
    for (long r = 0; r < mf.nr(); ++r, src += src_stride) {
        float vmax = 0;
        std::transform(
            src, src + mf.nc(), &buf(0,0),
            [&](auto x) { return vmax = std::max(vmax, std::abs(x)), x; });
        quantize_row(r, &buf(0,0), limit, vmax);
    }
}

template <typename T>
void qmat_t<T>::img2col(int limit, const img2col_base& gen,
                        const dlib::tensor& input, long n) {

    const auto context = core::job_context::this_context();
    const auto nthreads = context ? long(context->num_threads()) : 0;

    struct state {
        const img2col_base& gen;
        int limit;
        const float* const src;
        qmat_t& dest;
    };

    struct job : atomic_counter<int> {
        using I = atomic_counter::value_type;

        state& s;
        float* const buf;

        job(state& s, float* buf, long limit, long n, long i = 0,
            atomic_counter* link = nullptr)
            : atomic_counter(link,I(i*limit/n),I((i+1)*limit/n)),
              s(s), buf(buf) {
        }

        int operator()() {
            for (atomic_counter* counter = this; ; ) {
                const auto r = counter->next();
                if (r < counter->limit)
                    s.dest.quantize_row(r, buf, s.limit, s.gen(s.src, r, buf));
                else if ((counter = counter->link) == this)
                    break;
            }
            return 0;
        }
    };

    const auto src = input.host() + n*input.k()*input.nr()*input.nc();
    set_size(gen.mat_nr, gen.mat_nc);
    m_rhs_limit = 0; // only lhs has this value
    state s{gen,limit,src,*this};

    dlibx::aligned_matrix<float,64> buf(1+nthreads, nc());

    job this_thread{s,&buf(0,0),nr(),1+nthreads};
    auto prev = &this_thread;
    std::list<core::job_function<job> > jobs, done;
    for (auto n = nthreads; n > 0; --n) {
        jobs.emplace_back(s,&buf(n,0),nr(),1+nthreads,n,prev);
        prev = &jobs.back().fn;
    }
    this_thread.link = prev;
    for (auto& job : jobs)
        context->submit_absolute(core::job_queue::order_min, job);
    this_thread();
    while (!jobs.empty())
        done.splice(done.end(), jobs,
                    context->wait_for_one(jobs.begin(), jobs.end()));
    for (auto& job : done)
        *job; // re-throw any exceptions
}

template <typename T>
unsigned qmat_t<T>::serialize_bits() const {
    auto bits = 4u;
    for (long r = 0; r < nr(); ++r)
        bits = std::max(bits, bits_required(ptr(r), std::size_t(nc())));
    return bits;
}

template <typename T>
void qmat_t<T>::serialize(std::ostream& out) const {
    using error = dlib::serialization_error;
    using dlib::serialize;

    if (m_rhs_limit <= 0)
        throw error("Only lhs qmat may be serialized.");

    serialize("qmat", out);

    const auto bits = serialize_bits();
    serialize(bits, out);

    // dimensions
    serialize(nr(), out);
    serialize(nc(), out);

    // coefficients
    serialize(bfloat16_const_span(row_coeff,std::size_t(nr())), out);

    // values
    bits_writer writer(out, bits);
    for (long r = 0; r < nr(); ++r) {
        if (!writer)
            throw error("Stream error while serializing qmat.");
        for (auto it = ptr(r), end = it + nc(); it != end; ++it)
            writer(*it);
    }
    if (!writer.flush())
        throw error("Stream error while serializing qmat.");
}

template <typename T>
void qmat_t<T>::deserialize(std::istream& in, unsigned bits) {
    using error = dlib::serialization_error;
    using dlib::deserialize;

    // dimensions
    long nr, nc;
    deserialize(nr, in);
    deserialize(nc, in);
    if (nr < 0 || nc < 0)
        throw error("Negative dimensions found while deserializing qmat.");
    set_size(nr, nc);

    // coefficients
    deserialize(bfloat16_span(row_coeff,std::size_t(nr)), in);

    // compute this limit while reading values
    m_rhs_limit = std::numeric_limits<value_type>::max();

    // values
    bits_reader reader(in, bits);
    for (long r = 0; r < nr; ++r) {
        int sum = 0;
        for (auto it = ptr(r), end = it + nc; it != end; ++it) {
            *it = reader.get<value_type>();
            sum += std::abs(*it);
        }
        if (0 < sum) {
            // want sum * rhs_limit <= i32max
            const auto z = std::numeric_limits<int32_t>::max() / sum;
            m_rhs_limit = std::min(m_rhs_limit, z);
        }
        std::fill(&value(r,nc), &value(r+1,0), 0); // zero end of row padding
        if (!reader)
            throw error("Error reading value while deserializing qmat.");
    }
}

template <typename value_type, typename ITER>
static void deserialize_raw_seq(ITER dest, std::size_t n, std::istream& in) {
    static const dlib::byte_orderer bo;
    for (auto sbuf = in.rdbuf(); n > 0; --n, ++dest) {
        value_type d;
        if (sbuf->sgetn(reinterpret_cast<char*>(&d), sizeof(d)) != std::streamsize(sizeof(d))) {
            in.setstate(std::ios::badbit);
            throw dlib::serialization_error("Error reading data while deserializing qmat.");
        }
        bo.little_to_host(d);
        *dest = d;
    }
}

/// old format: 16-bit only -- todo remove someday
template <>
void qmat_t<int16_t>::deserialize_1(std::istream& in) {
    using error = dlib::serialization_error;

    long nr, nc;
    dlib::deserialize(nr, in);
    dlib::deserialize(nc, in);
    if (nr < 0 || nc < 0)
        throw error("Negative dimensions found while deserializing qmat.");
    set_size(nr, nc);

    static_assert(sizeof(float) == 4, "float must be 4 bytes");
    deserialize_raw_seq<float>(row_coeff, std::size_t(nr), in);

    for (long r = 0; r < nr; ++r) {
        deserialize_raw_seq<value_type>(ptr(r), std::size_t(nc), in);
        std::fill(&value(r,nc), &value(r+1,0), 0); // zero end of row padding
    }
    m_rhs_limit = calc_rhs_limit();
}

/// old format: 16-bit only -- todo remove someday
template <>
void qmat_t<int16_t>::deserialize_2(std::istream& in) {
    using error = dlib::serialization_error;

    long nr, nc;
    dlib::deserialize(nr, in);
    dlib::deserialize(nc, in);
    if (nr < 0 || nc < 0)
        throw error("Negative dimensions found while deserializing qmat.");
    set_size(nr, nc);

    static_assert(sizeof(float) == 4, "float must be 4 bytes");
    deserialize_raw_seq<float>(&coeff(0), std::size_t(nr), in);

    std::streambuf* sbuf = in.rdbuf();
    for (long r = 0; r < nr; ++r) {
        for (auto it = ptr(r), end = it + nc; it != end; ++it) {
            auto y = sbuf->sbumpc();
            if (y == EOF) {
                in.setstate(std::ios::badbit);
                throw error("Error reading data while deserializing qmat (EOF).");
            }
            if (y & 128) {
                // two byte encoding of value >= 128
                auto z = sbuf->sbumpc();
                if (z == EOF) {
                    in.setstate(std::ios::badbit);
                    throw error("Error reading data while deserializing qmat (EOF).");
                }
                y = 128 + (((z<<7) + (y&127)) & 32767);
            }
            assert(y >= 0 && y < 32768 + 128);
            // convert unsigned -> signed
            // odd becomes negative while even stays positive
            if (y&1) y = -y-1;
            *it = static_cast<value_type>(y/2);
        }
        std::fill(&value(r,nc), &value(r+1,0), 0); // zero end of row padding
    }
    m_rhs_limit = calc_rhs_limit();
}

std::shared_ptr<qmat> qmat::deserialize_shared(std::istream& in) {
    using error = dlib::serialization_error;

    std::string version;
    dlib::deserialize(version, in);
    if (version == "qmat") {
        unsigned bits = 0;
        dlib::deserialize(bits, in);
        if (!(4 <= bits && bits <= 16))
            throw error("Invalid number of bits while deserializing qmat.");
        if (bits <= ops::machine.max_8bit_bits) {
            auto r = std::make_shared<qmat_t<int8_t> >();
            r->deserialize(in, bits);
            return r;
        }
        else {
            auto r = std::make_shared<qmat_t<int16_t> >();
            r->deserialize(in, bits);
            // for compatibility with machines that use 8-bit calc
            // reduce rhs_limit if bits <= 8
            if (bits <= 8)
                r->reduce_rhs_limit(127);
            return r;
        }
    }
    else if (version == "q16_1") {
        auto r = std::make_shared<qmat_t<int16_t> >();
        r->deserialize_1(in);
        return r;
    }
    else if (version == "q16_2") {
        auto r = std::make_shared<qmat_t<int16_t> >();
        r->deserialize_2(in);
        return r;
    }
    throw dlib::serialization_error("Unexpected version '"+version+"' found while deserializing qmat.");
}

template <typename T>
static void mult_block(float* dest, long els_per_row,
                       const qmat_t<T>& lhs, long r_start, long r_end,
                       const qmat_t<T>& rhs, long c_start, long c_end) {
    auto lhs_coeff = &lhs.coeff(r_start);
    const auto rhs_coeff = &rhs.coeff(c_start);
    const auto rhs_value = rhs.ptr(c_start);
    const auto rhs_stride = unsigned(rhs.ptr(1) - rhs.ptr(0));
    const auto n = unsigned(c_end - c_start);
    dest += r_start*els_per_row + c_start;
    for (auto r = r_start; r < r_end; ++r, ++lhs_coeff, dest += els_per_row)
        ops::mult_row(dest,
                      *lhs_coeff, lhs.ptr(r), unsigned(lhs.nc()),
                      rhs_coeff, rhs_value, rhs_stride, n);
}

namespace {
    template <typename T, long BSIZE>
    struct mult_row_major {
        float* const dest;
        const long els_per_row;
        const qmat_t<T>& lhs;
        const qmat_t<T>& rhs;
        std::atomic<long>& next;
        const long mod;

        int operator()() {
            for (;;) {
                const auto i = next.fetch_add(1, std::memory_order_relaxed);
                const auto r_first = (i/mod)*BSIZE;
                if (r_first >= lhs.nr()) break;
                const auto c_first = (i%mod)*BSIZE;
                const auto r_last = std::min(r_first + BSIZE, lhs.nr());
                const auto c_last = std::min(c_first + BSIZE, rhs.nr());
                mult_block(dest, els_per_row,
                           lhs, r_first, r_last,
                           rhs, c_first, c_last);
            }
            return 0;
        }
    };

    template <typename T, long BSIZE>
    struct mult_col_major {
        float* const dest;
        const long els_per_row;
        const qmat_t<T>& lhs;
        const qmat_t<T>& rhs;
        std::atomic<long>& next;
        const long mod;

        int operator()() {
            for (;;) {
                const auto i = next.fetch_add(1, std::memory_order_relaxed);
                const auto c_first = (i/mod)*BSIZE;
                if (c_first >= rhs.nr()) break;
                const auto r_first = (i%mod)*BSIZE;
                const auto c_last = std::min(c_first + BSIZE, rhs.nr());
                const auto r_last = std::min(r_first + BSIZE, lhs.nr());
                mult_block(dest, els_per_row,
                           lhs, r_first, r_last,
                           rhs, c_first, c_last);
            }
            return 0;
        }
    };
}

template <typename FN, typename T>
static inline void mult_run(float* dest, long els_per_row,
                            const qmat_t<T>& lhs,
                            const qmat_t<T>& rhs,
                            long mod,
                            core::job_context* context) {
    std::atomic<long> next{0};
    std::list<core::job_function<FN> > jobs, done;
    for (auto n = context ? context->num_threads() : 0; n > 0; --n) {
        jobs.emplace_back(dest,els_per_row,lhs,rhs,next,mod);
        context->submit_absolute(core::job_queue::order_min,jobs.back());
    }
    FN{dest,els_per_row,lhs,rhs,next,mod}(); // this thread
    while (!jobs.empty())
        done.splice(done.end(), jobs,
                    context->wait_for_one(jobs.begin(), jobs.end()));
    for (auto& job : done)
        *job; // re-throw any exceptions
}

template <typename T>
dlibx::aligned_matrix<float,64>&
qmat_t<T>::mult_transpose_rhs(const qmat_t& rhs, dlibx::aligned_matrix<float,64>& prod) const {
    if (nr() > 0 && rhs.nr() > 0) {
        assert(nc() > 0 && nc() == rhs.nc());
        prod.set_size(nr(), rhs.nr());
        assert((63 & std::size_t(&prod(0,0))) == 0);  // aligned to 64 bytes

        const auto context = core::job_context::this_context();
        
        static constexpr auto BSIZE = 16;  // block size (rows and cols)

        if (prod.nr() >= prod.nc()) {
            const auto mod = (prod.nc()+(BSIZE-1)) / BSIZE;
            mult_run<mult_row_major<T,BSIZE> >(
                &prod(0,0),prod.elements_per_row(),*this,rhs,mod,context);
        }
        else {
            const auto mod = (prod.nr()+(BSIZE-1)) / BSIZE;
            mult_run<mult_col_major<T,BSIZE> >(
                &prod(0,0),prod.elements_per_row(),*this,rhs,mod,context);
        }
    }
    return prod;
}

static bool warn_context_not_found = false;

template <typename T>
void qmat_t<T>::fc(const dlib::tensor& input,
                   dlib::resizable_tensor& output) const {

    const auto context = core::job_context::this_context();
    const auto nthreads = context ? context->num_threads() : 0;
    if (!context && !warn_context_not_found) {
        warn_context_not_found = true;
        FILE_LOG(logWARNING) << "qmat: job_context not found -- using single thread/core";
    }

    if (input.num_samples() <= 0) return;
    DLIB_CASSERT(nc() == input.k()*input.nr()*input.nc());

    struct quantize_job {
        const dlib::tensor& input;
        qmat_t& rhs;
        const int limit;
        std::atomic<long>& next;

        auto operator()() {
            dlibx::aligned_matrix<float,64> buf(1,rhs.nc());
            for (;;) {
                const auto r = next.fetch_add(1, std::memory_order_relaxed);
                if (r >= input.num_samples()) break;
                auto src = input.host() + r*rhs.nc();
                float vmax = 0;
                std::transform(
                    src, src + rhs.nc(), &buf(0,0),
                    [&](auto x) {
                        return vmax = std::max(vmax, std::abs(x)), x;
                    });
                rhs.quantize_row(r, &buf(0,0), limit, vmax);
            }
            return 0;
        }
    };

    qmat_t rhs;
    rhs.set_size(long(input.num_samples()), nc());

    // quantize rhs first
    {
        std::atomic<long> next{0};
        std::vector<core::job_function<quantize_job> > jobs;
        if (auto n = std::min(nthreads, std::size_t(input.num_samples()-1)))
            for (jobs.reserve(n); n > 0; --n) {
                jobs.emplace_back(input,rhs,rhs_limit(),next);
                context->submit_absolute(core::job_queue::order_min,jobs.back());
            }
        quantize_job{input,rhs,rhs_limit(),next}(); // this thread
        if (!jobs.empty()) {
            FILE_LOG(logDETAIL) << "qmat::fc wait for quantize";
            context->wait_for_all(jobs.begin(), jobs.end());
            for (auto& job : jobs)
                *job; // re-throw any exceptions
        }
    }

    // for the calculation we need the output buffer to potentially be
    // a little larger than num_samples() * nr()
    // so we allocate it with the number of values per sample rounded up
    // to a multiple of 4
    // then restructure tensor to remove excess and resize when done
    const auto els_per_sample = 1 + ((nr() - 1) | 3);
    output.set_size(input.num_samples(), els_per_sample);

    struct mult_job {
        const qmat_t& lhs;
        const qmat_t& rhs;
        float* const dest;
        const long els_per_row;
        std::atomic<long>& next;

        int operator()() {
            const auto lhs_stride = unsigned(lhs.ptr(1) - lhs.ptr(0));
            const auto rhs_stride = unsigned(rhs.ptr(1) - rhs.ptr(0));
            for (;;) {
                const auto i = 4*next.fetch_add(1, std::memory_order_relaxed);
                if (i >= lhs.nr()) break;
                // for each sample produce 4 output values (from 4 lhs rows)
                const auto lhs_coeff = &lhs.coeff(i);
                const auto lhs_value = lhs.ptr(i);
                auto rhs_coeff = &rhs.coeff(0);
                auto rhs_value = rhs.ptr(0);
                auto dp = dest + i;
                for (long j = rhs.nr(); j > 0; --j,
                         ++rhs_coeff, rhs_value += rhs_stride,
                         dp += els_per_row)
                    // output is transposed so swap lhs and rhs in this call
                    ops::mult_row(
                        dp,
                        *rhs_coeff, rhs_value, unsigned(rhs.nc()),
                        lhs_coeff, lhs_value, lhs_stride, 4);
            }
            return 0;
        }
    };

    {
        std::atomic<long> next{0};
        std::vector<core::job_function<mult_job> > jobs;
        jobs.reserve(nthreads);
        for (auto n = nthreads; n > 0; --n) {
            jobs.emplace_back(*this,rhs,output.host(),els_per_sample,next);
            context->submit_absolute(core::job_queue::order_min,jobs.back());
        }
        mult_job{*this,rhs,output.host(),els_per_sample,next}(); // this thread
        if (!jobs.empty()) {
            FILE_LOG(logDETAIL) << "qmat::fc wait for multiply";
            context->wait_for_all(jobs.begin(), jobs.end());
            for (auto& job : jobs)
                *job; // re-throw any exceptions
        }
    }

    // pack results and resize output
    if (output.num_samples() > 1 && els_per_sample != nr()) {
        auto dest = output.host(), src = dest;
        auto n = output.num_samples() - 1;
        do {
            dest += nr(), src += els_per_sample;
            std::memmove(dest, src, size_t(nr())*sizeof(float));
        } while (--n > 0);
    }
    output.set_size(output.num_samples(), nr());
}


template <typename T>
void qmat_t<T>::conv1x1(const dlib::tensor& input,
                        dlib::resizable_tensor& output) const {

    if (m_rhs_limit <= 0)
        throw std::logic_error("qmat::conv() called on matrix not setup as LHS");
    if (nc() != input.k())
        throw std::logic_error("tensor has wrong number of channels for convolution");
    if (input.num_samples() <= 0) return;

    output.set_size(input.num_samples(),
                    nr(),  // numfilters
                    input.nr(),
                    input.nc());

    const auto context = core::job_context::this_context();
    const auto nthreads = context ? context->num_threads() : 0;
    if (!context && !warn_context_not_found) {
        warn_context_not_found = true;
        FILE_LOG(logWARNING) << "qmat: job_context not found -- using single thread/core";
    }

    // parallelize on blocks of block_size sequential pixels
    const auto block_size =
        [&]() {
            const auto num_px = long(input.nr() * input.nc());
            const auto x =
                long(num_px * input.num_samples() / long(1+2*nthreads));
            if (x < 1) return 1l;
            if (num_px <= x) return num_px;  // parallelize on whole samples
            const auto bps = (num_px + x - 1) / x;
            return std::max(1l, (num_px + bps - 1) / bps);
        }();

    struct state {
        const qmat_t& lhs;
        const dlib::tensor& input;
        float* const output_buffer;
        const long block_size;
        std::atomic<long> next{0};

        float const* const input_buffer;
        const long channel_size;
        const long input_sample;
        const long output_sample;
        const long blocks_per_sample;

        state(const qmat_t& lhs,
              const dlib::tensor& input,
              float* const output_buffer,
              long block_size)
            : lhs(lhs),
              input(input),
              output_buffer(output_buffer),
              block_size(block_size),
              input_buffer(input.host()),
              channel_size(long(input.nr() * input.nc())),
              input_sample(long(input.k()) * channel_size),
              output_sample(lhs.nr() * channel_size),
              blocks_per_sample((channel_size + block_size - 1) / block_size) {
        }

        void operator()() {
            qmat_t rhs;
            rhs.set_size(block_size,lhs.nc());
            const auto rhs_stride = unsigned(rhs.ptr(1) - rhs.ptr(0));
            const auto rhs_limit = lhs.m_rhs_limit;

            dlibx::aligned_matrix<float,64> abuf(1,std::max(rhs.nr(),rhs.nc()));
            auto buf = &abuf(0,0);

            const auto& bps = blocks_per_sample;
            for (const auto end = input.num_samples() * bps; ; ) {
                const auto ni = next.fetch_add(1,std::memory_order_relaxed);
                if (ni >= end) break;

                const auto sample_num = ni / bps;
                const auto pixel_start = (ni % bps) * block_size;
                const auto pixel_count =
                    std::min(block_size, channel_size - pixel_start);

                auto src =
                    input_buffer + sample_num * input_sample + pixel_start;
                for (long r = 0; r < pixel_count; ++r, ++src) {
                    auto sc = src;
                    auto bp = buf;
                    float vmax = 0;
                    for (auto i = input.k(); i > 0; --i, ++bp,
                             sc += channel_size)
                        vmax = std::max(vmax, std::abs(*bp = *sc));
                    rhs.quantize_row(r, buf, rhs_limit, vmax);
                }

                auto dest =
                    output_buffer + sample_num * output_sample + pixel_start;
                for (auto k = 0; k < lhs.nr(); ++k, dest += channel_size) {
                    ops::mult_row(
                        buf,
                        lhs.coeff(k), lhs.ptr(k), unsigned(lhs.nc()),
                        &rhs.coeff(0), rhs.ptr(0), rhs_stride,
                        unsigned(pixel_count));
                    std::copy_n(buf, pixel_count, dest);
                }
            }
        }
    };

    state s(*this, input, output.host_write_only(), block_size);

    if (nthreads <= 0) {
        s();
        return;
    }

    struct job {
        state& s;
        inline auto operator()() { s(); return 0; }
    };
    std::vector<core::job_function<job> > jobs;
    jobs.reserve(nthreads);
    for (auto n = nthreads; n > 0; --n) {
        jobs.emplace_back(s);
        context->submit_absolute(core::job_queue::order_min, jobs.back());
    }
    s(); // this thread
    context->wait_for_all(jobs.begin(), jobs.end());
    for (auto& job : jobs)
        *job; // re-throw any exceptions
}

template <typename T>
template <int s_nr, int s_nc, int s_dy, int s_dx>
void qmat_t<T>::conv(const dlib::tensor& input, dlib::tensor& output,
                     int m_sy, int m_sx,
                     int m_nr, int m_nc,
                     int m_dy, int m_dx) const {

    const auto context = core::job_context::this_context();
    const auto nthreads = context ? context->num_threads() : 0;
    if (!context && !warn_context_not_found) {
        warn_context_not_found = true;
        FILE_LOG(logWARNING) << "qmat: job_context not found -- using single thread/core";
    }

    // parallelize on output rows, unless there are too few then on pixels
    const auto block_size =
        output.nr() * output.num_samples() >= long(1+2*nthreads) ?
        long(output.nc()) : 1;

    struct state {
        const qmat_t& lhs;
        const dlib::tensor& input;
        dlib::tensor& output;
        const long block_size;
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

        const long blocks_per_sample;

        state(const qmat_t& lhs,
              const dlib::tensor& input,
              dlib::tensor& output,
              long block_size,
              int nr, int nc,
              int dy, int dx,
              int sy, int sx)
            : lhs(lhs),
              input(input),
              output(output),
              block_size(block_size),
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
              blocks_per_sample(output_channel / block_size) {
            assert(output_channel == blocks_per_sample * block_size);
        }

        void operator()() {
            qmat_t rhs;
            rhs.set_size(block_size,lhs.nc());
            const auto rhs_stride = unsigned(rhs.ptr(1) - rhs.ptr(0));
            const auto rhs_limit = lhs.m_rhs_limit;

            dlibx::aligned_matrix<float,64> abuf(1,std::max(rhs.nr(),rhs.nc()));
            auto buf = &abuf(0,0);
            
            const auto& bps = blocks_per_sample;
            for (const auto end = input.num_samples() * bps; ; ) {
                const auto ni = next.fetch_add(1,std::memory_order_relaxed);
                if (ni >= end) break;

                const auto sample_num = ni / bps;
                const auto output_ofs = block_size * (ni % bps);

                const auto input_row = sy * (output_ofs / output.nc());
                const auto input_col = sx * (output_ofs % output.nc());

                auto src = input_buffer + input_sample*sample_num
                    + input_row*input.nc() + input_col;
                for (long r = 0; r < rhs.nr(); ++r, src += sx) {
                    auto sc = src;
                    auto bp = buf;
                    float vmax = 0;
                    for (auto i = input.k(); i > 0; --i,
                             sc += input_channel) {
                        auto sr = sc;
                        for (auto j = nr(); j > 0; --j,
                                 sr += input_step) {
                            auto px = sr;
                            for (auto k = nc(); k > 0; --k, ++bp,
                                     px += dx())
                                vmax = std::max(vmax, std::abs(*bp = *px));
                        }
                    }
                    rhs.quantize_row(r, buf, rhs_limit, vmax);
                }

                auto dest =
                    output_buffer + sample_num*output_sample + output_ofs;
                for (auto k = 0; k < lhs.nr(); ++k, dest += output_channel) {
                    ops::mult_row(
                        buf,
                        lhs.coeff(k), lhs.ptr(k), unsigned(lhs.nc()),
                        &rhs.coeff(0), rhs.ptr(0), rhs_stride,
                        unsigned(block_size));
                    std::copy_n(buf, block_size, dest);
                }
            }
        }
    };

    state s(*this, input, output, block_size,
            m_nr, m_nc, m_dy, m_dx, m_sy, m_sx);

    if (nthreads <= 0) {
        s();
        return;
    }

    struct job {
        state& s;
        inline auto operator()() { s(); return 0; }
    };
    std::vector<core::job_function<job> > jobs;
    jobs.reserve(nthreads);
    for (auto n = nthreads; n > 0; --n) {
        jobs.emplace_back(s);
        context->submit_absolute(core::job_queue::order_min, jobs.back());
    }
    s(); // this thread
    context->wait_for_all(jobs.begin(), jobs.end());
    for (auto& job : jobs)
        *job; // re-throw any exceptions
}

template <typename T>
void qmat_t<T>::conv(const dlib::tensor& input,
                     dlib::resizable_tensor& output,
                     int nr, int nc, int dy, int dx,
                     int sy, int sx) const {
    if (nr == 1 && nc == 1 && sy == 1 && sx == 1) {
        conv1x1(input, output);
        return;
    }
    if (m_rhs_limit <= 0)
        throw std::logic_error("qmat::conv() called on matrix not setup as LHS");
    if (nr < 1 || nc < 1 || dy < 1 || dx < 1 || sy < 1 || sx < 1)
        throw std::invalid_argument("invalid convolution arguments");
    if (this->nc() != input.k()*nr*nc)
        throw std::logic_error("tensor has wrong kernel size for convolution");
    if (input.num_samples() <= 0) return;

    // sliding window size
    const auto wy = 1 + (nr-1) * dy;
    const auto wx = 1 + (nc-1) * dx;

    output.set_size(input.num_samples(),
                    this->nr(),  // num filters
                    1 + (input.nr() - wy) / sy,
                    1 + (input.nc() - wx) / sx);

    if (dy == dx) {
        if (nr == nc) {
            if (dy == 1) {
                switch (nr) {
                case 3: conv<3,3,1,1>(input, output, sy, sx); return;
                case 5: conv<5,5,1,1>(input, output, sy, sx); return;
                case 7: conv<7,7,1,1>(input, output, sy, sx); return;
                }
            }
            else if (nr == 3) {
                switch (dy) {
                case 2: conv<3,3,2,2>(input, output, sy, sx); return;
                case 3: conv<3,3,3,3>(input, output, sy, sx); return;
                case 5: conv<3,3,5,5>(input, output, sy, sx); return;
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

    // generic case
    conv<0,0,0,0>(input, output, sy, sx, nr, nc, dy, dx);
}

template <typename T>
template <int s_nr, int s_nc, int s_dy, int s_dx>
void qmat_t<T>::convdw(const dlib::tensor& input, dlib::tensor& output,
                       int m_sy, int m_sx,
                       int m_nr, int m_nc,
                       int m_dy, int m_dx) const {

    const auto context = core::job_context::this_context();
    const auto nthreads = context ? context->num_threads() : 0;
    if (!context && !warn_context_not_found) {
        warn_context_not_found = true;
        FILE_LOG(logWARNING) << "qmat: job_context not found -- using single thread/core";
    }

    // parallelize on input channels
    struct state {
        const qmat_t& lhs;
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

        state(const qmat_t& lhs,
              const dlib::tensor& input,
              dlib::tensor& output,
              int nr, int nc,
              int dy, int dx,
              int sy, int sx)
            : lhs(lhs),
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
            qmat_t rhs;
            rhs.set_size(output_channel,lhs.nc());
            const auto rhs_stride = unsigned(rhs.ptr(1) - rhs.ptr(0));
            const auto rhs_limit = lhs.m_rhs_limit;

            dlibx::aligned_matrix<float,64> abuf(1,std::max(rhs.nr(),rhs.nc()));
            auto buf = &abuf(0,0);

            for (const auto end = input.num_samples() * input.k(); ; ) {
                const auto ni = next.fetch_add(1,std::memory_order_relaxed);
                if (ni >= end) break;

                auto src = input_buffer + ni*input_channel;
                long out_col = 0;
                for (long r = 0; r < rhs.nr(); ++r, src += sx) {
                    auto sc = src;
                    auto bp = buf;
                    float vmax = 0;
                    for (auto j = nr(); j > 0; --j, sc += input_step) {
                        auto px = sc;
                        for (auto k = nc(); k > 0; --k, ++bp, px += dx())
                            vmax = std::max(vmax, std::abs(*bp = *px));
                    }
                    rhs.quantize_row(r, buf, rhs_limit, vmax);
                    if (++out_col >= output.nc()) {
                        src += end_of_row_delta;
                        out_col = 0;
                    }
                }

                auto dest = output_buffer + ni*mult*output_channel;
                for (auto k = (mult * ni) % long(output.k()),
                         end = k + mult; k < end; ++k,
                         dest += output_channel) {
                    ops::mult_row(
                        buf,
                        lhs.coeff(k), lhs.ptr(k), unsigned(lhs.nc()),
                        &rhs.coeff(0), rhs.ptr(0), rhs_stride,
                        unsigned(rhs.nr()));
                    std::copy_n(buf, rhs.nr(), dest);
                }
            }
        }
    };

    state s(*this, input, output, m_nr, m_nc, m_dy, m_dx, m_sy, m_sx);

    if (nthreads <= 0) {
        s();
        return;
    }

    struct job {
        state& s;
        inline auto operator()() { s(); return 0; }
    };
    std::vector<core::job_function<job> > jobs;
    jobs.reserve(nthreads);
    for (auto n = nthreads; n > 0; --n) {
        jobs.emplace_back(s);
        context->submit_absolute(core::job_queue::order_min, jobs.back());
    }
    s(); // this thread
    context->wait_for_all(jobs.begin(), jobs.end());
    for (auto& job : jobs)
        *job; // re-throw any exceptions
}

template <typename T>
void qmat_t<T>::convdw(const dlib::tensor& input,
                       dlib::resizable_tensor& output,
                       int nr, int nc, int dy, int dx,
                       int sy, int sx) const {
    if (m_rhs_limit <= 0)
        throw std::logic_error("qmat::conv() called on matrix not setup as LHS");
    if (nr < 1 || nc < 1 || dy < 1 || dx < 1 || sy < 1 || sx < 1)
        throw std::invalid_argument("invalid convolution arguments");
    if (this->nc() != nr*nc)
        throw std::logic_error("tensor has wrong kernel size for convolution");
    const auto mult = this->nr() / input.k();
    if (mult < 1 || this->nr() != mult * input.k())
        throw std::logic_error("tensor has wrong number of channels for convolution");
    if (input.num_samples() <= 0) return;

    // sliding window size
    const auto wy = 1 + (nr-1) * dy;
    const auto wx = 1 + (nc-1) * dx;

    output.set_size(input.num_samples(),
                    this->nr(),  // num filters
                    1 + (input.nr() - wy) / sy,
                    1 + (input.nc() - wx) / sx);

    if (nr == 3 && nc == 3 && dy == 1 && dx == 1) {
        convdw<3,3,1,1>(input, output, sy, sx);
        return;
    }

    // generic case
    convdw<0,0,0,0>(input, output, sy, sx, nr, nc, dy, dx);
}


// explicit template instantiation
template class dlibx::qmat_t<int16_t>;
template class dlibx::qmat_t<int8_t>;
