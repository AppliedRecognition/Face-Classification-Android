#include <boost/test/unit_test.hpp>
#include <applog/base_directory.hpp>
#include <applog/core.hpp>

#include <dlibx/qmat.hpp>
#include <dlibx/matrix_ops.hpp>
#include <dlibx/tensor.hpp>
#include <stdext/convert.hpp>

#include <random>
#include <chrono>

BOOST_AUTO_TEST_SUITE(dlibx)

static std::mt19937 rgen(11);

namespace {
    struct tensor_offset_view {
        const dlib::tensor& t;
        long rofs, cofs;

        struct row {
            const float* base;
            long nc, cofs;
            float operator()(long c) const {
                c += cofs;
                return 0 <= c && c < nc ? base[c] : 0.0f;
            }
        };
        auto operator()(long k) const {
            assert(0 <= k && k < t.k());
            return [this, chan = t.host() + k*t.nr()*t.nc()](long r) {
                r += rofs;
                return 0 <= r && r < t.nr() ?
                    row{chan + r*t.nc(), t.nc(), cofs} :
                    row{nullptr,0,0};
            };
        }
    };
}

namespace {
    int alignment_sum = 0;
    template <typename T>
    struct alignment_test {
        template <std::size_t N>
        auto test() const {
            int sum = 0;
            char buf[N];
            for (auto _len : { 1, 2, 3, 4, 7, 8, 9, 15, 16, 17,
                        31, 32, 33, 63, 64, 65, 127, 128, 129 }) {
                const auto len = unsigned(_len);
                auto alloc = stdx::make_aligned<T[],N>(len);
                BOOST_CHECK_EQUAL(std::size_t(alloc.get()) & (N-1), 0);
                alloc[0] = 5;
                alloc[len-1] = 10;
                const auto ofs = (len-1) & ~unsigned(N/sizeof(T));
                BOOST_REQUIRE(ofs < len);
                memcpy(buf, alloc.get() + ofs, N);
                sum += buf[0] + buf[N-1];

                stdx::aligned_ptr<T[]> other1(move(alloc));
                BOOST_CHECK(alloc.get() == nullptr);
                stdx::aligned_ptr<T[]> other2;
                BOOST_CHECK(other2.get() == nullptr);
                other2 = move(other1);
                BOOST_CHECK(other1.get() == nullptr);
                BOOST_CHECK(other2.get() != nullptr);
            }
            return sum;
        }
        alignment_test() {
            alignment_sum += test<8>();
            alignment_sum += test<16>();
            alignment_sum += test<32>();
            alignment_sum += test<64>();
            alignment_sum += test<128>();
        }
    };
}

BOOST_AUTO_TEST_CASE(aligned_alloc) {
    alignment_test<short>();
    alignment_test<int>();
    alignment_test<long>();
    FILE_LOG(logINFO) << "alignment dummy value: " << alignment_sum;
}

template <long nr, long nc>
static float* _img2col(float* dest, const dlib::tensor& t, long r, long c) {
    static_assert(nr > 0 && nc > 0 && (nr&nc&1), "invalid filter dimension");
    static constexpr auto hr = nr/2;
    static constexpr auto hc = nc/2;
    assert(0 <= r && r < t.nr());
    assert(0 <= c && c < t.nc());
    const tensor_offset_view v{t,r,c};
    for (long k = 0; k < t.k(); ++k) {
        auto chan = v(k);
        for (long dr = -hr; dr <= hr; ++dr) {
            auto row = chan(dr);
            for (long dc = -hc; dc <= hc; ++dc, ++dest)
                *dest = row(dc);
        }
    }
    return dest;
}

BOOST_AUTO_TEST_CASE(qmat_img2col_test) {
    FILE_LOG(logINFO) << "--";

    static constexpr auto filter_k = 2;
    static constexpr auto filter_nr = 3;
    static constexpr auto filter_nc = 5;
    static constexpr auto stride_y = 2;
    static constexpr auto stride_x = 3;
    static constexpr auto final_nr = 5;
    static constexpr auto final_nc = 4;
    dlib::resizable_tensor t(
        1,
        filter_k,
        filter_nr+stride_y*(final_nr-1),
        filter_nc+stride_x*(final_nc-1));
    {
        std::uniform_int_distribution<> dis(-256, 256);
        for (auto& x : t) x = float(dis(rgen));
    }

    qmat16 q0;
    q0.img2col(
        4096,
        img2col<filter_nr,filter_nc,1,1,0,0,compute_maxabs>(stride_y,stride_x,t),
        t,0);
    BOOST_REQUIRE_EQUAL(q0.nr(), final_nr*final_nc);
    BOOST_REQUIRE_EQUAL(q0.nc(), filter_k*filter_nr*filter_nc);

    dlib::matrix<float> m0(q0.nr(),q0.nc());
    for (long r = 0; r < m0.nr(); ++r) {
        const auto c = q0.coeff(r);
        const auto ptr = &q0.value(r,0);
        std::transform(ptr, ptr + q0.nc(), &m0(r,0), [c](auto x){return c*x;});
    }

    dlib::matrix<float> m1(m0.nr(),m0.nc());
    for (long j = 0; j < m1.nr(); ++j) {
        const auto r = (j/final_nc)*stride_y + filter_nr/2;
        const auto c = (j%final_nc)*stride_x + filter_nc/2;
        auto first = &m1(j,0);
        auto last = _img2col<filter_nr,filter_nc>(first, t, r, c);
        BOOST_REQUIRE_EQUAL(m1.nc(), last - first);
    }

    float e = 0;
    for (long r = 0; r < m0.nr(); ++r)
        for (long c = 0; c < m0.nc(); ++c) {
            const auto v0 = m0(r,c);
            const auto v1 = m1(r,c);
            e = std::max(e,std::abs(v0-v1));
            BOOST_CHECK_EQUAL(std::round(v0), std::round(v1));
        }
    FILE_LOG(logINFO) << "img2col error: " << e;
}

static auto is_8bit_safe() {
    qmat8 lhs, rhs;
    lhs.set_size(1,100);
    lhs.coeff(0) = 1;
    std::fill(&lhs.value(0,lhs.nc()), &lhs.value(1,0), 0);
    rhs.set_size(1,100);
    rhs.coeff(0) = 1;
    for (auto lhsv : { int8_t(127), int8_t(-127) }) {
        std::fill_n(&lhs.value(0,0), lhs.nc(), lhsv);
        for (auto rhsv : { int8_t(127), int8_t(-127) }) {
            std::fill_n(&rhs.value(0,0), rhs.nc(), rhsv);
            auto prod = lhs.mult_transpose_rhs(rhs);
            BOOST_REQUIRE_EQUAL(prod.nr(), 1);
            BOOST_REQUIRE_EQUAL(prod.nc(), 1);
            if (std::lround(prod(0,0)) != lhsv*rhsv*100)
                return false;
        }
    }
    return true;
}

template <int min_val, int max_val, typename QMAT>
static auto random_fill(QMAT& mat) {
    static_assert(min_val < 0 && 0 < max_val);
    std::uniform_int_distribution<int> umm(-1,1);
    std::uniform_int_distribution<int> ur(min_val,max_val);
    std::vector<unsigned> row_strategy(std::size_t(mat.nr()));
    for (unsigned i = 0; i < row_strategy.size(); ++i)
        row_strategy[i] = i%5;
    std::shuffle(row_strategy.begin(), row_strategy.end(), rgen);
    auto it = row_strategy.begin();
    for (long r = 0; r < mat.nr(); ++r) {
        mat.coeff(r) = float(r+1);
        auto p = &mat.value(r,0);
        std::fill(p, &mat.value(r+1,0), 0);
        switch (*it++) {
        case 0:  // all zero -- done
            break;
        case 1:  // max val
            std::fill_n(p, mat.nc(), max_val);
            break;
        case 2:  // min val
            std::fill_n(p, mat.nc(), min_val);
            break;
        case 3:  // min/zero/max random
            for (auto end = p + mat.nc(); p != end; ++p)
                switch (umm(rgen)) {
                case -1: *p = min_val; break;
                case 1:  *p = max_val; break;
                }
            break;
        case 4:  // uniform random
            for (auto end = p + mat.nc(); p != end; ++p)
                *p = stdx::convert_from(ur(rgen));
            break;
        default: assert(!"bad strategy");
        }
    }
}

template <typename QMAT>
static auto random_tail(QMAT& mat) {
    using T = typename QMAT::value_type;
    std::uniform_int_distribution<int> ur(
        std::numeric_limits<T>::min(), std::numeric_limits<T>::max());
    for (long r = 0; r < mat.nr(); ++r)
        for (auto p = &mat.value(r,mat.nc()),
                 end = &mat.value(r+1,0); p != end; ++p)
            *p = stdx::convert_from(ur(rgen));
}

template <typename DEST, typename SRC>
static auto copy(DEST& dmat, const SRC& smat) {
    dmat.set_size(smat.nr(), smat.nc());
    std::copy_n(&smat.coeff(0), dmat.nr(), &dmat.coeff(0));
    for (long r = 0; r < dmat.nr(); ++r) {
        std::copy_n(&smat.value(r,0), dmat.nc(), &dmat.value(r,0));
        std::fill(&dmat.value(r,dmat.nc()), &dmat.value(r+1,0), 0);
    }
}

BOOST_AUTO_TEST_CASE(qmat_mult_test) {
    const auto full_8bit = is_8bit_safe();

    if (full_8bit)
        FILE_LOG(logINFO) << "qmat: full 8-bit x 8-bit multiply (except -128)";
    else
        FILE_LOG(logWARNING) << "qmat: can only do 7-bit x 8-bit multiply";

    using clock = std::chrono::steady_clock;
    clock::duration t8{}, t16{};

    for (long nc = 7; nc < 100; nc += 9) {
        auto good = nc;
        for (long nr_lhs = 3; nr_lhs < 12 && good == nc; ++nr_lhs) {
            for (long nr_rhs = 3; nr_rhs < 12 && good == nc; ++nr_rhs) {
                qmat8 lhs8, rhs8;
                lhs8.set_size(nr_lhs, nc);
                if (full_8bit)
                    random_fill<-127,127>(lhs8);
                else
                    random_fill<-64,63>(lhs8);
                rhs8.set_size(nr_rhs, nc);
                random_fill<-127,127>(rhs8);
                random_tail(rhs8);

                qmat16 lhs16, rhs16;
                copy(lhs16, lhs8);
                copy(rhs16, rhs8);
                random_tail(rhs16);

                const auto start = clock::now();
                const auto m8 = lhs8.mult_transpose_rhs(rhs8);
                const auto mid = clock::now();
                const auto m16 = lhs16.mult_transpose_rhs(rhs16);
                t16 += clock::now() - mid;
                t8 += mid - start;
                BOOST_REQUIRE_EQUAL(m8.nr(), nr_lhs);
                BOOST_REQUIRE_EQUAL(m8.nc(), nr_rhs);
                BOOST_REQUIRE_EQUAL(m8.nr(), m16.nr());
                BOOST_REQUIRE_EQUAL(m8.nc(), m16.nc());

                for (long r = 0; r < m8.nr(); ++r)
                    for (long c = 0; c < m8.nc(); ++c) {
                        const auto v8 = m8(r,c);
                        const auto v16 = m16(r,c);
                        const auto d = std::abs(v8 - v16);
                        if (d >= 1) {
                            FILE_LOG(logWARNING) << '[' << nc << ']'
                                                 << '\t' << r << ',' << c
                                                 << '\t' << std::lround(v8)
                                                 << ' ' << std::lround(v16);
                            good = 0;
                            r = m8.nr();
                            break;
                        }
                    }
            }
        }
        BOOST_CHECK_EQUAL(nc, good);
    }

    // report timing
    const auto t_ratio = float(t8.count()) / float(t16.count());
    if (t8 <= t16)
        FILE_LOG(logINFO) << "qmat: 8-bit is faster than 16-bit ("
                          << t_ratio << ')';
    else
        FILE_LOG(logWARNING) << "qmat: 8-bit is SLOWER than 16-bit ("
                             << t_ratio << ')';
}

BOOST_AUTO_TEST_CASE(qmat_16_mult_test) {
    FILE_LOG(logINFO) << "qmat: 16 bit";

    // make sure we have more than one 64-byte cache line
    std::vector<int> vals;
    while (vals.size() <= 32)
        for (int x = -3; x <= 3; ++x)
            vals.push_back(x);
    const auto nc = long(vals.size());

    dlib::matrix<float> lhs;
    lhs.set_size(13,nc);
    for (long r = 0; r < lhs.nr(); ++r) {
        std::shuffle(vals.begin(), vals.end(), rgen);
        auto it = vals.begin();
        for (long c = 0; c < lhs.nc(); ++c)
            lhs(r,c) = float(*it++ * ((r+1)*(255/3))) / 16;
    }

    qmat16 lhsq;
    auto rhs_limit = lhsq.assign_lhs(lhs, 9); // bits
    BOOST_REQUIRE_EQUAL(rhs_limit, 32767);
    rhs_limit = 150; // override
    BOOST_REQUIRE_EQUAL(lhsq.nr(), lhs.nr());
    BOOST_REQUIRE_EQUAL(lhsq.nc(), lhs.nc());
    float lhs_error = 0;
    for (long r = 0; r < lhs.nr(); ++r) {
        for (long c = 0; c < lhs.nc(); ++c) {
            const auto z = lhsq.coeff(r) * lhsq.value(r,c);
            lhs_error = std::max(lhs_error, std::abs(lhs(r,c) - z));
        }
    }
    FILE_LOG(logINFO) << "lhs_error: " << lhs_error;
    BOOST_CHECK(lhs_error < 1e-5);

    dlib::matrix<float> rhs;
    rhs.set_size(11,nc);
    for (long r = 0; r < rhs.nr(); ++r) {
        std::shuffle(vals.begin(), vals.end(), rgen);
        auto it = vals.begin();
        for (long c = 0; c < rhs.nc(); ++c)
            rhs(r,c) = float((rhs_limit * *it++ / 3) * (r+1)) / 32;
    }

    qmat16 rhsq;
    FILE_LOG(logINFO) << "rhs_limit = " << rhs_limit;
    rhsq.assign_rhs(rhs, rhs_limit);
    BOOST_REQUIRE_EQUAL(rhsq.nr(), rhs.nr());
    BOOST_REQUIRE_EQUAL(rhsq.nc(), rhs.nc());
    float rhs_error = 0;
    for (long r = 0; r < rhs.nr(); ++r) {
        for (long c = 0; c < rhs.nc(); ++c) {
            const auto z = rhsq.coeff(r) * rhsq.value(r,c);
            rhs_error = std::max(rhs_error, std::abs(rhs(r,c) - z));
        }
    }
    FILE_LOG(logINFO) << "rhs_error: " << rhs_error;
    BOOST_CHECK(rhs_error < 1e-5);

    const dlib::matrix<float> prod = lhs * trans(rhs);
    BOOST_REQUIRE_EQUAL(prod.nr(), lhs.nr());
    BOOST_REQUIRE_EQUAL(prod.nc(), rhs.nr());

    const auto prodq = lhsq.mult_transpose_rhs(rhsq);
    BOOST_REQUIRE_EQUAL(prodq.nr(), prod.nr());
    BOOST_REQUIRE_EQUAL(prodq.nc(), prod.nc());

    float mult_error = 0;
    for (long r = 0; r < prod.nr(); ++r)
        for (long c = 0; c < prod.nc(); ++c) {
            const auto e = float(std::abs(prodq(r,c) - prod(r,c)));
            mult_error = std::max(mult_error, e);
        }
    FILE_LOG(logINFO) << "mult_error: " << mult_error;
    BOOST_CHECK(mult_error < 1e-5);
}

BOOST_AUTO_TEST_CASE(qmat_8_mult_test) {
    FILE_LOG(logINFO) << "qmat: 8 bit";

    {
        using namespace dlibx::ops;
        BOOST_REQUIRE(machine.description != nullptr);
        FILE_LOG(logINFO) << "machine: " << machine.description;
        std::array<float,16> rhs_coeff;
        rhs_coeff.fill(1.0f);
        dlibx::aligned_matrix<int8_t,64> rhs(3,128);
        std::fill_n(&rhs(0,0), 128, 127);
        std::fill_n(&rhs(1,0), 128, -127);
        std::fill_n(&rhs(2,0), 128, -128);
        dlibx::aligned_matrix<int8_t,64> lhs(1,128);
        for (auto v : { 127, -127, -128 } ) {
            std::fill_n(&lhs(0,0), 128, int8_t(v));
            std::array<float,16> dest;
            mult_row(dest.data(),
                     1.0f, &lhs(0,0), 128,
                     rhs_coeff.data(), &rhs(0,0),
                     unsigned(rhs.elements_per_row()), 3);

            if (std::abs(dest[0] - float(128*127*v)) > 1e-5)
                FILE_LOG(logWARNING) << "8-bit inner product with "
                                     << v << " * 127 doesn't work!";
            else if (v == -128)
                FILE_LOG(logINFO) << "8-bit inner product with "
                                  << v << " * 127 works";

            if (std::abs(dest[1] - float(128*-127*v)) > 1e-5)
                FILE_LOG(logWARNING) << "8-bit inner product with "
                                     << v << " * -127 doesn't work!";
            else if (v == -128)
                FILE_LOG(logINFO) << "8-bit inner product with "
                                  << v << " * -127 works";

            if (std::abs(dest[2] - float(128*-128*v)) > 1e-5)
                FILE_LOG(logWARNING) << "8-bit inner product with "
                                     << v << " * -128 doesn't work!";
            else
                FILE_LOG(logINFO) << "8-bit inner product with "
                                  << v << " * -128 works";

            if (v != -128) {
                BOOST_CHECK_EQUAL(dest[0], 128*127*v);
                BOOST_CHECK_EQUAL(dest[1], -128*127*v);
            }
        }
    }

    // make sure we have more than one 64-byte cache line
    std::vector<int> vals;
    while (vals.size() <= 64)
        for (int x = -7; x <= 7; ++x)
            vals.push_back(x);
    const auto nc = long(vals.size());

    dlib::matrix<float> lhs;
    lhs.set_size(13,nc);
    for (long r = 0; r < lhs.nr(); ++r) {
        std::shuffle(vals.begin(), vals.end(), rgen);
        auto it = vals.begin();
        for (long c = 0; c < lhs.nc(); ++c)
            lhs(r,c) = float(*it++ * ((r+1)*(63/7)));
    }

    qmat8 lhsq;
    const auto rhs_limit = lhsq.assign_lhs(lhs, 7); // bits
    BOOST_REQUIRE_EQUAL(rhs_limit, 127);
    BOOST_REQUIRE_EQUAL(lhsq.nr(), lhs.nr());
    BOOST_REQUIRE_EQUAL(lhsq.nc(), lhs.nc());
    float lhs_error = 0;
    for (long r = 0; r < lhs.nr(); ++r) {
        for (long c = 0; c < lhs.nc(); ++c) {
            const auto z = lhsq.coeff(r) * lhsq.value(r,c);
            lhs_error = std::max(lhs_error, std::abs(lhs(r,c) - z));
        }
    }
    FILE_LOG(logINFO) << "lhs_error: " << lhs_error;
    BOOST_CHECK(lhs_error < 1e-5);

    dlib::matrix<float> rhs;
    rhs.set_size(11,nc);
    for (long r = 0; r < rhs.nr(); ++r) {
        std::shuffle(vals.begin(), vals.end(), rgen);
        auto it = vals.begin();
        for (long c = 0; c < rhs.nc(); ++c)
            rhs(r,c) = float((127 * *it++ / 7) * (r+1));
    }

    qmat8 rhsq;
    FILE_LOG(logINFO) << "rhs_limit = " << rhs_limit;
    rhsq.assign_rhs(rhs, rhs_limit);
    BOOST_REQUIRE_EQUAL(rhsq.nr(), rhs.nr());
    BOOST_REQUIRE_EQUAL(rhsq.nc(), rhs.nc());
    float rhs_error = 0;
    for (long r = 0; r < rhs.nr(); ++r) {
        for (long c = 0; c < rhs.nc(); ++c) {
            const auto z = rhsq.coeff(r) * rhsq.value(r,c);
            rhs_error = std::max(rhs_error, std::abs(rhs(r,c) - z));
        }
    }
    FILE_LOG(logINFO) << "rhs_error: " << rhs_error;
    BOOST_CHECK(rhs_error < 1e-5);

    const dlib::matrix<float> prod = lhs * trans(rhs);
    BOOST_REQUIRE_EQUAL(prod.nr(), lhs.nr());
    BOOST_REQUIRE_EQUAL(prod.nc(), rhs.nr());

    const auto prodq = lhsq.mult_transpose_rhs(rhsq);
    BOOST_REQUIRE_EQUAL(prodq.nr(), prod.nr());
    BOOST_REQUIRE_EQUAL(prodq.nc(), prod.nc());

    float mult_error = 0;
    for (long r = 0; r < prod.nr(); ++r)
        for (long c = 0; c < prod.nc(); ++c) {
            const auto e = std::abs(prodq(r,c) - prod(r,c));
            mult_error = std::max(mult_error, e);
        }
    FILE_LOG(logINFO) << "mult_error: " << mult_error;
    BOOST_CHECK(mult_error < 1e-5);

    FILE_LOG(logINFO) << "qmat: done";
}


BOOST_AUTO_TEST_SUITE_END()
