#include <boost/test/unit_test.hpp>
#include <applog/base_directory.hpp>
#include <applog/core.hpp>

#include <dlibx/bfloat16.hpp>

#include <random>

BOOST_AUTO_TEST_SUITE(dlibx)

static bool isneg(float x) {
    union { float f; uint32_t u; } y;
    y.f = x;
    return y.u >> 31;
}
static auto ispos(float x) {
    return !isneg(x);
}
static auto iszero(float x) {
    return std::fpclassify(x) == FP_ZERO;
}
static auto issub(float x) {
    return std::fpclassify(x) == FP_SUBNORMAL;
}

template <std::size_t N>
static auto as_float(const uint32_t(&array)[N]) {
    struct wrap {
        const float* const m_begin;
        const float* const m_end;
        inline auto begin() const { return m_begin; }
        inline auto end() const { return m_end; }
    };
    auto ptr = reinterpret_cast<const float*>(array);
    return wrap { ptr, ptr + N };
}

static auto test_value(float x0, int fp_class, bool positive, bool exact) {
    BOOST_REQUIRE_EQUAL(fp_class, std::fpclassify(x0));
    BOOST_REQUIRE_EQUAL(positive, ispos(x0));

    auto x1 = x0;
    truncate_to_bfloat16(&x1, 1);
    BOOST_CHECK_EQUAL(fp_class, std::fpclassify(x1));
    BOOST_CHECK_EQUAL(positive, ispos(x1));
    if (fp_class != FP_NAN) {
        if (exact)
            BOOST_CHECK_EQUAL(x1,x0);
        else
            BOOST_CHECK(std::abs(x1) < std::abs(x0));
    }

    std::stringstream ss;
    serialize(bfloat16(&x0,1u),ss);
    serialize(bfloat16(&x1,1u),ss);
    BOOST_CHECK_EQUAL(ss.str().size(), 4);

    float x2[2];
    deserialize(bfloat16(x2),ss);
    for (auto y : x2) {
        BOOST_CHECK_EQUAL(fp_class, std::fpclassify(y));
        BOOST_CHECK_EQUAL(positive, ispos(y));
        if (fp_class != FP_NAN) {
            if (exact)
                BOOST_CHECK_EQUAL(y, x0);
            else
                BOOST_CHECK(std::abs(y) < std::abs(x0));
            BOOST_CHECK_EQUAL(y, x1);
        }
    }
}

BOOST_AUTO_TEST_CASE(bfloat16_test) {
    FILE_LOG(logINFO) << "--";

    static_assert(sizeof(float) == 4 && sizeof(uint32_t) == 4);

    // float_bit_pattern, fpclassify, exact_conversion
    const std::tuple<uint32_t,int,bool> standard_values[] = {
        { 0,           FP_ZERO,      true  },
        { 0x00010000u, FP_SUBNORMAL, true  },
        { 0x0001ffffu, FP_SUBNORMAL, false },
        { 0x00400000u, FP_SUBNORMAL, true  },
        { 0x00400001u, FP_SUBNORMAL, false },
        { 0x007f0000u, FP_SUBNORMAL, true  },
        { 0x007fffffu, FP_SUBNORMAL, false },
        { 0x00800000u, FP_NORMAL,    true  },
        { 0x00800001u, FP_NORMAL,    false },
        { 0x7f7f0000u, FP_NORMAL,    true  },
        { 0x7f7fffffu, FP_NORMAL,    false },
        { 0x7f800000u, FP_INFINITE,  true  },
        { 0x7f800001u, FP_NAN,       true  },
        { 0x7f808000u, FP_NAN,       true  },
        { 0x7f810000u, FP_NAN,       true  },
        { 0x7f900000u, FP_NAN,       true  },
        { 0x7fffffffu, FP_NAN,       true  },
    };
    for (auto& t : standard_values) {
        union { float f; uint32_t u; } xp, xn;
        xp.u = std::get<0>(t);
        test_value(xp.f, std::get<1>(t), true, std::get<2>(t));
        xn.u = std::get<0>(t) | 0x80000000u;
        test_value(xn.f, std::get<1>(t), false, std::get<2>(t));
    }

    // subnormal values so small they truncate to zero as bfloat16
    const uint32_t subnormal_values[] = {
        0x0001u, 0x80000001u,
        0x8000u, 0x80008000u
    };
    for (auto x : as_float(subnormal_values)) {
        BOOST_REQUIRE(issub(x));
        auto y = x;
        truncate_to_bfloat16(&y, 1);
        BOOST_CHECK(iszero(y));
        BOOST_CHECK_EQUAL(isneg(x),isneg(y));
    }
    
    FILE_LOG(logINFO) << "bfloat16: done";
}

namespace {
    unsigned bits_required(int x) {
        return dlibx::bits_required(&x,1);
    }
}

static std::mt19937 rgen(1);

template <typename T>
static void test_bits() {
    for (unsigned nbits = 2; nbits <= 16; ++nbits) {
        const auto z = 1l << (std::is_signed_v<T> ? nbits-1 : nbits);
        const auto min = std::is_signed_v<T> ? -z : 0;
        std::uniform_int_distribution<T> ud(T(min), T(z-1));
        using U = std::conditional_t<std::is_signed_v<T>,int,unsigned>;
        //FILE_LOG(logINFO) << min << ' ' << z-1 << ' ' << nbits;
        BOOST_REQUIRE_EQUAL(bits_required(U(z-1)), nbits);
        if (min) BOOST_REQUIRE_EQUAL(bits_required(U(min)), nbits);
        for (unsigned len = 1; len <= 2; ++len) {
            std::vector<T> vec(len);
            std::stringstream out;
            bits_writer bw(out, nbits);
            for (auto& x : vec) {
                x = ud(rgen);
                bw(x);
            }
            bw.flush();
            BOOST_CHECK(bw);
            BOOST_CHECK(bits_required(vec.data(),vec.size()) <= nbits);

            const auto bin = out.str();
            FILE_LOG(logDETAIL) << "bits=" << nbits
                                << " len=" << len
                                << " size=" << bin.size();
            BOOST_CHECK_EQUAL(bin.size(), (nbits*len+7)/8);

            std::stringstream in(bin);
            std::vector<T> v2(len);
            bits_reader br(in, nbits);
            for (auto& x : v2)
                x = br.get<T>();
            BOOST_CHECK(br);

            BOOST_CHECK(vec == v2);
            if (0) {
                std::stringstream ss;
                for (auto& x : bin)
                    ss << std::hex << ' ' << int(static_cast<unsigned char>(x));
                FILE_LOG(logINFO) << ss.str();
                for (unsigned i = 0; i < vec.size(); ++i)
                    FILE_LOG(logINFO) << '\t' << i << ' '
                                      << vec[i] << ' ' << v2[i];
            }
            
            char c;
            in.read(&c,1);
            BOOST_CHECK(in.eof());
        }
    }
}

BOOST_AUTO_TEST_CASE(serialize_bits_test) {
    FILE_LOG(logINFO) << "--";

    BOOST_CHECK_EQUAL(bits_required(0u), 1);
    BOOST_CHECK_EQUAL(bits_required(1u), 1);
    BOOST_CHECK_EQUAL(bits_required(2u), 2);
    BOOST_CHECK_EQUAL(bits_required(3u), 2);
    BOOST_CHECK_EQUAL(bits_required(4u), 3);
    BOOST_CHECK_EQUAL(bits_required(5u), 3);
    BOOST_CHECK_EQUAL(bits_required(6u), 3);
    BOOST_CHECK_EQUAL(bits_required(7u), 3);
    BOOST_CHECK_EQUAL(bits_required(8u), 4);
    BOOST_CHECK_EQUAL(bits_required(9u), 4);

    BOOST_CHECK_EQUAL(bits_required(-5), 4);
    BOOST_CHECK_EQUAL(bits_required(-4), 3);
    BOOST_CHECK_EQUAL(bits_required(-3), 3);
    BOOST_CHECK_EQUAL(bits_required(-2), 2);
    BOOST_CHECK_EQUAL(bits_required(-1), 1);
    BOOST_CHECK_EQUAL(bits_required(0), 1);
    BOOST_CHECK_EQUAL(bits_required(1), 2);
    BOOST_CHECK_EQUAL(bits_required(2), 3);
    BOOST_CHECK_EQUAL(bits_required(3), 3);
    BOOST_CHECK_EQUAL(bits_required(4), 4);

    test_bits<int16_t>();
    test_bits<uint16_t>();
    test_bits<int>();
    test_bits<unsigned>();
    test_bits<long>();
    FILE_LOG(logINFO) << "bits: done";
}

BOOST_AUTO_TEST_SUITE_END()
