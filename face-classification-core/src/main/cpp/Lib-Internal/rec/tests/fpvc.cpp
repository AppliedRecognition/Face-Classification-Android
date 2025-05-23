#include <boost/test/unit_test.hpp>

#include <random>

#include <rec/fpvc.hpp>
#include <applog/core.hpp>

using namespace rec::internal;

BOOST_AUTO_TEST_SUITE(rec)

BOOST_AUTO_TEST_CASE(fpvc_monotonic) {
    // ensure compress is monotonic
    auto prev = fpvc_unsigned_compress(0);
    BOOST_CHECK(prev == 0);
    for (int y = 1; y < 2048; ++y) {
        const auto x = fpvc_unsigned_compress(y);
        BOOST_CHECK(prev <= x);
        prev = x;
    }
}

BOOST_AUTO_TEST_CASE(fpvc_tables) {
    // verify tables
    for (unsigned x = 0; x < 128; ++x) {
        const auto y = fpvc_unsigned_decompress(x);
        BOOST_CHECK_EQUAL(fpvc_s16_decompress_table[x], y);
        BOOST_CHECK_EQUAL(fpvc_s16_decompress_table[255-x], -y);
        BOOST_CHECK_EQUAL(fpvc_f32_decompress_table[x], y);
        BOOST_CHECK_EQUAL(fpvc_f32_decompress_table[255-x], -y);
    }
}

BOOST_AUTO_TEST_CASE(fpvc_rounding) {
    // ensure x = compress(decompress(x)) and
    // x = compress(decompress(x)+delta) for both pos and neg delta
    auto prev = fpvc_unsigned_decompress(0);
    assert(prev == 0);
    for (unsigned x = 1; x < 128; ++x) {
        const auto y = fpvc_unsigned_decompress(x);
        const auto x2 = fpvc_unsigned_compress(y);
        BOOST_CHECK(x2 == x);
        BOOST_CHECK(prev < y);
        const auto gap = y - prev;
        const auto thres = y - gap/2;
        const auto x3 = fpvc_unsigned_compress(thres);
        BOOST_CHECK(x3 == x);
        const auto x4 = fpvc_unsigned_compress(thres-1)+1;
        BOOST_CHECK(x4 == x);
        prev = y;
    }
}

BOOST_AUTO_TEST_CASE(fpvc_vec) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> d;

    for (unsigned rep = 0; rep < 10; ++rep) {
        const unsigned N = 100*(rep+1);
        std::vector<float> vec;
        vec.reserve(N);
        float maxv = 0, norm = 0;
        for (unsigned i = 0; i < N; ++i) {
            vec.push_back(d(gen));
            const float y = fabsf(vec.back());
            if (maxv < y)
                maxv = y;
            norm += y*y;
        }
        norm = sqrtf(norm);
        
        const auto enc = fpvc_vector_compress(vec.begin(),vec.end());
        const auto dec = fpvc_vector_decompress(enc);
        BOOST_REQUIRE(dec.size() == N);

        float maxv2 = 0, norm2 = 0, rms = 0;
        for (unsigned i = 0; i < N; ++i) {
            const float y = fabsf(dec[i]);
            if (maxv2 < y)
                maxv2 = y;
            norm2 += y*y;
            const float d = dec[i] - vec[i];
            rms += d*d;
        }
        norm2 = sqrtf(norm2);
        rms = sqrtf(rms) / std::min(norm,norm2);
        
        const float merr = fabsf(maxv-maxv2) / std::min(maxv,maxv2);
        const float nerr = fabsf(norm-norm2) / std::min(norm,norm2);

        FILE_LOG(logINFO) << "N=" << N
                          << "  max=" << maxv
                          << "  norm=" << norm
                          << "  merr=" << merr
                          << "  nerr=" << nerr
                          << "  rms=" << rms;

        BOOST_CHECK(merr < 1e-7);
        BOOST_CHECK(nerr < 0.002);
        BOOST_CHECK(rms < 0.015625);  // 1/64

        // re-compressing decompressed vector must give exactly same result
        const auto enc2 = fpvc_vector_compress(dec.begin(),dec.end());
        BOOST_CHECK(enc.second == enc2.second);
        const float eerr =
            fabsf(enc.first-enc2.first) / std::min(enc.first,enc2.first);
        BOOST_CHECK(eerr < 1e-7);
    }
}

BOOST_AUTO_TEST_CASE(fp16vec_12_16) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int16_t> d(-2048,2047);

    std::vector<int16_t> vals;
    for (auto n = 48; n > 0; --n)
        vals.push_back(d(gen));
    vals.front() = 2047;
    vals.back() = -2048;

    for (unsigned n = 1; n <= vals.size(); ++n) {
        fp16vec v;
        v.coeff = 1.0f;
        v.resize(n);
        std::copy(vals.end() - n, vals.end(), v.begin());
        BOOST_REQUIRE_EQUAL(v.size(), n);
        const auto n12 = fp16vec_12_bytes(n);
        const auto n16 = fp16vec_16_bytes(n);

        std::vector<unsigned char> ser12;
        serialize_12(ser12, v);
        BOOST_REQUIRE_EQUAL(ser12.size(), n12);
        const auto r12 = deserialize_fp16vec_12(ser12.data(), n);
        BOOST_CHECK_EQUAL(r12.coeff, v.coeff);
        BOOST_CHECK_EQUAL(r12.size(), n);
        BOOST_CHECK(std::equal(r12.begin(), r12.end(), v.begin()));

        std::vector<unsigned char> ser16;
        serialize_16(ser16, v);
        BOOST_REQUIRE_EQUAL(ser16.size(), n16);
        const auto r16 = deserialize_fp16vec_16(ser16.data(), n);
        BOOST_CHECK_EQUAL(r16.coeff, v.coeff);
        BOOST_CHECK_EQUAL(r16.size(), n);
        BOOST_CHECK(std::equal(r16.begin(), r16.end(), v.begin()));
        
        /*
        std::stringstream ss;
        for (auto x : v.values)
            ss << ' ' << x;
        FILE_LOG(logINFO) << ss.str();
        */
    }
}

BOOST_AUTO_TEST_SUITE_END()

