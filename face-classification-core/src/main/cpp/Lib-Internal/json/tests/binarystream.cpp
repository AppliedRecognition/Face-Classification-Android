#include <boost/test/unit_test.hpp>

#include <applog/core.hpp>
#include <stdext/binarystream.hpp>

#include <random>

BOOST_AUTO_TEST_SUITE(json)

BOOST_AUTO_TEST_CASE(binarystream_tests) {
    {
        stdx::binarystream in({});
        BOOST_CHECK(in);
        char c;
        in >> c;
        BOOST_CHECK(in.eof());
    }

    {
        static constexpr auto str = "HelloWorld!";
        stdx::binarystream in({str,strlen(str)});
        std::string out;
        while (in) {
            char c;
            in >> c;
            if (in.eof()) break;
            out.push_back(c);
        }
        BOOST_CHECK(in.eof());
        BOOST_CHECK_EQUAL(str, out);
    }

    {
        std::mt19937 rgen(1);
        std::uniform_int_distribution<unsigned> ub(0, 255);

        static constexpr auto N = 1024*1024;
        std::vector<std::byte> orig;
        orig.reserve(N);
        for (auto n = N; n > 0; --n)
            orig.push_back(std::byte(ub(rgen)));
        BOOST_REQUIRE_EQUAL(N, orig.size());

        stdx::binarystream in({orig.data(),orig.size()});
        std::vector<std::byte> out;
        char buf[256];
        while (in) {
            auto len = ub(rgen);
            if (2 <= len && out.size() + len < orig.size()) {
                auto sbuf = in.rdbuf();
                BOOST_REQUIRE(sbuf);
                auto r = sbuf->sgetn(buf,len);
                BOOST_REQUIRE_EQUAL(r,len);
                auto p = reinterpret_cast<std::byte*>(buf);
                out.insert(out.end(),p,p+len);
            }
            else {
                auto sbuf = in.rdbuf();
                BOOST_REQUIRE(sbuf);
                auto c = sbuf->sbumpc();
                if (c == EOF) {
                    in.setstate(std::ios::eofbit);
                    break;
                }
                out.push_back(std::byte(c));
            }
        }
        BOOST_CHECK(in.eof());
        BOOST_REQUIRE_EQUAL(orig.size(), out.size());
        BOOST_REQUIRE_EQUAL(memcmp(orig.data(), out.data(), out.size()),0);
    }
}

BOOST_AUTO_TEST_SUITE_END()
