#include <boost/test/unit_test.hpp>

#include <raw_image/types.hpp>
#include <stdext/arg.hpp>
#include <stdext/span.hpp>
#include <stdext/binary.hpp>
#include <applog/core.hpp>

BOOST_AUTO_TEST_SUITE(raw_image)

static_assert(stdx::can_extract_pointer<plane_ptr,plane>::value);
static_assert(stdx::can_extract_pointer<plane_ptr,const plane>::value);
static_assert(stdx::can_extract_pointer<plane&,plane>::value);
static_assert(stdx::can_extract_pointer<plane&,const plane>::value);
static_assert(stdx::can_extract_pointer<const plane&,const plane>::value);
static_assert(!stdx::can_extract_pointer<plane&&,plane>::value);
static_assert(!stdx::can_extract_pointer<plane&&,const plane>::value);

BOOST_AUTO_TEST_CASE(span) {
    {
        plane raw;
        stdx::spanarg<plane> s(raw);
        BOOST_CHECK(!s.empty());
        BOOST_CHECK_EQUAL(s.size(), 1);
        stdx::spanarg<const plane> c(raw);
        BOOST_CHECK_EQUAL(c.size(), 1);
        stdx::span<const plane> c2(s);
        BOOST_CHECK_EQUAL(c2.size(), 1);
        stdx::spanarg<const plane> c3(c2);
        BOOST_CHECK_EQUAL(c3.size(), 1);
    }
    {
        const plane raw;
        //stdx::span<plane> s(raw);
        stdx::spanarg<const plane> c(raw);
        BOOST_CHECK_EQUAL(c.size(), 1);
    }

    {
        std::unique_ptr<plane> ptr;
        stdx::span<plane> s(ptr);
        BOOST_CHECK(s.empty());
        BOOST_CHECK_EQUAL(s.size(), 0);
        stdx::span<const plane> c1(ptr);
        BOOST_CHECK_EQUAL(c1.size(), 0);
        stdx::span<const plane> c2(s);
        BOOST_CHECK_EQUAL(c2.size(), 0);
        stdx::spanarg<const plane> c3(ptr);
        BOOST_CHECK_EQUAL(c3.size(), 0);
    }
    {
        auto ptr = std::make_unique<plane>();
        stdx::span<plane> s(ptr);
        BOOST_CHECK_EQUAL(s.size(), 1);
        stdx::span<const plane> c(ptr);
        BOOST_CHECK_EQUAL(c.size(), 1);
    }
    {
        auto ptr = std::make_unique<const plane>();
        //stdx::span<plane> s(ptr);
        stdx::span<const plane> c(ptr);
        BOOST_CHECK_EQUAL(c.size(), 1);
    }

    {
        plane arr[2] = { {}, {} };
        stdx::span<plane> s(arr);
        BOOST_CHECK_EQUAL(s.size(), 2);
        stdx::span<const plane> c1(arr);
        BOOST_CHECK_EQUAL(c1.size(), 2);
        stdx::spanarg<const plane> c2(arr);
        BOOST_CHECK_EQUAL(c2.size(), 2);
    }
    {
        const plane arr[2] = { {}, {} };
        //stdx::span<plane> s(arr);
        stdx::span<const plane> c(arr);
        BOOST_CHECK_EQUAL(c.size(), 2);
    }
    {
        std::array<plane,2> arr;
        stdx::span<plane> s(arr);
        BOOST_CHECK_EQUAL(s.size(), 2);
        stdx::span<const plane> c(arr);
        BOOST_CHECK_EQUAL(c.size(), 2);
    }
    {
        std::array<const plane,2> arr;
        //stdx::span<plane> s(arr);
        stdx::span<const plane> c(arr);
        BOOST_CHECK_EQUAL(c.size(), 2);
    }
    {
        std::vector<plane> arr(2);
        stdx::span<plane> s(arr);
        BOOST_CHECK_EQUAL(s.size(), 2);
        stdx::span<const plane> c1(arr);
        BOOST_CHECK_EQUAL(c1.size(), 2);
        stdx::spanarg<const plane> c2(arr);
        BOOST_CHECK_EQUAL(c2.size(), 2);
    }
    {
        const std::vector<plane> arr(2);
        //stdx::span<plane> s(arr);
        stdx::span<const plane> c(arr);
        BOOST_CHECK_EQUAL(c.size(), 2);
    }
    {
        stdx::binary b("hello");
        auto s0 = stdx::span<const std::byte>(b);
        auto s1 = stdx::span<const unsigned char>(b);
        auto s2 = stdx::span<const char>(b);
        BOOST_REQUIRE(!s0.empty());
        BOOST_CHECK_EQUAL(s0.size(),s1.size());
        BOOST_CHECK_EQUAL(s0.size(),s2.size());
        const void* d0 = s0.data();
        const void* d1 = s1.data();
        const void* d2 = s2.data();
        BOOST_CHECK_EQUAL(d0,b.data());
        BOOST_CHECK_EQUAL(d0,d1);
        BOOST_CHECK_EQUAL(d0,d2);
        BOOST_CHECK_EQUAL(s0.data(),b.data<std::byte>());
        BOOST_CHECK_EQUAL(s1.data(),b.data<unsigned char>());
        BOOST_CHECK_EQUAL(s2.data(),b.data<char>());
        BOOST_CHECK_EQUAL(s2[0],'h');
    }
}

BOOST_AUTO_TEST_SUITE_END()
