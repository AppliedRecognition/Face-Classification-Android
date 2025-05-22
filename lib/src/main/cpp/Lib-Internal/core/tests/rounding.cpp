#include <boost/test/unit_test.hpp>
#include <applog/core.hpp>

#include <stdext/rounding.hpp>

BOOST_AUTO_TEST_SUITE(core)

template <typename T>
static void test_limits() {
    const long min = std::numeric_limits<T>::min();
    const long max = std::numeric_limits<T>::max();
    BOOST_CHECK_EQUAL(min, stdx::round_to<T>(min-1));
    BOOST_CHECK_EQUAL(max, stdx::round_to<T>(max+1));
    BOOST_CHECK_EQUAL(min, stdx::round_to<T>(float(min-1)));
    BOOST_CHECK_EQUAL(max, stdx::round_to<T>(2*float(max)));
}

template <typename T>
static void check_equal(T a, T b) {
    BOOST_CHECK_EQUAL(a,b);
}

BOOST_AUTO_TEST_CASE(rounding) {
    BOOST_CHECK_EQUAL(-2, stdx::round_to<int>(-1.6));
    BOOST_CHECK_EQUAL(-1, stdx::round_to<int>(-1.4));
    BOOST_CHECK_EQUAL(-1, stdx::round_to<int>(-0.7));
    BOOST_CHECK_EQUAL(0, stdx::round_to<int>(-0.4));
    BOOST_CHECK_EQUAL(0, stdx::round_to<int>(0.4));
    BOOST_CHECK_EQUAL(1, stdx::round_to<int>(0.7));
    BOOST_CHECK_EQUAL(1, stdx::round_to<int>(1.4));
    BOOST_CHECK_EQUAL(2, stdx::round_to<int>(1.6));

    check_equal<int>(-2, stdx::round_from(-1.6));
    check_equal<int>(-1, stdx::round_from(-1.4));
    check_equal<int>(-1, stdx::round_from(-0.7));
    check_equal<int>(0, stdx::round_from(-0.4));
    check_equal<int>(0, stdx::round_from(0.4));
    check_equal<int>(1, stdx::round_from(0.7));
    check_equal<int>(1, stdx::round_from(1.4));
    check_equal<int>(2, stdx::round_from(1.6));

    test_limits<signed char>();
    test_limits<unsigned char>();
    test_limits<signed short>();
    test_limits<unsigned short>();

    // ensure integer rounding is constexpr
    static constexpr auto c0 = stdx::round_to<unsigned char>(-1);
    static constexpr auto c1 = stdx::round_to<unsigned char>(1);
    static constexpr auto c2 = stdx::round_to<unsigned char>(1000);
    static_assert(c0 == 0 && c1 == 1 && c2 == 255);
}

BOOST_AUTO_TEST_SUITE_END()
