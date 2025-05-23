#include <boost/test/unit_test.hpp>

#include <dlibx/rotated_box.hpp>

#include <applog/core.hpp>
//#include <random>

BOOST_AUTO_TEST_SUITE(dlibx)

BOOST_AUTO_TEST_CASE(rotated_box) {
    const auto rbox0 = raw_image::rotated_box {
        { 2,3 },
        4, 5,
        0.125
    };
    const auto chip0 = to_chip_details(rbox0);

    const auto rbox1 = to_rotated_box(chip0);
    BOOST_CHECK_EQUAL(rbox0.center.x, rbox1.center.x);
    BOOST_CHECK_EQUAL(rbox0.center.y, rbox1.center.y);
    BOOST_CHECK_EQUAL(rbox0.width,    rbox1.width);
    BOOST_CHECK_EQUAL(rbox0.height,   rbox1.height);
    BOOST_CHECK_EQUAL(rbox0.angle,    rbox1.angle);

    const auto chip1 = to_chip_details(rbox1);
    BOOST_CHECK_EQUAL(chip0.rect,  chip1.rect);
    BOOST_CHECK_EQUAL(chip0.angle, chip1.angle);

    ::dlib::chip_details chip2;
    chip2.rect = { 3, 5, 7, 16 };
    chip2.angle = -0.25;
    const auto rbox2 = to_rotated_box(chip2);

    const auto chip3 = to_chip_details(rbox2);
    BOOST_CHECK_EQUAL(chip2.rect,  chip3.rect);
    BOOST_CHECK_EQUAL(chip2.angle, chip3.angle);

    const auto rbox3 = to_rotated_box(chip2);
    BOOST_CHECK_EQUAL(rbox2.center.x, rbox3.center.x);
    BOOST_CHECK_EQUAL(rbox2.center.y, rbox3.center.y);
    BOOST_CHECK_EQUAL(rbox2.width,    rbox3.width);
    BOOST_CHECK_EQUAL(rbox2.height,   rbox3.height);
    BOOST_CHECK_EQUAL(rbox2.angle,    rbox3.angle);
}

BOOST_AUTO_TEST_SUITE_END()
