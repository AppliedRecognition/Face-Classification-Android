#include <boost/test/unit_test.hpp>
#include <applog/core.hpp>

#include <raw_image/transform.hpp>
#include <raw_image/points.hpp>

#include <dlib/geometry/vector.h>
#include <opencv2/core/core.hpp>


BOOST_AUTO_TEST_SUITE(raw_image)

template <typename PT>
static void test_round_point(const PT& p) {
    const dlib::point dp0 = round_from(p);
    const auto dp1 = round_to<dlib::point>(p);
    BOOST_CHECK_EQUAL(dp0.x(), dp1.x());
    BOOST_CHECK_EQUAL(dp0.y(), dp1.y());

    const dlib::dpoint ddp0 = round_from(p);
    const auto ddp1 = round_to<dlib::dpoint>(p);
    BOOST_CHECK_CLOSE(ddp0.x(), ddp1.x(), 1e-5);
    BOOST_CHECK_CLOSE(ddp0.y(), ddp1.y(), 1e-5);

    const cv::Point cvp0 = round_from(p);
    const auto cvp1 = round_to<cv::Point>(p);
    BOOST_CHECK_EQUAL(cvp0.x, cvp1.x);
    BOOST_CHECK_EQUAL(cvp0.y, cvp1.y);

    BOOST_CHECK_EQUAL(cvp0.x, dp0.x());
    BOOST_CHECK_EQUAL(cvp0.y, dp0.y());

    const cv::Point2f cvfp0 = round_from(p);
    const auto cvfp1 = round_to<cv::Point2f>(p);
    BOOST_CHECK_CLOSE(cvfp0.x, cvfp1.x, 1e-5);
    BOOST_CHECK_CLOSE(cvfp0.y, cvfp1.y, 1e-5);

    BOOST_CHECK_CLOSE(cvfp0.x, ddp0.x(), 1e-5);
    BOOST_CHECK_CLOSE(cvfp0.y, ddp0.y(), 1e-5);
}

BOOST_AUTO_TEST_CASE(point_rounding) {
    test_round_point(dlib::point(3,5));
    test_round_point(dlib::dpoint(3.14,5.56));
    test_round_point(cv::Point(3,5));
    test_round_point(cv::Point2d(3.14,5.56));
}

BOOST_AUTO_TEST_CASE(point_inverse) {
    plane img;
    img.width = 71;
    img.height = 97;
    const auto p0 = cv::Point(12,8);
    for (img.scale = -1; img.scale <= 1; ++img.scale) {
        for (img.rotate = 0; img.rotate < 8; ++img.rotate) {
            const auto p1 = to_image_point(p0, img);
            const auto p2 = to_original_point(p1, img);
            BOOST_CHECK_EQUAL(p0.x, p2.x);
            BOOST_CHECK_EQUAL(p0.y, p2.y);
            const auto p3 = to_original_point(p0, img);
            const auto p4 = to_image_point(p3, img);
            BOOST_CHECK_EQUAL(p0.x, p4.x);
            BOOST_CHECK_EQUAL(p0.y, p4.y);
        }
    }
}

BOOST_AUTO_TEST_CASE(point_on_image) {
    const auto i0 = create(11,17,pixel::gray8);
    std::fill_n(i0->data, i0->height*i0->bytes_per_line, 0);

    const auto p0 = cv::Point(2,3);
    i0->data[2 + 3*i0->bytes_per_line] = 123;

    for (i0->rotate = 0; i0->rotate < 8; ++i0->rotate) {
        const auto i1 = copy_rotate(*i0);
        const auto p1 = to_original_point(p0, *i0);
        BOOST_REQUIRE(p1.x >= 0 && p1.y >= 0);
        const auto px = i1->data + p1.x + unsigned(p1.y)*i1->bytes_per_line;
        BOOST_CHECK_EQUAL(*px, 123);
    }
}

BOOST_AUTO_TEST_SUITE_END()
