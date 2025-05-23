#include <boost/test/unit_test.hpp>
#include <applog/base_directory.hpp>
#include <applog/core.hpp>

#include <raw_image_io/io.hpp>
#include <raw_image/transform.hpp>


BOOST_AUTO_TEST_SUITE(raw_image)

static unsigned long
line_diff(const unsigned char* a, const unsigned char* b, unsigned n) {
    unsigned long e = 0;
    for ( ; n > 0; --n, ++a, ++b) {
        const auto d = *a - *b;
        e += unsigned(d*d);
    }
    return e;
}

static unsigned long
pixel_diff(const plane& a, const plane& b) {
    throw_if_invalid(a);
    throw_if_invalid(b);
    BOOST_CHECK(a.data != b.data);
    BOOST_CHECK_EQUAL(a.width, b.width);
    BOOST_CHECK_EQUAL(a.height, b.height);
    BOOST_CHECK_EQUAL(a.layout, b.layout);
    const auto bpp = bytes_per_pixel(a.layout);
    BOOST_CHECK(a.width*bpp <= a.bytes_per_line);
    BOOST_CHECK(b.width*bpp <= b.bytes_per_line);
    unsigned long e = 0;
    const auto* ap = a.data;
    const auto* bp = b.data;
    for (auto i = a.height; i > 0; --i,
             ap += a.bytes_per_line, bp += b.bytes_per_line)
        e += line_diff(ap,bp,a.width*bpp);
    return e / bpp;
}

/*
static bool same_pixels(const plane& a, const plane& b) {
    return pixel_diff(a, b) == 0;
}
*/


BOOST_AUTO_TEST_CASE(raw_image_multi_plane) {
    FILE_LOG(logINFO) << "multi-plane: start";

    const auto base_path =
        base_directory("lib-internal") / "raw_image_io" / "tests";
    const auto img_path = base_path / "image_037.jpg";

    const auto src_img = load(img_path, pixel::rgb24);
    FILE_LOG(logDETAIL) << "image: "
                        << src_img->width << 'x' << src_img->height;

    const auto chip =
        extract_region(*src_img, 200, 200, 100, 100, 13, 75, 75, pixel::rgb24);
    raw_image::plane_ptr chip0;
    
    for (unsigned rot = 0; rot < 8; ++rot) {
        auto r = copy_rotate(src_img, rot);
        auto nv21 = create_nv21(*r);
        auto z = copy(
            *nv21, rotate(nv21->front().rotate), src_img->layout);
        const auto img_err =
            float(pixel_diff(*src_img, *z)) / float(src_img->width*src_img->height);
        BOOST_CHECK(img_err < 2.25);

        auto c =
            extract_region(*nv21, 200, 200, 100, 100, 13, 75, 75, pixel::rgb24);
        const auto chip_err =
            float(pixel_diff(*chip, *c)) / float(c->width*c->height);
        BOOST_CHECK(chip_err < 3.125);
        float c0_err = 0;
        if (chip0) {
            c0_err = float(pixel_diff(*chip0, *c)) / float(c->width*c->height);
            BOOST_CHECK(c0_err < 0.125);
        }
        else
            chip0 = move(c);

        FILE_LOG(logDETAIL) << "== rot " << rot
                            << " error " << img_err
                            << ' ' << chip_err
                            << ' ' << c0_err;

        //std::stringstream ss;
        //ss << "test_" << rot << ".jpg";
        //save(ss.str(), copy(*nv21, pixel::rgb24));
    }

    FILE_LOG(logINFO) << "multi-plane: done";
}

BOOST_AUTO_TEST_SUITE_END()
