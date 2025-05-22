#include <cmath>
#include <boost/test/unit_test.hpp>
#include <applog/core.hpp>

#include <raw_image/core.hpp>
#include <raw_image/color_convert.hpp>


BOOST_AUTO_TEST_SUITE(raw_image)

static constexpr auto all_types = {
    pixel::gray8,
    pixel::yuv24_jpeg,
    pixel::bgr24,
    pixel::rgb24,
    pixel::argb32,
    pixel::abgr32,
    pixel::rgba32,
    pixel::bgra32,
};

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

static bool same_pixels(const plane& a, const plane& b) {
    return pixel_diff(a, b) == 0;
}

/*
static float pixel_error(const plane& a, const plane& b) {
    return std::sqrt(float(pixel_diff(a, b)) / float(a.width) / float(a.height));
}
*/

static void zero_alpha(const plane& img) {
    if (bytes_per_pixel(img) == 4) {
        auto line = img.data;
        if (img.layout == pixel::rgba32 ||
            img.layout == pixel::bgra32)
            line += 3;
        for (auto i = img.height; i > 0; --i, line += img.bytes_per_line) {
            auto px = line;
            for (auto j = img.width; j > 0; --j, px += 4)
                *px = 0;
        }
    }
}

static void do_rbg_tests(const plane& src) {
    for (auto dest_type : all_types) {
        auto rgb0 = copy(src, pixel::bgr24);
        auto rgb1 = copy(src, pixel::rgb24);
        auto rgb2 = copy(src, pixel::argb32);

        if (bytes_per_pixel(src.layout) >= 4)
            zero_alpha(*rgb2);

        const auto d0 = copy(rgb0, dest_type);
        const auto d1 = copy(rgb1, dest_type);
        BOOST_CHECK(same_pixels(*d0,*d1));

        const auto d2 = copy(rgb2, dest_type);
        BOOST_CHECK(same_pixels(*d0,*d2));

        if (auto p = convert(*rgb0, dest_type)) {
            BOOST_CHECK_EQUAL(bytes_per_pixel(dest_type), 4);
            rgb0 = move(p);
        }
        BOOST_CHECK(same_pixels(*rgb0,*d0));

        if (auto p = convert(*rgb1, dest_type)) {
            BOOST_CHECK_EQUAL(bytes_per_pixel(dest_type), 4);
            rgb1 = move(p);
        }
        BOOST_CHECK(same_pixels(*rgb0,*rgb1));

        BOOST_REQUIRE(!convert(*rgb2, dest_type));
        BOOST_CHECK(same_pixels(*rgb0,*rgb2));
    }
}

static void do_yuv_tests(const plane& src) {
    const auto y0 = copy(src, pixel::gray8);

    auto yuv = copy(src, pixel::yuv24_jpeg);
    const auto y1 = copy(yuv, pixel::gray8);
    BOOST_CHECK(same_pixels(*y0,*y1));

    BOOST_REQUIRE(!convert(*yuv, pixel::gray8));
    BOOST_CHECK(same_pixels(*y0,*yuv));
}

static void do_gray_tests(const plane& src) {
    if (src.layout != pixel::gray8) return;
    auto yuv = copy(src, pixel::yuv24_jpeg);
    auto rgb0 = copy(src, pixel::rgb24);
    auto rgb1 = copy(yuv, pixel::rgb24);
    BOOST_CHECK(same_pixels(*rgb0,*rgb1));
    BOOST_REQUIRE(!convert(*yuv, pixel::gray8));
    BOOST_CHECK(same_pixels(*yuv,src));
    BOOST_REQUIRE(!convert(*rgb0, pixel::gray8));
    BOOST_CHECK(same_pixels(*rgb0,src));
}

static void do_expand_tests(const plane& src) {
    if (src.layout == pixel::argb32) return;
    for (auto mid_type : all_types) {
        if (mid_type == pixel::argb32) continue;
        for (auto final_type : all_types) {
            if (bytes_per_pixel(final_type) <= bytes_per_pixel(mid_type))
                continue;
            auto big = copy(src, pixel::argb32);

            auto mid = copy(big, mid_type);
            BOOST_REQUIRE(!convert(*big, mid_type));
            BOOST_CHECK(same_pixels(*mid, *big));

            auto fin = copy(mid, final_type);
            BOOST_REQUIRE(!convert(*big, final_type));
            BOOST_CHECK(same_pixels(*fin, *big));
        }
    }
}

static constexpr auto noteq(const std::array<uint8_t, 4>& a, const std::array<uint8_t, 4>& b) {
    return a[0] != b[0] || a[1] != b[1] || a[2] != b[2] || a[3] != b[3];
}

template <pixel_color c1, pixel_color c2>
static void constexpr_tests() {
    {
        static constexpr auto a1 = to_layout<pixel::rgb24>(c1);
        static constexpr auto b1 = to_layout<pixel::bgr24>(c1);
        static_assert(a1[0] == b1[2] && a1[1] == b1[1] && a1[2] == b1[0]);

        static constexpr auto a2 = to_layout<pixel::rgb24>(c2);
        static constexpr auto b2 = to_layout<pixel::bgr24>(c2);
        static_assert(a2[0] == b2[2] && a2[1] == b2[1] && a2[2] == b2[0]);
        static_assert(noteq(a1,a2));
    }

    {
        static constexpr auto u1 = to_layout<pixel::uv16_jpeg>(c1);
        static constexpr auto v1 = to_layout<pixel::vu16_jpeg>(c1);
        static_assert(u1[0] == v1[1] && u1[1] == v1[0] && u1[2] == v1[2]);

        static constexpr auto u2 = to_layout<pixel::uv16_jpeg>(c2);
        static constexpr auto v2 = to_layout<pixel::vu16_jpeg>(c2);
        static_assert(u2[0] == v2[1] && u2[1] == v2[0] && u2[3] == v2[3]);
        static_assert(noteq(u1,u2));
    }

    {
        static constexpr auto u1 = to_layout<pixel::uv16_nv21>(c1);
        static constexpr auto v1 = to_layout<pixel::vu16_nv21>(c1);
        static_assert(u1[0] == v1[1] && u1[1] == v1[0]);

        static constexpr auto u2 = to_layout<pixel::uv16_nv21>(c2);
        static constexpr auto v2 = to_layout<pixel::vu16_nv21>(c2);
        static_assert(u2[0] == v2[1] && u2[1] == v2[0]);
        static_assert(noteq(u1,u2));
    }
}


BOOST_AUTO_TEST_CASE(raw_image_color_convert) {
    FILE_LOG(logINFO) << "color conversions: start";

    constexpr_tests<color_black, color_red>();
    constexpr_tests<color_green, color_blue>();
    constexpr_tests<color_cyan, color_yellow>();
    constexpr_tests<color_magenta, color_white>();

    // random pixel data
    std::vector<unsigned char> data;
    data.reserve(1024);
    unsigned x = 0;
    for (auto p : { 37u, 199u, 41u, 79u })
        for (auto i = 256; i > 0; --i)
            data.push_back((x += p) & 0xff);

    for (auto src_type : all_types) {
        plane src;
        src.data = data.data();
        src.layout = src_type;
        for (auto w : { 7u, 8u, 9u, 13u }) {
            src.width = w;
            src.bytes_per_line = w * bytes_per_pixel(src_type);
            for (auto h : { 3u, 4u, 5u, 12u }) {
                src.height = h;
                do_rbg_tests(src);
                do_yuv_tests(src);
                do_gray_tests(src);
                do_expand_tests(src);
            }
        }
    }

    // rgb -> yuv -> rgb
    const auto px = create(1,1,pixel::rgba32);
    BOOST_CHECK(manages_pixel_buffer(px));
    unsigned n0 = 0, n1 = 0, n2 = 0;
    for (int32_t v = 0; v < 256*256*256; v += 17) {
        *reinterpret_cast<int32_t*>(px->data) = v;
        BOOST_REQUIRE(!convert(*px, pixel::yuv24_jpeg));
        BOOST_REQUIRE_EQUAL(px->layout, pixel::yuv24_jpeg);
        BOOST_REQUIRE(!convert(*px, pixel::rgba32));
        BOOST_REQUIRE_EQUAL(px->layout, pixel::rgba32);
        const auto y = *reinterpret_cast<const int32_t*>(px->data);

        if (const auto e0 = (v>>16) - (y>>16))
            BOOST_CHECK_EQUAL(e0*e0, 1);
        else
            ++n0;
        
        if (const auto e1 = ((v>>8)&0xff) - ((y>>8)&0xff))
            BOOST_CHECK(e1*e1 <= 4);
        else
            ++n1;

        if (const auto e2 = (v&0xff) - (y&0xff))
            BOOST_CHECK_EQUAL(e2*e2, 1);
        else
            ++n2;
    }
    BOOST_CHECK(n0 >= 128*256*256/17);
    BOOST_CHECK(n1 >= 128*256*256/17);
    BOOST_CHECK(n2 >= 128*256*256/17);

    FILE_LOG(logINFO) << "color conversions: done";
}

BOOST_AUTO_TEST_SUITE_END()
