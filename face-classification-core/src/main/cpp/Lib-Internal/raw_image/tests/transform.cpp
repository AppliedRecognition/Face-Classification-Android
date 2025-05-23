#include <cmath>
#include <boost/test/unit_test.hpp>
#include <applog/core.hpp>

#include <raw_image/transform.hpp>
#include <raw_image/reader.hpp>


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

static bool same_pixels(const plane& a, const plane& b) {
    return pixel_diff(a, b) == 0;
}

static float pixel_error(const plane& a, const plane& b) {
    return std::sqrt(float(pixel_diff(a, b)) / float(a.width) / float(a.height));
}

static plane_ptr warp_rotate(const plane& img, unsigned rotate) {
    BOOST_REQUIRE(rotate < 4);
    const auto sw = (img.rotate&1) ? img.height : img.width;
    const auto sh = (img.rotate&1) ? img.width : img.height;
    const auto cx = float(sw)/2;
    const auto cy = float(sh)/2;
    const auto dw = (rotate&1) ? sh : sw;
    const auto dh = (rotate&1) ? sw : sh;
    return extract_region(
        img, cx, cy, float(dw), float(dh), float(rotate)*90, dw, dh);
}

static void test_quadrant_rotate(const plane& orig, unsigned rotate) {
    BOOST_REQUIRE(rotate < 4);
    BOOST_REQUIRE_EQUAL(orig.width&1,  0);
    BOOST_REQUIRE_EQUAL(orig.height&1, 0);

    auto i0 = copy_rotate(orig);
    BOOST_REQUIRE_EQUAL(i0->rotate, 0);
    const auto oqw = float(i0->width)/2;
    const auto oqh = float(i0->height)/2;

    in_place_rotate(*i0, rotate);
    const auto qw = i0->width / 2;
    const auto qh = i0->height / 2;

    const plane qn[] = {
        crop(*i0, 0,   0, qw, qh),
        crop(*i0, qw,  0, qw, qh),
        crop(*i0, qw, qh, qw, qh),
        crop(*i0, 0,  qh, qw, qh)
    };

    auto cx = oqw / 2;
    auto cy = oqh / 2;
    for (auto i = 0u; i < 4; ++i) {
        switch (i) {
        case 1: cx += oqw; break;
        case 2: cy += oqh; break;
        case 3: cx -= oqw; break;
        default: break;
        }
        const auto qi = extract_region(
            orig, cx, cy, float(qw), float(qh), float(rotate)*90, qw, qh);
        auto e = pixel_error(qn[(i-rotate)&3], *qi);
        if (fabsf(e) >= 1e-5) {
            FILE_LOG(logWARNING) << orig.rotate << ' ' << rotate << ' ' << i << ' ' << e;
            BOOST_CHECK(fabsf(e) < 1e-5);
            break;
        }
    }
}

static void do_tests(const plane& orig) {
    {
        auto c = copy(orig);
        BOOST_CHECK(same_pixels(orig, *c));
        copy_pixels(orig, *c);
        BOOST_CHECK(same_pixels(orig, *c));
    }
    {
        auto proc = copy_flip(orig);
        in_place_flip(*proc);
        BOOST_CHECK(same_pixels(orig, *proc));
    }
    {
        auto proc = copy_mirror(orig);
        in_place_mirror(*proc);
        BOOST_CHECK(same_pixels(orig, *proc));
    }
    {
        auto proc = copy_transpose(orig);
        throw_if_invalid(*proc);
        in_place_transpose(*proc);
        BOOST_CHECK(same_pixels(orig, *proc));
    }

    // rotate (basic test)
    for (unsigned r = 0; r < 8; ++r) {
        auto proc = copy_rotate(orig, r);
        switch (r) {
        case 0:
            BOOST_CHECK(same_pixels(orig, *proc));
            BOOST_CHECK_EQUAL(proc->rotate, 0);
            break;
        case 1: BOOST_CHECK_EQUAL(proc->rotate, 3); break;
        case 2: BOOST_CHECK_EQUAL(proc->rotate, 2); break;
        case 3: BOOST_CHECK_EQUAL(proc->rotate, 1); break;
        case 4: BOOST_CHECK_EQUAL(proc->rotate, 4); break;
        case 5: BOOST_CHECK_EQUAL(proc->rotate, 5); break;
        case 6: BOOST_CHECK_EQUAL(proc->rotate, 6); break;
        case 7: BOOST_CHECK_EQUAL(proc->rotate, 7); break;
        default: assert(!"cannot happen");
        }
        auto proc2 = copy(reader::construct(orig), rotate(r));
        BOOST_CHECK(same_pixels(*proc, *proc2));
        in_place_rotate(*proc);
        BOOST_CHECK(same_pixels(orig, *proc));
    }
    
    // rotate (advanced)
    BOOST_REQUIRE_EQUAL(orig.rotate, 0);
    for (auto o = orig; o.rotate < 8; ++o.rotate) {
        const auto target = copy_rotate(o);
        BOOST_REQUIRE_EQUAL(target->rotate, 0);
        for (unsigned rot0 = 0; rot0 < 8; ++rot0) {
            auto proc = copy_rotate(o, rot0);
            BOOST_CHECK((rot0 == o.rotate) == (proc->rotate == 0));
            in_place_rotate(*proc);
            BOOST_CHECK(proc->rotate == 0);
            BOOST_CHECK(same_pixels(*target, *proc));
        }
    }

    // use extract_region to crop
    for (auto x = 0u; x < orig.width; ++x)
        for (auto y = 0u; y < orig.height; ++y)
            for (auto w = orig.width - x; w > 0; --w)
                for (auto h = orig.height - y; h > 0; --h) {
                    const auto i0 = crop(orig, x, y, w, h);
                    const auto cx = float(2*x+w)/2;
                    const auto cy = float(2*y+h)/2;
                    const auto i1 =
                        extract_region(orig, cx, cy, float(w), float(h), 0,
                                       w, h);
                    auto e = pixel_error(i0, *i1);
                    if (fabsf(e) >= 1e-5) {
                        FILE_LOG(logWARNING) << w << 'x' << h << '+' << x << '+' << y << ' ' << e;
                        BOOST_CHECK(fabsf(e) < 1e-5);
                        return;
                    }
                }

    // use extract_region to scale
    if (((orig.width|orig.height)&1) == 0) {
        const auto w = float(orig.width);
        const auto h = float(orig.height);
        auto i0 = extract_region(orig, w/2, h/2, w, h, 0,
                                 orig.width/2, orig.height/2);
        i0->scale = 1;
        auto i1 = extract_region(*i0, w/2, h/2, w, h, 0,
                                 orig.width/2, orig.height/2);
        auto e1 = pixel_error(*i0, *i1);
        BOOST_CHECK(fabsf(e1) < 1e-5);
        auto i2 = copy_resize(orig, orig.width/2, orig.height/2);
        auto e2 = pixel_error(*i0, *i2);
        BOOST_CHECK(fabsf(e2) < 1e-5);
    }
    
    // use extract_region to rotate (basic test)
    for (unsigned r = 0; r < 4; ++r) {
        auto i0 = copy_rotate(orig, r);
        auto i1 = warp_rotate(orig, r);
        auto e = pixel_error(*i0, *i1);
        if (fabsf(e) >= 1e-5) {
            FILE_LOG(logWARNING) << "rot " << r << " err " << e;
            BOOST_CHECK(fabsf(e) < 1e-5);
            return;
        }
    }

    // use extract_region to rotate (advanced)
    if (orig.width >= 4 && orig.height >= 4) {
        BOOST_REQUIRE_EQUAL(orig.rotate, 0);
        for (auto o = orig; o.rotate < 8; ++o.rotate) {
            const auto target = copy_rotate(o);
            BOOST_REQUIRE_EQUAL(target->rotate, 0);
            for (unsigned rot0 = 0; rot0 < 4; ++rot0) {
                auto i0 = copy_rotate(*target, rot0);
                auto i1 = warp_rotate(o, rot0);
                auto e = pixel_error(*i0, *i1);
                if (fabsf(e) >= 1e-5) {
                    FILE_LOG(logWARNING) << o.rotate << ' ' << rot0 << ' ' << e;
                    BOOST_CHECK(fabsf(e) < 1e-5);
                    return;
                }
            }
            if (((o.width|o.height)&1) == 0) {
                //FILE_LOG(logINFO) << "quads: " << orig.width << 'x' << orig.height;
                for (unsigned r = 0; r < 4; ++r)
                    test_quadrant_rotate(o, r);
            }
        }
    }

        /*
        if (orig.width == 14 && orig.height == 16 &&
            orig.layout == pixel::gray8) {
            std::stringstream ss;
            ss << r << ".jpg";
            save("ta"+ss.str(), *i0);
            save("tb"+ss.str(), *i1);
        }
        */
}

static void do_tests(pixel_layout color, unsigned w, unsigned h) {
    //FILE_LOG(logINFO) << "size: " << w << 'x' << h << ' ' << int(color);
    auto img = create(w, h, color);
    unsigned c = 66;
    for (auto p = img->data,
             end = img->data + h*img->bytes_per_line; p != end; ++p)
        *p = (c += 113) & 0xff;
    auto copy = *img;
    throw_if_invalid(*img);
    do_tests(*img);
    throw_if_invalid(*img);
    c = 66;
    for (auto p = copy.data,
             end = copy.data + h*copy.bytes_per_line; p != end; ++p)
        BOOST_CHECK_EQUAL(*p, (c += 113) & 0xff);
}


BOOST_AUTO_TEST_CASE(raw_image) {
    FILE_LOG(logINFO) << "transform: start";

    for (auto h : { 1u, 2u, 5u, 8u, 10u, 13u, 16u } )
        for (auto w : { 1u, 2u, 5u, 8u, 10u, 13u, 16u } )
            for (auto color : { pixel::gray8, pixel::bgr24, pixel::argb32 } )
                do_tests(color, w, h);

    FILE_LOG(logINFO) << "transform: done";
}

BOOST_AUTO_TEST_SUITE_END()
