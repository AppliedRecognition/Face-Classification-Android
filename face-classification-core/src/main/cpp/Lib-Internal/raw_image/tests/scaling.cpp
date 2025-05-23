#include <cmath>
#include <boost/test/unit_test.hpp>
#include <applog/core.hpp>

#include <raw_image/reader.hpp>
#include <raw_image/concat.hpp>


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

static auto make_test_image(unsigned width, unsigned height,
                            pixel_layout layout) {
    BOOST_REQUIRE(0 < width && width < 16);
    BOOST_REQUIRE(0 < height && height < 16);
    const auto bpp = bytes_per_pixel(layout);
    BOOST_REQUIRE(0 < bpp && bpp <= 4);
    auto img = create(width, height, layout);
    auto line = img->data;
    for (unsigned j = 1; j <= height; ++j, line += img->bytes_per_line) {
        auto px = line;
        for (unsigned i = 1; i <= width; ++i, px += bpp) {
            px[0] = ((j<<4) + i) & 0xff;
            switch (bpp) {
            case 4:
                px[3] = px[0] ^ 0xff;
                [[fallthrough]];
            case 3:
                px[2] = px[0] ^ 0xf0;
                [[fallthrough]];
            case 2:
                px[1] = px[0] ^ 0x0f;
                //[[fallthrough]];
            }
        }
    }
    return img;
}
                       

// make block of single pixel (replicated)
static auto make_block(unsigned width, unsigned height,
                       pixel_layout layout,
                       const unsigned char* px) {
    BOOST_REQUIRE(width > 0 && height > 0);
    const auto bpp = bytes_per_pixel(layout);
    BOOST_REQUIRE(0 < bpp && bpp <= 4);
    auto img = create(width, height, layout);
    auto row = img->data;
    for (auto j = height; j > 0; --j, row += img->bytes_per_line) {
        auto el = row;
        for (auto i = width; i > 0; --i, el += bpp)
            memcpy(el, px, bpp);
    }
    return img;
}

static auto upscale(const plane& img,
                    unsigned sw, unsigned sh) {
    const auto bpp = bytes_per_pixel(img.layout);
    std::vector<plane_ptr> rows, cols;
    rows.reserve(img.height);
    cols.reserve(img.width);
    auto line = img.data;
    for (auto j = img.height; j > 0; --j, line += img.bytes_per_line) {
        auto px = line;
        for (auto i = img.width; i > 0; --i, px += bpp)
            cols.push_back(make_block(sw, sh, img.layout, px));
        rows.push_back(concat_horz(cols.begin(), cols.end()));
        cols.clear();
    }
    return concat_vert(rows.begin(), rows.end());
}

static auto test_scale(const plane& src,
                       unsigned w, unsigned h) {
    auto scaler = scale_area(reader::construct(src), w, h);
    BOOST_REQUIRE(scaler);
    auto dest = create(
        scaler->width(), scaler->height(), scaler->layout());
    scaler->copy_to(*dest, dest->bytes_per_line);
    return dest;
}



BOOST_AUTO_TEST_CASE(raw_image_scaling) {
    FILE_LOG(logINFO) << "scaling: start";

    // scale tests
    for (auto cs : { pixel::gray8, pixel::uv16_nv21,
                pixel::yuv24_jpeg, pixel::bgra32 }) {
        for (auto h : { 1, 3, 4, 10, 15 })
            for (auto w : { 1, 6, 12, 15 }) {
                const auto src = make_test_image(unsigned(w), unsigned(h), cs);
                FILE_LOG(logDETAIL) << diag(src);
                for (unsigned sh = 1; sh <= 5; ++sh)
                    for (unsigned sw = 1; sw <= 5; ++sw) {
                        auto up1 = upscale(*src, sw, sh);

                        auto up2 = test_scale(*src, up1->width, up1->height);
                        BOOST_CHECK(same_pixels(*up2, *up1));

                        auto down1 = test_scale(*up1, src->width, src->height);
                        BOOST_CHECK(same_pixels(*down1, *src));
                    }
            }
    }

    {
        plane img;
        for (auto line : read_lines_of<pixel::rgb24>(img)) {
            for (auto px : line) {
                auto red = px[0], green = px[1], blue = px[2];
                auto sum = red + green + blue;
                (void)sum;
            }
        }
        for (auto line : read_lines_bpp<3>(img)) {
            for (auto px : line) {
                auto c0 = px[0], c1 = px[1], c2 = std::get<2>(px);
                auto sum = c0 + c1 + c2;
                (void)sum;
                //auto compile_error = std::get<3>(px);
            }
        }
        for (auto line : read_lines_of<pixel::gray8>(img)) {
            for (auto px : line) {
                unsigned gray = px;
                (void)gray;
            }
        }
        for (auto line : read_lines_bpp<1>(img)) {
            for (auto px : line) {
                unsigned c0 = px;
                (void)c0;
            }
        }
    }

    FILE_LOG(logINFO) << "scaling: done";
}

BOOST_AUTO_TEST_SUITE_END()
