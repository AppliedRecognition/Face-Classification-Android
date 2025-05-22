#include <boost/test/unit_test.hpp>
#include <applog/base_directory.hpp>
#include <applog/core.hpp>

#include <raw_image_io/io.hpp>
#include <raw_image/pixels.hpp>

#include <filesystem>
#include <fstream>


BOOST_AUTO_TEST_SUITE(raw_image)

template <typename T>
static constexpr auto sqr(T x) { return x*x; }

static auto operator==(const plane& a, const plane& b) {
    if (a.width != b.width || a.height != b.height || a.layout != b.layout)
        return false;
    const auto line_bytes = a.width * bytes_per_pixel(a.layout);
    auto linea = a.data, lineb = b.data;
    unsigned long long e = 0;
    for (auto j = a.height; j > 0; --j,
             linea += a.bytes_per_line, lineb += b.bytes_per_line)
        for (auto pa = linea, pb = lineb,
                 end = lineb + line_bytes; pb != end; ++pa, ++pb)
            e += unsigned(sqr(*pa - *pb));
    if (e > 0) {
        FILE_LOG(logERROR) << "image compare error: " << e;
        return false;
    }
    return true;
}

BOOST_AUTO_TEST_CASE(image_save_png) {
    const auto base_path =
        base_directory("lib-internal") / "raw_image_io" / "tests";

    FILE_LOG(logINFO) << "save png: start";

    const auto orig_rgb = load(base_path / "image_077.jpg", pixel::rgb24);

    // add alpha channel
    const auto orig_rgba = copy(orig_rgb, pixel::rgba32);
    {
        unsigned y = 0;
        for (auto&& line : pixels_bpp<4>(orig_rgba)) {
            const auto a =
                static_cast<unsigned char>((y++) * 256 / orig_rgba->height);
            for (auto& px : line)
                px[3] = a;
        }
    }

    const auto orig_gray8 = copy(orig_rgb, pixel::gray8);

    // create 16-bit grayscale image
    const auto orig_gray16 =
        create(orig_rgb->width, orig_rgb->height, pixel::a16_le);
    {
        const pixels_bpp<3> src(orig_rgb);
        auto sline = src.begin();
        for (auto&& dline : pixels<uint16_t>(orig_gray16)) {
            auto spx = (*sline).begin();
            ++sline;
            for (auto& dpx : dline) {
                dpx = uint16_t(97 + ((*spx)[0]+(*spx)[1]+(*spx)[2])*171/2);
                ++spx;
            }
        }
    }

    // save as png
    save(orig_rgb,    base_path / "test_rgb.png");
    save(orig_rgba,   base_path / "test_rgba.png",   png);
    save(orig_gray8,  base_path / "test_gray8.png",  png, jpeg(50));
    save(orig_gray16, base_path / "test_gray16.png", png, jpeg(50));

    // re-load them
    const auto png_rgb    = load(base_path / "test_rgb.png");
    const auto png_rgba   = load(base_path / "test_rgba.png");
    const auto png_gray8  = load(base_path / "test_gray8.png");
    const auto png_gray16 = load(base_path / "test_gray16.png", pixel::a16_le);

    // should be exact match (png is lossless)
    BOOST_CHECK_EQUAL(*orig_rgb,    *png_rgb);
    BOOST_CHECK_EQUAL(*orig_rgba,   *png_rgba);
    BOOST_CHECK_EQUAL(*orig_gray8,  *png_gray8);
    BOOST_CHECK_EQUAL(*orig_gray16, *png_gray16);

    FILE_LOG(logINFO) << "save png: done";
}

BOOST_AUTO_TEST_SUITE_END()
