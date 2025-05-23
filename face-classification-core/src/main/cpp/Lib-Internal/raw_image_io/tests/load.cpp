#include <boost/test/unit_test.hpp>
#include <applog/base_directory.hpp>
#include <applog/core.hpp>

#include <raw_image_io/io.hpp>

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
    e /= line_bytes;
    e /= a.height;
    if (e > 2) {
        FILE_LOG(logERROR) << "image compare error: " << e;
        return false;
    }
    return true;
}

template <typename... Args>
static auto altload(const std::filesystem::path& path, Args&&... args) {
    // todo: try out memory mapped file here
    std::ifstream in(path, std::ios_base::binary);
    std::vector<char> buf(std::istreambuf_iterator<char>(in), {});
    const auto bin = stdx::binary(move(buf));
    return from_binary(bin, std::forward<Args>(args)...);
}

BOOST_AUTO_TEST_CASE(image_loading) {
    const auto base_path =
        base_directory("lib-internal") / "raw_image_io" / "tests";

    FILE_LOG(logINFO) << "load: start";

    const auto orig_rgb = load(base_path / "image_077.jpg", pixel::rgb24);
    const auto orig_yuv = copy(orig_rgb, pixel::yuv);
    const auto orig_gray = copy(orig_yuv, pixel::gray8);

    const auto load_color =
        [&](auto filename, rotate rot) {
            auto path = base_path / filename;
            auto img_default = load(path,rot);
            auto alt_default = altload(path,rot);
            BOOST_CHECK_EQUAL(*img_default, *alt_default);
            auto img_rgb = load(path, pixel::rgb24, rot);
            auto alt_rgb = altload(path, pixel::rgb24, rot);
            BOOST_CHECK(img_rgb->layout == pixel::rgb24);
            BOOST_CHECK_EQUAL(*img_rgb, *alt_rgb);
            BOOST_CHECK_EQUAL(*img_rgb, *copy(img_default,pixel::rgb24));
            BOOST_CHECK_EQUAL(*img_rgb, *copy(orig_rgb,rot));
            auto img_yuv = load(path, pixel::yuv, rot);
            auto alt_yuv = altload(path, pixel::yuv, rot);
            BOOST_CHECK(img_yuv->layout == pixel::yuv);
            BOOST_CHECK_EQUAL(*img_yuv, *alt_yuv);
            BOOST_CHECK_EQUAL(*img_yuv, *copy(img_default,pixel::yuv));
            BOOST_CHECK_EQUAL(*img_yuv, *copy(orig_yuv,rot));
        };

    const auto load_gray =
        [&](auto filename, rotate rot, bool load_must_be_gray) {
            auto path = base_path / filename;
            auto img_default = load(path, rot);
            auto alt_default = altload(path, rot);
            BOOST_CHECK_EQUAL(*img_default, *alt_default);
            BOOST_CHECK(!load_must_be_gray ||
                        img_default->layout == pixel::gray8);
            auto img_gray = load(path, pixel::gray8, rot);
            auto alt_gray = altload(path, pixel::gray8, rot);
            BOOST_CHECK(img_gray->layout == pixel::gray8);
            BOOST_CHECK_EQUAL(*img_gray, *alt_gray);
            BOOST_CHECK_EQUAL(*img_gray, *copy(img_default,pixel::gray8));
            BOOST_CHECK_EQUAL(*img_gray, *copy(orig_gray,rot));
        };

    for (unsigned r = 0; r < 8; ++r) {
        load_color("image_077.jpg", rotate(r));
        load_color("image_077.png", rotate(r));
        load_color("image_077.tiff", rotate(r));
        load_gray("image_077_bw.jpg", rotate(r), true);
        load_gray("image_077_bw.png", rotate(r), true);
        load_gray("image_077_bw.tiff", rotate(r), false);
    }

    FILE_LOG(logINFO) << "load: done";
}

BOOST_AUTO_TEST_SUITE_END()
