#include <boost/test/unit_test.hpp>
#include <applog/base_directory.hpp>
#include <applog/core.hpp>

#include <dlibx/dnn_input_yuv.hpp>

#include <raw_image/core.hpp>
#include <raw_image/reader.hpp>

#include <dlibx/linear_regression.hpp>

#include <random>

BOOST_AUTO_TEST_SUITE(dlibx)

static std::mt19937 rgen(1);
static std::uniform_int_distribution<unsigned> uv_distr(0, 255);
static std::normal_distribution<float> norm_distr;

static auto
make_yuv(unsigned width, unsigned height, float mean, float stddev) {
    auto img = create(width, height, raw_image::pixel::yuv);
    auto line = img->data;
    for (auto j = height; j > 0; line += img->bytes_per_line, --j) {
        auto px = line;
        for (auto i = width; i > 0; px += 3, --i) {
            auto y = std::lround(mean + stddev * norm_distr(rgen));
            using uchar = unsigned char;
            px[0] = uchar(std::clamp(y, 0l, 255l));
            px[1] = uchar(uv_distr(rgen));
            px[2] = uchar(uv_distr(rgen));
        }
    }
    return img;
}

static auto verify_yuv(const raw_image::plane& img, float const* t) {
    // y
    auto line = img.data;
    core::linear_regression<float> reg;
    reg.reserve(img.width * img.height);
    for (auto j = img.height; j > 0; line += img.bytes_per_line, --j) {
        auto px = line;
        for (auto i = img.width; i > 0; ++t, px += 3, --i)
            reg.add(float(*px), 1, *t);
    }
    auto c = reg.compute();
    BOOST_REQUIRE_EQUAL(c.size(), 2);
    float ssr = 0;
    line = img.data;
    t -= img.width * img.height;
    for (auto j = img.height; j > 0; line += img.bytes_per_line, --j) {
        auto px = line;
        for (auto i = img.width; i > 0; ++t, px += 3, --i) {
            auto d = c[0] + c[1]**t - float(*px);
            ssr += d*d;
        }
    }
    ssr /= float(img.height);
    ssr /= float(img.width);

    // u
    line = img.data + 1;
    for (auto j = img.height; j > 0; line += img.bytes_per_line, --j) {
        auto px = line;
        for (auto i = img.width; i > 0; ++t, px += 3, --i)
            BOOST_CHECK_EQUAL(unsigned(*px), 128 + 128**t);
    }

    // v
    line = img.data + 2;
    for (auto j = img.height; j > 0; line += img.bytes_per_line, --j) {
        auto px = line;
        for (auto i = img.width; i > 0; ++t, px += 3, --i)
            BOOST_CHECK_EQUAL(unsigned(*px), 128 + 128**t);
    }

    return std::array<float,3> { c[0], c[1], ssr };
}

BOOST_AUTO_TEST_CASE(input_yuv_test) {
    FILE_LOG(logINFO) << "--";

    dlibx::input_yuv_normalized input;
    
    for (auto width : { 5u, 11u, 17u, 23u }) {
        for (auto height : { 7u, 13u, 19u, 29u }) {
            FILE_LOG(logINFO) << "yuv: " << width << 'x' << height;
            std::vector<raw_image::plane_ptr> imgs;
            for (unsigned i = 0; i < 4; ++i)
                imgs.push_back(
                    make_yuv(width, height,
                             float(130 - 10*i),
                             float(5 + 8*i)));
            dlib::resizable_tensor t;
            input.to_tensor(imgs.begin(), imgs.end(), t);
            BOOST_REQUIRE_EQUAL(t.num_samples(), imgs.size());
            BOOST_REQUIRE_EQUAL(t.k(), 3);
            BOOST_REQUIRE_EQUAL(t.nr(), height);
            BOOST_REQUIRE_EQUAL(t.nc(), width);
            auto data = t.host();
            float mean = 130, stddev = 5;
            for (auto&& p : imgs) {
                auto arr = verify_yuv(*p, data);
                //FILE_LOG(logINFO) << '\t' << arr[0] << '\t' << arr[1] << '\t' << arr[2];
                BOOST_CHECK(std::abs(arr[0] - mean) < 8);
                BOOST_CHECK(std::abs(arr[1] - stddev) < 4);
                BOOST_CHECK_SMALL(arr[2], 1e-7f);
                data += 3*width*height;
                mean -= 10;
                stddev += 8;
            }
        }
    }

    FILE_LOG(logINFO) << "input_yuv: done";
}

BOOST_AUTO_TEST_SUITE_END()
