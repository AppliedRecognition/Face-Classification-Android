#include <cmath>
#include <boost/test/unit_test.hpp>
#include <applog/core.hpp>

#include <raw_image/adjust.hpp>
#include <raw_image/transform.hpp>
#include <raw_image/pixels.hpp>

#include <random>


BOOST_AUTO_TEST_SUITE(raw_image)

static std::mt19937 rgen(1);
static std::normal_distribution<float> norm_distr;

static void test_dim(unsigned dim, unsigned nreps = 1) {
    //FILE_LOG(logINFO) << "--";
    FILE_LOG(logINFO) << "matrix: " << dim << 'x' << dim;

    const auto in = create(dim, dim, pixel::f32);
    const auto out = create(dim, dim, pixel::f32);

    for (auto rep = nreps; rep > 0; --rep) {

        // random matrix
        for (auto&& line : pixels<float>(in))
            for (auto& px : line)
                px = norm_distr(rgen);

        if (0) {
            for (auto&& line : pixels<float>(in)) {
                std::stringstream ss;
                for (auto z : line)
                    ss << ' ' << z;
                FILE_LOG(logINFO) << "mat:" << ss.str();
            }
        }

        // compute inverse
        auto inv = matrix_inverse(*in);
        if (0) {
            for (auto&& line : pixels<float>(inv)) {
                std::stringstream ss;
                for (auto z : line)
                    ss << ' ' << z;
                FILE_LOG(logINFO) << "inv:" << ss.str();
            }
        }

        if (0 && dim == 2) {
            const pixels<float> a(inv);
            const pixels<float> b(in);
            auto m00 = a[0][0]*b[0][0] + a[0][1]*b[1][0];
            auto m01 = a[1][0]*b[0][0] + a[1][1]*b[1][0];
            auto m10 = a[0][0]*b[0][1] + a[0][1]*b[1][1];
            auto m11 = a[1][0]*b[0][1] + a[1][1]*b[1][1];
            FILE_LOG(logINFO) << m00 << ' ' << m01 << ' ' << m10 << ' ' << m11;
        }

        // invalidate output
        memset(out->data, 0xff, out->height*out->bytes_per_line);

        // multiply
        matrix_multiply(reader::construct(in,rotate(5)),*inv)
            ->copy_to(*out);
        if (0) {
            for (auto&& line : pixels<float>(out)) {
                std::stringstream ss;
                for (auto z : line)
                    ss << ' ' << z;
                FILE_LOG(logINFO) << "out:" << ss.str();
            }
        }

        // verify
        unsigned j = 0;
        for (auto&& line : pixels<float>(out)) {
            for (unsigned i = 0; i < dim; ++i) {
                auto z = line[i];
                if (i == j) z -= 1;
                BOOST_CHECK(std::abs(z) < 0.05);
            }
            ++j;
        }
    }
}

BOOST_AUTO_TEST_CASE(raw_image_matrix) {
    FILE_LOG(logINFO) << "matrix: start";

    for (unsigned dim = 1; dim <= 10; ++dim)
        test_dim(dim, 10);

    FILE_LOG(logINFO) << "matrix: done";
}

BOOST_AUTO_TEST_SUITE_END()
