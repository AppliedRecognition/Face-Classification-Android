#include <boost/test/unit_test.hpp>
#include <applog/core.hpp>

#include <raw_image/linear_regression.hpp>
#include <random>

BOOST_AUTO_TEST_SUITE(raw_image)

template <typename T, typename GEN>
static void do_test2(GEN&& gen) {
    static constexpr auto ofs = T(123);
    static constexpr auto coeff = T(M_PI);
    std::uniform_real_distribution<T> distr;
    std::normal_distribution<T> noise(0,T(1)/2);
    raw_image::linear_regression<T> reg;
    reg.reserve(1000);
    for (auto i = 1000; i > 0; --i) {
        const auto x = distr(gen);
        const auto z = ofs + coeff*x + noise(gen);
        reg.add(z, 1, x);
    }
    const auto result = reg.compute();
    BOOST_REQUIRE_EQUAL(2, result.size());
    FILE_LOG(logDETAIL) << "offset\t" << result[0] << '\t' << ofs;
    FILE_LOG(logDETAIL) << "coeff\t" << result[1] << '\t' << coeff;
    BOOST_CHECK(std::abs(result[0] - ofs) < 0.25);
    BOOST_CHECK(std::abs(result[1] - coeff) < 0.25);
}

template <typename T, typename GEN>
static void do_test3(GEN&& gen) {
    std::uniform_real_distribution<T> distr;
    std::normal_distribution<T> noise(0,T(1)/16);
    const auto target = std::vector<T>{
        distr(gen) - 0.5f,
        -1 - distr(gen),
        1 + distr(gen)
    };
    raw_image::linear_regression<T> reg;
    for (T x = -5; x <= 5; x += 1)
        for (T y = -5; y <= 5; y += 1) {
            const auto z = target[0] + x*target[1] + y*target[2] + noise(gen);
            reg.add(z, {1, x, y});
        }
    const auto result = reg.compute();
    BOOST_REQUIRE_EQUAL(target.size(), result.size());
    for (unsigned i = 0; i < target.size(); ++i) {
        BOOST_CHECK(std::abs(target[i] - result[i]) < T(1)/64);
        FILE_LOG(logDETAIL) << "reg[" << i << "]: "
                            << target[i] << '\t' << result[i];
    }
}

BOOST_AUTO_TEST_CASE(regression_test) {
    std::mt19937 gen(1);
    do_test2<float>(gen);
    //do_test2<double>(gen);
    do_test3<float>(gen);
    //do_test3<double>(gen);
}

BOOST_AUTO_TEST_SUITE_END()
