#include <boost/test/unit_test.hpp>
#include <applog/base_directory.hpp>
#include <applog/core.hpp>

#include <dlibx/dnn_lmcon.hpp>

#include <core/context.hpp>
#include <core/thread_data.hpp>
#include <core/job_queue.hpp>

#include <random>

BOOST_AUTO_TEST_SUITE(dlibx)

template <long K, long NR, long NC, int SY, int SX>
using con_type = dlib::con_<K,NR,NC,SY,SX,
                            SY!=1 ? 0 : int(NR/2),
                            SX!=1 ? 0 : int(NC/2)>;

template <long K, long NR, long NC, int SY, int SX>
static auto make_lmcon(const con_type<K,NR,NC,SY,SX>& c) {
    return dlibx::lm_con_<K,NR,NC,SY,SX>(c);
}

static std::mt19937 rgen(1);

template <typename DISTR>
static void set_random(dlib::tensor& t, DISTR&& distr) {
    for (auto d = t.host_write_only(), end = d + t.size(); d != end; ++d)
        *d = float(distr(rgen));
}

static void set_random(dlib::tensor& t) {
    std::uniform_int_distribution<> dis(-3, 3);
    set_random(t,dis);
}

static void require_same_size(const dlib::tensor& t0, const dlib::tensor& t1) {
    BOOST_REQUIRE_EQUAL(t0.num_samples(), t1.num_samples());
    BOOST_REQUIRE_EQUAL(t0.k(), t1.k());
    BOOST_REQUIRE_EQUAL(t0.nr(), t1.nr());
    BOOST_REQUIRE_EQUAL(t0.nc(), t1.nc());
}

static auto mean_var(const dlib::tensor& t0, const dlib::tensor& t1) {
    require_same_size(t0,t1);
    const auto s =
        std::inner_product(t0.host(), t0.host() + t0.size(),
                           t1.host(), 0.0f,
                           [](auto a, auto b) { return a + b; },
                           [](auto a, auto b) { return (a-b)*(a-b); });
    return s / float(t0.size());
}

static void check_equal(const dlib::tensor& t0, const dlib::tensor& t1) {
    BOOST_CHECK_EQUAL(mean_var(t0,t1), 0.0f);
}

namespace {
    struct input {
        dlib::resizable_tensor data;

        auto nr() const { return data.nr(); }
        auto nc() const { return data.nc(); }
        auto num_samples() const { return data.num_samples(); }

        const dlib::tensor& get_output() const {
            return data;
        }

        dlib::resizable_tensor gradient;
        dlib::tensor& get_gradient_input() { return gradient; }

        input(long k, long nr, long nc, long nsamples = 1) {
            data.set_size(nsamples,k,nr,nc);
            std::uniform_int_distribution<> int3(-3, 3);
            set_random(data, int3);
            gradient.copy_size(data);
            std::uniform_real_distribution<float> real1(-1, 1);
            set_random(gradient, real1);
        }
    };
}

template <typename CON1, typename CON2>
static constexpr void test_samples_impl(CON1&&, CON2&&) {} // base case

template <typename CON1, typename CON2, typename S0, typename... SAMPLES>
static auto
test_samples_impl(CON1&& con1, CON2&& con2, S0&& s0, SAMPLES&&... samples) {
    FILE_LOG(logDETAIL) << "input: " << s0.data.nr() << 'x' << s0.data.nc();

    // forward
    dlib::resizable_tensor out1, out2;
    con1.forward(s0,out1);
    con2.forward(s0,out2);
    check_equal(out1,out2);

    // input to backward()
    dlib::resizable_tensor gradient_input;
    gradient_input.copy_size(out1);
    std::uniform_real_distribution<float> ud(-1, 1);
    set_random(gradient_input, ud);

    // gradient of parameters
    dlib::resizable_tensor pg1;
    pg1.copy_size(con1.get_layer_params());
    set_random(pg1, ud);
    auto pg2 = pg1;

    // backward
    auto sub1 = s0, sub2 = s0;
    con1.backward(gradient_input, sub1, pg1);
    con2.backward(gradient_input, sub2, pg2);
    if (s0.num_samples() <= 1)
        check_equal(pg1,pg2);
    else // parallelization causes some variation in the parameter gradient
        BOOST_CHECK(mean_var(pg1,pg2) < 1e-8);
    check_equal(sub1.gradient,sub2.gradient);

    {
        // quantize
        std::stringstream ss;
        set_parameter_format(ss, dlibx::quantize(12));
        serialize(con2, ss);
        std::decay_t<CON2> qcon;
        deserialize(qcon, ss);
        BOOST_CHECK(qcon.get_shared_qfilt());
        dlib::resizable_tensor out3;
        qcon.forward(s0,out3);
        const auto var = mean_var(out2,out3);
        FILE_LOG(logDETAIL) << "quantize variance: " << var;
        BOOST_CHECK(var < 5e-4);
    }

    test_samples_impl(std::forward<CON1>(con1), std::forward<CON2>(con2),
                      std::forward<SAMPLES>(samples)...);
    return out1;
}

template <typename CON, typename... SAMPLES>
static auto test_samples(CON&& con, SAMPLES&&... samples) {
    return test_samples_impl(con, make_lmcon(con),
                             std::forward<SAMPLES>(samples)...);
}

static auto run_tests(int num_samples = 1) {
    const auto sample_small = input(5,3,1,num_samples);
    const auto sample_medium = input(5,10,12,num_samples);
    const auto sample_large = input(5,29,23,num_samples);
    const auto sample_giant = input(5,17,73,num_samples);

    {
        FILE_LOG(logINFO) << "lmcon: 1x1 output (" << num_samples << " samples)";
        con_type<7,3,1,2,2> fm;
        fm.setup(sample_small);
        set_random(fm.get_layer_params());
        auto fo = test_samples(fm, sample_small);
        BOOST_CHECK_EQUAL(fo.nr(), 1);
        BOOST_CHECK_EQUAL(fo.nc(), 1);
    }

    {
        FILE_LOG(logINFO) << "lmcon: stride 3x2 no padding (" << num_samples << " samples)";
        con_type<7,5,3,3,2> fm;
        fm.setup(sample_medium);
        set_random(fm.get_layer_params());
        test_samples(fm, sample_medium, sample_large);
    }

    {
        FILE_LOG(logINFO) << "lmcon: stride 1 with padding (" << num_samples << " samples)";
        con_type<7,5,3,1,1> fm;
        fm.setup(sample_medium);
        set_random(fm.get_layer_params());
        test_samples(
            fm, sample_small, sample_medium, sample_large, sample_giant);
    }

    {
        FILE_LOG(logINFO) << "lmcon: pointwise 1x1 (" << num_samples << " samples)";
        con_type<11,1,1,1,1> fm;
        fm.setup(sample_medium);
        set_random(fm.get_layer_params());
        test_samples(
            fm, sample_small, sample_medium, sample_large, sample_giant);
    }
    return 0;
}

template <int dilate, typename QUEUE>
static auto test_dilate(QUEUE&& q, int num_samples = 1) {
    const auto sample = input(5,17,73,num_samples);

    dlibx::lm_con_<10,3,3,1,1,1,1,dilate,dilate> con{};
    con.setup(sample);
    set_random(con.get_layer_params());

    // this just checks if the output for serial vs parallelized
    dlib::resizable_tensor out1, out2;
    con.forward(sample,out1);
    q.run([&]{ con.forward(sample,out2); return 0; });
    FILE_LOG(logINFO) << "output: " << out1.num_samples() << 'x'
                      << out1.k() << 'x' << out1.nr() << 'x' << out1.nc();
    check_equal(out1,out2);

    {
        // quantize
        std::stringstream ss;
        set_parameter_format(ss, dlibx::quantize(12));
        serialize(con, ss);
        decltype(con) qcon;
        deserialize(qcon, ss);
        BOOST_CHECK(qcon.get_shared_qfilt());
        dlib::resizable_tensor out3, out4;
        qcon.forward(sample,out3);
        q.run([&]{ qcon.forward(sample,out4); return 0; });
        check_equal(out3,out4);
        const auto var = mean_var(out2,out3);
        FILE_LOG(logDETAIL) << "quantize variance: " << var;
        BOOST_CHECK(var < 5e-4);
    }
}

BOOST_AUTO_TEST_CASE(lmcon_test) {
    FILE_LOG(logINFO) << "--";

    // not parallelized
    run_tests();
    run_tests(3);

    // parallelized
    core::context_settings cs;
    cs.min_threads = 2;
    cs.max_threads = 4;
    const auto c = core::context::construct(cs);

    using namespace std::chrono_literals;
    std::this_thread::sleep_for(2ms);
    //std::this_thread::yield();

    FILE_LOG(logINFO) << "lmcon: parallelized";
    c->threads().run([]{ return run_tests(); });
    c->threads().run([]{ return run_tests(2); });
    c->threads().run([]{ return run_tests(5); });
    c->threads().run([]{ return run_tests(10); });

    FILE_LOG(logINFO) << "lmcon: dilated convolution";
    for (auto s : { 1, 2, 5 }) {
        test_dilate<2>(c->threads(), s);
        test_dilate<3>(c->threads(), s);
        test_dilate<5>(c->threads(), s);
    }

    FILE_LOG(logINFO) << "lmcon: done";
}

BOOST_AUTO_TEST_SUITE_END()
