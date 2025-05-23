#include <boost/test/unit_test.hpp>
#include <applog/base_directory.hpp>
#include <applog/core.hpp>

#include <dlibx/dnn_fc_dynamic.hpp>

#include <core/context.hpp>
#include <core/thread_data.hpp>
#include <core/job_queue.hpp>

#include <random>

BOOST_AUTO_TEST_SUITE(dlibx)

static std::mt19937 rgen(17);

namespace {
    struct subnet {
        const dlib::tensor& data;
        auto nr() const { return data.nr(); }
        auto nc() const { return data.nc(); }
        const dlib::tensor& get_output() const { return data; }
    };
}

BOOST_AUTO_TEST_CASE(fc_qmat_test) {
    dlib::resizable_tensor input;
    {
        input.set_size(17,83);
        std::uniform_int_distribution<int> ud(-1,1);
        for (auto dest = input.host_write_only(),
                 end = dest + input.size(); dest != end; ++dest)
            *dest = float(ud(rgen));
    }
    const subnet sub{input};

    qmat16 lhs;
    {
        lhs.set_size(97,83);
        std::fill(&lhs.value(0,0), &lhs.value(lhs.nr(),0), 0);
        std::uniform_int_distribution<int16_t> ud(-5,5);
        for (auto r = 0; r < lhs.nr(); ++r) {
            lhs.coeff(r) = float(r+1);
            for (auto c = 0; c < lhs.nc(); ++c)
                lhs.value(r,c) = ud(rgen);
        }
        lhs.reset_rhs_limit(4);
        BOOST_REQUIRE_EQUAL(lhs.rhs_limit(), 4);
    }

    fc_dynamic_<97,NO_BIAS> fc;
    {
        fc.setup(sub);
        dlib::tensor& params = fc.get_layer_params();
        BOOST_REQUIRE_EQUAL(params.size(), lhs.nr()*lhs.nc());
        // note: params are transposed relative to rhs
        BOOST_REQUIRE_EQUAL(params.num_samples(), lhs.nc());
        auto dest = params.host_write_only();
        for (auto c = 0; c < lhs.nc(); ++c)
            for (auto r = 0; r < lhs.nr(); ++r)
                *dest++ = lhs.coeff(r) * lhs.value(r,c);
    }

    dlib::resizable_tensor out1;
    fc.forward(sub, out1);
    BOOST_REQUIRE_EQUAL(out1.num_samples(), input.num_samples());

    // todo: setup context and run this multi-thread
    dlib::resizable_tensor out2;
    lhs.fc(input, out2);
    BOOST_REQUIRE_EQUAL(out2.num_samples(), input.num_samples());
    BOOST_REQUIRE_EQUAL(out2.size(), out1.size());

    {
        auto s1 = out1.host(), s2 = out2.host();
        for (auto n = 0; n < input.num_samples(); ++n) {
            for (auto i = lhs.nr(); i > 0; --i, ++s1, ++s2)
                BOOST_CHECK_EQUAL(*s1, *s2);
        }
        BOOST_REQUIRE_EQUAL(s1, out1.host() + out1.size());
    }

    dlib::resizable_tensor out3;
    {
        core::context_settings cs;
        cs.min_threads = cs.max_threads = 2;
        auto c = core::context::construct(cs);
        c->threads().run([&] { lhs.fc(input, out3); return 0; });
    }
    BOOST_REQUIRE_EQUAL(out3.num_samples(), input.num_samples());
    BOOST_REQUIRE_EQUAL(out3.size(), out1.size());

    {
        auto s1 = out1.host(), s3 = out3.host();
        for (auto n = 0; n < input.num_samples(); ++n) {
            for (auto i = lhs.nr(); i > 0; --i, ++s1, ++s3)
                BOOST_CHECK_EQUAL(*s1, *s3);
        }
        BOOST_REQUIRE_EQUAL(s3, out3.host() + out3.size());
    }
}

BOOST_AUTO_TEST_SUITE_END()
