#include <boost/test/unit_test.hpp>
#include <applog/base_directory.hpp>
#include <applog/core.hpp>

#include <rec/prototype.hpp>
#include <core/context.hpp>

BOOST_AUTO_TEST_SUITE(rec)

BOOST_AUTO_TEST_CASE(random_prototype) {
    core::context_settings cs;
    const auto context = core::context::construct(cs);
    const auto models_path = base_directory("lib-internal") / "models";

    static constexpr auto proto_ver = 16;

    {
        std::array<rec::prototype_ptr, 128> vec;
        FILE_LOG(logINFO) << "random_prototype: generate random";
        for (auto& x : vec)
            x = rec::prototype::random(context, proto_ver);
        std::vector<float> scores;
        scores.reserve(vec.size()*(vec.size()-1)/2);
        FILE_LOG(logINFO) << "random_prototype: compare random";
        for (auto it = vec.begin(), end = vec.end(); it != end; ++it)
            for (auto jt = next(it); jt != end; ++jt)
                scores.push_back(compare(*it,*jt));
        std::sort(scores.begin(), scores.end());
        const auto median = scores[scores.size()/2];
        FILE_LOG(logDETAIL) << scores.front() << '\t'
                            << median << '\t'
                            << scores.back();
        BOOST_CHECK_LT(scores.front(), -1.5f);
        BOOST_CHECK_LT(std::abs(median), 0.05f);
        BOOST_CHECK_LT(1.5f, scores.back());
    }

    {
        FILE_LOG(logINFO) << "random_prototype: generate related";
        const auto base = rec::prototype::random(context, proto_ver);
        for (float target = 5.0f; target > -5.25f; target -= 0.5f) {
            auto x = rec::prototype::random(context, base, target);
            auto score = compare(base, x);
            BOOST_CHECK_LT(std::abs(score-target), 0.01);
            FILE_LOG(logDETAIL) << '\t' << target << '\t' << score;
        }
    }

    FILE_LOG(logINFO) << "random_prototype: done";
}

BOOST_AUTO_TEST_SUITE_END()
