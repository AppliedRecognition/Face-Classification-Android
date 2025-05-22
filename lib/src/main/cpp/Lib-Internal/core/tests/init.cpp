#include <boost/test/unit_test.hpp>
#include <applog/core.hpp>

#include <set>

#include <core/context.hpp>
#include <core/job_queue.hpp>
#include <core/thread_data.hpp>
#include <core/thread_set.hpp>

using namespace core;

namespace {
    struct visit_threads {
        thread_set& ts;
        
        explicit visit_threads(thread_set& ts)
            : ts(ts) {}

        template <typename TC>
        std::array<std::size_t,4> operator()(TC& tc) {
            auto i = ts.visit(&tc);
            auto j0 = get<std::size_t>(tc.data.thread,  i);
            auto j1 = get<std::size_t>(tc.data.context, i);
            auto j2 = get<std::size_t>(tc.data.global,  i);
            FILE_LOG(logINFO) << "visit "
                              << i << ' ' << j0 << ' ' << j1 << ' ' << j2;
            ts.wait();
            return {i,j0,j1,j2};
        }
    };
}

static void test_context() {
    context_settings settings;
    settings.min_threads = settings.max_threads = 4;
    settings.use_simd = false;

    auto c = context::construct(settings);

    BOOST_CHECK_EQUAL(c->num_threads(), settings.max_threads);
    
    thread_set ts(c->num_threads());
    using job_type = job_function<visit_threads>;
    std::list<job_type> thread_list;
    auto& queue = c->threads();
    // start a job on each thread
    // each job waits until all have run to ensure they are on distinct threads
    // ** this will lock-up if we don't have at least one thread per job **
    for (unsigned i = 0; i < ts.num_threads; ++i) {
        thread_list.emplace_back(ts);
        queue.submit(thread_list.back());
    }
    std::array<std::set<std::size_t>,4> id_set;
    for (auto& x : thread_list) {
        queue.wait(x);
        const auto& a = x.get();
        assert(a.size() == id_set.size());
        for (unsigned i = 0; i < a.size(); ++i)
            id_set[i].insert(a[i]);
        BOOST_CHECK_EQUAL(a[0],a[1]);
    }
    BOOST_CHECK_EQUAL(id_set[0].size(), ts.num_threads);
    BOOST_CHECK_EQUAL(id_set[1].size(), ts.num_threads);
    BOOST_CHECK_EQUAL(id_set[2].size(), 1);
    BOOST_CHECK_EQUAL(id_set[3].size(), 1);

    BOOST_CHECK_EQUAL(*id_set[2].begin(), get<std::size_t>(c->data().context));
    BOOST_CHECK_EQUAL(*id_set[3].begin(), get<std::size_t>(c->data().global));
}

BOOST_AUTO_TEST_SUITE(core)

BOOST_AUTO_TEST_CASE(sdk_load) {
    FILE_LOG(logINFO) << "core: init";
    test_context();
    FILE_LOG(logINFO) << "core: done";
}

BOOST_AUTO_TEST_SUITE_END()
