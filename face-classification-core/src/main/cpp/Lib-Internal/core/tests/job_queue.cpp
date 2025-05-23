#include <boost/test/unit_test.hpp>
#include <applog/core.hpp>

#include <core/job_queue.hpp>

using namespace core;


namespace {
    struct thread_data {};
}

namespace core_test_ns {
    struct interrupt_job {
        bool has_run = false;
        bool interrupted = false;
        job::base<thread_data>* subjob = nullptr;

        int operator()(const thread_data&) {
            BOOST_CHECK(!interrupted);
            BOOST_CHECK(!has_run);
            has_run = true;
            return 0;
        }

        friend void interrupt(interrupt_job& job) {
            BOOST_CHECK(!job.has_run);
            BOOST_CHECK(!job.interrupted);
            job.interrupted = true;
            if (job.subjob) job.subjob->interrupt_job();
        }
    };
}

namespace {
    struct anon_job {
        bool has_run = false;
        bool interrupted = false;
        job::base<thread_data>* subjob = nullptr;

        int operator()(const thread_data&) {
            BOOST_CHECK(!interrupted);
            BOOST_CHECK(!has_run);
            has_run = true;
            return 0;
        }

        friend void interrupt(anon_job& job) {
            BOOST_CHECK(!job.has_run);
            BOOST_CHECK(!job.interrupted);
            job.interrupted = true;
            if (job.subjob) job.subjob->interrupt_job();
        }
    };
}

template <typename T>
static void check_interrupted(const T& job) {
    BOOST_CHECK(!job.fn.has_run);
    BOOST_CHECK(job.fn.interrupted);
}

static void test_interrupt() {
    {
        job::function<thread_data, anon_job> j0;
        job::function<thread_data, anon_job> j1;
        job::function<thread_data, core_test_ns::interrupt_job> j2;
        job::function<thread_data, core_test_ns::interrupt_job> j3;
        job::queue<thread_data> queue;
        queue.submit(j0);
        queue.submit(j1);
        queue.submit(j2);
        queue.submit(j3);
        j0.fn.subjob = &j3;
        j2.fn.subjob = &j1;
        j0.interrupt_job();
        j2.interrupt_job();
        check_interrupted(j0);
        check_interrupted(j1);
        check_interrupted(j2);
        check_interrupted(j3);
    }
}

namespace {
    std::atomic<int> actual_seq(0);

    struct child_order_job {
        const int expect_seq;
        child_order_job(int seq) : expect_seq(seq) {}

        template <typename JC>
        std::pair<int,int> operator()(JC&) {
            return { actual_seq++, expect_seq };
        }
    };

    template <int T> struct order_job;

    template <> struct order_job<0> {
        const int expect_seq;
        order_job(int seq) : expect_seq(seq) {}

        template <typename JC>
        std::vector<std::pair<int,int> >
        operator()(JC& jc) {
            BOOST_CHECK_EQUAL(JC::this_context(), &jc);
            
            std::vector<std::pair<int,int> > r;
            r.emplace_back(actual_seq++, expect_seq);

            using job_type = job::function<thread_data, child_order_job>;
            std::list<job_type> jobs;
            
            jobs.emplace_back(expect_seq + 2);
            jc.submit(jobs.back(), job::relative_order(JC::order_max));
            jobs.emplace_back(expect_seq + 2);
            jc.submit(jobs.back(), job::relative_order(JC::order_max/2));
            jobs.emplace_back(expect_seq + 2);
            jc.submit(jobs.back(), job::relative_order(1));

            jobs.emplace_back(expect_seq + 1);
            jc.submit(jobs.back(), job::relative_order(JC::order_min));
            jobs.emplace_back(expect_seq + 1);
            jc.submit(jobs.back(), job::relative_order(JC::order_min/2));
            jobs.emplace_back(expect_seq + 1);
            jc.submit(jobs.back(), job::relative_order(-1));

            jobs.emplace_back(expect_seq + (jc.job.order() < 0 ? 1 : 2));
            jc.submit(jobs.back());

            while (!jobs.empty()) {
                const auto it = jc.wait_for_one(jobs.begin(), jobs.end());
                r.push_back(**it);
                jobs.erase(it);
            }
            
            return r;
        }
    };

    template <> struct order_job<1> {
        const int expect_seq;
        order_job(int seq) : expect_seq(seq) {}

        template <typename JC>
        std::vector<std::pair<int,int> >
        operator()(JC& jc) {
            BOOST_CHECK_EQUAL(JC::this_context(), &jc);
            
            std::vector<std::pair<int,int> > r;
            r.emplace_back(actual_seq++, expect_seq);

            using job_type = job::function<thread_data, child_order_job>;
            std::list<job_type> jobs;
            
            job_type j0(expect_seq + 2);
            jc.submit(j0, job::relative_order(JC::order_max));
            job_type j1(expect_seq + 2);
            jc.submit(j1, job::relative_order(JC::order_max/2));
            job_type j2(expect_seq + 2);
            jc.submit(j2, job::relative_order(1));
            
            job_type j3(expect_seq + 1);
            jc.submit(j3, job::relative_order(JC::order_min));
            job_type j4(expect_seq + 1);
            jc.submit(j4, job::relative_order(JC::order_min/2));
            job_type j5(expect_seq + 1);
            jc.submit(j5, job::relative_order(-1));
            
            job_type j6(expect_seq + (jc.job.order() < 0 ? 1 : 2));
            jc.submit(j6);

            jc.wait(j0,j1,j2,j3,j4,j5,j6);

            r.push_back(*j0);
            r.push_back(*j1);
            r.push_back(*j2);
            r.push_back(*j3);
            r.push_back(*j4);
            r.push_back(*j5);
            r.push_back(*j6);
            
            return r;
        }
    };
}

template <int T>
static void test_order_0() {
    using queue_type = job::queue<thread_data>;
    using job_type = job::function<thread_data, order_job<T> >;
    queue_type queue;
    static constexpr auto order_min = queue_type::order_min;
    static constexpr auto order_max = queue_type::order_max;

    job_type j0{0*16};
    queue.submit_absolute(order_min, j0);

    job_type j1{1*16};
    queue.submit_absolute(order_min/2, j1);

    job_type j2{2*16};
    queue.submit_absolute(-1, j2);

    job_type j3{3*16};
    queue.submit(j3);

    job_type j4{4*16};
    queue.submit_absolute(1, j4);

    job_type j5{5*16};
    queue.submit_absolute(order_max/2, j5);

    job_type j6{6*16};
    queue.submit_absolute(order_max, j6);

    queue.wait(j6,j1,j3,j5,j0,j2,j4);

    std::vector<std::pair<int,int> > final;
    for (auto& p : { &*j0, &*j1, &*j2, &*j3, &*j4, &*j5, &*j6 })
        final.insert(final.end(), p->begin(), p->end());
    std::sort(final.begin(), final.end());
    for (std::size_t i = 1; i < final.size(); ++i)
        BOOST_CHECK(final[i-1].second <= final[i].second);
    BOOST_CHECK(final.size() == 7*8);
}

template <int T>
static void test_order_1() {
    using queue_type = job::queue<thread_data>;
    using job_type = job::function<thread_data, order_job<T> >;
    queue_type queue;
    static constexpr auto order_min = queue_type::order_min;
    static constexpr auto order_max = queue_type::order_max;

    std::list<job_type> jobs;

    jobs.emplace_back(4*16);
    queue.submit_absolute(1, jobs.back());

    jobs.emplace_back(5*16);
    queue.submit_absolute(order_max/2, jobs.back());

    jobs.emplace_back(6*16);
    queue.submit_absolute(order_max, jobs.back());
    
    jobs.emplace_back(0*16);
    queue.submit_absolute(order_min, jobs.back());

    jobs.emplace_back(1*16);
    queue.submit_absolute(order_min/2, jobs.back());

    jobs.emplace_back(2*16);
    queue.submit_absolute(-1, jobs.back());

    jobs.emplace_back(3*16);
    queue.submit(jobs.back());

    std::vector<std::pair<int,int> > final;
    while (!jobs.empty()) {
        const auto it = queue.wait_for_one(jobs.begin(), jobs.end());
        final.insert(final.end(), (**it).begin(), (**it).end());
        jobs.erase(it);
    }
    std::sort(final.begin(), final.end());
    for (std::size_t i = 1; i < final.size(); ++i)
        BOOST_CHECK(final[i-1].second <= final[i].second);
    BOOST_CHECK(final.size() == 7*8);
}


BOOST_AUTO_TEST_SUITE(core)

BOOST_AUTO_TEST_CASE(job_queue) {
    FILE_LOG(logINFO) << "job_queue: start";
    test_interrupt();

    FILE_LOG(logINFO) << "== order 0 0";
    test_order_0<0>();
    FILE_LOG(logINFO) << "== order 0 1";
    test_order_0<1>();
    FILE_LOG(logINFO) << "== order 1 0";
    test_order_1<0>();
    FILE_LOG(logINFO) << "== order 1 1";
    test_order_1<1>();

    {   // std::vector of jobs and copying job_function
        using queue_type = job::queue<thread_data>;
        queue_type queue;
        struct job {
            const int x;
            job(int x) : x(x) {}
            int operator()() const { return x; }
        };
        std::vector<core::job::function<thread_data,job> > jobs1;
        jobs1.reserve(5);
        for (auto i = 0; i < 10; ++i)
            jobs1.emplace_back(i);
        auto jobs2 = jobs1;
        for (auto& job : jobs1) queue.submit_absolute(0,job);
        for (auto& job : jobs2) queue.submit_absolute(0,job);
        BOOST_CHECK_EXCEPTION(auto c = jobs1, std::logic_error,
                              [](auto&){return true;});
        BOOST_CHECK_EXCEPTION(auto j = std::move(jobs1.front()),
                              std::logic_error,
                              [](auto&){return true;});
        BOOST_CHECK_EQUAL(jobs2.size(), 10);
        for (std::size_t i = 0; i < jobs2.size(); ++i) {
            auto& job = jobs2[i]; 
            queue.wait(job);
            BOOST_CHECK_EQUAL(*job,i);
        }
        queue.wait_for_all(jobs1.begin(), jobs1.end());
        for (auto& job : jobs1) { *job; }
        jobs1.clear();
    }

    {   // run lambda and this_context
        using queue_type = job::queue<thread_data>;
        using context_type = job::context<thread_data>;
        queue_type queue;
        const auto r = queue.run([&] {
                const auto jc = context_type::this_context();
                BOOST_CHECK(jc && &jc->owner() == queue.get_pool().get());
                return 66;
            } );
        BOOST_CHECK_EQUAL(r, 66);
    }
    
    FILE_LOG(logINFO) << "job_queue: done";
}

BOOST_AUTO_TEST_SUITE_END()
