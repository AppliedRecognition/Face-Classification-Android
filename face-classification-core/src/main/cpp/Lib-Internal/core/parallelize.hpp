#pragma once

#include <core/context.hpp>
#include <core/thread_data.hpp>
#include <core/job_queue.hpp>

#include <atomic>
#include <vector>

namespace core {

    /** \brief Multi-threaded execution of function over elements of vector.
     *
     * RandIter can be any random access iterator so this method works
     * over any kind of array, vector or span.
     * The function is invoked with each element of the span.
     */
    template <typename RandIter, typename Size, typename Func>
    void parallelize(RandIter first, Size len, Func op,
                     job_context* jc, std::size_t nthreads,
                     job::order_type order = job::order_min,
                     std::size_t job_max_threads = std::size_t(-1)) {
        if (nthreads <= 0 || !jc) {
            for ( ; 0 < len; --len, ++first)
                op(*first);
        }
        else {
            std::atomic<Size> next{0};
            auto func =
                [first=std::move(first),len,&next,
                 op=std::move(op)]() mutable {
                    for (;;) {
                        const auto i =
                            next.fetch_add(1,std::memory_order_relaxed);
                        if (i < len)
                            op(first[i]);
                        else break;
                    }
                    return 0;
                };
            using fn = job_function<decltype(func)>;
            std::vector<fn> jobs(nthreads+1, func);
            for (auto& job : jobs) {
                job.set_max_threads(job_max_threads);
                jc->submit_absolute(order,job);
            }
            jc->wait_for_all(jobs.begin(), jobs.end());
            for (auto& job : jobs)
                *job; // re-throw any exceptions
        }
    }
    template <typename RandIter, typename Size, typename Func>
    inline auto parallelize(RandIter first, Size len, Func&& func,
                            job_context* jc) {
        auto nthreads = jc ? jc->num_threads() : 0;
        return parallelize(first, len, std::forward<Func>(func), jc, nthreads);
    }
    template <typename RandIter, typename Size, typename Func>
    inline auto parallelize(RandIter first, Size len, Func&& func,
                            job_context& jc) {
        return parallelize(first, len, std::forward<Func>(func), &jc);
    }
    template <typename RandIter, typename Size, typename Func>
    inline auto parallelize(RandIter first, Size len, Func&& func,
                            job_context& jc, std::size_t nthreads,
                            job::order_type order = job::order_min) {
        return parallelize(first, len, std::forward<Func>(func),
                           &jc, nthreads, order);
    }


    /** \brief Multi-threaded execution of a method.
     *
     * The STATE object must have operator()(index).
     * The method will be involed by (nthreads+1) threads and
     * passed each index value in [0, end).
     */
    template <typename STATE, typename INDEX>
    auto parallelize(STATE&& s, INDEX end,
                     job_context* jc, std::size_t nthreads,
                     job::order_type order = job::order_min,
                     std::size_t job_max_threads = std::size_t(-1)) {
        if (nthreads <= 0 || !jc) {
            for (INDEX i = 0; i < end; ++i)
                s(i);
        }
        else {
            std::atomic<INDEX> next{0};
            auto func =
                [&s,end,&next]() {
                    for (;;) {
                        const auto i =
                            next.fetch_add(1,std::memory_order_relaxed);
                        if (i < end)
                            s(i);
                        else break;
                    }
                    return 0;
                };
            using fn = job_function<decltype(func)>;
            std::vector<fn> jobs(nthreads+1, func);
            for (auto& job : jobs) {
                job.set_max_threads(job_max_threads);
                jc->submit_absolute(order,job);
            }
            jc->wait_for_all(jobs.begin(), jobs.end());
            for (auto& job : jobs)
                *job; // re-throw any exceptions
        }
    }
    template <typename STATE, typename INDEX>
    inline auto parallelize(STATE&& s, INDEX end, job_context* jc) {
        auto nthreads = jc ? jc->num_threads() : 0;
        return parallelize(s, end, jc, nthreads);
    }
    template <typename STATE, typename INDEX>
    inline auto parallelize(STATE&& s, INDEX end,
                            job_context& jc, std::size_t nthreads,
                            job::order_type order = job::order_min) {
        return parallelize(s, end, &jc, nthreads, order);
    }
    template <typename STATE, typename INDEX>
    inline auto parallelize(STATE&& s, INDEX end, job_context& jc) {
        return parallelize(s, end, &jc);
    }


    /** \brief Multi-threaded execution of a method.
     *
     * The STATE object must have operator()().
     * The method will be invoked by (nthreads+1) threads.
     */
    template <typename STATE>
    auto parallelize(STATE&& s, job_context* jc, std::size_t nthreads,
                     job::order_type order = job::order_min) {
        if (nthreads <= 0 || !jc)
            s();
        else {
            struct state {
                STATE& s;
                inline auto operator()() { s(); return 0; }
            };
            std::vector<job_function<state> > jobs(nthreads,state{s});
            for (auto& job : jobs)
                jc->submit_absolute(order, job);
            s(); // this thread
            jc->wait_for_all(jobs.begin(), jobs.end());
            for (auto& job : jobs)
                *job; // re-throw any exceptions
        }
    }
    template <typename STATE>
    inline auto parallelize(STATE&& s, job_context* jc) {
        auto nthreads = jc ? jc->num_threads() : 0;
        return parallelize(s, jc, nthreads);
    }
    template <typename STATE>
    inline auto parallelize(STATE&& s,
                            job_context& jc, std::size_t nthreads,
                            job::order_type order = job::order_min) {
        return parallelize(s, &jc, nthreads, order);
    }
    template <typename STATE>
    inline auto parallelize(STATE&& s, job_context& jc) {
        return parallelize(s, &jc);
    }
}
