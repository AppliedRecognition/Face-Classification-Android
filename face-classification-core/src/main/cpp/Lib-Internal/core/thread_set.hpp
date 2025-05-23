#pragma once

#include <cassert>
#include <set>
#include <mutex>
#include <condition_variable>

namespace core {
    /** \brief Helper for creating jobs that must execute exactly once
     * in each thread before any return.
     *
     * In each thread first call visit with some thread unique pointer
     * to get an integer indicating order of arrival.
     * Then (after doing some work) call wait() to wait until all threads
     * have reached visit().
     */
    struct thread_set {
        const std::size_t num_threads;
        std::set<const void*> set;
        std::mutex m;
        std::condition_variable done;

        explicit thread_set(std::size_t num_threads)
            : num_threads(num_threads) {}

        std::size_t visit(const void* t) {
            std::unique_lock<std::mutex> lock(m);
            const auto p = set.insert(t);
            // have to assert here because otherwise deadlock
            // the most common reason for this assert triggering is
            // failure to call wait()
            assert(p.second);
            done.notify_all();
            return set.size();
        }

        void wait() {
            std::unique_lock<std::mutex> lock(m);
            while (set.size() < num_threads)
                done.wait(lock);
        }
    };
}
