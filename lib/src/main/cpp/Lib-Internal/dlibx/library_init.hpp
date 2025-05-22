#pragma once

#include <atomic>

namespace dlibx {

    /** \brief External library initialization.
     *
     * Currently only openblas requires initialization if it's being used.
     * Choose between blas_openblas.cpp or blas_other.cpp as needed.
     */
    class library_init_rec {
        std::atomic_flag done = ATOMIC_FLAG_INIT;

        static void openblas_init();
        static void init();

    public:
        inline void operator()() {
            if (!done.test_and_set(std::memory_order_relaxed)) init();
        }
    };
    extern library_init_rec library_init;
}
