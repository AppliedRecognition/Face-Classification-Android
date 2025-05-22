#pragma once

namespace core {

    /** \brief Settings required to construct context object.
     */
    struct context_settings {
        /** \brief Minimum number of threads.
         *
         * If min_threads < max_threads, then the total number of threads to
         * use is determined by calling std::thread::hardware_concurrency().
         * The result of this call is clamped to be within the bounds
         * specified.
         *
         * If min_threads >= max_threads, then the total number of threads
         * will be the minimum of these 2 numbers, but at least 1.
         *
         * The number of additional threads started will be one less than
         * the total number of threads determined above.
         * This is because the main thread, the one making calls to have
         * work performed, counts as one of the threads.
         * Therefore, if std::min(min_threads,max_threads) <= 1, then
         * no additional threads are started.
         */
        unsigned min_threads = 1;

        /** \brief Maximum number of threads.
         * \sa min_threads for details
         */
        unsigned max_threads = 1;

        /** \brief Make use of CPU SIMD instructions.
         *
         * This setting currently does nothing!
         * In the past it did what is described below.
         *
         * Set to true only in the case where the cpu has been confirmed to
         * have the necessary SIMD feature.
         * For X86, some version (yet to be determined) of SSE is required.
         * For ARM, the base level of NEON is required.
         */
        bool use_simd = false;
    };
}
