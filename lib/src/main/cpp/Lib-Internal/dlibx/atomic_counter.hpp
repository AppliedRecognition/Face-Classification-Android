#pragma once

#include <atomic>

namespace dlibx {

    /** \brief Circularly linked list of atomic counters.
     *
     * Each thread gets it's own counter to count through first.
     * Then the linked list is followed to help other threads finish.
     *
     * Warning: if used as base for job function object and these objects
     * are stored in std::list, ensure none of them are deallocated until
     * all jobs have finished.
     */
    template <typename T, T multiplier = 1>
    class atomic_counter {
    public:
        using value_type = T;
    private:
        std::atomic<value_type> value;
        const value_type offset;
    public:
        const value_type limit;
        atomic_counter* link;

        inline value_type next() {
            return offset +
                multiplier * value.fetch_add(1,std::memory_order_relaxed);
        }

        atomic_counter(atomic_counter* link,
                       value_type offset,
                       value_type limit)
            : value(0), offset(offset), limit(limit), link(link) {}
    };
}
