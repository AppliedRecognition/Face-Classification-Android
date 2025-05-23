#pragma once

#include "thread_set.hpp"
#include <applog/core.hpp>
#include <cstring>
#include <errno.h>
#include <sys/time.h>
#include <sys/resource.h>

namespace core {
    /** \brief Job function to call setpriority() on each worker thread.
     *
     * This is a linux specific method.
     */
    struct setpriority_job {
        thread_set& ts;
        const int priority;
        const applog::log_level loglevel;

        setpriority_job(int priority, thread_set& ts,
                        applog::log_level loglevel = logINFO)
            : ts(ts), priority(priority), loglevel(loglevel) {
        }

        template <typename JC>
        auto operator()(JC&& jc) {
            const auto n = unsigned(ts.visit(&jc.data));
            const auto prev = getpriority(PRIO_PROCESS,0);
            if (auto err = setpriority(PRIO_PROCESS,0,priority))
                FILE_LOG(logERROR) << "setpriority(): " << strerror(errno);
            const auto cur = getpriority(PRIO_PROCESS,0);
            if (cur == priority)
                FILE_LOG(loglevel) << "setpriority() now " << cur
                                   << " (was " << prev << ')';
            else
                FILE_LOG(logWARNING) << "setpriority() attempted " << priority
                                     << " but got " << cur
                                     << " (was " << prev << ')';
            ts.wait();
            return n;
        }
    };
}
