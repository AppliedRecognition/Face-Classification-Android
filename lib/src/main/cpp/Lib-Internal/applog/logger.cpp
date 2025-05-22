
#include "logger.hpp"

#define __FBLIB_APPLOG_PRIVATE_INTERNAL_USE_ONLY__
#include "internal.hpp"

#include <cassert>

using namespace applog;


static const char* level_to_string(log_level level) {
    static const char* strings[] = {
        "FATAL", "ERROR", "WARNING", "INFO", "DEBUG", "DETAIL", "TRACE" };
    return level < 0 ? "NONE" : strings[level];
}


/**************** class logger ****************/

void logger::detail::init_stream(const time_point& now,
                                 log_level level,
                                 applog::internal::thread& rec,
                                 const std::string& module_extra) {
    static const char* const indent[] = {
        ": ", ": ", ": ", ": ", ": ",
        ": \t",   // logDETAIL
        ":  \t\t"  // logTRACE
    };
    static_assert(logDETAIL == 5 && logTRACE == 6,
                  "indent array needs update");
    assert(level >= 0 && level < 7);
    stream << "- " << now.local_time_of_day().data()
           << ' ' << rec.thread_name()
           << ' ' << level_to_string(level)
           << indent[level]
           << rec.module_tags()
           << module_extra;
}

logger::detail* logger::init(log_level level) {
    if ((thread = applog::internal::thread::get())) {
        const auto now = thread->global->now();
        const auto detail = thread->enter();
        try {
            // determine which sinks we intend to write to
            const auto lock = thread->global->get_shared_lock();
            for (auto& p : *thread->global) {
                p.second.check_day(p.first,now);
                const auto& levels = p.second.get_levels(p.first,*thread);
                assert(!levels.empty());
                if (level <= levels.back())
                    detail->sinks.push_back(p.first);
            }
            thread->sink_levels_updated();
        
            // if we have at least one sink to write to, initialize stream
            if (!detail->sinks.empty()) {
                detail->init_stream(now,level,*thread);
                return detail;
            }
        }
        catch (...) {
            thread->leave(detail);
            throw;
        }
        thread->leave(detail);
    }
    return nullptr;
}

logger::detail* logger::init(log_level level, const module& module) {
    if ((thread = applog::internal::thread::get())) {
        const auto now = thread->global->now();
        const auto detail = thread->enter();
        try {
            // determine which sinks we intend to write to
            const auto lock = thread->global->get_shared_lock();
            for (auto& p : *thread->global) {
                p.second.check_day(p.first,now);
                // enter module specified as argument
                const auto& levels = p.second.get_levels(p.first,*thread);
                if (level <= module::detail::enter(module,p.first,levels.back()))
                    detail->sinks.push_back(p.first);
            }
            thread->sink_levels_updated();

            // if we have at least one sink to write to, initialize stream
            if (!detail->sinks.empty()) {
                detail->init_stream(now,level,*thread,
                                    thread->extra_module_tag(module));
                return detail;
            }
        }
        catch (...) {
            thread->leave(detail);
            throw;
        }
        thread->leave(detail);
    }
    return nullptr;
}

logger::~logger() {
    if (thread) thread->leave(state);
}

void logger::_flush() {
    assert(state);
    state->stream << std::endl;
    for (const auto& ptr : state->sinks) {
        try {
            ptr->write_log(state->stream.str());
        }
        catch (const std::exception&) {
            // what to do?
        }
    }
    thread->leave(state);
    // state = nullptr;  < this is done in outer method
}

std::ostream& logger::operator()() const {
    assert(state);
    return state->stream;
}


