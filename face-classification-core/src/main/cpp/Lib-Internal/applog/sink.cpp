
#include "sink.hpp"

#define __FBLIB_APPLOG_PRIVATE_INTERNAL_USE_ONLY__
#include "internal.hpp"


using namespace applog;


void sink::add_sink(shared_ptr sink) {
    if (const auto internal = applog::internal::global::get()) {
        const auto lock = internal->get_unique_lock();
        internal->insert(sink);
    }
    else throw std::runtime_error("cannot add sink: logging not available");
}

sink::lock_type sink::lock_and_reset_sink(shared_ptr sink) {
    if (const auto internal = applog::internal::global::get()) {
        auto lock = internal->get_unique_lock();
        if (internal->reset(sink))
            return std::make_shared<decltype(lock)>(std::move(lock));
    }
    return {};
}

void sink::remove_sink(shared_ptr sink) {
    if (const auto internal = applog::internal::global::get()) {
        const auto lock = internal->get_unique_lock();
        internal->erase(sink);
    }
}
