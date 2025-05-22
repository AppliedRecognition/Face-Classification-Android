#pragma once

#include "object_store.hpp"
#include "context_settings.hpp"

namespace applog {
    class section;
}

namespace core {
    struct context_data {
        object_store<true>& global;
        object_store<true>& context;

        inline const context_settings& settings() const {
            return get<const context_settings>(context);
        }
    };

    class thread_data : public context_data {
    public:
        object_store<false> thread;

        explicit thread_data(
            object_store<true>& global,
            object_store<true>& context,
            bool register_thread = false);

        explicit thread_data(
            context_data& cd,
            bool register_thread = false)
            : thread_data(cd.global, cd.context, register_thread) {}

        /// copy constructor creates a new thread object store
        thread_data(thread_data& td) : thread_data(td.global, td.context) {}

        ~thread_data();

    private:
        const std::unique_ptr<applog::section> section;
    };
}
