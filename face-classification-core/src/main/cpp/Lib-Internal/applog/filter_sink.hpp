#pragma once

#include <map>
#include "module.hpp"
#include "sink.hpp"

namespace applog {

    /** \brief Specialization with basic log level management.
     */
    class filter_sink : public sink {
    public:
        using shared_ptr = std::shared_ptr<filter_sink>;

        inline log_level get_base_level() const {
            return base_level;
        }

        static inline void set_base_level(shared_ptr sink, log_level level) {
            if (sink) {
                lock_type lock = lock_and_reset_sink(sink);
                sink->base_level = level;
            }
        }

        filter_sink(log_level base_level = logTRACE) 
            : base_level(base_level) {
        }

    private:
        log_level module_entered(const module&, log_level) const override;

        log_level base_level;
        using module_levels_type = std::map<module,log_level>;
        module_levels_type module_levels;
    };

}

