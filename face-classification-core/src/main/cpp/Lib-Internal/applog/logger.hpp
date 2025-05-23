#pragma once

#include <ostream>
#include "levels.hpp"

#ifndef APPLOG_MINIMUM_LEVEL
#define APPLOG_MINIMUM_LEVEL logTRACE
#endif

namespace applog {
    namespace internal {
        struct thread;
    }

    class module;

    /** \brief Stream type object to write a single log line.
     *
     * Don't use this class directly.  Instead use the APPLOG() macro below.
     */
    class logger {
    public:
        explicit logger(log_level level)
            : state(logNONE < level && level <= APPLOG_MINIMUM_LEVEL ?
                    init(level) : nullptr) {
        }
        explicit logger(log_level level, const module& m)
            : state(logNONE < level && level <= APPLOG_MINIMUM_LEVEL ?
                    init(level,m) : nullptr) {
        }
        ~logger();

        /// \returns true if log line can be written
        inline bool good() const { return state; }
        
        /// \returns stream to write log line to
        std::ostream& operator()() const;
        
        /// flush log line to sinks
        inline void flush() {
            _flush();
            state = nullptr;
        }

    private:
        struct detail;
        internal::thread* thread = nullptr;
        detail* state;

        detail* init(log_level level);
        detail* init(log_level level, const module& m);
        void _flush();

        logger(logger&&) = delete;
        logger(const logger&) = delete;
        logger& operator=(logger&&) = delete;
        logger& operator=(const logger&) = delete;

        friend struct internal::thread;
    };
}


/** \brief Macro to construct logger object only if level is sufficient.
 */
#define APPLOG(level)                                                   \
    for (applog::logger _applog_logger_object_(level);                  \
         _applog_logger_object_.good();                                 \
         _applog_logger_object_.flush()) _applog_logger_object_()

#define FILE_LOG(level)                                                 \
    for (applog::logger _applog_logger_object_(level);                  \
         _applog_logger_object_.good();                                 \
         _applog_logger_object_.flush()) _applog_logger_object_()

#define MODLOG(module,level)                                            \
    for (applog::logger _applog_logger_object_(level,module);           \
         _applog_logger_object_.good();                                 \
         _applog_logger_object_.flush()) _applog_logger_object_()


