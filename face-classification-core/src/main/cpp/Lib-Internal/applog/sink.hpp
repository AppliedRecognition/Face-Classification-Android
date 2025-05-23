#pragma once

#include <string>
#include <memory>

#include "levels.hpp"


namespace applog {
    class module;

    /** \brief Abstract base class for all logging sinks.
     */
    class sink {
    public:
        using shared_ptr = std::shared_ptr<sink>;
        using lock_type = std::shared_ptr<void>;

        /** \brief Add a new logging sink.
         */
        static void add_sink(shared_ptr sink);

        /** \brief Take global unique lock on applog and reset sink when
         * released.
         *
         * Use this method if the log level policy of a sink has changed and
         * one wants this change to take effect in all thread immediately.
         *
         * The module_entered() method will not be called while this lock is
         * held.  
         * The write_log() method may still be called by other threads.
         *
         * If the sink is not currently active (hasn't been added), the
         * returned value will not represent a lock.
         */
        static lock_type lock_and_reset_sink(shared_ptr sink);

        /** \brief Remove a logging sink.
         */
        static void remove_sink(shared_ptr sink);
    

        /** \brief Get revised log level given entry into module.
         *
         * Within each new thread, this method will be called for the first
         * time with a nameless non-thread module to get the base log level.
         *
         * This method may be called in multiple threads simultaneously.
         *
         * \param[in] module module that is being entered
         * \param[in] prev_level log level in force before entry to module
         * \return new log level
         */
        virtual log_level module_entered(const module& m, 
                                         log_level prev_level) const = 0;

        /** \brief Write a log line.
         *
         * The log_line is expected to contain the terminating end-of-line
         * characters.
         *
         * This method may be called in multiple threads simultaneously.
         *
         * \param[in] log_line string to write to log
         * \param[in] day_msg true if this is a date message
         * \param[in] new_day true if this is the first line written on new day
         */
        virtual void write_log(const std::string& log_line, 
                               bool day_msg = false, 
                               bool new_day = false) = 0;

        virtual ~sink() {}
    };

}


