#pragma once

#include <string>
#include <memory>
#include "levels.hpp"


namespace applog {
    namespace internal {
        struct global;
        struct thread;
    }

    /** \brief Module flags.
     *
     * <dl>
     *   <dt>NUMBER</dt>
     *     <dd>append unique index to description</dd>
     *   <dt>THREAD</dt>
     *     <dd>description will appear in the thread section of log message</dd>
     * </dl>
     */
    enum module_flag {
        flagNONE   = 0,
        flagNUMBER = 1,
        flagTHREAD = 2
    };

    /** \brief Tag for a logical module within the application.
     *
     */
    class module {
    public:
        /** \brief Construct a non-thread module with no name.
         */
        module();

        /** \brief Construct a named module. 
         */
        explicit module(const std::string& description, int flags = flagNONE);

        /** \brief Set a description for the module.
         *
         * This method overwrites any previously set description.
         */
        void set_description(const std::string& description); 

        /** \brief Get current description of module.
         */
        const std::string& get_description(); 

        /** \brief Register a submodule.
         *
         * When entering a submodule with section, or by specifying the
         * submodule directly to logger, all parent modules will also be
         * entered.
         * Thus when log message output is limited to a parent, log messages
         * associated with decendent modules will also be output.
         *
         * A call to this method that would lead to a cycle in the module
         * hierarchy will result in an exception.
         */
        void register_submodule(module submodule);


        inline bool operator==(const module& other) const {
            return state == other.state;
        }
        inline bool operator!=(const module& other) const {
            return state != other.state;
        }
        inline bool operator<(const module& other) const {
            return state < other.state;
        }
    
        /// \brief Internal detail (made public to simplify internal use).
        struct detail;

    private:
        module(std::shared_ptr<detail> state) : state(state) {}
        std::shared_ptr<detail> state;
        friend class section;
        friend struct internal::global;
        friend struct internal::thread;
    };


    /** \brief Object to mark scope of module section.
     */
    class section {
    public:
        /** \brief Begin section of specified module.
         *
         * Recursive re-entry into a module is allowed.
         *
         * If level is not logNONE, section entry and exit will be logged
         * provided that module has a non-empty description.
         * Recursive entry into a module (including subsequent entry into
         * a parent module) will not be logged.
         *
         * At most one thread module may be entered at a given time within a
         * single thread.
         */
        section(module m, log_level level = logNONE);

        ~section();

    private:
        section(const section&); ///< disabled
        section& operator=(const section&); ///< disabled

        static std::string thread_push_back(
            const std::shared_ptr<module::detail>& ptr, log_level ll);

        std::shared_ptr<module::detail> module_ptr;
        std::string module_type;
        log_level level;
    };

    // return list of currently active threads
    std::string report_threads();
}


/** \brief Create module for thread and start section.
 */
#define REGISTER_NUMBERED_THREAD(name)                                  \
    applog::section                                                     \
    APPLOG_THREAD_SECTION(                                              \
        applog::module(name,applog::flagTHREAD|applog::flagNUMBER),     \
        logINFO)



