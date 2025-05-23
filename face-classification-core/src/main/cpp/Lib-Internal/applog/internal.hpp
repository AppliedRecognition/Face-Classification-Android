#pragma once

// internal include file (don't include in files outside applog library)
#ifndef __FBLIB_APPLOG_PRIVATE_INTERNAL_USE_ONLY__
#error applog/internal.hpp is private (do not include)
#endif

#include "module.hpp"
#include "sink.hpp"
#include "logger.hpp"
#include "time_point.hpp"

#include <atomic>
#include <list>
#include <map>
#include <mutex>
#include <optional>
#include <set>
#include <shared_mutex>
#include <sstream>
#include <vector>

namespace std_shared_mutex_compat {
    using shared_mutex = std::shared_timed_mutex;
}
namespace std {
    namespace shared_mutex_selector {
        using namespace std_shared_mutex_compat;
        using type = shared_mutex;
    }
}
namespace applog {
    /// shared_mutex if available, otherwise shared_timed_mutex
    using shared_mutex = std::shared_mutex_selector::type;

    using module_shared_ptr = std::shared_ptr<module::detail>;

    struct module::detail {
        inline bool is_thread() const {
            return (flags & flagTHREAD) != 0;
        }
        static bool is_thread(const module& module) {
            return module.state->is_thread();
        }
        const std::string& get_description() const {
            return description;
        }
        static const std::string& get_description(const module& module) {
            return module.state->description;
        }
        void set_description(const std::string& desc) {
            description = desc;
        }

        void insert_parent(const module_shared_ptr& m) {
            parents.insert(m);
        }

        static void insert_parents(std::set<module_shared_ptr>& dest,
                                   const module_shared_ptr& module);

        bool find_ancestor(const module_shared_ptr& ancestor) const;

        static log_level enter(const module_shared_ptr& module,
                               sink::shared_ptr sink, 
                               log_level prev_level);
        static log_level enter(const module& module,
                               sink::shared_ptr sink, 
                               log_level prev_level);

        detail() noexcept : flags(0) {}
        detail(std::string description, int flags);

    private:
        log_level enter_parents(sink::shared_ptr sink, 
                                log_level prev_level) const;

        std::string description;
        const int flags;
        std::set<module_shared_ptr> parents;

        using lock_type = std::lock_guard<std::mutex>;
        static std::mutex number_mutex;
        static std::map<std::string,unsigned long long> number_map;
    };


    /** \brief Per logging instance state.
     */
    struct logger::detail {
        std::vector<sink::shared_ptr> sinks;
        std::ostringstream stream;

        void init_stream(const time_point& now,
                         log_level level,
                         applog::internal::thread& rec,
                         const std::string& module_extra = {});

        inline void reset(const std::ios& other) {
            sinks.clear();
            stream.copyfmt(other);
            stream.rdbuf()->str(std::string{});
            stream.clear();
        }
    };


    /** \brief Per thread object.
     */
    struct internal::thread {
        const std::shared_ptr<internal::global> global;
        std::unique_ptr<section> thread_section;

        logger::detail* enter();
        void leave(logger::detail*);

        const std::string& thread_name();
            
        // returns module type if leave must be logged
        const char* push_back(const module_shared_ptr& module,
                              log_level enter_level);
            
        void update_sink_levels(const sink::shared_ptr& sink,
                                std::vector<log_level>& levels) const;
            
        void sink_levels_updated();
            
        void invalidate_levels();
            
        const std::string& module_tags();
        std::string extra_module_tag(const module_shared_ptr& m);
        std::string extra_module_tag(const module& m);
            
        void cleanup();

        /// returns nullptr if logging not available
        static thread* get();
            
        thread(std::shared_ptr<internal::global> global_ptr,
               std::atomic<bool>* available_flag = nullptr);
        ~thread();

        // per thread sink levels (key is pointer to sink object)
        std::map<const void*,
                 std::weak_ptr<std::vector<log_level> > > m_sink_levels;

    private:
        std::atomic<bool>* const available_flag;

        using modules_vector_type =
            std::vector<std::weak_ptr<module::detail> >;
        modules_vector_type::size_type m_valid_levels = 0;
        modules_vector_type m_modules;
            
        std::string m_thread_name;
        std::weak_ptr<module::detail> m_thread_module;
            
        std::optional<std::string> m_module_tags;
        std::set<module_shared_ptr> m_module_tag_set;

        std::list<std::string>::iterator m_thread_iterator;

        std::list<logger::detail> m_instance;
        std::list<logger::detail>::iterator m_instance_iter;
    };


    /** \brief Global static (singlton) state object.
     */
    struct internal::global {
        const module base_module;
        const std::ostringstream base_stream{};

        using lock_type = std::shared_ptr<void>;

        struct sink_record {
            using levels_type = std::vector<log_level>;
            
            void check_day(const sink::shared_ptr& sink,
                           const time_point& now);
            
            const levels_type&
            get_levels(const sink::shared_ptr& sink, thread& rec);

            void reset();

            sink_record();

        private:
            std::mutex m_day_mutex;
            day_number_type prev_day = day_number_type{0};

            std::shared_ptr<std::list<levels_type> > thread_levels;
            std::mutex m_levels_mutex;
        };


        /// returns nullptr if global state not available
        static std::shared_ptr<global> get();

        /** \brief Get shared lock on applog subsystem.
         */
        inline auto get_shared_lock() const {
            return std::shared_lock<shared_mutex>(m_sink_mutex);
        }

        /** \brief Get unique lock on applog subsystem.
         */
        inline auto get_unique_lock() {
            return std::unique_lock<shared_mutex>(m_sink_mutex);
        }

        using sink_map_type = std::map<sink::shared_ptr,sink_record>;
        using iterator = sink_map_type::iterator;
        using const_iterator = sink_map_type::const_iterator;

        inline void insert(const sink::shared_ptr& sink) {
            m_sink_map[sink];
        }

        // returns true if the sink is present, false if not
        inline bool reset(const sink::shared_ptr& sink) {
            sink_map_type::iterator it = m_sink_map.find(sink);
            if (it == m_sink_map.end()) return false;
            it->second.reset();
            return true;
        }

        inline void erase(const sink::shared_ptr& sink) {
            m_sink_map.erase(sink);
        }

        inline auto begin() {
            return m_sink_map.begin();
        }
        inline auto end() {
            return m_sink_map.end();
        }

        inline auto add_thread(const std::string& name) {
            std::lock_guard<std::mutex> lock(m_thread_mutex);
            return m_thread_list.insert(m_thread_list.end(),name);
        }

        inline void erase_thread(std::list<std::string>::iterator it) {
            std::lock_guard<std::mutex> lock(m_thread_mutex);
            m_thread_list.erase(it);
        }

        std::string report_threads();

        static time_point now(global* ptr);
        inline time_point now() { return now(this); }
        
        global() = default;

    private:
        std::mutex m_thread_mutex;
        std::list<std::string> m_thread_list;

        mutable shared_mutex m_sink_mutex;
        sink_map_type m_sink_map;

        std::mutex m_time_mutex;
    };

}
