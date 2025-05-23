#pragma once

#include <memory>
#include "context_settings.hpp"

namespace core {

    class context;
    using context_ptr = std::unique_ptr<context>;

    // forward declarations
    namespace job {
        template <typename DATA>
        class queue;
        template <typename DATA>
        class context;
        template <typename DATA>
        class external_job;
        template <typename DATA, typename R>
        class result;
        template <typename DATA, typename FN>
        class function;
    }

    struct context_data;
    class thread_data;
    using job_queue = job::queue<thread_data>;
    using job_context = job::context<thread_data>;
    template <typename R>
    using job_result = job::result<thread_data,R>;
    template <typename FN>
    using job_function = job::function<thread_data,FN>;

    
    /** \brief Multi-threading context.
     *
     * In addition to managing the threads used for parallelization,
     * this object can also store arbitrary data (e.g. settings, models, etc.).
     * In addition to this, each thread has a thread_data object which
     * can similarly store arbitrary data but is thread specific.
     * There is also global data which is shared across all context objects
     * but will be cleaned up when the last context is destroyed.
     *
     * This class is an abstract base class.
     * Use the construct() method to get an instance.
     */
    class context {
    public:
        virtual ~context() = default;


        /** \brief Construct from settings.
         */
        static context_ptr construct(const context_settings& settings);
        

        /** \brief Access to context and global data.
         */
        virtual context_data& data() = 0;
        virtual const context_data& data() const = 0;
        inline operator context_data&() { return data(); }
        inline operator const context_data&() const { return data(); }


        /** \brief Access to settings used to construct context.
         */
        const context_settings& settings() const;

        
        /** \brief Actual number of threads.
         *
         * Note that the main thread (thread calling this method) counts
         * as one.  
         * Therefore, the number of additional worker threads that were
         * started is one less than the number returned.
         *
         * This value will differ from the one in settings if the latter
         * is zero.
         * This value will always be non-zero.
         */
        virtual std::size_t num_threads() const = 0;

        
        /** \brief Job queue.
         */
        virtual job_queue& threads() = 0;


    protected:
        context() = default;
        context(context&&) = delete;
        context& operator=(context&&) = delete;
        context(const context&) = delete;
        context& operator=(const context&) = delete;
    };


    /** \brief Either run job to obtain context or access already running job.
     *
     * This object is meant to be used as a method argument to either
     * accept a context object in which to run the job, or
     * access an already running job_context object.
     */
    class active_job {
    public:
        ~active_job();
        active_job(job_context& jc);
        active_job(job_queue& queue);
        active_job(core::context& context) : active_job(context.threads()) {}
        active_job(core::context* context);

        template <typename U, typename = std::enable_if_t<std::is_convertible<decltype(std::declval<U&&>().get()),core::context*>::value> >
        active_job(U&& ptr)
            : active_job(static_cast<core::context*>(std::forward<U>(ptr).get())) {}

        active_job(active_job&& other);

        job_context& context();
        inline operator job_context&() { return context(); }

        /// returns nullptr if context or queue not available at construction
        inline auto* queue_ptr() const { return q; }
        
    private:
        std::unique_ptr<job::external_job<thread_data> > main;
        job_queue* q;
        job_context* jc;
    };
}
