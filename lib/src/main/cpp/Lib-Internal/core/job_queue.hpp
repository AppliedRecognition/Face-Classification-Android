#pragma once

#include <algorithm>
#include <atomic>
#include <stdexcept>
#include <limits>
#include <type_traits>
#include <thread>
#include <vector>
#include <mutex>
#include <condition_variable>
#include <stdext/options_tuple.hpp>
#include <stdext/identity.hpp>

#include <boost/intrusive/set.hpp>


namespace core {
    namespace job {

        // order (job priority) type
        using order_type = int;
        constexpr auto order_min =
            std::numeric_limits<order_type>::min();
        constexpr auto order_max =
            std::numeric_limits<order_type>::max();

        // job options
        enum class absolute_order : order_type {};
        enum class relative_order : order_type {};
        //using order = relative_order;

        template <typename T>
        constexpr void option_apply(T& t, relative_order d) {
            auto& o = std::get<absolute_order>(t);
            const auto delta = order_type(d);
            const auto order = order_type(o);
            if (delta > 0 && order > order_max - delta)
                o = absolute_order(order_max);
            else if (delta < 0 && order < order_min - delta)
                o = absolute_order(order_min);
            else
                o = absolute_order(order + delta);
        }

        struct can_run_now_tag;
        using can_run_now_option = stdx::option_bool<can_run_now_tag>;
        const can_run_now_option can_run_now{true};

        struct return_to_parent_tag;
        using return_to_parent_option = stdx::option_bool<return_to_parent_tag>;
        const return_to_parent_option return_to_parent{true};


        /// thrown to interrupt job
        struct interrupt_signal {};


        // forward declarations
        template <typename DATA> class external_job;
        template <typename DATA> class queue;
        template <typename DATA> class pool;
        template <typename DATA> class context;


        /** \brief Base class for jobs.  
         *
         * This class is abstract.  
         * Use function<> for concrete jobs.
         */
        template <typename DATA>
        class base : public boost::intrusive::set_base_hook<> {
        protected:
            enum s { PENDING, ACTIVE, ABANDONED, VALUE, EXCEPTION };
            std::atomic<int> state{PENDING};
            std::atomic<bool> interrupt_pending{false};
            bool m_restrict_reentry = false;
            std::size_t m_max_threads = std::size_t(-1);
            const void* m_submitter = nullptr;
            pool<DATA>* m_owner = nullptr;
            order_type m_order;
            
            base() = default;
            base(base&&) = delete;
            base& operator=(base&&) = delete;
            base& operator=(const base&) = delete;

            /// can be copied if not submitted
            base(const base& other)
                 : m_restrict_reentry(other.m_restrict_reentry),
                   m_max_threads(other.m_max_threads) {
                if (other.m_owner || other.m_submitter)
                    throw std::logic_error("cannot copy submitted job");
            }

            virtual ~base() = default;

            virtual void run(context<DATA>&) = 0;

            virtual void run_interrupt_method() = 0;

            template <typename> friend class pool;
            template <typename> friend class external_job;

        public:
            /** \brief Limit return value for job_context::num_threads().
             *
             * If the job method uses the result of job_context::num_threads()
             * to determine the number of sub-jobs to run, then this
             * setting can be used to limit the job's parallelism.
             */
            inline void set_max_threads(std::size_t n) {
                m_max_threads = n;
            }
            inline auto max_threads() const { return m_max_threads; }

            /** \brief Allow job to wait for jobs it did not submit.
             *
             * Deadlock can occur if a job waits for sub-jobs that were
             * inherited (ie. submitted before the job started running).
             * This happens because the job may find itself running within
             * the context of one of it's sub-jobs (further down the stack).
             * Enable this option for any such job to prevent deadlock.
             *
             * Note that the usual pattern of submitting sub-jobs from within
             * a running job and then waiting for them does not have this
             * deadlock problem, and therefore, does not need this option.
             *
             * The use of this option comes with a performance cost as it
             * limits the running of this job from within wait().
             * An explicit wait() for this job may still cause it to execute
             * within the context of the job that called wait().
             *
             * This method may only be called before the job has been
             * submitted.
             */
            void can_inherit_jobs(bool enable = true) {
                if (m_owner && m_restrict_reentry != enable)
                    throw std::logic_error("cannot change job inherit status after submit");
                m_restrict_reentry = enable;
            }
            inline auto restrict_reentry() const {
                return m_restrict_reentry;
            }

            /** \brief Claim ownership of job.
             *
             * Throws an exception if job is already owned.
             */
            void claim(pool<DATA>& owner, order_type order,
                       const void* submitter = nullptr) {
                if (m_owner)
                    throw std::logic_error("job already submitted");
                m_owner = &owner;
                m_order = order;
                m_submitter = submitter;
            }

            /** \brief Access owner of job.
             *
             * \returns nullptr if job not submitted
             */
            inline auto owner() const {
                return m_owner;
            }

            /** \brief Access opaque pointer to submitter of this job.
             *
             * Note that this pointer is const void* and is to be used
             * for comparison purposes only.
             */
            inline auto submitter() const {
                return m_submitter;
            }

            /** \brief Test if this job was submitted by other job.
             */
            inline bool submitted_by(const base& other) const {
                return m_submitter == &other;
            }

            /** \brief Access to order value job was submitted with.
             *
             * Undefined if job has not been submitted yet.
             */
            inline order_type order() const {
                return m_order;
            }

            /// sort by order (higher priority jobs first)
            inline bool operator<(const base& other) const {
                return order() <= other.order() &&
                    (order() < other.order() || this < &other);
            }

            inline bool is_pending() const {
                return state.load(std::memory_order_acquire) == PENDING;
            }
            inline bool is_active() const {
                return state.load(std::memory_order_acquire) == ACTIVE;
            }
            inline bool is_done() const {
                return state.load(std::memory_order_acquire) > ACTIVE;
            }

            /** \brief Mark job as interrupted.
             *
             * If the job is pending and an attempt is made to run it, or
             * if it is active and it either submits or waits for a sub-job,
             * then it will be terminated and marked as abandoned.
             * In other cases, the job will complete normally.
             */
            void interrupt_job() {
                interrupt_pending.store(true, std::memory_order_release);
                if (is_pending())
                    run_interrupt_method();
            }

            /** \brief Check if interrupt flag set and throw signal if it is.
             */
            inline void throw_if_interrupted() const {
                if (interrupt_pending.load(std::memory_order_acquire))
                    throw interrupt_signal();
            }
            template <typename ITER, typename FN>
            inline void
            throw_if_interrupted_i(ITER it, ITER end, FN&& fn) const {
                if (interrupt_pending.load(std::memory_order_acquire)) {
                    for ( ; it != end; ++it)
                        fn(*it).interrupt_job();
                    throw interrupt_signal();
                }
            }

            /** \brief Run job with specified data.
             *
             * Job must be in pending state and owned prior to making this
             * call.  
             * The overload that accepts a lock will unlock prior to running
             * the job and re-lock afterwards.
             */
            void run(std::unique_lock<std::mutex>& lock,
                     DATA& data, order_type order) {
                const auto prev =
                    state.exchange(ACTIVE,std::memory_order_release);
                assert(prev == PENDING);
                // note: unlock must happen after job.state transition to ACTIVE
                lock.unlock();
                context tc(order, this, data);
                run(tc);
                lock.lock();
            }
            inline void run(DATA& data) {
                const auto prev =
                    state.exchange(ACTIVE,std::memory_order_release);
                assert(prev == PENDING);
                context tc(m_order, this, data);
                run(tc);
            }
        };


        /** \brief Context / thread specific data for running job.
         *
         * An instance of this class is passed as the only argument to
         * a job method.  
         * It provides access to thread specific data for the job, and
         * allows the job to submit and wait for sub-jobs.
         */
        template <typename DATA>
        class context {
        public:
            using order_type = job::order_type;
            static constexpr auto order_min = job::order_min;
            static constexpr auto order_max = job::order_max;

            using data_type = DATA;
            using base = job::base<DATA>;

        private:
            static thread_local context* this_context_ptr;
            context* const parent_context;

            const order_type order;  ///< effective order <= job.order()

            context(context&&) = delete;
            context(const context&) = delete;
            context& operator=(context&&) = delete;
            context& operator=(const context&) = delete;

        public:
            /** \brief Current job.
             */
            const base& job;

            /** \brief Thread specific data.
             */
            data_type& data;
            operator data_type&() { return data; }
            operator const data_type&() const { return data; }

            /** \brief Constructor.
             */
            explicit context(order_type order, const base* job, data_type& data)
                : parent_context(this_context_ptr),
                  order(std::min(order,job->order())),
                  job(*job),
                  data(data) {
                assert(job && job->owner());
                assert(parent_context == nullptr ||
                       parent_context->job.owner() != job->owner() ||
                       &parent_context->data == &data);
                this_context_ptr = this;
            }

            /** \brief Destructor.
             */
            ~context() {
                this_context_ptr = parent_context;
            }

            /** \brief Access to pool that owns this context.
             */
            inline pool<DATA>& owner() const {
                return *job.owner();
            }
            
            /** \brief Number of additional threads running.
             */
            inline std::size_t num_threads() const {
                return std::min(job.max_threads(), owner().num_threads());
            }

            /** \brief Submit a job.
             *
             * Priority can be either same as caller, an absolute value,
             * or a value relative to caller.
             *
             * \sa queue::submit() for complete details
             *
             * This method is an interrupt point.
             * If job_interrupted() has been called for the current job, 
             * it will terminate at this point.
             *
             * \param[in] fn function object
             * \param[in] opts job options
             */
            template <typename... Opts>
            void submit(base& fn, Opts&&... opts) {
                const auto ot = stdx::options_tuple<absolute_order,can_run_now_option,return_to_parent_option>(absolute_order(job.order()), std::forward<Opts>(opts)...);
                job.throw_if_interrupted();
                fn.claim(owner(), order_type(std::get<absolute_order>(ot)),
                         std::get<return_to_parent_option>(ot) ?
                         job.submitter() : static_cast<const void*>(&job));
                if (std::get<can_run_now_option>(ot) &&
                    fn.owner()->can_run_now(fn))
                    fn.run(data);
                else
                    fn.owner()->queue_job(fn);
            }
            inline void submit_absolute(order_type order, base& fn) {
                submit(fn, absolute_order(order));
            }

            /** \brief Wait for multiple jobs to complete.
             *
             * Unless the can_inherit_jobs() flag is set, this method may only
             * be used to wait for jobs that were submitted by the calling
             * job.
             *
             * \sa queue::wait() for complete details
             *
             * Note that other jobs run by this method will be using
             * the same thread specific data that the caller has access to.
             *
             * This method is an interrupt point.
             * If job_interrupted() has been called for the current job, 
             * an exception is thrown to terminate it.
             * Furthermore, the jobs that were to be waited for will
             * also be interrupted.
             */
            template <typename... Jobs>
            inline void wait(Jobs&... jobs) {
                base* arr[] = { &jobs... };
                if (!job.restrict_reentry())
                    for (auto p : arr)
                        if (!p->submitted_by(job))
                            throw std::invalid_argument("illegal wait for inherited job");
                const auto begin = std::begin(arr);
                const auto end =
                    std::remove_if(begin, std::end(arr),
                                   [](auto p) { return p->is_done(); });
                if (begin == end) {
                    job.throw_if_interrupted();
                    return;
                }
                std::sort(begin, end,
                          [](auto* a, auto* b) {
                              return a->order() < b->order();
                          });
                const auto deref = [](auto*p) -> auto& { return *p; };
                job.throw_if_interrupted_i(begin, end, deref);
                job.owner()->template wait_all<true>(
                    "wait sub", data, order, begin, end, deref, true);
            }

            /** \brief Wait for all jobs in sequence to complete.
             *
             * If same_priority, then pending jobs are run in the order
             * they appear in the sequence. 
             * If not, pending jobs are run highest priority first.
             */
            template <typename ITER, typename FN>
            inline void wait_for_all(ITER first, ITER last, FN&& fn,
                                     bool same_priority = true) {
                if (!job.restrict_reentry())
                    for (auto it = first; it != last; ++it)
                        if (!fn(*it).submitted_by(job))
                            throw std::invalid_argument("illegal wait for inherited job");
                job.throw_if_interrupted_i(first, last, fn);
                for ( ; first != last; ++first) {
                    if (fn(*first).is_done()) continue;
                    job.owner()->template wait_all<true>(
                        "wait sub", data, order,
                        first, last, std::forward<FN>(fn),
                        same_priority);
                    break;
                }
            }
            template <typename ITER>
            inline void wait_for_all(ITER first, ITER last) {
                wait_for_all(first, last, stdx::identity());
            }

            /** \brief Wait for any one job to complete.
             *
             * Unless the can_inherit_jobs() flag is set, this method may only
             * be used to wait for jobs that were submitted by the calling
             * job.
             *
             * \sa queue::wait_for_one() for complete details
             *
             * Note that other jobs run by this method will be using
             * the same thread specific data that the caller has access to.
             *
             * This method is an interrupt point.
             * If job_interrupted() has been called for the current job, 
             * an exception is thrown to terminate it.
             * Furthermore, the jobs that were to be waited for will
             * also be interrupted.
             */
            template <typename ITER, typename FN>
            inline ITER wait_for_one(ITER first, ITER last, FN&& fn) {
                job.throw_if_interrupted_i(first, last, fn);
                if (first == last || fn(*first).is_done()) return first;
                if (!job.restrict_reentry() && !fn(*first).submitted_by(job))
                    throw std::invalid_argument("illegal wait for inherited job");
                return job.owner()->template wait_one<true>(
                    "wait sub", data, order, first, last, std::forward<FN>(fn));
            }
            template <typename ITER>
            inline ITER wait_for_one(ITER first, ITER last) {
                return wait_for_one(first, last, stdx::identity());
            }

            /** \brief Return completed job.
             *
             * If any job in the sequence is not pending and not active, then
             * it is returned as complete.
             * Otherwise, last is returned.
             */
            template <typename ITER, typename FN>
            static inline ITER try_for_one(ITER first, ITER last, FN&& fn) {
                for ( ; first != last; ++first)
                    if (fn(*first).is_done())
                        return first;
                return last;
            }
            template <typename ITER>
            static inline ITER try_for_one(ITER first, ITER last) {
                for ( ; first != last; ++first)
                    if ((*first).is_done())
                        return first;
                return last;
            }

            /** \brief Get pointer to context for current job.
             *
             * If owner specified, only return context if it belongs to owner.
             * Otherwise return any available context.
             * Returns null if not running within specified job pool.
             */
            static context*
            this_context(const pool<DATA>* owner = nullptr) {
                if (this_context_ptr &&
                    (owner == nullptr || owner == &this_context_ptr->owner()))
                    return this_context_ptr;
                return nullptr;
            }
            static context* this_context(const queue<DATA>* owner);
        };
        template <typename DATA>
        thread_local context<DATA>* context<DATA>::this_context_ptr = nullptr;
        
        /** \brief Helper struct to invoke job method.
         */
        template <typename, typename FN, typename = void>
        struct function_traits {
            using result_type = std::decay_t<std::invoke_result_t<FN&> >;
            template <typename U>
            static inline result_type
            invoke(FN& fn, U&&) { return fn(); }
        };
        template <typename DATA, typename FN>
        struct function_traits<DATA,FN,std::void_t<decltype(std::declval<FN&>()(std::declval<context<DATA>&>()))> > {
            using result_type =
                std::decay_t<std::invoke_result_t<FN&,context<DATA>&> >;
            static inline result_type
            invoke(FN& fn, context<DATA>& jc) { return fn(jc); }
        };
        template <typename DATA, typename FN>
        using function_result_t =
            typename function_traits<DATA,FN>::result_type;
        template <typename FN, typename DATA>
        inline auto invoke(FN& fn, context<DATA>& jc) {
            return function_traits<DATA,FN>::invoke(fn,jc);
        }

        
        /** \brief Pool of internal worker threads and queue of pending jobs.
         *
         * This object manages worker threads that will execute jobs and
         * does all scheduling of jobs.
         *
         * Job order (priority) minutiae:
         *
         * Lower order mean higher priority and vice versa.
         * order_min is the highest priority while order_max is the lowest.
         *
         * While in the pending queue each job has the order it was submitted
         * with.
         * While active (running) a job may have a lower effective order.
         *
         * When waiting for subjobs to complete, the calling thread may run
         * other unrelated jobs if they have an order which is strictly
         * less than the minimum of: the calling job's effective order and
         * the order of all of the subjobs being waited for.
         * If the main thread calls wait(), then the minimum is over the order
         * of the subjobs being waited for.
         * If no high priority unrelated jobs are pending, then the calling
         * thread will either run the pending subjob with lowest order, or
         * do a hard wait if all subjobs are active.
         *
         * To ensure that no unrelated jobs get run while waiting, either
         * the calling job or the subjob must have order order_min.
         */
        template <typename DATA>
        class pool {
        public:
            using data_type = DATA;
            using base = job::base<DATA>;
            using context = job::context<DATA>;

        private:
            using job_set_type = boost::intrusive::set<base, boost::intrusive::constant_time_size<false> >;
            job_set_type pending_set;

            std::mutex m;
            std::condition_variable job_submitted, job_finished;
            std::vector<std::thread> thread_list;
            bool shutdown = false;

            static inline void log_lock(const char*) {}
            static inline void log_unlock(const char*) {}

            struct thread_method {
                pool& obj;
                template <typename... Args>
                void operator()(Args&&... args) {
                    data_type data{std::forward<Args>(args)...};
                    std::unique_lock<std::mutex> lock(obj.m);
                    log_lock("thread");
                    for (;;) {
                        while (!obj.shutdown && obj.pending_set.empty()) {
                            log_unlock("thread");
                            obj.job_submitted.wait(lock);
                            log_lock("thread");
                        }
                        if (obj.shutdown) break;
                        obj.run_job(lock, data, obj.pending_set.begin());
                    }
                    log_unlock("thread");
                }
            };

            void run_job(std::unique_lock<std::mutex>& lock,
                         data_type& data,
                         const typename job_set_type::iterator& it,
                         order_type order = job::order_max) {
                assert(lock);
                assert(it != pending_set.end());
                auto& job = *it;
                pending_set.erase(it); // note: ref still valid after erase
                log_unlock("run_job");
                job.run(lock, data, order);
                log_lock("run_job");
                job_finished.notify_all();
            }

            // run first job from queue with job.order < order
            // if restrict_reentry, then only if !job.restrict_reentry
            // return true if a job was run, false if no suitable job found
            template <bool restrict_reentry>
            bool run_other(std::unique_lock<std::mutex>& lock,
                           data_type& data, order_type order) {
                if (shutdown) return false;
                for (auto it = pending_set.begin(), 
                         end = pending_set.end(); it != end; ++it) {
                    if (it->order() >= order) break;
                    if (!restrict_reentry || !it->restrict_reentry()) {
                        run_job(lock,data,it,order);
                        return true;
                    }
                }
                return false;
            }

            pool(pool&&) = delete;
            pool(const pool&) = delete;
            pool& operator=(pool&&) = delete;
            pool& operator=(const pool&) = delete;

        public:
            pool() = default;

            /** \brief Destructor.
             *
             * Signal threads to shutdown and wait for them to exit.
             * Any remaining pending jobs are marked as abandoned.
             */
            ~pool() {
                std::unique_lock<std::mutex> lock(m);
                shutdown = true;
                if (!thread_list.empty()) {
                    // wait for threads to finish
                    lock.unlock();
                    job_submitted.notify_all();
                    for (auto& t : thread_list)
                        t.join();
                    lock.lock();
                }
                // mark all remaining pending jobs as abandoned
                for (auto& job : pending_set) {
                    assert(job.owner() == this);
                    const auto prev = 
                        job.state.exchange(job.ABANDONED,
                                           std::memory_order_release);
                    assert(prev == job.PENDING);
                }
            }

            /** \brief Start a new thread.
             * \param[in] args arguments to construct thread data object
             */
            template <typename... Args>
            void start_thread(Args&&... args) {
                std::thread t(thread_method{*this},std::forward<Args>(args)...);
                thread_list.push_back(move(t));
            }

            /** \brief Number of internal threads running in pool.
             */
            inline std::size_t num_threads() const {
                return thread_list.size();
            }

            /** \brief Test if job can be run without queuing.
             *
             * If this pool has no threads, and either the pending set is empty
             * or the job has priority, then the job may be run immediately
             * instead of being queued.
             */
            bool can_run_now(base& fn) const {
                return thread_list.empty() &&
                    (pending_set.empty() ||
                     fn.order() <= pending_set.begin()->order());
            }

            /** \brief Insert job into queue of pending jobs.
             *
             * The job must have already been claim()'d for this pool
             * and be in the pending state.
             */
            void queue_job(base& fn) noexcept {
                assert(fn.is_pending() && fn.owner() == this);
                std::unique_lock<std::mutex> lock(m);
                log_lock("submit");
                const auto p = pending_set.insert(fn);
                assert(p.second);
                log_unlock("submit");
                lock.unlock();
                job_submitted.notify_all();
                job_finished.notify_all();
            }

            /** \brief Remove job from queue.
             *
             * If the job has not yet started, remove it from the queue.
             * If the job is active, wait for it to complete.
             */
            void remove_job(base& job) noexcept {
                std::unique_lock<std::mutex> lock(m);
                log_lock("remove");
                int s;
                while ((s = job.state.load(std::memory_order_acquire)) == base::ACTIVE) {
                    log_unlock("wait remove");
                    job_finished.wait(lock);
                    log_lock("wait remove");
                }
                if (s == base::PENDING)
                    pending_set.erase(job);
                log_unlock("remove");
            }

            /** \brief Wait for all jobs to complete.
             *
             * This method should not be called directly. 
             * Use the wait() methods in class context and class queue.
             */
            template <bool restrict_reentry, typename ITER, typename FN>
            void wait_all(const char* tag, data_type& data, order_type order,
                          ITER first, ITER last, FN&& fn,
                          bool run_first_pending) {
                std::unique_lock<std::mutex> lock(m);
                log_lock(tag);
                for (auto pending_first = first, pending_last = last; ; ) {
                    // find minimum order pending job to run
                    auto mo = job::order_max;
                    base* mj = nullptr;
                    for ( ; pending_first != pending_last; ++pending_first) {
                        auto p = std::addressof(fn(*pending_first));
                        if (p->is_pending()) {
                            mo = p->order(), mj = p;
                            break;
                        }
                    }
                    if (!run_first_pending) {
                        for (auto it = pending_first,
                                 end = pending_last; it != end; ++it) {
                            auto p = std::addressof(fn(*it));
                            if (p->is_pending()) {
                                pending_last = std::next(it);
                                if (mo > p->order())
                                    mo = p->order(), mj = p;
                            }
                        }
                    }
                    if (!mj) break; // all jobs are active or done
                    if (mj->owner() != this)
                        throw std::invalid_argument("job not owned by pool");
                    run_job(lock, data, pending_set.iterator_to(*mj), order);
                }
                assert(lock);
                // all jobs are active, wait for them to finish
                auto thres = order; // minimum order of subjobs
                for (auto jt = first; jt != last; ++jt)
                    thres = std::min(thres, fn(*jt).order());
                if (thres < order)
                    ++thres;
                for ( ; first != last; ++first) {
                    auto& job = fn(*first);
                    while (job.is_active()) {
                        // to avoid hard wait, run other jobs
                        if (!run_other<restrict_reentry>(lock,data,thres)) {
                            log_unlock("wait for job");
                            job_finished.wait(lock);
                            log_lock("wait for job");
                        }
                    }
                }
                log_unlock(tag);
            }

            /** \brief Wait for at least one job to complete.
             *
             * This method should not be called directly. 
             * Use the wait() methods in class context and class queue.
             */
            template <bool restrict_reentry, typename ITER, typename FN>
            ITER wait_one(const char* tag,
                          data_type& data, order_type order,
                          ITER first, ITER last, FN&& fn) {
                std::unique_lock<std::mutex> lock(m);
                log_lock(tag);
                for (;;) {
                    auto top_job = pending_set.end();
                    auto top_iter = first;
                    
                    // find either pending job with lowest order
                    // or a job that is neither pending nor active (complete)
                    for (auto jt = first; jt != last; ++jt) {
                        auto& job = fn(*jt);
                        if (job.owner() != this)
                            throw std::invalid_argument("job not owned by pool");
                        switch (job.state.load(std::memory_order_acquire)) {
                        case base::PENDING:
                            if (top_job == pending_set.end() ||
                                job.order() < top_job->order()) {
                                top_job = pending_set.iterator_to(job);
                                top_iter = jt;
                            }
                        case base::ACTIVE:
                            break;
                        default:
                            log_unlock(tag);
                            return jt;
                        }
                    }
                    if (top_job == pending_set.end())
                        break;  // all jobs active
                    run_job(lock,data,top_job,order);
                    log_unlock(tag);
                    return top_iter;
                }
       
                // all jobs are active, wait for any one to finish
                auto thres = order; // minimum order of subjobs
                for (auto jt = first; jt != last; ++jt)
                    thres = std::min(thres, fn(*jt).order());
                if (thres < order)
                    ++thres;
                for (;;) {
                    // to avoid hard wait, run other jobs
                    if (!run_other<restrict_reentry>(lock,data,thres)) {
                        log_unlock("wait one wait");
                        job_finished.wait(lock);
                        log_lock("wait one wait");
                    }
                    for (auto jt = first; jt != last; ++jt)
                        if (fn(*jt).is_done()) {
                            log_unlock(tag);
                            return jt;
                        }
                }
                log_unlock(tag);
            }
        };

        /** \brief Thread data for external "main" thread and access to pool.
         *
         * This object holds the thread specific data for the "main" thread
         * along with a pool object to schedule jobs to.
         * Multiple instances of this object may share the same pool to
         * support multiple external threads queuing to and sharing the 
         * same pool of internal threads.
         */
        template <typename DATA>
        class queue {
        public:
            using order_type = job::order_type;
            static constexpr auto order_min = job::order_min;
            static constexpr auto order_max = job::order_max;

            using data_type = DATA;
            using base = job::base<DATA>;
            using context = job::context<DATA>;
            using pool = job::pool<DATA>;

            struct data_unlock {
                std::mutex* mux;
                data_unlock(std::mutex* mux = nullptr) : mux(mux) {}
                void operator()(data_type* ptr) const {
                    if (ptr && mux) mux->unlock();
                }
            };
            using data_ptr = std::unique_ptr<data_type,data_unlock>;

        private:
            /** \brief Thread specific data for main thread.
             */
            data_type data_;
            std::mutex data_mux;

            const std::shared_ptr<pool> poolptr;
            
            // run job that has not been queued
            void run_in_place(base& fn, context* jc) {
                if (jc)
                    fn.run(jc->data);
                else {
                    std::lock_guard<std::mutex> dlock(data_mux);
                    fn.run(data_);
                }
            }

            queue(queue&&) = delete;
            queue(const queue&) = delete;
            queue& operator=(queue&&) = delete;
            queue& operator=(const queue&) = delete;
            
        public:
            /** \brief Default constructor.
             *
             * Constructs data with no arguments and a new pool object.
             */
            queue() : data_{}, poolptr(std::make_shared<pool>()) {}

            /** \brief Constructor.
             *
             * Constructs a new pool object.
             *
             * \param[in] args arguments to construct main thread data object
             */
            template <typename A0, typename... Args,
                      typename = std::enable_if_t<
                          !std::is_same_v<std::decay_t<A0>,
                                          std::shared_ptr<pool> > > >
            explicit queue(A0&& a0, Args&&... args)
                : data_{std::forward<A0>(a0), std::forward<Args>(args)...},
                  poolptr(std::make_shared<pool>()) {
            }

            /** \brief Construct with a shared pool object.
             *
             * This queue will share a pool object with other queue objects.
             */
            template <typename... Args>
            explicit queue(const std::shared_ptr<pool>& pp, Args&&... args)
                : data_{std::forward<Args>(args)...},
                  poolptr(pp) {
                assert(pp);
            }

            /** \brief Pointer to data with internal mutex locked until
             * pointer object is destroyed.
             */
            inline auto get_data() {
                data_mux.lock();
                return data_ptr(&data_, &data_mux);
            }

            /** \brief Access to shared pool.
             */
            inline auto& get_pool() const {
                return poolptr;
            }

            /** \brief Number of additional threads running in pool.
             */
            inline auto num_threads() const {
                return poolptr->num_threads();
            }

            /** \brief Start an internal thread in pool.
             */
            template <typename... Args>
            inline void start_thread(Args&&... args) {
                poolptr->start_thread(std::forward<Args>(args)...);
            }
            
            /** \brief Run job now.
             *
             * Construct function object and run job now with access
             * to a context object.
             */
            template <typename FN, typename... Args>
            auto run_emplace(Args&&... args);

            template <typename FN>
            inline auto run(FN&& fn) {
                return run_emplace<std::decay_t<FN> >(std::forward<FN>(fn));
            }
            
            /** \brief Run function object now.
             */
            void run_job(base& fn) {
                const auto jc = context::this_context(this);
                fn.claim(*poolptr,
                         jc ? jc->job.order() : 0,
                         jc ? &jc->job : nullptr);
                run_in_place(fn, jc);
            }
            
            /** \brief Submit a job.
             *
             * The job method must take a single context& argument
             * and return a non-void object or pointer.
             * If an exception is thrown, it will be captured and re-thrown
             * in get().
             *
             * If not specified, order defaults to 0.
             * Lower order values convey higher priority.
             *
             * If can_run_now is true, no additional threads have been
             * started (num_threads() is zero) and the order of the job
             * is at least as low as the lowest currently queued (or no
             * jobs are currently queued), then the job may be run directly
             * by this method instead of being queued.
             * Otherwise, the job is queued and this method will return
             * immediately.
             *
             * \param[in] fn function object
             * \param[in] opts job options
             */
            template <typename... Opts>
            void submit(base& fn, Opts&&... opts) {
                stdx::options_tuple<
                    absolute_order,
                    can_run_now_option,
                    return_to_parent_option> ot;
                const auto jc = context::this_context(this);
                const void* submitter = nullptr;
                if (jc) {
                    jc->job.throw_if_interrupted();
                    ot.apply(absolute_order(jc->job.order()),
                             std::forward<Opts>(opts)...);
                    submitter = std::get<return_to_parent_option>(ot) ?
                        jc->job.submitter() :
                        static_cast<const void*>(&jc->job);
                }
                else
                    ot.apply(std::forward<Opts>(opts)...);
                fn.claim(*poolptr,
                         order_type(std::get<absolute_order>(ot)),
                         submitter);
                if (std::get<can_run_now_option>(ot) &&
                    poolptr->can_run_now(fn))
                    run_in_place(fn, jc);
                else
                    poolptr->queue_job(fn);
            }
            inline void submit_absolute(order_type order, base& fn, 
                                        bool can_run_now = false) {
                submit(fn, absolute_order(order),
                       job::can_run_now(can_run_now));
            }

            /** \brief Wait for multiple jobs to complete.
             *
             * If a job has not started yet, then that job will be run
             * within the caller's thread (ie. this method).
             * If none of the specified jobs are pending, but some are
             * active in other threads, and other unrelated jobs are
             * pending, then the lowest order pending job will be run. 
             * If there are no jobs pending, then the thread will wait for
             * either the jobs to complete or a new job to be submitted.
             */
            template <typename... Jobs>
            inline void wait(Jobs&... jobs) {
                base* arr[] = { &jobs... };
                const auto begin = std::begin(arr);
                const auto end =
                    std::remove_if(begin, std::end(arr),
                                   [](auto p) { return p->is_done(); });
                if (begin == end) return;
                std::sort(begin, end,
                          [](auto* a, auto* b) {
                              return a->order() < b->order();
                          });
                const auto deref = [](auto*p) -> auto& { return *p; };
                if (const auto jc = context::this_context(this))
                    jc->wait_for_all(begin, end, deref);
                else {
                    std::lock_guard<std::mutex> dlock(data_mux);
                    poolptr->template wait_all<false>(
                        "wait", data_, job::order_max,
                        begin, end, deref, true);
                }
            }

            /** \brief Wait for all jobs in sequence to complete.
             *
             * If same_priority, then pending jobs are run in the order
             * they appear in the sequence. 
             * If not, pending jobs are run highest priority first.
             */
            template <typename ITER, typename FN>
            inline void wait_for_all(ITER first, ITER last, FN&& fn,
                                     bool same_priority = true) {
                for ( ; first != last; ++first) {
                    if (fn(*first).is_done()) continue;
                    if (const auto jc = context::this_context(this))
                        jc->wait_for_all(first, last, std::forward<FN>(fn),
                                         same_priority);
                    else {
                        std::lock_guard<std::mutex> dlock(data_mux);
                        poolptr->template wait_all<false>(
                            "wait", data_, job::order_max,
                            first, last, std::forward<FN>(fn),
                            same_priority);
                    }
                    break;
                }
            }
            template <typename ITER>
            inline void wait_for_all(ITER first, ITER last) {
                wait_for_all(first, last, stdx::identity());
            }

            /** \brief Wait for any one job to complete.
             *
             * If any job in the sequence is not pending and not active,
             * then it is returned as complete.
             * If any jobs are pending, the job with the lowest order will
             * be run.  
             * If all jobs are currently active, then some other unrelated job
             * may be run.
             * If no jobs are pending, then this method will wait for either
             * one of the active jobs to complete or a new job to be submitted.
             */
            template <typename ITER, typename FN>
            inline ITER wait_for_one(ITER first, ITER last, FN&& fn) {
                if (first == last || fn(*first).is_done()) return first;
                if (const auto jc = context::this_context(this))
                    return jc->wait_for_one(first, last, std::forward<FN>(fn));
                else {
                    std::lock_guard<std::mutex> dlock(data_mux);
                    return poolptr->template wait_one<false>(
                        "wait", data_, job::order_max,
                        first, last, std::forward<FN>(fn));
                }
            }
            template <typename ITER>
            inline ITER wait_for_one(ITER first, ITER last) {
                return wait_for_one(first,last,stdx::identity());
            }

            /** \brief Return completed job.
             *
             * If any job in the sequence is not pending and not active, then
             * it is returned as complete.
             * Otherwise, last is returned.
             */
            template <typename ITER, typename FN>
            static inline ITER try_for_one(ITER first, ITER last, FN&& fn) {
                for ( ; first != last; ++first)
                    if (fn(*first).is_done())
                        return first;
                return last;
            }
            template <typename ITER>
            static inline ITER try_for_one(ITER first, ITER last) {
                for ( ; first != last; ++first)
                    if ((*first).is_done())
                        return first;
                return last;
            }
        };

        template <typename DATA>
        context<DATA>* context<DATA>::this_context(const queue<DATA>* owner) {
            return this_context(owner ? owner->get_pool().get() : nullptr);
        }

        /** \brief Result of job.
         *
         * This class is abstract.
         * Use function<> for concrete jobs.
         *
         * This class may be used (by reference or pointer) to represent
         * a job with a specific type of result but without having to
         * be specific regarding the function that produced that result.
         */
        template <typename DATA, typename R>
        class result : public base<DATA> {
            static_assert(!std::is_void<R>::value,
                          "job cannot return void");
            static_assert(!std::is_reference<R>::value,
                          "job cannot return reference");

        protected:
            using base = job::base<DATA>;

            union {
                unsigned char uninitialized[sizeof(R)];
                R value;
                std::exception_ptr exception;
            };
            result() {}

            /// can be copied if not submitted (ie. uninitialized)
            result(const result& other) : base(other) {}

        public:
            using result_type = R;
            using reference = result_type&;
            using const_reference = const result_type&;
            using pointer = result_type*;
            using const_pointer = const result_type*;

            /** \brief Destructor.
             * 
             * Destructing a job that is pending will cancel the job.
             * If the job is currently executing in another thread, then
             * this method will block and wait for it to finish executing.
             */
            ~result() {
                switch (this->state.load(std::memory_order_acquire)) {
                case base::VALUE:
                    value.~R();
                    break;
                case base::EXCEPTION:
                    exception.~exception_ptr();
                    break;
                case base::PENDING:
                case base::ABANDONED:
                    break;
                // note: must not be ACTIVE
                default:
                    assert(!"job state corrupt");
                }
            }

            /** \brief Get job result.
             *
             * If the job has not yet executed, this method will throw an
             * std::runtime_error exception.
             * If the job has executed but threw an exception, then this
             * method will re-throw that exception.
             */
            reference get() {
                switch (this->state.load(std::memory_order_acquire)) {
                case base::VALUE:
                    return value;
                case base::EXCEPTION:
                    std::rethrow_exception(exception);
                case base::ABANDONED:
                    throw std::runtime_error("job abandoned");
                default:
                    throw std::runtime_error("job pending or active");
                }
            }
            const_reference get() const {
                switch (this->state.load(std::memory_order_acquire)) {
                case base::VALUE:
                    return value;
                case base::EXCEPTION:
                    std::rethrow_exception(exception);
                case base::ABANDONED:
                    throw std::runtime_error("job abandoned");
                default:
                    throw std::runtime_error("job pending or active");
                }
            }
            inline reference operator*() {
                return get();
            }
            inline const_reference operator*() const {
                return get();
            }
            inline pointer operator->() {
                return &get();
            }
            inline const_pointer operator->() const {
                return &get();
            }
        };

        /** \brief Check if any job failed with an exception and rethrow.
         *
         * This method will throw an exception if any of the jobs are in
         * a state other than completed with a return value.
         * This includes throwing std::runtime_error if a job is pending,
         * active or abandoned.
         */
        template <typename ITER>
        inline void rethrow_exceptions(ITER first, ITER last) {
            for ( ; first != last; ++first) **first;
        }


        namespace defaults {
            /** \brief Default interrupt method (does nothing).
             *
             * Only selected when arg lookup fails to find job friend method.
             */
            template <typename U> void interrupt(U&&) {}
        }

        /** \brief Object holding specific invokable job.
         *
         * An instance of this class holds both the job method and the
         * result of the job (or exception) once the job has completed.
         *
         * The job method may either take a single argument of type
         * context<DATA>& or DATA&, or no argument.
         * The former is called if both are present.
         *
         * The return type of the method cannot be void.
         *
         * If the job method throws an exception, the exception will be
         * captured and then re-thrown by get().
         *
         * If the job is interrupted by interrupt_job() and it has not
         * yet started to run, then it is possible to have a custom method
         * called to interrupt subjobs and do other cleanup.  
         * Enable this by declaring within FN a friend method:
         *   friend void interrupt(FN&);
         */
        template <typename DATA, typename FN,
                  typename R = function_result_t<DATA,FN> >
        class function_r : public result<DATA, R> {
        public:
            using result_type = R;
            FN fn;

            /** \brief Constructor.
             * 
             * \param[in] args arguments to job method
             */
            function_r() : fn{} {}
            template <typename A0, typename... Args, 
                      typename = std::enable_if_t<
                          (sizeof...(Args) > 0 ||
                           !std::is_convertible<A0, const function_r&>::value)> >
            function_r(A0&& arg0, Args&&... args)
                : fn{std::forward<A0>(arg0), std::forward<Args>(args)...} {
            }

            /// can be copied if not submitted (base will throw otherwise)
            function_r(const function_r& other)
                : result<DATA,R>(other), fn{other.fn} {}

            ~function_r() {
                if (this->m_owner && !this->is_done()) {
                    try {
                        this->interrupt_job();
                    }
                    catch (const std::exception&) {
                    }
                    this->m_owner->remove_job(*this);
                }
            }
            
        private:
            std::atomic_flag interrupted = ATOMIC_FLAG_INIT;

            void run_interrupt_method() override final {
                using defaults::interrupt;
                if (!interrupted.test_and_set())
                    interrupt(fn);
            }

            void run(context<DATA>& context) override final {
                try {
                    this->throw_if_interrupted();
                    new (&this->value) result_type(invoke(fn,context));
                    const auto prev = 
                        this->state.exchange(this->VALUE, 
                                             std::memory_order_release);
                    assert(prev == this->ACTIVE);
                }
                catch (const interrupt_signal&) {
                    const auto prev = 
                        this->state.exchange(this->ABANDONED, 
                                             std::memory_order_release);
                    assert(prev == this->ACTIVE);
                }
                catch (const std::exception&) {
                    new (&this->exception)
                        std::exception_ptr(std::current_exception());
                    const auto prev = 
                        this->state.exchange(this->EXCEPTION, 
                                             std::memory_order_release);
                    assert(prev == this->ACTIVE);
                }
            }
        };

        template <typename DATA, typename FN>
        class function final : public function_r<DATA, FN> {
            using base = function_r<DATA,FN>;
        public:
            using base::base;
        };


        /** \brief Support external job method run on main thread.
         *
         * This object provides access to the main thread data and job context.
         */
        template <typename DATA>
        class external_job {
            typename queue<DATA>::data_ptr data;
            struct nop { int operator()() const { return 0; } };
            function<DATA,nop> job;

            static auto* setup(function<DATA,nop>& job, pool<DATA>& q) {
                job.claim(q,0);
                job.state.store(job.ACTIVE, std::memory_order_release);
                return &job;
            }

            external_job(external_job&&) = delete;
            external_job(const external_job&) = delete;
            external_job& operator=(external_job&&) = delete;
            external_job& operator=(const external_job&) = delete;

        public:
            job::context<DATA> context;

            explicit external_job(queue<DATA>& q)
                : data(q.get_data()),
                  job(),
                  context(0, setup(job, *q.get_pool()), *data) {
            }
            ~external_job() {
                // note: the job destructor will assert if in ACTIVE state
                job.state.store(job.ABANDONED, std::memory_order_release);
            }
        };


        template <typename DATA>
        template <typename FN, typename... Args>
        auto queue<DATA>::run_emplace(Args&&... args) {
            function<DATA,FN> fn(std::forward<Args>(args)...);
            run_job(fn);
            return std::move(*fn);
        }
    }
}
