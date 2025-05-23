
#include "context.hpp"
#include "job_queue.hpp"
#include "thread_data.hpp"

#include <mutex>
#include <applog/core.hpp>


using namespace core;


static auto shared_global() {
    static std::weak_ptr<object_store<true> > weak;
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);
    if (auto p = weak.lock())
        return p;
    auto p = std::make_shared<object_store<true> >();
    weak = p;
    return p;
}

namespace {
    struct impl final : public context {
        std::shared_ptr<object_store<true> > gstore;
        object_store<true> cstore;
        context_data cdata;
        job_queue queue;

        impl(const context_settings& settings)
            : gstore(shared_global()),
              cdata{*gstore, cstore},
              queue(*gstore, cstore) {
            auto n = settings.min_threads < settings.max_threads ?
                std::thread::hardware_concurrency() : settings.max_threads;
            if (n < settings.min_threads)
                n = settings.min_threads;
            if (n > settings.max_threads)
                n = settings.max_threads;
            while (1 + queue.num_threads() < n)
                queue.start_thread(
                    std::ref(*gstore), std::ref(cstore), true);
            emplace<const context_settings>(cstore, settings);
        }

        context_data& data() override {
            return cdata;
        }
        const context_data& data() const override {
            return cdata;
        }
        std::size_t num_threads() const override {
            return 1 + queue.num_threads();
        }
        job_queue& threads() override {
            return queue;
        }
    };
}

context_ptr context::construct(const context_settings& settings) {
    return std::make_unique<impl>(settings);
}

const context_settings& context::settings() const {
    return get<const context_settings>(data().context);
}

static inline auto& verify_context(core::context* ptr) {
    if (!ptr) throw std::invalid_argument("context is nullptr");
    return *ptr;
}

active_job::~active_job() = default;

active_job::active_job(job_context& jc) : q(nullptr), jc(&jc) {}

active_job::active_job(job_queue& queue) : q(&queue), jc(nullptr) {}

active_job::active_job(core::context* ptr)
    : active_job(verify_context(ptr).threads()) {
}

active_job::active_job(active_job&& other) = default;

job_context& active_job::context() {
    if (!jc) {
        assert(q);
        jc = job_context::this_context(q);
        if (!jc) {
            main = std::make_unique<job::external_job<thread_data> >(*q);
            jc = &main->context;
        }
    }
    return *jc;
}
