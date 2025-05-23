
#include "detection.hpp"
#include "image.hpp"
#define __FBLIB_DET_PRIVATE_INTERNAL_USE_ONLY__
#include "internal_landmarks.hpp"

#include <core/thread_set.hpp>

#include <set>
#include <condition_variable>

#include <applog/core.hpp>


using namespace det;
using namespace det::internal;


void det::set_models_loader(stdx::arg<core::context> context,
                            models::loader_function loader) {
    if (!context) {
        FILE_LOG(logERROR) << "set_models_loader: invalid context";
        throw std::invalid_argument("invalid context argument");
    }
    core::emplace<const models_loader>(
        context->data().context, models_loader{std::move(loader)});
}

namespace det {
    namespace internal {
        struct detector_factory_record {
            std::vector<detector_factory_function> factories;
            detector_ptr loaded;
            std::mutex mux;
        };
    }
}

using detector_map_type = std::map<unsigned, detector_factory_record>;

void internal::insert_factory(core::context_data& data,
                              unsigned detver,
                              detector_factory_function func) {
    auto& map = core::emplace<detector_map_type>(data.context);
    map[detver].factories.push_back(func);
}

detector_base&
internal::load_face_detector(core::context_data& data,
                             const detection_settings& settings) {
    auto& map = core::get<detector_map_type>(data.context);
    const auto ver = settings.detector_version;
    const auto it = map.find(ver);
    if (it == map.end())
        throw std::invalid_argument("unknown detector version");
    std::lock_guard lock(it->second.mux);
    if (!it->second.loaded) {
        for (auto& f : it->second.factories) {
            try {
                it->second.loaded = f(data, settings);
                if (it->second.loaded)
                    break; // success
            }
            catch (const std::exception& e) {
                if (&f == &it->second.factories.back()) {
                    FILE_LOG(logERROR) << "while loading detector "
                                       << ver << ' ' << e.what();
                    throw;
                }
                FILE_LOG(logINFO) << "while loading detector "
                                  << ver << ' ' << e.what()
                                  << " (trying next option)";
            }
        }
        if (!it->second.loaded) {
            FILE_LOG(logERROR) << "while loading detector "
                               << ver << " factories returned null pointer";
            throw std::runtime_error("failed to load face detector");
        }
    }
    return *it->second.loaded;
}

namespace {
    struct load_cascades {
        core::thread_set& ts;
        const detection_settings* settings;
        detector_base* obj;

        template <typename JC>
        int operator()(JC& jc) {
            const auto n = unsigned(ts.visit(&jc));
            obj->prepare_thread(jc, *settings, n);
            ts.wait();
            return 0;
        }
    };
}

void det::prepare_detection(stdx::arg<core::context> context,
                            const detection_settings& settings) {
    if (!context) {
        FILE_LOG(logERROR) << "prepare_detection: invalid context";
        throw std::invalid_argument("invalid context argument");
    }
    auto&& data = context->data();
    FILE_LOG(logINFO) << "det::prepare_detection: start";
    if (settings.detector_version) {
        auto& obj = load_face_detector(data, settings);

        auto& queue = context->threads();
        core::thread_set ts(queue.num_threads()+1);
        using job_type = core::job_function<load_cascades>;
        std::list<job_type> thread_list;
        // start a job on each thread to load cascades -- each job waits
        // until all have run to ensure they are on distinct threads
        // ** this will lock-up if we don't have at least one thread per job **
        for (unsigned i = 0; i < ts.num_threads; ++i) {
            thread_list.emplace_back(ts, &settings, &obj);
            queue.submit(thread_list.back());
        }
        while (!thread_list.empty()) {
            const auto it =
                queue.wait_for_one(thread_list.begin(), thread_list.end());
            it->get();  // to "get" exceptions
            thread_list.erase(it);
        }
    }
    load_landmark_detectors(data, settings.landmark_detection);
    FILE_LOG(logINFO) << "det::prepare_detection: done";
}

const detection_input&
internal::verify_no_rotation(const detection_input& input) {
    if (input.image.rotate&3) {
        FILE_LOG(logERROR) << "face detection on image with rotation "
                           << input.image.rotate << " (incorrect settings)";
        throw std::invalid_argument("face detector requires unrotated image");
    }
    return input;
}

namespace {
    struct detection_cleanup_job {
        std::unique_ptr<internal::detection_state> state;
        detection_cleanup_job(internal::detection_state* p) : state(p) {}

        template <typename U>
        int operator()(U&&) {
            state.reset();
            return 0;
        }
    };

    class detection_cleanup {
    public:
        using job_type = core::job_function<detection_cleanup_job>;

    private:
        std::atomic<job_type*> job;
        
        detection_cleanup(detection_cleanup&&) = delete;
        detection_cleanup(const detection_cleanup&) = delete;
        detection_cleanup& operator=(detection_cleanup&&) = delete;
        detection_cleanup& operator=(const detection_cleanup&) = delete;

    public:
        detection_cleanup() : job(nullptr) {}
        ~detection_cleanup() {
            delete job.load(std::memory_order_acquire);
        }
        void set(job_type* p) {
            delete job.exchange(p);
        }
    };

}

struct internal::detection_state {
    core::context_data& cdata;
    core::job::pool<core::thread_data>& pool;
    core::job_queue* queue;
    std::unique_ptr<output_base> output_constructor;
    detection_input input;
    internal::detection_result d;

    explicit detection_state(core::active_job& job,
                             std::unique_ptr<output_base> output_constructor,
                             bool low_latency = true)
        : cdata(job.context().data),
          pool(job.context().owner()),
          queue(job.queue_ptr()),
          output_constructor(move(output_constructor)) {
        input.output_constructor = this->output_constructor.get();
        input.low_latency = low_latency;
        if (!input.output_constructor)
            throw std::invalid_argument("output_constructor is nullptr");
    }
};

void detection_state_deleter::operator()(detection_state* p) {
    if (p->d.next) {
        p->d.next->interrupt_job();
        if (p->pool.num_threads() > 0) {
            auto c = new detection_cleanup::job_type(p);
            core::get<detection_cleanup>(p->cdata.context).set(c);
            c->can_inherit_jobs();
            c->claim(p->pool, -1);
            p->pool.queue_job(*c);
            return; // don't delete p
        }
    }
    delete p;
}

detection_state_ptr
internal::start_detect_faces(core::active_job& job,
                             const detection_settings& settings,
                             const image_struct* image,
                             std::unique_ptr<output_base> output_constructor,
                             bool low_latency, json::value* diag) {
    if (!image) {
        FILE_LOG(logERROR) << "detect_faces: invalid image";
        throw std::invalid_argument("invalid image argument");
    }

    auto& detobj = load_face_detector(job.context(), settings);

    detection_state_ptr result(
        new detection_state(job, move(output_constructor), low_latency));
    result->input.settings = settings;

    if (settings.detector_version <= 3)
        result->input.image = get_raw_from_image(image, gray);
    else
        result->input.image = get_raw_from_image(image, color);

    auto dj = detobj.detection_job(result->input, diag);
    result->d.next.reset(new core::job_function<decltype(dj)>(move(dj)));

    job.context().submit(
        *result->d.next,
        core::job::relative_order(
            low_latency ? core::job_queue::order_max/2+1 : 1));
    
    return result;
}

namespace {
    struct output_constructor {
        template <typename... Args>
        output_constructor(Args&&...) {}

        template <typename... Args>
        inline face_coordinates
        operator()(face_coordinates& fc, Args&&...) const { return move(fc); }
    };
}

det::detection_handle<face_coordinates>
det::start_detect_faces(core::active_job context,
                        const detection_settings& settings,
                        stdx::arg<const image_struct> image,
                        low_latency_option latency_option,
                        json::value* diag) {
    return start_detect_faces(
        std::move(context), settings, image,
        output_constructor{}, latency_option, diag);
}


detection_state_ptr
internal::start_detect_landmarks(
    core::active_job& job,
    const landmark_settings& landmarks,
    const image_struct* image,
    stdx::forward_iterator<const detected_coordinates&> first, 
    stdx::forward_iterator<const detected_coordinates&> last,
    std::unique_ptr<output_base> output_constructor) {
    
    if (!image) {
        FILE_LOG(logERROR) << "detect_landmarks: invalid image";
        throw std::invalid_argument("invalid image argument");
    }

    detection_state_ptr result(
        new detection_state(job, move(output_constructor)));
    result->input.settings.landmark_detection = landmarks;
    auto detectors = load_landmark_detectors(job.context(), landmarks);

    if (first != last) {
        auto p = new core::job_function<internal::landmark_jobs>();
        result->d.next.reset(p);
        unsigned idx = 0;
        result->input.image = get_raw_from_image(image, gray);
        auto& queue = job.context();
        for (core::job_queue::order_type o = 0; first != last; ++first, ++idx) {
            p->fn.job_list.emplace_back(
                face_coordinates(*first), result->input, detectors, idx);
            queue.submit(p->fn.job_list.back(),
                         core::job::relative_order(o += 8));
        }
        p->can_inherit_jobs();
        queue.submit(
            *p, core::job::relative_order(core::job_queue::order_max/2));
    }

    return result;
}

det::detection_handle<face_coordinates>
det::start_detect_landmarks(
    core::active_job context,
    const landmark_settings& landmarks,
    stdx::arg<const image_struct> image,
    stdx::forward_iterator<const detected_coordinates&> first, 
    stdx::forward_iterator<const detected_coordinates&> last) {

    return start_detect_landmarks(
        std::move(context), landmarks, image,
        move(first), move(last), output_constructor{});
}

face_list_type
det::detect_landmarks(
    core::active_job context,
    const landmark_settings& landmarks,
    stdx::arg<const image_struct> image,
    stdx::forward_iterator<const detected_coordinates&> first, 
    stdx::forward_iterator<const detected_coordinates&> last) {

    face_list_type result;
    const auto n = distance(first,last);
    if (n > 0) {
        result.reserve(std::size_t(n));
        auto h = start_detect_landmarks(
            std::move(context), landmarks, image, first, last);
        std::move(h.begin(), h.end(), back_inserter(result));
    }
    return result;
}

std::vector<any_ptr> internal::get_some(detection_state& handle) {
    if (!handle.d.faces.empty())
        FILE_LOG(logERROR) << "internal state corrupt in det::get_some";
    std::vector<any_ptr> result;
    if (const auto jc = core::job_context::this_context(&handle.pool)) {
        while (handle.d.next && result.empty()) {
            jc->wait(*handle.d.next);
            handle.d = std::move(**handle.d.next);
            result.swap(handle.d.faces);
        }
    }
    else if (auto q = handle.queue) {
        while (handle.d.next && result.empty()) {
            q->wait(*handle.d.next);
            handle.d = std::move(**handle.d.next);
            result.swap(handle.d.faces);
        }
    }
    else throw std::runtime_error("job context not available");
    return result;
}
