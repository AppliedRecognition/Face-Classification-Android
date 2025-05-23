
#define __FBLIB_DET_PRIVATE_INTERNAL_USE_ONLY__
#include "internal_landmarks.hpp"

#include <applog/core.hpp>


using namespace det;
using namespace det::internal;

namespace det {
    namespace internal {
        struct landmarks_factory_record {
            std::vector<landmarks_factory_function> factories;
            landmarks_ptr loaded;
            std::mutex mux;
        };
    }
}

using landmarks_map_type =
    std::map<det::landmark_options, landmarks_factory_record>;

void internal::insert_factory(core::context_data& data,
                              det::landmark_options lmver,
                              landmarks_factory_function func) {
    auto& map = core::emplace<landmarks_map_type>(data.context);
    map[lmver].factories.push_back(func);
}

std::vector<landmarks_base const*>
internal::load_landmark_detectors(core::context_data& data,
                                  const landmark_settings& settings) {
    std::vector<landmarks_base const*> vec;
    auto& map = core::get<landmarks_map_type>(data.context);
    for (auto o = unsigned(settings.landmarks), i = 1u; o; i <<= 1) {
        if (o & i) {
            o ^= i; // reset bit to zero
            const auto lmver = landmark_options(i);
            const auto it = map.find(lmver);
            if (it == map.end())
                throw std::invalid_argument(
                    "unknown landmark detector version");
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
                            FILE_LOG(logERROR)
                                << "while loading landmark detector "
                                << unsigned(lmver) << ' ' << e.what();
                            throw;
                        }
                        FILE_LOG(logINFO) << "while loading landmark detector "
                                          << unsigned(lmver) << ' ' << e.what()
                                          << " (trying next option)";
                    }
                }
                if (!it->second.loaded) {
                    FILE_LOG(logERROR) << "while loading landmark detector "
                                       << unsigned(lmver)
                                       << " factories returned null pointer";
                    throw std::runtime_error(
                        "failed to load landmark detector");
                }
            }
            vec.push_back(it->second.loaded.get());
        }
    }
    return vec;
}

static int x(const detected_coordinates& dc) {
    return stdx::round_from((dc.eye_left.x + dc.eye_right.x) / 2);
}
static int y(const detected_coordinates& dc) {
    return stdx::round_from((dc.eye_left.y + dc.eye_right.y) / 2);
}
static int ed(const detected_coordinates& dc) {
    const auto dx = dc.eye_left.x - dc.eye_right.x;
    const auto dy = dc.eye_left.y - dc.eye_right.y;
    return stdx::round_from(std::sqrt(dx*dx + dy*dy));
}

std::pair<unsigned,any_ptr>
landmark_detection_job::operator()(core::job_context& jc) {

    face_coordinates result = std::move(initial_position);
    APPLOG_CHECK(!result.empty());
    FILE_LOG(logDETAIL) << "job: [" << jc.job.order() << "] landmarks "
                        << x(result.back()) << 'x' << y(result.back())
                        << " ed " << ed(result.back());

    auto output = input.output_constructor->copy(result, jc);

    const auto cc = input.settings.landmark_detection.contrast_correction;
    for (auto* ptr : detectors) {
        auto d = (*ptr)(result.back(), input.image, jc.data, cc);
        if (!d.landmarks.empty())
            result.emplace_back(std::move(d));
        else
            break; // no further landmark detection
    }

    return { idx, (*output)(result,jc) };
}

detection_result landmark_jobs::operator()(core::job_context& jc) {
    FILE_LOG(logDETAIL) << "job: [" << jc.job.order() << "] final";

    detection_result result;
    result.faces.reserve(job_list.size());

    while (!job_list.empty()) {
        const auto it = result.faces.empty() ?
            jc.wait_for_one(job_list.begin(), job_list.end()) :
            jc.try_for_one(job_list.begin(), job_list.end());

        if (it == job_list.end()) {
            assert(!result.faces.empty());
            assert(pending.empty() || pending.begin()->first > expected_idx);
            auto p = new core::job_function<landmark_jobs>();
            result.next.reset(p);
            p->fn.job_list = move(job_list);
            p->fn.pending = move(pending);
            p->fn.expected_idx = expected_idx;
            p->can_inherit_jobs();
            jc.submit(*p, core::job::return_to_parent);
            break;
        }

        auto& r = **it;
        if (r.first <= expected_idx) {
            expected_idx = std::max(expected_idx, r.first + 1);
            result.faces.emplace_back(std::move(r.second));
            while (!pending.empty() && pending.begin()->first <= expected_idx) {
                expected_idx =
                    std::max(expected_idx, pending.begin()->first + 1);
                result.faces.emplace_back(std::move(pending.begin()->second));
                pending.erase(pending.begin());
            }
        }
        else
            pending.insert(std::move(r));

        job_list.erase(it);
    }
    return result;
}

namespace det {
    namespace internal {
        void interrupt(landmark_jobs& j) {  // friend method
            for (auto& job : j.job_list)
                job.interrupt_job();
        }
    }
}

detection_result
internal::landmark_detection(core::job_context& jc,
                             const detection_input& input,
                             std::vector<face_coordinates> faces) {

    detection_result result;
    auto detectors =
        load_landmark_detectors(jc, input.settings.landmark_detection);

    if (faces.size() == 1) {
        // optimization: do work here
        result.faces.emplace_back(
            internal::landmark_detection_job{
                std::move(faces.front()),input,move(detectors),0u}(jc).second);
    }
    else if (faces.size() > 1) {
        // spawn subjob for each face
        auto p = new core::job_function<internal::landmark_jobs>;
        result.next.reset(p);
        auto o = core::job::order_type(0);
        for (auto& f : faces) {
            p->fn.job_list.emplace_back(std::move(f),input,detectors,0u);
            jc.submit(p->fn.job_list.back(),
                      core::job::relative_order(o+=8),
                      core::job::can_run_now(!input.low_latency));
        }
        p->can_inherit_jobs();
        jc.submit(*p, core::job::absolute_order(core::job::order_max),
                  core::job::can_run_now(!input.low_latency),
                  core::job::return_to_parent);
    }
    return result;
}
