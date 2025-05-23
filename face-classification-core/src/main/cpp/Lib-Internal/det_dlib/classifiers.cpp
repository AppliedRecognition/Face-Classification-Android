
#include "classifiers.hpp"

#define __FBLIB_DET_PRIVATE_INTERNAL_USE_ONLY__
#include "internal.hpp"

#include <dlibx/net_vector.hpp>

#include <applog/core.hpp>

#include <mutex>


using namespace det;

namespace det {
    namespace internal {
        struct classifier_master {
            struct model_detail {
                dlibx::net::vector model;
                classifier_model_type detail;
                model_detail() : detail{{},{},model} {}
            };
            std::map<std::string, model_detail> map;
            std::mutex mux;

            static void set_path(model_detail& rec, std::string& path) {
                if (!path.empty()) {
                    FILE_LOG(logINFO) << "loading classifier: " << path;
                    rec.detail.path = move(path);
                }
                else
                    FILE_LOG(logINFO) << "loading classifier: "
                                      << rec.detail.name;
            }

            static void
            load(model_detail& rec, std::istream& in, std::string path) {
                set_path(rec, path);
                deserialize(rec.model, in);
            }

            static void
            load(model_detail& rec, stdx::binary bin, std::string path) {
                set_path(rec, path);
                stdx::binarystream in(move(bin));
                deserialize(rec.model, in);
            }

            static void load(model_detail& rec, const core::context_data& data);

            template <typename... Args>
            const classifier_model_type*
            load(std::string_view name, Args&&... args);
        };

        struct classifier_thread {
            std::map<const classifier_model_type*, dlibx::net::vector> map;

            auto& operator[](const classifier_model_type* attr) {
                if (!attr)
                    throw std::invalid_argument("classifier is nullptr");
                auto& nv = map[attr];
                if (nv.empty())
                    nv = attr->model;
                return nv;
            }
        };
    }
}

void internal::classifier_master::
load(model_detail& rec, const core::context_data& data) {
    const auto& loader = get_loader(data);
    auto&& r = loader(models::format::dlib, models::type::classifier,
                      rec.detail.name);
    auto& vec = r.models;
    if (!vec.empty()) {
        set_path(rec, r.path);
        auto& var = vec.front();
        if (auto p = std::get_if<models::istream_ptr>(&var)) {
            if (auto s = p->get())
                deserialize(rec.model, *s);
        }
        else if (auto p = std::get_if<stdx::binary>(&var))
            if (!p->empty()) {
                stdx::binarystream in(*p);
                deserialize(rec.model, in);
            }
    }
}

template <typename... Args>
const classifier_model_type*
internal::classifier_master::load(std::string_view name, Args&&... args) {
    if (name.empty())
        throw std::invalid_argument("empty classifier name");
    std::lock_guard<std::mutex> lock(mux);
    const auto p = map.try_emplace(std::string(name));
    auto& rec = p.first->second;
    if (rec.model.empty()) {
        rec.detail.name = p.first->first;
        load(rec,std::forward<Args>(args)...);
        if (rec.model.empty()) {
            FILE_LOG(logERROR) << "model data not found for classifier '"
                               << name << "'";
            throw std::runtime_error("classifier model data not found");
        }
    }
    return &rec.detail;
}

classifier_model_type const*
det::load_classifier(stdx::arg<core::context> context,
                     std::string_view classifier_name) {
    if (!context) {
        FILE_LOG(logERROR) << "load_classifier: invalid context";
        throw std::invalid_argument("invalid context argument");
    }
    auto&& data = context->data();
    auto& master = core::emplace<internal::classifier_master>(data.context);
    return master.load(classifier_name, data);
}

classifier_model_type const*
det::load_classifier(stdx::arg<core::context> context,
                     std::string_view classifier_name,
                     stdx::arg<std::istream> from_stream,
                     std::string path) {
    if (!from_stream) {
        FILE_LOG(logERROR) << "load_classifier: invalid stream";
        throw std::invalid_argument("invalid stream argument");
    }
    if (!context) {
        FILE_LOG(logERROR) << "load_classifier: invalid context";
        throw std::invalid_argument("invalid context argument");
    }
    auto&& data = context->data();
    auto& master = core::emplace<internal::classifier_master>(data.context);
    return master.load(classifier_name, *from_stream, move(path));
}

classifier_model_type const*
det::load_classifier(stdx::arg<core::context> context,
                     std::string_view classifier_name,
                     stdx::binary from_binary,
                     std::string path) {
    if (!context) {
        FILE_LOG(logERROR) << "load_classifier: invalid context";
        throw std::invalid_argument("invalid context argument");
    }
    auto&& data = context->data();
    auto& master = core::emplace<internal::classifier_master>(data.context);
    return master.load(classifier_name, move(from_binary), move(path));
}


namespace {
    struct attr_job {
        stdx::span<const raw_image::plane> image;
        const std::vector<raw_image::point2f>& pts;
        const classifier_model_type* model;

        inline auto operator()(core::job_context& jc) {
            using obj = internal::classifier_thread;
            auto& net = core::emplace<obj>(jc.data.thread)[model];
            std::vector<float> vec;
            net(net.extract(image,pts),vec);
            return vec;
        }
    };
}

static auto to_pts(const detected_coordinates& dc) {
    std::vector<raw_image::point2f> pts;
    if (dc.landmarks.empty()) {
        pts.reserve(2);
        pts.push_back(raw_image::round_from(dc.eye_left));
        pts.push_back(raw_image::round_from(dc.eye_right));
    }
    else {
        pts.reserve(dc.landmarks.size());
        for (auto& p : dc.landmarks)
            pts.push_back(raw_image::round_from(p));
    }
    return pts;
}

template <typename T>
static inline auto& verify_deref(const std::unique_ptr<T>& ptr) {
    if (!ptr)
        throw std::logic_error(
            "detect_classifiers must be constructed with initial configuration");
    return *ptr;
}


/**************** apply_classifier() ****************/

std::vector<float>
det::apply_classifier(stdx::arg<core::context> context,
                      classifier_model_type const* attr,
                      const stdx::spanarg<const raw_image::plane>& image,
                      const detected_coordinates& face) {
    if (!context) {
        FILE_LOG(logERROR) << "extract_classifier: invalid context";
        throw std::invalid_argument("invalid context argument");
    }
    if (!attr) {
        FILE_LOG(logERROR) << "extract_classifier: null classifier pointer";
        throw std::invalid_argument("null classifier pointer");
    }
    return context->threads().run_emplace<attr_job>(image, to_pts(face), attr);
}


/**************** class apply_classifiers ****************/

struct apply_classifiers::internal_state {
    const internal_config& config;
    internal_state(const internal_config& config) : config(config) {}

    // for detection_classifiers
    std::vector<raw_image::point2f> pts;
    std::vector<core::job_function<attr_job> > jobs;
};

apply_classifiers::~apply_classifiers() = default;

apply_classifiers::apply_classifiers(apply_classifiers&&) = default;
apply_classifiers& apply_classifiers::operator=(apply_classifiers&&) = default;

apply_classifiers::apply_classifiers(
    const stdx::spanarg<const raw_image::plane>& image,
    std::vector<const classifier_model_type*> detection_classifiers,
    std::vector<std::pair<const classifier_model_type*,float> > landmark_classifiers)
    : config(new internal_config{
            {image.begin(),image.end()},
            move(detection_classifiers),
            move(landmark_classifiers)
        }) {
}

apply_classifiers::apply_classifiers(const apply_classifiers& other,
                                     const face_coordinates& fc,
                                     core::job_context& jc)
    : state(new internal_state(verify_deref(other.config))) {

    // start detection_classifier jobs
    if (!state->config.detection_classifiers.empty()) {
        if (fc.empty())
            throw std::invalid_argument("empty face_coordinates object");
        state->pts = to_pts(fc.back());
        state->jobs.reserve(state->config.detection_classifiers.size());
        for (auto p : state->config.detection_classifiers) {
            state->jobs.emplace_back(state->config.image, state->pts, p);
            jc.submit(state->jobs.back());
        }
    }
}

stdx::span<const raw_image::plane>
apply_classifiers::image() const {
    return state ? state->config.image : config->image;
}

face_coordinates_with_classifiers
apply_classifiers::operator()(face_coordinates& fc, core::job_context& jc) {
    if (fc.empty())
        throw std::invalid_argument("empty face_coordinates object");

    assert(state);
    auto& config = state->config;

    face_coordinates_with_classifiers result(move(fc));
    result.classifiers.reserve(config.detection_classifiers.size() +
                               config.landmark_classifiers.size());
    result.classifiers.resize(config.detection_classifiers.size());

    if (!config.landmark_classifiers.empty()) {
        const auto q = result.back().confidence;
        const auto pts = to_pts(result.back());
        if (config.landmark_classifiers.size() == 1) {
            // optimization: run here instead of queuing job
            auto& p = config.landmark_classifiers.front();
            if (p.second <= q)
                result.classifiers.emplace_back(
                    p.first, attr_job{config.image, pts, p.first}(jc));
        }
        else {
            // start landmark_classifier jobs
            std::vector<core::job_function<attr_job> > jobs;
            jobs.reserve(config.landmark_classifiers.size());
            for (auto p : config.landmark_classifiers) {
                if (p.second <= q) {
                    jobs.emplace_back(config.image, pts, p.first);
                    jc.submit(jobs.back());
                }
            }
            // wait for jobs and tally results
            jc.wait_for_all(jobs.begin(), jobs.end());
            for (auto& job : jobs)
                result.classifiers.emplace_back(job.fn.model, move(*job));
        }
    }

    // wait for detection_classifier jobs and tally results
    jc.wait_for_all(state->jobs.begin(), state->jobs.end());
    auto dest = result.classifiers.begin();
    for (auto& job : state->jobs)
        *dest++ = { job.fn.model, move(*job) };

    return result;
}
