
#include "compare.hpp"
#include "internal_multiface.hpp"

#include <core/context.hpp>
#include <core/job_queue.hpp>
#include <core/thread_data.hpp>

#include <dlibx/atomic_counter.hpp>

#include <applog/core.hpp>

#include <cmath>
#include <list>

using namespace rec;

std::pair<unsigned,unsigned> rec::index_decompress(std::size_t k) {
    const auto j = std::size_t(std::floor((1+std::sqrt(8*k+1))/2));
    return { unsigned(k-j*(j-1)/2), unsigned(j) };
}

std::vector<compare_result>
internal::compare(core::job_context* context,
                  prototype const* const* faces,
                  std::size_t nfaces, rec::variant variant) {

    if (nfaces <= 1) return {};  // nothing to do
    if (faces == nullptr)
        throw std::invalid_argument("null pointer exception");
    if (nfaces > std::numeric_limits<unsigned>::max())
        throw std::invalid_argument("too many prototypes");

    // verify faces
    version_type version = 0;
    for (unsigned j = 0; j < nfaces; ++j) {
        if (const auto proto = faces[j]) {
            if (const auto ver = proto->version) {
                if (version == 0)
                    version = ver;
                else if (version != ver)
                    throw std::invalid_argument("prototype version mismatch");
            }
            else throw std::invalid_argument("prototype has invalid version");
        }
        else throw std::invalid_argument("prototype is nullptr");
    }
    struct compare_state {
        prototype const* const* faces;
        const std::size_t nfaces;
        const rec::variant variant;
        compare_result* dest;
        std::atomic<unsigned> next;
    };
    std::vector<compare_result> r(nfaces*(nfaces-1)/2);
    compare_state state{faces, nfaces, variant, r.data(), {0}};

    struct job {
        compare_state& state;
        auto operator()() {
            const auto m1 = unsigned(state.nfaces - 1);
            for (;;) {
                const auto k =
                    state.next.fetch_add(1, std::memory_order_relaxed);
                if (k >= m1) break;
                const auto j = m1 - k;
                const auto& fj = *state.faces[j];
                auto src = state.faces;
                auto dest = state.dest + j*(j-1)/2;
                for (unsigned i = 0; i < j; ++i, ++dest, ++src)
                    *dest = compare(fj, *src, state.variant);
            }
            return 0;
        }
    };
    std::vector<core::job_function<job> > jobs;
    if (context) {
        jobs.reserve(context->num_threads());
        for (auto n = context->num_threads(); n > 0; --n)
            jobs.emplace_back(state);
        for (auto& job : jobs)
            context->submit(job, core::job::relative_order(-1));
    }
    job{state}(); // this thread
    if (context)
        context->wait_for_all(jobs.begin(), jobs.end());
    for (auto& job : jobs) *job; // re-throw any exceptions
    return r;
}

namespace {
    struct compare_state {
        const version_type version;
        const std::function<rec::multiface const&(std::size_t)> mffn;
        const std::size_t nmfs;
        prototype const* const* faces;
        const std::size_t nfaces;
        const float score_threshold;
        const rec::variant variant;

        const internal::multiface& update_mf(unsigned i) const {
            auto& mf = mffn(i);
            if (mf.empty())
                throw std::invalid_argument("multiface has no faces");
            if (mf.version() != version)
                throw std::invalid_argument("multiface version mismatch");
            return mf;
        }

        template <typename VEC>
        void process_mf(VEC& dest, unsigned j, float* buf) const {
            const auto& mf = update_mf(j);
            mf.compare_to_n(faces, nfaces, variant, buf);
            for (unsigned i = 0; i < nfaces; ++i)
                if (buf[i] >= score_threshold)
                    dest.emplace_back(buf[i], i, j);
        }

        template <typename VEC>
        void process_face(VEC& dest, unsigned i) const {
            const auto fi = faces + i;
            for (unsigned j = 0; j < nmfs; ++j) {
                const auto score = compare(mffn(j), *fi, variant);
                if (score >= score_threshold)
                    dest.emplace_back(score, i, j);
            }
        }

        template <typename JOB, typename QUEUE>
        auto run(QUEUE& queue) const {
            JOB this_thread{*this,1+queue.num_threads()};
            auto prev = &this_thread;
            std::list<core::job_function<JOB> > jobs;
            for (auto n = queue.num_threads(); n > 0; --n) {
                jobs.emplace_back(*this,1+queue.num_threads(),n,prev);
                prev = &jobs.back().fn;
            }
            this_thread.link = prev;
            for (auto& job : jobs)
                queue.submit(job, core::job::relative_order(-1));
            auto r = this_thread();
            queue.wait_for_all(jobs.begin(), jobs.end());
            auto total_size = r.size();
            for (auto& job : jobs)
                total_size += job->size();
            r.reserve(total_size);
            for (auto& job : jobs) {
                auto& vec = *job;
                r.insert(r.end(), vec.begin(), vec.end());
            }
            assert(r.size() == total_size);
            return r;
        }

        template <typename JOB>
        inline auto run() const {
            JOB this_thread{*this,1};
            this_thread.link = &this_thread;
            return this_thread();
        }
    };
}

std::vector<std::tuple<compare_result,unsigned,unsigned> >
internal::compare(
    core::job_context* context,
    prototype const* const* faces, std::size_t nfaces,
    std::function<rec::multiface const&(std::size_t)> mffn, std::size_t nmfs,
    float score_threshold, rec::variant variant) {

    if (nfaces == 0 || nmfs == 0) return {};  // nothing to do
    if (faces == nullptr)
        throw std::invalid_argument("null pointer exception");
    if (nfaces > std::numeric_limits<unsigned>::max())
        throw std::invalid_argument("too many prototypes");
    if (nmfs > std::numeric_limits<unsigned>::max())
        throw std::invalid_argument("too many multifaces");

    // verify faces
    version_type version = 0;
    for (unsigned j = 0; j < nfaces; ++j) {
        if (const auto proto = faces[j]) {
            if (const auto ver = proto->version) {
                if (version == 0)
                    version = ver;
                else if (version != ver)
                    throw std::invalid_argument("prototype version mismatch");
            }
            else throw std::invalid_argument("prototype has invalid version");
        }
        else throw std::invalid_argument("prototype is nullptr");
    }

    const compare_state state { version,
            move(mffn), nmfs, faces, nfaces,
            score_threshold, variant };

    if (nfaces <= nmfs) {
        struct job : dlibx::atomic_counter<unsigned> {
            using I = atomic_counter::value_type;
            const compare_state& state;

            job(const compare_state& state,
                std::size_t nthreads, std::size_t i = 0,
                job* prev = nullptr)
                : atomic_counter(prev,
                                 I(i*state.nmfs/nthreads),
                                 I((i+1)*state.nmfs/nthreads)),
                  state(state) {
            }

            auto operator()() {
                std::unique_ptr<float[]> buf(new float[state.nfaces]);
                std::vector<std::tuple<compare_result,unsigned,unsigned> > r;
                for (atomic_counter* counter = this; ; ) {
                    const auto i = counter->next();
                    if (i < counter->limit)
                        state.process_mf(r,i,buf.get());
                    else if ((counter = counter->link) == this)
                        return r;
                }
            }
        };
        return context ? state.run<job>(*context) : state.run<job>();
    }

    else { // nfaces > nmfs
        // update mfs (could be parallelized)
        for (unsigned i = 0; i < nmfs; ++i)
            state.update_mf(i);

        struct job : dlibx::atomic_counter<unsigned> {
            using I = atomic_counter::value_type;
            const compare_state& state;

            job(const compare_state& state,
                std::size_t nthreads, std::size_t i = 0,
                job* prev = nullptr)
                : atomic_counter(prev,
                                 I(i*state.nfaces/nthreads),
                                 I((i+1)*state.nfaces/nthreads)),
                  state(state) {
            }

            auto operator()() {
                std::vector<std::tuple<compare_result,unsigned,unsigned> > r;
                for (atomic_counter* counter = this; ; ) {
                    const auto j = counter->next();
                    if (j < counter->limit)
                        state.process_face(r,j);
                    else if ((counter = counter->link) == this)
                        return r;
                }
            }
        };
        return context ? state.run<job>(*context) : state.run<job>();
    }
}

template <bool TIGHT>
std::vector<std::vector<unsigned> >
static cluster(core::job_context*, float threshold,
               prototype const* const* faces, std::size_t nfaces,
               rec::variant variant) {

    std::vector<std::vector<unsigned> > r;
    if (nfaces <= 1) {
        // nothing to do
        if (nfaces == 1)
            r.push_back({0});
        return r;
    }
    if (faces == nullptr)
        throw std::invalid_argument("null pointer exception");
    if (nfaces > std::numeric_limits<unsigned>::max())
        throw std::invalid_argument("too many prototypes");

    struct compat {
        std::vector<bool> vec;
        std::vector<bool>::iterator iter;
        compat(std::size_t N) : vec(N), iter(vec.begin()) {}
        inline bool push(bool x) { *iter = x; ++iter; return x; }
        inline auto idx(unsigned i, unsigned j) const {
            return i < j ? (j*(j-1))/2 + i : (i*(i-1))/2 + j;
        }
        inline bool operator()(unsigned i, unsigned j) const {
            return vec[idx(i,j)];
        }
        inline bool operator()(const std::list<unsigned>& ei,
                               const std::list<unsigned>& ej) {
            for (auto i : ei)
                for (auto j : ej)
                    if (!(*this)(i,j)) {
                        if (i != ei.front() || j != ej.front())
                            vec[idx(ei.front(),ej.front())] = false;
                        return false;
                    }
            return true;
        }
    };
    std::optional<compat> compatible;

    // compute pair-wise scores
    // todo: call parallelized compare() above instead
    const auto N = nfaces*(nfaces-1)/2;
    FILE_LOG(logDETAIL) << "rec::cluster: doing "
                        << N << " comparisons of "
                        << nfaces << " faces";
    std::vector<std::tuple<float,unsigned,unsigned> > scores;
    scores.reserve(N);
    if (TIGHT) compatible.emplace(N);
    for (unsigned i = 1; i < nfaces; ++i) {
        auto& pi = *faces[i];
        for (unsigned j = 0; j < i; ++j) {
            const auto s = compare(pi, faces[j], variant);
            if (TIGHT) {
                if (compatible->push(s >= threshold))
                    scores.emplace_back(s,i,j);
            }
            else if (s >= threshold)
                scores.emplace_back(s,i,j);
        }
    }

    if (TIGHT) {
        // sort scores (largest first)
        FILE_LOG(logDETAIL) << "rec::cluster: sort";
        std::sort(scores.begin(), scores.end(), std::greater<>{});
    }

    // clusters
    struct cluster_rec {
        cluster_rec* leader;
        std::list<unsigned> els;
        cluster_rec() : leader(this) {}
        cluster_rec(cluster_rec&&) = delete;
        cluster_rec(const cluster_rec&) = delete;
        cluster_rec* get_leader() {
            if (this != leader)
                leader = leader->get_leader();
            return leader;
        }
    };
    std::vector<cluster_rec> clusters(nfaces);
    for (unsigned i = 0; i < nfaces; ++i)
        clusters[i].els.push_back(i);

    // form clusters
    FILE_LOG(logDETAIL) << "rec::cluster: clustering " << scores.size()
                        << " matches with threshold " << threshold;
    for (auto& t : scores) {
        auto* ci = clusters[std::get<1>(t)].get_leader();
        auto* cj = clusters[std::get<2>(t)].get_leader();
        if (ci != cj && (!TIGHT || (*compatible)(ci->els, cj->els))) {
            // merge clusters
            cj->els.splice(cj->els.begin(), move(ci->els));
            ci->leader = cj;
        }
    }
    const auto num_clusters =
        std::count_if(clusters.begin(), clusters.end(),
                      [](const auto& c) { return c.leader == &c; });
    FILE_LOG(logDETAIL) << "rec::cluster: " << num_clusters << " clusters";
    assert(num_clusters > 0);

    // create final clusters
    r.resize(std::size_t(num_clusters));
    auto jt = r.begin();
    for (auto& c : clusters) {
        if (c.leader == &c) {
            assert(!c.els.empty());
            jt->reserve(c.els.size());
            jt->assign(c.els.begin(), c.els.end());
            ++jt;
        }
    }
    assert(jt == r.end());
    FILE_LOG(logDETAIL) << "rec::cluster: done";
    return r;
}

std::vector<std::vector<unsigned> >
internal::cluster_loose(
    core::job_context* context, float threshold,
    prototype const* const* faces, std::size_t nfaces, rec::variant variant) {
    return cluster<false>(context, threshold, faces, nfaces, variant);
}
std::vector<std::vector<unsigned> >
internal::cluster_tight(
    core::job_context* context, float threshold,
    prototype const* const* faces, std::size_t nfaces, rec::variant variant) {
    return cluster<true>(context, threshold, faces, nfaces, variant);
}

std::vector<std::vector<unsigned> >
internal::cluster_count(
    core::job_context*, unsigned count,
    prototype const* const* faces, std::size_t nfaces, rec::variant variant) {

    if (nfaces <= 1) return {};  // nothing to do
    if (faces == nullptr)
        throw std::invalid_argument("null pointer exception");
    if (nfaces > std::numeric_limits<unsigned>::max())
        throw std::invalid_argument("too many prototypes");
    if (count >= nfaces) {
        std::vector<std::vector<unsigned> > r(nfaces);
        for (unsigned i = 0; i < nfaces; ++i)
            r[i].push_back(i);
        return r;
    }
    if (count <= 0)
        count = 1;

    // compute pair-wise scores
    // todo: call parallelized compare() above instead
    const auto N = nfaces*(nfaces-1)/2;
    FILE_LOG(logDETAIL) << "rec::cluster: doing "
                        << N << " comparisons of "
                        << nfaces << " faces";
    std::vector<std::tuple<float,unsigned,unsigned> > scores;
    scores.reserve(N);
    for (unsigned i = 1; i < nfaces; ++i) {
        auto& pi = *faces[i];
        for (unsigned j = 0; j < i; ++j)
            scores.emplace_back(compare(pi, faces[j], variant),i,j);
    }

    // sort scores (largest first)
    FILE_LOG(logDETAIL) << "rec::cluster: sort";
    std::sort(scores.begin(), scores.end(), std::greater<>{});

    // clusters
    struct cluster_rec {
        cluster_rec* leader;
        std::list<unsigned> els;
        cluster_rec() : leader(this) {}
        auto get_leader() const {
            auto r = leader;
            while (r != r->leader)
                r = r->leader;
            return r;
        }
    };
    std::vector<cluster_rec> clusters(nfaces);
    for (unsigned i = 0; i < nfaces; ++i)
        clusters[i].els.push_back(i);

    // form clusters
    FILE_LOG(logDETAIL) << "rec::cluster: forming " << count << " clusters";
    auto remaining = unsigned(nfaces - count);
    for (auto& t : scores) {
        auto* ci = clusters[std::get<1>(t)].get_leader();
        auto* cj = clusters[std::get<2>(t)].get_leader();
        if (ci != cj) {
            // merge clusters (note: j < i)
            if (cj->els.size() >= ci->els.size()) {
                cj->els.splice(cj->els.end(), move(ci->els));
                ci->leader = cj;
            }
            else { // cluster ci is larger
                ci->els.splice(ci->els.end(), move(cj->els));
                cj->leader = ci;
            }
            if (--remaining == 0)
                break;
        }
    }

    // create final clusters
    std::vector<std::vector<unsigned> > r{count};
    auto jt = r.begin();
    for (auto& c : clusters) {
        if (c.leader == &c) {
            assert(!c.els.empty());
            jt->reserve(c.els.size());
            jt->assign(c.els.begin(), c.els.end());
            ++jt;
        }
    }
    assert(jt == r.end());

    // sort largest clusters first
    std::sort(r.begin(), r.end(),
              [](const auto& a, const auto& b) {
                  return a.size() > b.size();
              });
    FILE_LOG(logDETAIL) << "rec::cluster: done";
    return r;
}

std::vector<internal::tree_node>
internal::make_tree(core::job_context*,
                    prototype const* const* faces, std::size_t nfaces,
                    rec::variant variant) {

    if (nfaces <= 1) {
        if (nfaces == 1)
            throw std::logic_error("cannot construct tree with one element");
        return {};  // nothing to do
    }
    if (faces == nullptr)
        throw std::invalid_argument("null pointer exception");
    if (nfaces > std::size_t(std::numeric_limits<int>::max()))
        throw std::invalid_argument("too many prototypes");

    // compute pair-wise scores
    // todo: call parallelized compare() above instead
    const auto N = nfaces*(nfaces-1)/2;
    FILE_LOG(logDETAIL) << "rec::tree: doing "
                        << N << " comparisons of "
                        << nfaces << " faces";
    std::vector<std::tuple<float,unsigned,unsigned> > scores;
    scores.reserve(N);
    for (unsigned i = 1; i < nfaces; ++i) {
        auto& pi = *faces[i];
        for (unsigned j = 0; j < i; ++j)
            scores.emplace_back(compare(pi, faces[j], variant), i, j);
    }

    // sort scores (largest first)
    FILE_LOG(logDETAIL) << "rec::tree: sort";
    std::sort(scores.begin(), scores.end(), std::greater<>{});

    // the final tree and location of each face
    // location may point to anywhere within tree that has that face
    std::vector<tree_node> tr;
    tr.reserve(nfaces - 1);
    std::vector<unsigned> loc(nfaces,std::numeric_limits<unsigned>::max());

    // returns large value if face is not in tree yet
    const auto find_root = [&](auto i) {
        auto root = loc[i];
        if (root < tr.size())
            while (root != tr[root].parent)
                root = tr[root].parent;
        return root;
    };

    // form tree
    FILE_LOG(logDETAIL) << "rec::tree: forming";
    for (auto& t : scores) {
        auto i = std::get<1>(t);
        auto ri = find_root(i);
        auto j = std::get<2>(t);
        auto rj = find_root(j);
        if (ri == rj && ri < tr.size()) {
            loc[i] = loc[j] = ri;
            continue;  // faces are already in the same tree
        }

        tr.emplace_back();
        auto& node = tr.back();
        node.parent = unsigned(tr.size()-1);
        node.score = std::get<0>(t);

        auto si = ri < tr.size() ? tr[ri].size : 1;
        auto sj = rj < tr.size() ? tr[rj].size : 1;
        node.size = si + sj;
        if (si < sj) {
            std::swap(i,j);
            std::swap(ri,rj);
        }

        // ri is the larger tree so it goes on the left
        if (ri < tr.size()) {
            node.left = int(ri);
            tr[ri].parent = node.parent;
        }
        else
            node.left = -int(i)-1;

        if (rj < tr.size()) {
            node.right = int(rj);
            tr[rj].parent = node.parent;
        }
        else
            node.right = -int(j)-1;

        loc[i] = loc[j] = node.parent;
        if (tr.size() + 1 == nfaces)
            break;
    }
    assert(tr.back().size == nfaces);

    FILE_LOG(logDETAIL) << "rec::tree: done";
    return tr;
}
