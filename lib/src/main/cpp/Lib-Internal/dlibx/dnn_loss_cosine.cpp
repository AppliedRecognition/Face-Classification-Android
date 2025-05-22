
#include "dnn_loss_cosine.hpp"

#include <core/parallelize.hpp>

#include <applog/core.hpp>

#include <numeric>  // todo: use blas instead
#include <atomic>


using namespace dlibx;

static std::atomic<int> s_thread{0};
struct thread_count {
    const int id;
    thread_count() : id(s_thread.fetch_add(1)) {}
};
static const thread_local thread_count t_id;

// number of distinct pairs (n choose 2)
static constexpr auto pairs(unsigned n) {
    return n * (n-1) / 2;
}

static constexpr auto eps = 1e-20f;

template <typename T>
static constexpr auto sqr(T x) { return x*x; }

namespace {
    struct score_tuple {
        int match;  ///< -1 for non-match and +1 for match
        float score;
        unsigned i, j;
    };
    inline bool operator<(const score_tuple& a, const score_tuple& b) {
        return a.match <= b.match && (a.match < b.match || a.score < b.score);
    }
}

double loss_cosine_::compute_loss_value_and_gradient(
    const dlib::tensor& embedding,
    training_label_type const* labels,
    dlib::tensor& grad) const {

    DLIB_CASSERT(embedding.num_samples() == grad.num_samples());
    DLIB_CASSERT(embedding.k() == grad.k());
    DLIB_CASSERT(embedding.nr() == 1 && embedding.nc() == 1);
    DLIB_CASSERT(grad.nr() == 1 && grad.nc() == 1);
    if (embedding.size() <= 0)
        return 0;

    const auto inner_product =
        [k = embedding.k()] (const float* a, const float* b) {
            // todo: use blas
            return std::inner_product(a, a + k, b, 0.0f);
        };

    /// compute dest += ac*av + bc*bv
    const auto sum_gradient =
        [k = embedding.k()] (float* dest,
                             float ac, const float* av,
                             float bc, const float* bv) {
            // todo: can we used blas for this
            for (auto n = k; n > 0; --n, ++av, ++bv, ++dest)
                *dest += ac**av + bc**bv;
        };

    // context for parallelization
    auto context = core::job_context::this_context();
    if (context && context->num_threads() <= 0) context = nullptr;

    // compute inverse of norm of each sample
    const auto nsamples = unsigned(embedding.num_samples());
    const auto invnorm = std::make_unique<float[]>(nsamples);
    parallelize(
        [&, k = embedding.k()] (unsigned i) {
            const auto v = embedding.host() + k*i;
            const auto norm = std::sqrt(inner_product(v,v));
            invnorm[i] = norm < eps ? (1/eps) : (1/norm);
        }, nsamples, context);
    
    // compute cosine distance (score) between all pairs of samples
    // for each score we also keep track of sample indices and whether
    // they have same label (match) or not (mismatch)
    const auto nscores = pairs(nsamples);
    const auto scores = std::make_unique<score_tuple[]>(nscores);
    const auto scores_end = scores.get() + nscores;
    parallelize(
        [&, k = embedding.k()] (unsigned i) {
            // compare sample i with samples i+1, i+2, ..., nsamples-1
            const auto si = embedding.host() + k*i;
            const auto ci = invnorm[i];
            const auto li = labels[i];
            auto dest = scores.get() + (nscores - pairs(nsamples-i));
            auto sj = si;
            for (auto j = i + 1; j < nsamples; ++j, ++dest) {
                sj += k;
                dest->match = labels[j] == li ? 1 : -1;
                dest->score = ci * invnorm[j] * inner_product(si,sj);
                dest->i = i;
                dest->j = j;
            }
        }, nsamples-1, context);

    if (1) {
        // verify results (disable when not debugging)
        for (auto p = scores.get(); p != scores_end; ++p) {
            assert(p->match == 1 || p->match == -1);
            assert(p->i < p->j && p->j < nsamples);
            assert(-1.001f < p->score && p->score < 1.001f);
        }
    }

    // sort by category and score
    // non-matches are first followed by matches
    // scores ascending within each category
    std::sort(scores.get(), scores_end);

    // find boundry between non-match and match
    const auto match_start =
        std::find_if(scores.get(), scores_end,
                     [](const auto& t) { return t.match == 1; });
    if (match_start == scores.get() || match_start == scores_end)
        throw std::runtime_error("loss_cosine requires both matches and non-matches");
    assert(match_start[-1].match == -1 && match_start[0].match == 1);

    // we are interested in scores between:
    //   score_lo = worst (lowest)  match    - margin, and
    //   score_hi = worst (highest) nonmatch + margin
    // if score_lo is actually higher than score_hi, then loss will be zero,
    // but we still calculate a gradient using the worst pair of each category
    const auto score_lo = match_start[0].score - margin;
    const auto score_hi = match_start[-1].score + margin;

    if (0) {
        // test gradient calculation
        const auto k = unsigned(embedding.k());
        std::vector<float> tmp;
        for (auto p = scores.get(); p != scores_end; ++p) {
            const auto ei = embedding.host() + k*p->i;
            const auto ej = embedding.host() + k*p->j;
            tmp.assign(k,0.0f);
            sum_gradient(
                tmp.data(), invnorm[p->j],ej, -p->score*invnorm[p->i],ei);
            const auto n2 = inner_product(tmp.data(), tmp.data());
            const auto err = std::abs(1 - sqr(p->score) - n2);
            if (err > 1e-5) {
                FILE_LOG(logWARNING) << err << '\t' << n2 << '\t' << p->score;
                assert(err < 1e-5);
            }
        }
    }

    // for gradient calculation gi = ej / |ej|  -  score * ei / |ei|
    // this method computes the expected norm |gi| from the score
    // however, if the score is too close to +1 or -1, then a small non-zero
    // value is returned to avoid divide by zero issues
    const auto gradient_norm_from_score =
        [](float s) {
            const auto n = 1 - sqr(s);
            return n <= 1e-10 ? 1e-5f : std::sqrt(n);
        };

    // compute loss, determine an equal number of match and nonmatches,
    // and compute total weight of gradients in each class
    //
    // gradients are scaled such that their norm is proportional to score:
    // for matches the norm will be min(1, 1 - score), while
    // for non-matches norm will be max(0, score).
    // Non-matches with score <= 0 are not used.
    double loss = 0;
    float total_match = 0, total_nonmatch = 0;
    unsigned pair_count = 0;
    for (auto match = match_start, nonmatch = match_start; match != scores_end && nonmatch != scores.get(); ++match) {
        --nonmatch; ///< predecrement
        if (nonmatch->score < score_lo || score_hi < match->score) break;
        loss += score_hi - match->score;
        loss += nonmatch->score - score_lo;
        total_match += 1.0f - std::max(0.0f, match->score);
        total_nonmatch += std::max(0.0f, nonmatch->score);
        ++pair_count;
    }
    if (pair_count == 0) {
        pair_count = 1; // use at least 1 pair from each category
        total_match += 1.0f - std::max(0.0f, match_start->score);
        total_nonmatch += std::max(0.0f, match_start[-1].score);
    }
    total_match    = std::max(total_match, 1e-5f); // to avoid div by zero
    total_nonmatch = std::max(total_nonmatch, 1e-5f);
    loss /= (2*pair_count);

    if (0) {
        float norm_min = 1 / invnorm[0], norm_max = 0;
        for (unsigned i = 0; i < nsamples; ++i) {
            auto n = 1 / invnorm[i];
            norm_min = std::min(norm_min, n);
            norm_max = std::max(norm_max, n);
        }
        FILE_LOG(logINFO) << t_id.id //<< static_cast<void const*>(this)
                          << '\t' << "loss: " << loss
                          << "  range: " << score_lo << ' ' << score_hi
                          << "  count: " << pair_count
                          << "  norms: " << norm_min << ' ' << norm_max;
        std::stringstream ss;
        for (auto p = scores_end; p != scores.get(); ) {
            --p;
            ss << ' ' << p->score;
        }
        FILE_LOG(logINFO) << t_id.id << ss.str();
    }


    // Compute gradients:
    //   https://math.stackexchange.com/questions/1923613/partial-derivative-of-cosine-similarity
    //   gi[k] += weight * (cj*ej[k] - s*ci*ei[k])
    //   gj[k] += weight * (ci*ei[k] - s*cj*ej[k])
    // Note that the norm of each gradient (excluding the weight coefficient)
    // is sqrt(1-score^2).
    // The gradient from each pair is weighted such that the total
    // contribution from matches is equal in magnitude to the contribution
    // from non-matches.
    const auto k = embedding.k();
    for (auto p = match_start, end = p + pair_count; p != end; ++p) {
        const auto gi = grad.host() + k*p->i;
        const auto gj = grad.host() + k*p->j;
        const auto ei = embedding.host() + k*p->i;
        const auto ej = embedding.host() + k*p->j;
        const auto n = gradient_norm_from_score(p->score);
        const auto w = -0.5f * (1 - std::max(0.0f,p->score)) / n / total_match;
        const auto s = w * p->score;
        sum_gradient(gi, w*invnorm[p->j],ej, -s*invnorm[p->i],ei);
        sum_gradient(gj, w*invnorm[p->i],ei, -s*invnorm[p->j],ej);
    }
    for (auto p = match_start, end = p - pair_count; p != end; ) {
        --p; ///< pre-decrement
        const auto gi = grad.host() + k*p->i;
        const auto gj = grad.host() + k*p->j;
        const auto ei = embedding.host() + k*p->i;
        const auto ej = embedding.host() + k*p->j;
        const auto n = gradient_norm_from_score(p->score);
        const auto w = 0.5f * std::max(0.0f,p->score) / n / total_nonmatch;
        const auto s = w * p->score;
        sum_gradient(gi, w*invnorm[p->j],ej, -s*invnorm[p->i],ei);
        sum_gradient(gj, w*invnorm[p->i],ei, -s*invnorm[p->j],ej);
    }

    if (0) {
        // verify gradient vectors are perpendicular to embedding vectors
        auto e = embedding.host();
        auto g = grad.host();
        auto in = invnorm.get();
        float lim = 0;
        for (auto n = nsamples; n > 0; --n, e += k, g += k, ++in) {
            const auto x = std::abs(inner_product(e,g) * *in * *in);
            lim = std::max(lim, x);
        }
        if (lim > 1e-5) {
            FILE_LOG(logWARNING) << lim;
            assert(lim < 1e-5);
        }
    }

    // divide gradients by norm of input vector
    if (1) {
        auto g = grad.host();
        auto in = invnorm.get();
        for (auto n = nsamples; n > 0; --n, g += k, ++in) {
            if (*in < 0.99f || 1.01f < *in)
                std::transform(g, g+k, g,
                               [z = *in] (float a) {
                                   return a*z;
                               });
        }
    }
    
    // move all vectors in direction of unit norm
    if (0) {
        auto e = embedding.host();
        auto g = grad.host();
        auto in = invnorm.get();
        for (auto n = nsamples; n > 0; --n, e += k, g += k, ++in) {
            std::transform(g, g+k, e, g,
                           [z = 0.5f*(1-*in)] (float a, float b) {
                               return a + b*z;
                           });
        }
    }
    
    return loss;
}



