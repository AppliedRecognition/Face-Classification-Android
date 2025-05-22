
#include "dnn_loss_metric.hpp"

#include <applog/core.hpp>

using namespace dlibx;

namespace {
    struct sample_pair {
        int match;  ///< -1 for non-match and +1 for match
        float dist;
        unsigned r, c;
    };
}

double loss_metric_dynamic_::compute_loss_value_and_gradient(
    const dlib::tensor& embedding,
    training_label_type const* labels,
    dlib::tensor& grad) const {

    DLIB_CASSERT(embedding.num_samples() == grad.num_samples());
    DLIB_CASSERT(embedding.k() == grad.k());
    DLIB_CASSERT(embedding.nr() == 1 && embedding.nc() == 1);
    DLIB_CASSERT(grad.nr() == 1 && grad.nc() == 1);
    if (embedding.size() <= 0)
        return 0;

    // compute inner product of every sample vector vs. every other
    const auto num_samples = unsigned(embedding.num_samples());
    temp.set_size(num_samples, num_samples);
    dlib::tt::gemm(0, temp, 1, embedding, false, embedding, true);

    // compute distances (distance^2 at first) for all pairs of samples
    std::vector<sample_pair> samples;
    samples.reserve(std::size_t(num_samples)*std::size_t(num_samples-1)/2);

    const float* d = temp.host();
    for (unsigned r = 0; r < num_samples; ++r) {
        const auto xx = d[r*num_samples + r];
        const auto x_label = labels[r];
        for (unsigned c = r + 1; c < num_samples; ++c) {
            const auto y_label = labels[c];
            auto& p = samples.emplace_back();
            p.match = (x_label == y_label) ? 1 : -1;
            p.r = r;
            p.c = c;
            const auto yy = d[c*num_samples + c];
            const auto xy = d[r*num_samples + c];
            p.dist = std::max(0.0f, xx + yy - 2*xy); ///< square of distance
        }
    }

    // sort by category and distance
    // matches first followed by non-matches
    // distance ascending within each category
    std::sort(samples.begin(), samples.end(),
              [](const auto& a, const auto& b) {
                  return a.match >= b.match &&
                      (a.match > b.match || a.dist < b.dist);
              });
    
    // find boundry between non-match and match
    const auto middle =
        std::find_if(samples.begin(), samples.end(),
                     [](const auto& t) { return t.match == -1; });
    if (middle == samples.begin() || middle == samples.end())
        throw std::runtime_error(
            "loss_metric requires both matches and non-matches");
    assert(middle[-1].match == 1 && middle[0].match == -1);
    const auto num_match = middle - samples.begin();

    // initialize gradient and loss
    grad_mul.copy_size(temp);
    float* const gm = grad_mul.host();
    std::fill_n(gm, grad_mul.size(), 0.0f);
    double loss = 0;
    unsigned samples_used = 0;

    // walk match and mismatch lists taking an equal number from each
    // until the difference between the distances is at least 2*margin
    for (auto match = middle, mismatch = middle; match != samples.begin() && mismatch != samples.end(); ++mismatch, ++samples_used) {
        --match; // pre-decrement
        match->dist = std::sqrt(match->dist);
        mismatch->dist = std::sqrt(mismatch->dist);
        if (mismatch->dist - match->dist > 2*margin)
            break;
        {
            loss += match->dist;
            const auto r = match->r;
            const auto c = match->c;
            const auto z = 1.0f / std::max(match->dist, 0.001f);
            gm[r*num_samples + r] += z;
            gm[c*num_samples + c] += z;
            gm[r*num_samples + c] -= z;
            gm[c*num_samples + r] -= z;
        }
        {
            loss -= mismatch->dist;
            const auto r = mismatch->r;
            const auto c = mismatch->c;
            const auto z = 1.0f / std::max(mismatch->dist, 0.001f);
            gm[r*num_samples + r] -= z;
            gm[c*num_samples + c] -= z;
            gm[r*num_samples + c] += z;
            gm[c*num_samples + r] += z;
        }
    }
    assert(samples_used <= num_match);
    loss += 2 * margin * float(samples_used);
    loss /= float(num_match);

    // scale gradient coefficients
    std::transform(gm, gm + grad_mul.size(), gm,
                   [c = 0.5f/float(num_match)](float x) {
                       return x * c;
                   });

    dlib::tt::gemm(0, grad, 1, grad_mul, false, embedding, false); 

    return loss;
}
