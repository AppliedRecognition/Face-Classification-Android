#pragma once

#include "prototype.hpp"
#include "multiface.hpp"

#include <core/context.hpp>
#include <stdext/identity.hpp>
#include <stdext/iterator_range.hpp>
#include <optional>

namespace rec {

    using stdx::identity;


    /** \brief Compare all prototypes to each other.
     *
     * This method does not compare prototypes to themselves.
     * It also avoids comparing prototype A to B and B to A as the result
     * would be the same.
     * Thus the total number of comparisons is N*(N-1)/2 for N prototypes.
     *
     * The result vector is "compressed".
     * Use the method index_compress(i,j) to find the vector index
     * associated with prototype indicies i and j.
     * Use the method index_decompress(k) to find the prototype indicies
     * associated with vector index k.
     *
     * The prototypes may be stored in any kind of sequence container (vector,
     * list, set, map, etc.) and passed as begin(), end() for first, last.
     * The adaptor function will be used to convert the dereferenced value
     * to a const prototype& (possibly via prototype_ptr).
     * If the dereference already provides something which can be implicitly
     * converted, then use identity() as the adaptor (default function).
     */
    template <typename ITER, typename ADAPTOR = identity>
    std::vector<compare_result>
    compare(std::optional<core::active_job> context,
            ITER first, ITER last, ADAPTOR&& adaptor = {},
            variant var = variant::none);

    /** \brief Compute vector index from prototype index pair.
     *
     * This method is symmetric: the result for (i,j) is the same as for (j,i).
     * The result is invalid if i == j.
     */
    inline std::size_t index_compress(std::size_t i, std::size_t j) {
        return i < j ? (j*(j-1))/2 + i : (i*(i-1))/2 + j;
    }

    /** \brief Compute prototype index pair from vector index.
     *
     * Always returns first < second.
     */
    std::pair<unsigned,unsigned> index_decompress(std::size_t k);

    /** \brief Increment prototype index pair in compare all result.
     *
     * To iterate through the results from multi-prototype comparison,
     * start at index {0,1} and use this method to increment the index pair.
     *
     * This method assumes idx.first < idx.second pre and post condition.
     * As a special case, note that {0,0} will increment to {0,1}.
     */
    inline void index_increment(std::pair<unsigned,unsigned>& idx) {
        if (++idx.first >= idx.second)
            idx.first = 0, ++idx.second;
    }


    /** \brief Compare multiple prototypes to multiple multifaces.
     *
     * The prototypes may be stored in any kind of sequence container (vector,
     * list, set, map, etc.) and passed as begin(), end() for first, last.
     * The adaptor function will be used to convert the dereferenced value
     * to a const prototype& (possibly via prototype_ptr).
     * If the dereference already provides something which can be implicitly
     * converted, then use identity() as the adaptor.
     *
     * Similarly, the multifaces may be in any kind of sequence container.
     * Their adaptor function must produce a const multiface&.
     * Note that a subject object will implicitly convert to const multiface&.
     *
     * The results will include all matches with score greater than or equal
     * to the specified score threshold.
     */
    template <typename P_ITER, typename P_ADAPTOR,
              typename MF_ITER, typename MF_ADAPTOR>
    std::vector<std::tuple<float, P_ITER, MF_ITER> >
    compare(std::optional<core::active_job> context,
            P_ITER p_first, P_ITER p_last, P_ADAPTOR&& p_adaptor,
            MF_ITER mf_first, MF_ITER mf_last, MF_ADAPTOR&& mf_adaptor,
            float score_threshold = 0, variant var = variant::none);


    /** \brief Cluster faces with "loose" membership.
     *
     * Any pair of faces that compare with score >= threshold will be placed
     * in the same cluster.
     * Note that in clusters of 3 or more faces, any 2 faces may be
     * arbitrarily far apart.
     */
    template <typename ITER, typename ADAPTOR = identity>
    std::vector<std::vector<ITER> >
    cluster_loose(std::optional<core::active_job> context, float threshold,
                  ITER first, ITER last, ADAPTOR&& adaptor = {},
                  variant var = variant::none);


    /** \brief Cluster faces with "tight" membership.
     *
     * All pairs of faces within a cluster compare with score >= threshold.
     * Clusters are formed using a greedy method starting with the
     * best matching pair of faces first.
     * Note that some pairs of faces that match within the threshold may not
     * be placed in the same cluster.
     */
    template <typename ITER, typename ADAPTOR = identity>
    std::vector<std::vector<ITER> >
    cluster_tight(std::optional<core::active_job> context, float threshold,
                  ITER first, ITER last, ADAPTOR&& adaptor = {},
                  variant var = variant::none);


    /** \brief Cluster faces into a specific number of clusters.
     *
     * Faces are added to clusters using a greedy method (best matching
     * faces first) until a specific number of clusters have been formed.
     */
    template <typename ITER, typename ADAPTOR = identity>
    std::vector<std::vector<ITER> >
    cluster_count(std::optional<core::active_job> context, unsigned count,
                  ITER first, ITER last, ADAPTOR&& adaptor = {},
                  variant var = variant::none);


    /** \brief Sort prototypes such that ones most like the group are first.
     *
     * Faces are clustered from best match (highest score) to lowest.
     * Each cluster is kept in order and when joining 2 clusters, the
     * larger one is placed first.
     */
    template <typename ITER, typename ADAPTOR = identity>
    std::vector<ITER>
    order(std::optional<core::active_job> context,
          ITER first, ITER last, ADAPTOR&& adaptor = {},
          variant var = variant::none);
}


/**** implementation ****/

namespace rec {
    namespace internal {
        std::vector<compare_result>
        compare(core::job_context*, prototype const* const*, std::size_t,
                variant);
    }
    template <typename ITER, typename ADAPTOR>
    std::vector<compare_result>
    compare(std::optional<core::active_job> context,
            ITER first, ITER last, ADAPTOR&& adaptor,
            variant var) {
        std::vector<const prototype*> protos;
        for ( ; first != last; ++first)
            protos.emplace_back(
                stdx::pointer_to<const prototype>(adaptor(*first)));
        return internal::compare(
            context ? &context->context() : nullptr,
            protos.data(), protos.size(), var);
    }

    namespace internal {
        std::vector<std::tuple<compare_result,unsigned,unsigned> >
        compare(core::job_context*, prototype const* const*, std::size_t,
                std::function<rec::multiface const&(std::size_t)>, std::size_t,
                float, variant);
    }
    template <typename P_ITER, typename P_ADAPTOR,
              typename MF_ITER, typename MF_ADAPTOR>
    std::vector<std::tuple<float, P_ITER, MF_ITER> >
    compare(std::optional<core::active_job> context,
            P_ITER p_first, P_ITER p_last, P_ADAPTOR&& p_adaptor,
            MF_ITER mf_first, MF_ITER mf_last, MF_ADAPTOR&& mf_adaptor,
            float score_threshold, variant var) {

        const stdx::iterator_range<P_ITER> p_iter(p_first, p_last);
        const auto nprotos = p_iter.size();
        if (nprotos <= 0) return {};
        std::vector<const prototype*> protos;
        protos.reserve(std::size_t(nprotos));
        for (auto&& obj : p_iter)
            protos.emplace_back(
                stdx::pointer_to<const prototype>(p_adaptor(obj)));

        const stdx::iterator_range<MF_ITER> mf_iter(mf_first, mf_last);
        const auto nmfs = mf_iter.size();
        if (nmfs <= 0) return {};

        const auto ir = internal::compare(
            context ? &context->context() : nullptr,
            protos.data(), protos.size(),
            [&](auto i) -> rec::multiface const& {
                return mf_adaptor(*mf_iter[i]);
            }, std::size_t(nmfs), score_threshold, var);

        std::vector<std::tuple<float, P_ITER, MF_ITER> > r;
        r.reserve(ir.size());
        for (auto& t : ir)
            r.emplace_back(std::get<0>(t),
                           p_iter[std::get<1>(t)],
                           mf_iter[std::get<2>(t)]);
        return r;
    }

    namespace internal {
        std::vector<std::vector<unsigned> >
        cluster_loose(core::job_context*, float,
                prototype const* const*, std::size_t, variant);
        std::vector<std::vector<unsigned> >
        cluster_tight(core::job_context*, float,
                prototype const* const*, std::size_t, variant);
        std::vector<std::vector<unsigned> >
        cluster_count(core::job_context*, unsigned,
                prototype const* const*, std::size_t, variant);
    }

    template <typename ITER, typename ADAPTOR>
    std::vector<std::vector<ITER> >
    cluster_loose(std::optional<core::active_job> context, float threshold,
                  ITER first, ITER last, ADAPTOR&& adaptor,
                  variant var) {
        const stdx::iterator_range<ITER> iter(first, last);
        const auto nprotos = iter.size();
        if (nprotos <= 0) return {};
        std::vector<const prototype*> protos;
        protos.reserve(std::size_t(nprotos));
        for (auto&& obj : iter)
            protos.emplace_back(
                stdx::pointer_to<const prototype>(adaptor(obj)));
        const auto ir =
            internal::cluster_loose(
                context ? &context->context() : nullptr,
                threshold, protos.data(), protos.size(), var);
        std::vector<std::vector<ITER> > r(ir.size());
        auto it = r.begin();
        for (auto& vec : ir) {
            it->reserve(vec.size());
            for (auto i : vec)
                it->emplace_back(iter[i]);
            ++it;
        }
        return r;
    }

    template <typename ITER, typename ADAPTOR>
    std::vector<std::vector<ITER> >
    cluster_tight(std::optional<core::active_job> context, float threshold,
                  ITER first, ITER last, ADAPTOR&& adaptor,
                  variant var) {
        const stdx::iterator_range<ITER> iter(first, last);
        const auto nprotos = iter.size();
        if (nprotos <= 0) return {};
        std::vector<const prototype*> protos;
        protos.reserve(std::size_t(nprotos));
        for (auto&& obj : iter)
            protos.emplace_back(
                stdx::pointer_to<const prototype>(adaptor(obj)));
        const auto ir =
            internal::cluster_tight(
                context ? &context->context() : nullptr,
                threshold, protos.data(), protos.size(), var);
        std::vector<std::vector<ITER> > r(ir.size());
        auto it = r.begin();
        for (auto& vec : ir) {
            it->reserve(vec.size());
            for (auto i : vec)
                it->emplace_back(iter[i]);
            ++it;
        }
        return r;
    }

    template <typename ITER, typename ADAPTOR>
    std::vector<std::vector<ITER> >
    cluster_count(std::optional<core::active_job> context, unsigned count,
                  ITER first, ITER last, ADAPTOR&& adaptor,
                  variant var) {
        const stdx::iterator_range<ITER> iter(first, last);
        const auto nprotos = iter.size();
        if (nprotos <= 0) return {};
        std::vector<const prototype*> protos;
        protos.reserve(std::size_t(nprotos));
        for (auto&& obj : iter)
            protos.emplace_back(
                stdx::pointer_to<const prototype>(adaptor(obj)));
        const auto ir =
            internal::cluster_count(
                context ? &context->context() : nullptr,
                count, protos.data(), protos.size(), var);
        std::vector<std::vector<ITER> > r(ir.size());
        auto it = r.begin();
        for (auto& vec : ir) {
            it->reserve(vec.size());
            for (auto i : vec)
                it->emplace_back(iter[i]);
            ++it;
        }
        return r;
    }

    namespace internal {
        struct tree_node {
            /// negative value is leaf_value = -value-1
            /// non-negative value is index of sub-tree (branch)
            int left, right;
            unsigned parent;
            unsigned size;
            float score;

            /// depth first search, left before right
            /// func(unsigned) is called with leaf values (face index)
            template <typename FUNC>
            void dfs(const std::vector<tree_node>& tree, FUNC&& func) const {
                if (left < 0) func(unsigned(-left-1));
                else tree[left].dfs(tree, func);
                if (right < 0) func(unsigned(-right-1));
                else tree[right].dfs(tree, std::forward<FUNC>(func));
            }
        };
        std::vector<tree_node>
        make_tree(core::job_context*, prototype const* const*, std::size_t,
                  rec::variant);
    }

    template <typename ITER, typename ADAPTOR>
    std::vector<ITER>
    order(std::optional<core::active_job> context,
          ITER first, ITER last, ADAPTOR&& adaptor, variant var) {
        const stdx::iterator_range<ITER> iter(first, last);
        const auto nprotos = iter.size();
        std::vector<ITER> r;
        r.reserve(nprotos);
        if (nprotos <= 2) {
            for ( ; first != last; ++first)
                r.emplace_back(first);
            return r;
        }
        std::vector<const prototype*> protos;
        protos.reserve(std::size_t(nprotos));
        for (auto&& obj : iter)
            protos.emplace_back(
                stdx::pointer_to<const prototype>(adaptor(obj)));
        const auto tr =
            internal::make_tree(
                context ? &context->context() : nullptr,
                protos.data(), protos.size(), var);
        tr.back().dfs(tr, [&](auto i) { r.emplace_back(iter[i]); });
        return r;
    }
}
