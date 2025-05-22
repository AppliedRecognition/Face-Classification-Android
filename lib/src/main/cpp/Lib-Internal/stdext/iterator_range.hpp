#pragma once

#include <iterator>
#include <vector>

namespace stdx {

    /** \brief Random access to an iterator range [first, last).
     */
    template <typename ITER, typename = void>
    class iterator_range {
        using size_type = typename std::vector<ITER>::size_type;
        std::vector<ITER> m_iters;
        ITER m_last;

    public:
        static constexpr auto is_random_access_iterator = false;
        
        iterator_range(ITER first, ITER last)
            : m_last(last) {
            for ( ; first != last; ++first)
                m_iters.emplace_back(first);
        }

        inline bool empty() const {
            return m_iters.empty();
        }
        inline auto size() const {
            return m_iters.size();
        }
        inline auto& operator[](size_type i) const {
            return m_iters[i];
        }
        inline auto begin() const {
            return m_iters.empty() ? m_last : m_iters.front();
        }
        inline auto end() const {
            return m_last;
        }
    };

    template <typename ITER>
    class iterator_range<
        ITER,
        std::enable_if_t<
            std::is_same<typename std::iterator_traits<ITER>::iterator_category,
                         std::random_access_iterator_tag>::value> > {
        ITER m_first, m_last;

    public:
        static constexpr auto is_random_access_iterator = true;

        iterator_range(ITER first, ITER last)
            : m_first(first), m_last(last) {}

        inline bool empty() const {
            return m_first == m_last;
        }
        inline auto size() const {
            return std::distance(m_first, m_last);
        }
        template <typename I>
        inline auto operator[](I i) const {
            return m_first + i;
        }
        inline auto begin() const {
            return m_first;
        }
        inline auto end() const {
            return m_last;
        }
    };

}
