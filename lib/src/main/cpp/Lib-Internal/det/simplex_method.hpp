#pragma once

#include <cassert>
#include <vector>
#include <map>

namespace simplex {

    using coeff_type = float;
    using vertex_type = std::vector<coeff_type>;

    inline vertex_type& operator+=(vertex_type& a, const vertex_type& b) {
        assert(a.size() == b.size());
        auto it = b.begin();
        for (auto& x : a) {
            x += *it;
            ++it;
        }
        return a;
    }

    inline vertex_type& operator-=(vertex_type& a, const vertex_type& b) {
        assert(a.size() == b.size());
        auto it = b.begin();
        for (auto& x : a) {
            x -= *it;
            ++it;
        }
        return a;
    }

    inline vertex_type& operator*=(vertex_type& a, coeff_type b) {
        for (auto& x : a) x *= b;
        return a;
    }

    inline vertex_type& operator/=(vertex_type& a, coeff_type b) {
        for (auto& x : a) x /= b;
        return a;
    }

    template <typename VERT, typename T>
    VERT reflect(const VERT& mid, const VERT& v, T coeff) {
        auto r = mid;
        r -= v;
        r *= coeff;
        r += mid;
        return r;
    }

    
    class state {
    public:
        using error_type = float;
        using internal_type = std::multimap<error_type, vertex_type>;
        using value_type = internal_type::value_type;
        using size_type = internal_type::size_type;
        using iterator = internal_type::const_iterator;
        
    private:
        internal_type vert_map;

    public:
        inline bool empty() const { return vert_map.empty(); }
        inline size_type size() const { return vert_map.size(); }

        inline iterator begin() const { return vert_map.begin(); }
        inline iterator end() const { return vert_map.end(); }

        inline iterator best() const { return vert_map.begin(); }
        inline iterator worst() const { return prev(vert_map.end()); }

        inline size_type coeff_count() const {
            return vert_map.empty() ? 0 : vert_map.begin()->second.size();
        }

        std::pair<coeff_type,coeff_type> coeff_minmax(size_type i) const {
            auto it = vert_map.begin();
            std::pair<coeff_type,coeff_type> r;
            r.second = r.first = it->second[i];
            for (++it; it != vert_map.end(); ++it) {
                if (r.first > it->second[i])
                    r.first = it->second[i];
                else if (r.second < it->second[i])
                    r.second = it->second[i];
            }
            return r;
        }
        
        inline void insert(error_type err, vertex_type&& v) {
            vert_map.emplace(err, std::move(v));
        }

        inline void replace_worst(error_type err, vertex_type&& v) {
            const auto it = worst();
            vert_map.emplace(err, std::move(v));
            vert_map.erase(it);
        }

        inline void swap(state& other) {
            vert_map.swap(other.vert_map);
        }

        template <typename ERRFN>
        void init(vertex_type base, const vertex_type& delta,
                  const ERRFN& errfn, float frac = 0.125) {
            assert(base.size() == delta.size());
            for (size_type i = 0; i < base.size(); ++i) {
                auto vert = base;
                for (unsigned j = 0; j < vert.size(); ++j) {
                    if (j != i)
                        vert[j] += frac * delta[j];
                    else if (frac >= 0)
                        vert[j] -= delta[j];
                    else
                        vert[j] += delta[j];
                }
                insert(errfn(vert), move(vert));
            }
            insert(errfn(base), move(base));
        }

        template <typename ERRFN>
        state(vertex_type base, const vertex_type& delta,
              const ERRFN& errfn, float frac = 0.125) {
            init(base, delta, errfn, frac);
        }

        state() = default;
    };

    
    template <typename ERRFN>
    void step(state& s, ERRFN&& errfn,
              float alpha = 1.0f,
              float beta = 0.5f,
              float gamma = 2.0f) {

        assert(s.size() >= 2);

        // midpoint of all but worst vertex
        const auto& worst_v = s.worst()->second;
        auto it = s.begin();
        auto mid = it->second;
        for (const auto end = s.end(); ++it != end; )
            mid += it->second;
        mid -= worst_v;
        mid /= coeff_type(s.size() - 1);

        // reflect worst about midpoint
        auto v0 = reflect(mid, worst_v, alpha);
        auto e0 = errfn(v0);

        if (e0 < s.best()->first) {
            // better than best -- expand reflection
            auto v1 = reflect(mid, worst_v, gamma);
            auto e1 = errfn(v1);
            if (e1 <= s.best()->first)
                // also good -- accept expanded vertex
                s.replace_worst(e1, std::move(v1));
            else
                // accept reflected vertex
                s.replace_worst(e0, std::move(v0));
        }

        else if (e0 < s.worst()->first) {
            // better than worst -- accept reflected vertex
            s.replace_worst(e0, std::move(v0));
        }
        
        else {
            // worse than (or equal to) worst -- contract vertex
            auto v1 = reflect(mid, worst_v, beta);
            auto e1 = errfn(v1);
            if (e1 < s.worst()->first)
                // better -- accept contracted vertex
                s.replace_worst(e1, std::move(v1));
            else {
                // shrink all vertices towards best vertex
                state ns;
                const auto best = s.best();
                auto best_v = best->second;
                for (auto it = s.begin(), end = s.end(); it != end; ++it)
                    if (it != best) {
                        auto v = reflect(best_v, it->second, beta);
                        ns.insert(errfn(v), std::move(v));
                    }
                ns.insert(errfn(best_v), std::move(best_v));
                s.swap(ns);
            }
        }
    }

    template <typename ERRFN, typename PRED>
    unsigned step_until(state& s, ERRFN&& errfn, PRED&& pred,
                        unsigned max_steps = 1000,
                        float alpha = 1.0f,
                        float beta = 0.5f,
                        float gamma = 2.0f) {
        unsigned n = 0;
        do {
            simplex::step(s,errfn,alpha,beta,gamma);
        } while (++n < max_steps && !pred(const_cast<const state&>(s)));
        return n;
    }

    struct spread_all {
        float limit;
        spread_all(float limit) : limit(limit) {}

        bool operator()(const state& s) const {
            for (std::size_t i = 0, n = s.coeff_count(); i < n; ++i) {
                const auto r = s.coeff_minmax(i);
                const auto z = r.second - r.first;
                assert(z >= 0);
                if (z > limit) return false;
            }
            return true;
        }
    };
}
