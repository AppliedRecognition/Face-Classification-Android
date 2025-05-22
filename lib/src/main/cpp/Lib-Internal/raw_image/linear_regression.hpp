#pragma once

#include <algorithm>
#include <stdexcept>
#include <vector>

namespace raw_image {
    /** \brief Linear regression.
     *
     * Only float is supported for the type.
     */
    template <typename T>
    class linear_regression {
    public:
        using value_type = T;
        using size_type = typename std::vector<value_type>::size_type;

    private:
        std::vector<value_type> X, Z;
        size_type ncols = 0, rows_to_reserve = 0;

    public:
        linear_regression() = default;
        explicit linear_regression(size_type reserve)
            : rows_to_reserve(reserve) {}

        void reserve(size_type nrows) {
            rows_to_reserve = nrows;
            if (ncols > 0) {
                X.reserve(nrows*ncols);
                Z.reserve(nrows);
            }
        }

        inline bool empty() const {
            return Z.empty();
        }

        inline auto size() const {
            return Z.size();
        }

        template <typename Iter>
        void insert_back(value_type z, Iter first, Iter last) {
            if (auto n = std::distance(first, last)) {
                if (X.empty()) {
                    if (n < 0)
                        throw std::logic_error("invalid iterator range");
                    ncols = size_type(n);
                    if (rows_to_reserve > 0) {
                        X.reserve(rows_to_reserve * ncols);
                        Z.reserve(rows_to_reserve);
                    }
                }
                else if (n < 0 || ncols != size_type(n))
                    throw std::invalid_argument("inconsistent number of coefficients");
                X.insert(X.end(), first, last);
                Z.push_back(z);
            }
        }

        inline void add(value_type z, std::initializer_list<value_type> xs) {
            insert_back(z, xs.begin(), xs.end());
        }

        template <typename... Args>
        inline void add(value_type z, value_type x0, Args... xi) {
            add(z, { x0, xi... });
        }

        std::vector<value_type> compute() const;

        // sum of square of residuals
        value_type ssr(const std::vector<value_type>& coeff) const;
    };
}
