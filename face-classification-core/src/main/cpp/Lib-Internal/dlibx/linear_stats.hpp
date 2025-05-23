#pragma once

#include <algorithm>
#include <stdexcept>

namespace core {

    template<typename T>
    class mean_variance {
    public:
        using value_type = T;

    private:
        unsigned n = 0;
        value_type sum{},sqr{};

        struct default_square {
            inline value_type operator()(value_type x) const {
                return x*x;
            }
        };

        
    public:
        inline bool empty() const {
            return n == 0;
        }

        inline auto size() const {
            return n;
        }

        value_type mean() const {
            return n > 0 ? sum/value_type(n) : value_type();
        }
    
        template<typename SQR_FN>
        value_type var(SQR_FN&& sqrfn) const {
            const auto nv = value_type(n);
            return n > 0 ? sqr/nv - sqrfn(sum/nv) : value_type();
        }

        inline value_type var() const {
            return var(default_square());
        }

        void insert(value_type x, value_type x2, unsigned weight) {
            if (n == 0) {
                sum = weight*x;
                sqr = weight*x2;
            }
            else {
                sum += weight*x;
                sqr += weight*x2;
            }
            n += weight;
        }

        void insert(value_type x, value_type x2) {
            if (n == 0) {
                sum = x;
                sqr = x2;
            }
            else {
                sum += x;
                sqr += x2;
            }
            ++n;
        }

        inline void insert(value_type x) {
            insert(x,x*x);
        }

        mean_variance& operator+=(value_type x) {
            insert(x);
            return *this;
        }        

        void erase(value_type x, value_type x2, unsigned weight) {
            if (n < weight)
                throw std::invalid_argument("attempt to erase value not present");
            sum -= weight*x;
            sqr -= weight*x2;
            n -= weight;
        }

        void erase(value_type x, value_type x2) {
            if (n < 1)
                throw std::invalid_argument("attempt to erase value not present");
            sum -= x;
            sqr -= x2;
            --n;
        }

        inline void erase(value_type x) {
            erase(x,x*x);
        }

        mean_variance& operator-=(value_type x) {
            erase(x);
            return *this;
        }        
    };

    template <typename T>
    mean_variance<T> operator+(const mean_variance<T>& a, T b) {
        mean_variance<T> result(a);
        result += b;
        return result;
    }

    template <typename T>
    mean_variance<T> operator-(const mean_variance<T>& a, T b) {
        mean_variance<T> result(a);
        result -= b;
        return result;
    }
}
