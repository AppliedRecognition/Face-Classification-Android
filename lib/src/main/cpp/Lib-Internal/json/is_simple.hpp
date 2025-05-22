#pragma once

#include "types.hpp"

namespace json {

    namespace detail {
        template <typename T, typename = void>
        struct is_simple {
            static constexpr inline bool test(const T&) {
                return true;
            }
        };

        template <typename ITER>
        inline bool is_simple_array(ITER first, ITER last) {
            using T = typename std::decay<decltype(*first)>::type;
            for (; first != last; ++first)
                if (!is_simple<T>::test(*first))
                    return false;
            return true;
        }

        template <typename T>
        struct is_simple<T, typename std::enable_if<is_array_type<T>::value>::type> {
            static inline bool test(const T& arr) {
                return is_simple_array(std::begin(arr), std::end(arr));
            }
        };

        template <typename ITER>
        inline bool is_simple_object(ITER first, ITER last) {
            using T = typename std::decay<decltype(first->second)>::type;
            return first == last ||
                (is_simple<T>::test(first->second) && ++first == last);
        }

        template <typename T>
        struct is_simple<T, typename std::enable_if<is_object_type<T>::value>::type> {
            static inline bool test(const T& obj) {
                return is_simple_object(std::begin(obj), std::end(obj));
            }
        };

        template <>
        struct is_simple<value, void> {
            static inline bool test(const value& v) {
                if (is_type<array>(v))
                    return is_simple<array>::test(get_array(v));
                if (is_type<object>(v))
                    return is_simple<object>::test(get_object(v));
                return true;
            }
        };
    }
    

    /** \brief Determine if value should receive simple formatting.
     *
     * All values are simple except: <ul>
     *   <li>a non-string sequence container (array or object) containing at
     *       least one non-simple value</li>
     *   <li>a pair associative container (object) containing two or
     *       more values</li>
     * </ul>
     */
    template <typename T>
    inline bool is_simple(const T& val) {
        return detail::is_simple<T>::test(val);
    }
}

