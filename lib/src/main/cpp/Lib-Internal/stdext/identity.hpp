#pragma once

#include <utility>

namespace stdx {

    /** \brief Identity function as in C++20.
     */
    struct identity {
        template <typename U>
        constexpr auto operator()(U&& t) const noexcept
            -> decltype(std::forward<U>(t)) {
            return std::forward<U>(t);
        }
    };
    
    /** \brief Function to call std::get<N>() on argument.
     */
    template <std::size_t N>
    struct get_n {
        template <typename U>
        constexpr auto operator()(U&& t) const noexcept
            -> decltype(std::get<N>(std::forward<U>(t))) {
            return std::get<N>(std::forward<U>(t));
        }
    };

    /** \brief Function to call std::get<T>() on argument.
     */
    template <typename T>
    struct get_t {
        template <typename U>
        constexpr auto operator()(U&& t) const noexcept
            -> decltype(std::get<T>(std::forward<U>(t))) {
            return std::get<T>(std::forward<U>(t));
        }
    };

}
