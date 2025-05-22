#pragma once

#include <tuple>

namespace stdx {

    /** \brief Default method to apply option to tuple is assignment.
     */
    namespace options_tuple_default {
        template <typename T, typename Opt>
        constexpr void option_apply(T& t, Opt&& opt) {
            std::get<std::decay_t<Opt> >(t) = std::forward<Opt>(opt);
        }
    }

    /** \brief Construct tuple from set of option objects.
     */
    template <typename... Ts>
    struct options_tuple : std::tuple<Ts...> {
        static constexpr void apply() {} // base case

        template <typename Opt, typename... Opts>
        constexpr void apply(Opt&& opt, Opts&&... opts) {
            using namespace options_tuple_default;
            option_apply(*this, std::forward<Opt>(opt));
            apply(std::forward<Opts>(opts)...);
        }

        template <typename T, std::size_t... I>
        constexpr void apply_tuple(const T& t, std::index_sequence<I...>) {
            apply(std::get<I>(t)...);
        }
        template <typename... Us>
        constexpr void apply(const options_tuple<Us...>& other) {
            apply_tuple(other, std::make_index_sequence<sizeof...(Us)>{});
        }

        template <typename... Opts>
        constexpr options_tuple(Opts&&... opts) {
            apply(std::forward<Opts>(opts)...);
        }
    };

    /** \brief Boolean option type.
     */
    template <typename Tag>
    struct option_bool {
        bool b;
        constexpr explicit option_bool(bool b = false) : b(b) {}
        constexpr auto operator()(bool b) const { return option_bool(b); }
        constexpr operator bool() const { return b; }
    };
}
