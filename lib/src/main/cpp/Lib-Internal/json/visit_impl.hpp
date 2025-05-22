#pragma once

#include <array>
#include <cassert>
#include <functional>
#include <utility>
#include <variant>

namespace stdx {

    struct bad_variant_access : std::exception {
        const char* what() const noexcept override {
            return "bad_variant_access: std::variant is valueless";
        }
    };

    template <class Visitor, class Variant>
    struct visit_return_type;
    template <class Visitor, class... Types>
    struct visit_return_type<Visitor, std::variant<Types...>&>
        : std::common_type<std::invoke_result_t<Visitor, Types&>...> {};
    template <class Visitor, class... Types>
    struct visit_return_type<Visitor, std::variant<Types...>&&>
        : std::common_type<std::invoke_result_t<Visitor, Types&&>...> {};
    template <class Visitor, class... Types>
    struct visit_return_type<Visitor, const std::variant<Types...>&>
        : std::common_type<std::invoke_result_t<Visitor, const Types&>...> {};
    template <class Visitor, class... Types>
    struct visit_return_type<Visitor, const std::variant<Types...>&&>
        : std::common_type<std::invoke_result_t<Visitor, const Types&>...> {};
    template <class Visitor, class Variant>
    using visit_return_type_t =
        typename visit_return_type<Visitor, Variant>::type;
    
    template <class R, class Visitor, class Variant, std::size_t I = 0>
    constexpr R visit_invoke(Visitor&& vis, Variant&& var) {
        auto* p = std::get_if<I>(&var);
        assert(p);
        using T = std::conditional_t<
            std::is_rvalue_reference<Variant&&>::value,
            std::add_rvalue_reference_t<decltype(*p)>,
            std::add_lvalue_reference_t<decltype(*p)> >;
        return std::invoke(std::forward<Visitor>(vis), static_cast<T>(*p));
    }

    template <class R, class Visitor, class Variant, std::size_t... Is>
    constexpr auto visit_table_(std::index_sequence<Is...>) {
        using FN = decltype(&visit_invoke<R,Visitor,Variant>);
        std::array<FN, sizeof...(Is)> table = {
            &visit_invoke<R,Visitor,Variant,Is>...
        };
        return table;
    }
    template <class R, class Visitor, class Variant>
    constexpr auto visit_table =
        visit_table_<R,Visitor,Variant>(
            std::make_index_sequence<
            std::variant_size_v<std::decay_t<Variant> > >{});


    /** \brief Alternate implementation of std::visit().
     */
    template <class R, class Visitor, class Variant>
    constexpr R visit(Visitor&& vis, Variant&& var) {
        const auto& table = visit_table<R,Visitor,Variant>;
        const auto i = var.index();
        if (i >= table.size())
            throw bad_variant_access{};
        return table[i](std::forward<Visitor>(vis),std::forward<Variant>(var));
    }

    template <class Visitor, class Variant>
    constexpr decltype(auto) visit(Visitor&& vis, Variant&& var) {
        return visit<visit_return_type_t<Visitor&&, Variant&&> >(
            std::forward<Visitor>(vis), std::forward<Variant>(var));
    }
}
