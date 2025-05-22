#pragma once

#include <type_traits>

namespace stdx {

    /** \brief Type trait to identify path like objects.
     *
     * Matches std::filesystem::path and boost::filesystem::path.
     */
    template <typename PATH, typename = void>
    struct is_path : std::false_type {};
    template <typename PATH>
    struct is_path<PATH, std::enable_if_t<std::is_same_v<std::decay_t<decltype(std::declval<const PATH&>().generic_string())>,std::string> > >
        : std::true_type {};

    template <typename PATH>
    constexpr auto is_path_v = is_path<PATH>::value;

}
