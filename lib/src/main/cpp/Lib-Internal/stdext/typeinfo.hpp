#pragma once

namespace stdx {
    struct type_info {};

    template <typename T>
    const type_info type_info_v{};

    /** \brief Alternative to typeid that does not require rtti.
     *
     * type_ptr<T> == type_ptr<U> if is_same<T,U>
     * type_ptr<T> != type_ptr<U> if !is_same<T,U>
     */
    template <typename T>
    type_info const* const typeptr = &type_info_v<T>;
}
