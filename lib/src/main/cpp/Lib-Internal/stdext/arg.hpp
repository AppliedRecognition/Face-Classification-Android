#pragma once

#include <type_traits>
#include <utility>

namespace stdx {
    /** \brief Extract pointer to object from either reference or smart pointer.
     *
     * Note: this method will not accept r-value references, even if T is
     * a const type.  So it is generally safe to store the pointer extracted
     * by this method.
     */
    template <typename T, typename U>
    constexpr std::enable_if_t<std::is_lvalue_reference<decltype(*std::declval<U&&>())>::value && std::is_convertible<decltype(*std::declval<U&&>()),T&>::value, T*> pointer_to(U&& ptr) {
        return ptr ? &static_cast<T&>(*std::forward<U>(ptr)) : nullptr;
    }
    template <typename T, typename U>
    constexpr std::enable_if_t<std::is_convertible_v<U&&,T&> && !std::is_rvalue_reference_v<U&&>, T*>
    pointer_to(U&& ref) {
        return &static_cast<T&>(std::forward<U>(ref));
    }

    /** \brief Test if argument will be accepted by pointer_to() method.
     */
    template <typename From, typename To, typename = void>
    struct can_extract_pointer : std::false_type {};
    template <typename From, typename To>
    struct can_extract_pointer<From, To, std::enable_if_t<std::is_pointer_v<decltype(pointer_to<To>(std::declval<From&&>()))> > > : std::true_type {};
    template <typename From, typename To>
    constexpr bool can_extract_pointer_v = can_extract_pointer<From,To>::value;


    /** \brief Generalized method argument -- accepts reference or pointer.
     *
     * Do not explicitly construct objects of this type.
     * It is used implicitly when calling methods that do not care whether
     * their argument is a reference or some kind of (smart) pointer.
     *
     * Warning: when T is const the arg(U&&) constructor will accept
     * an r-value reference.  Do not store a pointer or reference to
     * the object as it may be dangling.
     */
    template <typename T>
    struct arg {
        static_assert(!std::is_reference<T>::value, "T must not be reference");

        T& operator*() const           { return *ptr; }
        T* operator->() const          { return ptr; }
        T* get() const                 { return ptr; }
        explicit operator bool() const { return ptr; }

        template <typename U>
        arg(U&& ptr, std::enable_if_t<std::is_lvalue_reference<decltype(*std::declval<U&&>())>::value && std::is_convertible<decltype(*std::declval<U&&>()),T&>::value>* = nullptr)
            : ptr(ptr ? &static_cast<T&>(*std::forward<U>(ptr)) : nullptr) {}

        template <typename U>
        arg(U&& ref, std::enable_if_t<std::is_convertible<U&&,T&>::value>* = nullptr)
            : ptr(&static_cast<T&>(std::forward<U>(ref))) {}

    private:
        T* const ptr;

        // no move or copy -- object must be used as method argument only
        arg(arg&&) = delete;
        arg(const arg&) = delete;
    };
}
