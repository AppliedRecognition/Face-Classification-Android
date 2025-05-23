#pragma once

#include <type_traits>
#include <utility>

namespace stdx {

    /** \brief Extract pointer to data from array or continuous container.
     *
     * Similar to C++17 std::data().
     *
     * Also, if given a smart pointer (any object with a get() method
     * returning a pointer), then that pointer is returned.
     */
    template <typename T, std::size_t N>
    constexpr auto data(T(&arr)[N]) noexcept {
        return static_cast<T*>(arr);
    }
    template <typename T>
    constexpr auto data(T&& container, std::enable_if_t<std::is_pointer<decltype(std::declval<T&>().data())>::value>* = nullptr) {
        return container.data();
    }
    template <typename T>
    constexpr auto data(T&& container, std::enable_if_t<std::is_pointer<decltype(std::declval<T&>().get())>::value>* = nullptr) {
        return container.get();
    }

    /** \brief Extract size from array or container object.
     *
     * Similar to C++17 std::size().
     *
     * Also, if given a smart pointer (any object with a get() method
     * returning a pointer), then this method returns 1 if the pointer is
     * non-null and 0 otherwise.
     */
    template <typename T, std::size_t N>
    constexpr auto size(T(&)[N]) noexcept {
        return N;
    }
    template <typename T>
    constexpr auto size(T&& container, std::enable_if_t<std::is_integral<decltype(std::declval<T&>().size())>::value>* = nullptr) {
        return container.size();
    }
    template <typename T>
    constexpr auto size(T&& container, std::enable_if_t<std::is_pointer<decltype(std::declval<T&>().get())>::value>* = nullptr) {
        return container.get() != nullptr ? 1u : 0;
    }

    /** \brief Test if types have the same object size.
     */
    template <typename U, typename V, typename = void>
    struct is_same_size : std::false_type {};
    template <typename U, typename V>
    struct is_same_size<U,V,std::enable_if_t<sizeof(std::decay_t<U>) == sizeof(std::decay_t<V>)> >
        : std::true_type {};



    /** \brief Object that refers to a contiguous sequence of objects.
     *
     * Similar to C++20 std::span.
     *
     * This object can also be constructed from a smart pointer having a get()
     * method and returning a compatible pointer.  It will be treated as
     * a container having 0 or 1 element.
     */
    template <class T>
    class span {
    public:
        using element_type = T;
        using value_type = std::remove_cv_t<T>;
        using index_type = std::size_t;
        using difference_type = std::ptrdiff_t;
        using pointer = T*;
        using const_pointer = const T*;
        using reference = T&;
        using const_reference = const T&;

        using iterator = T*;
        using const_iterator = const T*;


        /** \brief Construct empty span.
         */
        constexpr span() noexcept = default;

        /** \brief Construct from pointer and count.
         */
        template <typename U>
        constexpr span(U* ptr, index_type count,
                       std::enable_if_t<
                       std::is_convertible_v<U*,T*> &&
                       is_same_size<T,U>::value>* = nullptr) noexcept
            : m_data(ptr), m_size(count) {
        }

        /** \brief Construct from pointer range.
         */
        template <typename U>
        constexpr span(U* first, U* last,
                       std::enable_if_t<
                       std::is_convertible_v<U*,T*> &&
                       is_same_size<T,U>::value>* = nullptr) noexcept
            : m_data(first), m_size(std::size_t(last - first)) {
        }

        /** \brief Construct from container.
         *
         * A container is either a C-style array, an object having both
         * data() and size() member methods, or a smart pointer having
         * a get() member method.
         * Example containers: std::array, std::vector and std::unique_ptr.
         *
         * \sa stdx::data() and stdx::size()
         */
        template <class C>
        constexpr span(C& c, std::enable_if_t<
                       std::is_integral_v<decltype(stdx::size(std::declval<C&>()))> &&
                       std::is_convertible_v<decltype(stdx::data(std::declval<C&>())),T*> &&
                       is_same_size<std::remove_pointer_t<decltype(stdx::data(std::declval<C&>()))>,T>::value>* = nullptr)
            : m_data(static_cast<T*>(stdx::data(c))), m_size(stdx::size(c)) {
        }

        /** \brief Specialization if c.data() returns void* and sizeof(T) == 1.
         *
         * For example, from stdx::binary.
         */
        template <class C>
        constexpr span(C& c, std::enable_if_t<
                       std::is_integral_v<decltype(stdx::size(std::declval<C&>()))> &&
                       std::is_void_v<std::remove_pointer_t<decltype(stdx::data(std::declval<C&>()))> > &&
                       is_same_size<char,T>::value>* = nullptr)
            : m_data(static_cast<T*>(stdx::data(c))), m_size(stdx::size(c)) {
        }

        /** \brief Construct from other kind of span with compatible pointer.
         */
        template <class U>
        constexpr span(const stdx::span<U>& s, std::enable_if_t<
                       std::is_convertible_v<U*,T*> &&
                       is_same_size<T,U>::value>* = nullptr) noexcept
            : span(s.data(), s.size()) {
        }

        constexpr span(const span&) noexcept = default;
        constexpr span& operator=(const span&) noexcept = default;

        constexpr bool empty() const noexcept { return m_size == 0; }
        constexpr index_type size() const noexcept { return m_size; }
        constexpr pointer data() const noexcept { return m_data; }
        constexpr reference front() const { return *m_data; }
        constexpr reference back() const { return *(m_data+m_size-1); }
        constexpr reference operator[](index_type idx) const {
            return *(m_data+idx);
        }

        constexpr span first(std::size_t count) const {
            return { m_data, count };
        }
        constexpr span last(std::size_t count) const {
            return { m_data+m_size-count, count };
        }
        constexpr span subspan(std::size_t offset, std::size_t count) const {
            return { m_data+offset, count };
        }
        constexpr span subspan(std::size_t offset) const {
            return { m_data+offset, m_size-offset };
        }

        constexpr iterator begin() const noexcept { return m_data; }
        constexpr const_iterator cbegin() const noexcept { return m_data; }
        constexpr iterator end() const noexcept { return m_data+m_size; }
        constexpr const_iterator cend() const noexcept { return m_data+m_size; }


    private:
        T* m_data = nullptr;
        std::size_t m_size = 0;
    };


    /** \brief Subclass to be used as method argument.
     *
     * This object can also be constructed from a reference to an object
     * that can be converted to T&.
     * This includes temporary objects (rvalue references).
     * The results is a span of size 1.
     *
     * This object is not movable or copyable as it should be used as a
     * method argument only.
     */
    template <class T>
    class spanarg : public span<T> {
    public:
        /** \brief Contruct from reference to object.
         * \post size() == 1
         */
        template <typename U>
        constexpr spanarg(U&& ref, std::enable_if_t<std::is_convertible<U&&,T&>::value>* = nullptr) noexcept
            : span<T>(&static_cast<T&>(ref), 1) {}

        /** \brief Contruct from pointer.
         *
         * Size is 1 if and only if pointer is non-null.
         */
        constexpr spanarg(T* ptr) noexcept
            : span<T>(ptr, ptr ? 1 : 0) {
        }

        /// accept R-value reference
        template <class C>
        constexpr spanarg(C&& c, std::enable_if_t<
                          std::is_integral_v<decltype(stdx::size(std::declval<C&>()))> &&
                          std::is_convertible_v<decltype(stdx::data(std::declval<C&>())),T*> &&
                          is_same_size<std::remove_pointer_t<decltype(stdx::data(std::declval<C&>()))>,T>::value>* = nullptr)
            : span<T>(static_cast<T*>(stdx::data(c)), stdx::size(c)) {
        }

        // not sure why this has to be repeated?
        template <class U>
        constexpr spanarg(const stdx::span<U>& s, std::enable_if_t<
                          std::is_convertible_v<U*,T*> &&
                          is_same_size<T,U>::value>* = nullptr) noexcept
            : span<T>(s.data(), s.size()) {
        }

        using span<T>::span;

        spanarg(spanarg&&) = delete;
        spanarg(const spanarg&) = delete;
        spanarg& operator=(spanarg&&) = delete;
        spanarg& operator=(const spanarg&) = delete;
    };
}
