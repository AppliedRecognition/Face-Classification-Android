#pragma once

#include <cstddef>
#include <memory>
#include <type_traits>

namespace stdx {

    /** \brief Custom destruction policy for std::unique_ptr.
     *
     * If the stored std::byte* pointer is not null, then delete[] it
     * instead of the argument.
     * Otherwise, delete the argument as std::default_delete<T> would.
     */
    template <typename T>
    struct delete_bytes;
    template <typename T>
    struct delete_bytes<T[]> {
        std::byte* ptr;
        void operator()(T* arg) const {
            if (ptr) delete[] ptr;
            else delete[] arg;
        }
        constexpr delete_bytes(std::byte* ptr = nullptr) noexcept : ptr(ptr) {}
        constexpr delete_bytes(delete_bytes&& other) noexcept
            : ptr(other.ptr) {
            other.ptr = nullptr;
        }
        constexpr delete_bytes& operator=(delete_bytes&& other) noexcept {
            if (this != &other) {
                ptr = other.ptr;
                other.ptr = nullptr;
            }
            return *this;
        }
        delete_bytes(const delete_bytes& other) = delete;
        constexpr delete_bytes& operator=(const delete_bytes& other) = delete;
    };

    /** \brief Specialization of std::unique_ptr with custom deleter.
     */
    template <typename T>
    using aligned_ptr = std::unique_ptr<T, delete_bytes<T> >;


    /** \brief Allocate an array with the specified alignment.
     *
     * Not only will the address of the first element have the specified
     * alignment, but the memory allocation is guaranteed to be large enough
     * that an integer number of blocks of N bytes is accessable starting
     * at that address.
     *
     * This method requires that the type be trivial as no initialization
     * is done, and that the object size is such that an integer number of
     * objects fits within each N byte block.
     * N must be a power of 2.
     *
     * Implementation note: the C++17 method std::aligned_alloc() can be
     * used to make this method much simplier, but it is likely not available
     * on the iOS 10 platform (which we are still targetting).
     */
    template <typename T, std::size_t N>
    inline std::enable_if_t<std::is_array_v<T>, aligned_ptr<T> >
    make_aligned(std::size_t len) {
        using U = std::remove_extent_t<T>;
        static_assert(std::is_trivial_v<U>);
        static_assert(N >= sizeof(U) && (N & (N-1)) == 0,
                      "N must be at least sizeof(T) and a power of 2");
        static_assert(N % sizeof(U) == 0,
                      "an integer number of objects must fill alignment block");
        const auto block = 1 + ((len*sizeof(U)-1) | (N-1));
        auto space = N + block;
        const auto alloc = new std::byte[space];
        void* ptr = alloc;
        if (!std::align(N, block, ptr, space)) {
            delete[] alloc;
            throw std::bad_alloc();
        }
        return { static_cast<U*>(ptr), alloc };
    }
}
