#pragma once

#include <string>
#include <memory>

namespace stdx {

    /** \brief Extension of std::char_traits to turn both assign() methods
     * into no operations (to support uninitialized allocation).
     */
    template<typename CharT>
    struct no_assign_char_traits : public std::char_traits<CharT> {
        static void assign(CharT&, const CharT&) {
        }
        static CharT* assign(CharT* s, std::size_t, CharT) {
            return s;
        }
    };

    template <typename T, typename ALLOC>
    inline void uninitialized_resize(
        std::basic_string<T,std::char_traits<T>,ALLOC>& s,
        std::size_t n) {
        // This is a little ugly but should be a reasonably safe way to allocate
        // a fixed size string for use as a buffer without initializing it.
        s.reserve(n);
        reinterpret_cast<std::basic_string<T,no_assign_char_traits<T>,ALLOC>&>(s).resize(n);
    }

    template <typename T = unsigned char>
    struct uninitialized_buffer
        : std::unique_ptr<T, std::default_delete<T[]> > {

        uninitialized_buffer() = default;

        explicit uninitialized_buffer(std::size_t n)
            : std::unique_ptr<T, std::default_delete<T[]> >(new T[n]) {
        }

        inline T& operator[](std::size_t i) const { return this->get()[i]; }
    };
}


