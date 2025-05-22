#pragma once

#include "types.hpp"
#include <stdext/arg.hpp>
#include <stdext/span.hpp>
#include <iterator>
#include <stdexcept>

namespace raw_image {
    /** \brief Read-write random access to the pixels of an image.
     *
     * This wrapper appears like a container of lines.  
     * One may use operator[] or a for loop to access the image per line.
     *
     * Each line will dereference to an std::span<T> with the correct width.
     *
     * To directly access a pixel use operator()(x,y).  
     * This gives the same result as pixels[y][x].
     *
     * If either T is const or this object is const then the access is
     * read-only.
     */
    template <typename T>
    class pixels {
    public:
        raw_image::plane const& plane;

        using pixel_type = T;
        using value_type = stdx::span<T>;
        using size_type = decltype(plane.height);

        static auto verify_pointer(const raw_image::plane* ptr) {
            if (!ptr || bytes_per_pixel(ptr->layout) != sizeof(T))
                throw std::logic_error("pixel type has wrong size");
            return ptr;
        }

        template <typename U, typename = std::enable_if_t<stdx::can_extract_pointer_v<U, const raw_image::plane> > >
        pixels(U&& plane) : plane(*verify_pointer(stdx::pointer_to<const raw_image::plane>(plane))) {}

        inline auto width() const { return plane.width; }
        inline auto height() const { return plane.height; }

        inline auto size() const { return plane.height; }
        inline bool empty() const { return plane.height <= 0; }

        
        // line access
        template <typename J>
        inline auto ptr(J y) {
            using I = std::ptrdiff_t;
            return reinterpret_cast<T*>(
                plane.data + I(y) * I(plane.bytes_per_line));
        }
        template <typename J>
        inline T const* ptr(J y) const {
            using I = std::ptrdiff_t;
            return reinterpret_cast<T*>(
                plane.data + I(y) * I(plane.bytes_per_line));
        }
        template <typename J>
        inline auto operator[](J y) {
            return value_type(ptr(y), plane.width);
        }
        template <typename J>
        inline auto operator[](J y) const {
            return stdx::span<const T>(ptr(y), plane.width);
        }

        // pixel access
        template <typename I, typename J>
        inline pixel_type& operator()(I x, J y) {
            return ptr(y)[x];
        }
        template <typename I, typename J>
        inline pixel_type const& operator()(I x, J y) const {
            return ptr(y)[x];
        }

        // iterator
        template <typename U>
        class iterator_t {
            raw_image::plane const* plane = nullptr;
            uint8_t* pos = nullptr;

            explicit iterator_t(raw_image::plane const& plane, uint8_t* pos)
                : plane(&plane), pos(pos) {}
            explicit iterator_t(raw_image::plane const& plane)
                : iterator_t(plane, plane.data) {}

            friend class pixels<T>;
            
        public:
            using iterator_category = std::random_access_iterator_tag;
            using difference_type = std::ptrdiff_t;
            using value_type = stdx::span<U>;
            using pointer = value_type*;
            using reference = value_type&;

            iterator_t() = default;
            iterator_t(const iterator_t&) = default;
            iterator_t& operator=(const iterator_t&) = default;

            // if U is const, then enable copy construct from non-const other
            template <typename V, typename =
                      std::enable_if_t<std::is_const_v<U> &&
                                       !std::is_same_v<U,V> > >
            iterator_t(const iterator_t<V>& other)
                : plane(other.plane), pos(other.pos) {
            }
            
            inline auto operator*() const {
                return stdx::span<U>(reinterpret_cast<T*>(pos), plane->width);
            }
            template <typename J>
            inline auto operator[](J y) const {
                using I = std::ptrdiff_t;
                const auto line = reinterpret_cast<T*>(
                    pos + I(y) * I(plane->bytes_per_line));
                return stdx::span<U>(line, plane->width);
            }

            template <typename V>
            inline bool operator==(const iterator_t<V>& other) const {
                return pos == other.pos;
            }
            template <typename V>
            inline bool operator!=(const iterator_t<V>& other) const {
                return pos != other.pos;
            }
            template <typename V>
            inline bool operator<(const iterator_t<V>& other) const {
                return pos < other.pos;
            }
            template <typename V>
            inline bool operator<=(const iterator_t<V>& other) const {
                return pos <= other.pos;
            }
            template <typename V>
            inline bool operator>(const iterator_t<V>& other) const {
                return pos > other.pos;
            }
            template <typename V>
            inline bool operator>=(const iterator_t<V>& other) const {
                return pos >= other.pos;
            }

            inline auto& operator++() {
                pos += plane->bytes_per_line;
                return *this;
            }
            inline auto operator++(int) {
                auto r = *this;
                pos += plane->bytes_per_line;
                return r;
            }
            template <typename J>
            inline auto& operator+=(J y) {
                using I = std::ptrdiff_t;
                pos += I(y) * I(plane->bytes_per_line);
                return *this;
            }
            template <typename J>
            inline auto operator+(J y) {
                auto r = *this;  r += y;  return r;
            }
            template <typename J>
            inline friend auto operator+(J y, iterator_t i) {
                i += y;
                return i;
            }

            inline auto& operator--() {
                pos -= plane->bytes_per_line;
                return *this;
            }
            inline auto operator--(int) {
                auto r = *this;
                pos -= plane->bytes_per_line;
                return r;
            }
            template <typename J>
            inline auto& operator-=(J y) {
                using I = std::ptrdiff_t;
                pos -= I(y) * I(plane->bytes_per_line);
                return *this;
            }
            template <typename J>
            inline auto operator-(J y) {
                auto r = *this;  r -= y;  return r;
            }

            template <typename V>
            inline auto operator-(const iterator_t<V>& other) const {
                using I = std::ptrdiff_t;
                return (pos - other.pos) / I(plane->bytes_per_line);
            }
        };
        using iterator = iterator_t<T>;
        using const_iterator = iterator_t<const T>;

        inline auto begin() { return iterator(plane); }
        inline auto begin() const { return const_iterator(plane); }
        inline auto cbegin() const { return const_iterator(plane); }

        inline auto end() { return iterator(plane, plane.data + plane.height*plane.bytes_per_line); }
        inline auto end() const { return const_iterator(plane, plane.data + plane.height*plane.bytes_per_line); }
        inline auto cend() const { return const_iterator(plane, plane.data + plane.height*plane.bytes_per_line); }
    };


    /** \brief For 8-bit images pixels are an array of uint8_t.
     */
    template <unsigned BPP>
    using pixels_bpp =
        pixels<std::conditional_t<BPP != 1, std::array<uint8_t,BPP>, uint8_t> >;
}


