#pragma once

#include "typeinfo.hpp"
#include <iterator>
#include <type_traits>
#include <memory>


namespace stdx {

    /** \brief Base class to wrap any forward iterator.
     *
     * Provides access to wrapped iterator and comparison.
     * For comparison, the wrapped iterators must have exactly the same type.
     * For example, if some "iterator" and "const_iterator" are different
     * types, then they may not be compared -- an exception will be thrown.
     */
    class forward_iterator_base {
    protected:
        template <typename T>
        using decay_t = std::decay_t<T>;
        
        template <typename T, typename R = void>
        using enable_if_not_base =
            std::enable_if_t<!std::is_base_of<
                                 forward_iterator_base,
                                 decay_t<T> >::value, R>;
        
        struct forward_base {
            virtual ~forward_base() = default;
            virtual forward_base* new_copy() const = 0;
            using obj_pair = std::pair<const void*, const stdx::type_info*>;
            virtual obj_pair get() const = 0;
            virtual bool equal(const forward_base* other) const = 0;
            virtual std::ptrdiff_t distance(const forward_base*) const = 0;
            virtual void advance(std::ptrdiff_t n) = 0;
            virtual void incr() = 0;
        };

        template <typename T>
        struct forward_type : public forward_base {
            virtual T deref() const = 0;
        };

        template <typename T, typename ITER>
        struct forward_iter : public forward_type<T> {
            ITER it;

            template <typename U>
            explicit forward_iter(U&& it) : it(std::forward<U>(it)) {}

            using obj_pair = forward_base::obj_pair;
            static obj_pair get(const forward_iterator_base* p) {
                if (p->ptr) return p->ptr->get();
                else return { nullptr, nullptr };
            }
            template <typename U>
            static enable_if_not_base<U, obj_pair> get(const U* p) {
                return { p, stdx::typeptr<U> };
            }
            obj_pair get() const override {
                return get(&it);
            }

            static bool equal(const forward_iterator_base* p,
                              const forward_base* other) {
                if (p->ptr) return p->ptr->equal(other);
                throw std::bad_typeid();
            }
            template <typename U>
            static enable_if_not_base<U, bool>
            equal(const U* p, const forward_base* other) {
                const auto r = other->get();
                if (r.second == stdx::typeptr<U>)
                    return *p == *static_cast<const U*>(r.first);
                throw std::bad_typeid();
            }
            bool equal(const forward_base* other) const override {
                return equal(&it, other);
            }

            static std::ptrdiff_t distance(const forward_iterator_base* p,
                                           const forward_base* other) {
                if (p->ptr) return p->ptr->distance(other);
                throw std::bad_typeid();
            }
            template <typename U>
            static enable_if_not_base<U, std::ptrdiff_t>
            distance(const U* p, const forward_base* other) {
                const auto r = other->get();
                if (r.second == stdx::typeptr<U>)
                    return std::distance(*p, *static_cast<const U*>(r.first));
                throw std::bad_typeid();
            }
            std::ptrdiff_t distance(const forward_base* other) const override {
                return distance(&it, other);
            }

            void advance(std::ptrdiff_t n) override { std::advance(it,n); }
            void incr() override { ++it; }
        };

        template <typename ITER>
        struct forward_void final : public forward_iter<void,ITER> {
            template <typename U, typename... Args>
            explicit forward_void(U&& it, Args&&...)
                : forward_iter<void,ITER>(std::forward<U>(it)) {}

            forward_base* new_copy() const override {
                return new forward_void(*this);
            }
            void deref() const override {}
        };

        template <typename T, typename ITER>
        struct forward_obj_proper final : public forward_iter<T,ITER> {
            template <typename U>
            explicit forward_obj_proper(U&& it)
                : forward_iter<T,ITER>(std::forward<U>(it)) {}

            forward_base* new_copy() const override {
                return new forward_obj_proper(*this);
            }
            T deref() const override {
                return *this->it;
            }
        };

        template <typename T, typename ITER>
        using forward_obj = typename std::conditional<
            std::is_same<T,void>::value,
            forward_void<ITER>, forward_obj_proper<T,ITER> >::type;

        template <typename T, typename ITER, typename ADAPT>
        struct adapt_obj_proper final : public forward_iter<T,ITER> {
            ADAPT adapt;

            template <typename U, typename V>
            adapt_obj_proper(U&& it, V&& adapt)
                : forward_iter<T,ITER>(std::forward<U>(it)), 
                  adapt(std::forward<V>(adapt)) {}

            forward_base* new_copy() const override {
                return new adapt_obj_proper(*this);
            }
            T deref() const override {
                return adapt(*this->it);
            }
        };

        template <typename T, typename ITER, typename ADAPT>
        using adapt_obj = typename std::conditional<
            std::is_same<T,void>::value,
            forward_void<ITER>, adapt_obj_proper<T,ITER,ADAPT> >::type;

        
        std::unique_ptr<forward_base> ptr;
        forward_iterator_base(forward_base* ptr) : ptr(ptr) {}


    public:
        forward_iterator_base() = default;
        
        forward_iterator_base(forward_iterator_base&& it)
            : ptr(move(it.ptr)) {
        }
            
        forward_iterator_base(const forward_iterator_base& it)
            : ptr(it.ptr ? it.ptr->new_copy() : nullptr) {
        }

        forward_iterator_base& operator=(forward_iterator_base&& it) {
            ptr = move(it.ptr);
            return *this;
        }

        forward_iterator_base& operator=(const forward_iterator_base& it) {
            ptr.reset(it.ptr ? it.ptr->new_copy() : nullptr);
            return *this;
        }

        template <typename ITER>
        inline const decay_t<ITER>& get() const {
            if (ptr) {
                using U = decay_t<ITER>;
                const auto p = ptr->get();
                if (p.second == stdx::typeptr<U>)
                    return *static_cast<const U*>(p.first);
            }
            throw std::bad_cast();
        }

        inline bool operator==(const forward_iterator_base& rhs) const {
            if (ptr && rhs.ptr) return ptr->equal(rhs.ptr.get());
            throw std::bad_typeid();
        }
        inline bool operator!=(const forward_iterator_base& rhs) const {
            return !operator==(rhs);
        }

        inline std::ptrdiff_t
        distance(const forward_iterator_base& other) const {
            if (ptr && other.ptr) return ptr->distance(other.ptr.get());
            throw std::bad_typeid();
        }

        inline void advance(std::ptrdiff_t n) {
            if (ptr) ptr->advance(n);
        }
    };

    template <typename ITER>
    inline const typename std::decay<ITER>::type&
    get(const forward_iterator_base& it) {
        return it.template get<ITER>();
    }

    inline std::ptrdiff_t
    distance(const forward_iterator_base& a, const forward_iterator_base& b) {
        return a.distance(b);
    }

    inline void advance(forward_iterator_base& it, std::ptrdiff_t n) {
        it.advance(n);
    }

    
    /** \brief Wrapper for any forward iterator which dereferences to a type
     * convertable to T.
     *
     * The value returned by dereferencing the wrapped iterator is either
     * implicitly converted to T, or optionally,
     * an adaptor function may be provided to convert the value.
     *
     * The return type for operator* is T, 
     * which will only be a reference if T is a reference.
     * Also, operator-> is only available if T is a reference.
     *
     * In the special case where T is void, dereference is not available
     * and any iterator may be wrapped.
     */
    template <typename T>
    class forward_iterator final : public forward_iterator_base {
        
        template <typename U>
        using can_convert_from =
            std::enable_if_t<std::is_convertible<U,T>::value &&
                             !std::is_same<U,T>::value>;

        template <typename ITER>
        using can_deref_from =
            std::enable_if_t<std::is_convertible<
                                 decltype(*std::declval<ITER>()),T>::value &&
                             !std::is_base_of<forward_iterator_base,
                                              decay_t<ITER> >::value>;

        // this requirement is to ensure forward_iterator doesn't work
        // with stdx::arg (which could lead to dangling references)
        static_assert(!std::is_object_v<T> ||
                      std::is_copy_constructible_v<T> ||
                      std::is_move_constructible_v<T>,
                      "forward_iterator<T>: type T must be movable or copyable");

    public:
        using iterator_category = std::forward_iterator_tag;
        using difference_type = std::ptrdiff_t;
        using value_type = std::remove_reference_t<T>;
        using pointer = value_type*;
        using reference = T;

        forward_iterator() = default;
        forward_iterator(forward_iterator&& it) = default;
        forward_iterator(const forward_iterator& it) = default;
        forward_iterator& operator=(forward_iterator&& it) = default;
        forward_iterator& operator=(const forward_iterator& it) = default;

        template <typename U, typename = can_convert_from<U> >
        forward_iterator(forward_iterator<U>&& it)
            : forward_iterator_base(
                new forward_obj<T, forward_iterator<U> >(std::move(it))) {
        }

        template <typename U, typename = can_convert_from<U> >
        forward_iterator(const forward_iterator<U>& it)
            : forward_iterator_base(
                new forward_obj<T, forward_iterator<U> >(it)) {
        }
        
        template <typename ITER, typename = can_deref_from<ITER> >
        forward_iterator(ITER&& it) 
            : forward_iterator_base(
                new forward_obj<T, decay_t<ITER> >(
                    std::forward<ITER>(it))) {
        }

        template <typename ITER, typename ADAPT>
        forward_iterator(ITER&& it, ADAPT&& adaptor) 
            : forward_iterator_base(
                new adapt_obj<T, decay_t<ITER>, decay_t<ADAPT> >(
                    std::forward<ITER>(it),
                    std::forward<ADAPT>(adaptor))) {
        }

        inline forward_iterator& operator++() {
            ptr->incr();
            return *this;
        }
        inline forward_iterator operator++(int) {
            forward_iterator result(*this);
            operator++();
            return result;
        }

        inline forward_iterator operator+(std::ptrdiff_t delta) const {
            forward_iterator result(*this);
            result.ptr->advance(delta);
            return result;
        }

        inline T operator*() const {
            return static_cast<const forward_type<T>*>(ptr.get())->deref();
        }

        inline typename std::conditional<
            std::is_reference<T>::value,
            typename std::remove_reference<T>::type*, void>::type
        operator->() const {
            return &operator*();
        }
    };

    template <typename T, typename U>
    inline forward_iterator<T>
    static_iterator_cast(const forward_iterator<U>& iter) {
        return { iter, [](U x) -> T { return static_cast<T>(x); } };
    }
}
