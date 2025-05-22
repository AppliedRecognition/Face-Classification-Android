#pragma once

#include <array>
#include <cassert>
#include <cstring>
#include <string>
#include <vector>
#include <memory>


namespace stdx {
    using std::move;


    /** \brief Container for holding read-only binary data.
     */
    class binary {
        using ptr_type = std::shared_ptr<const void>;

        static ptr_type make_copy(const void* p, std::size_t n) {
            const auto pc = static_cast<const unsigned char*>(p);
            const auto buf = 
                std::make_shared<std::vector<unsigned char> >(pc,pc+n);
            return ptr_type(buf,buf->data());
        }
        
    public:
        using size_type = std::size_t;


        /** \name Construct / Modify
         */
        //@{
        constexpr binary() noexcept : ptr(), len(0) {}

        binary(std::shared_ptr<const void> p, size_type n) noexcept
            : ptr(move(p)), len(n) {}

        template <typename T, std::size_t N>
        binary(T(&data)[N], std::enable_if_t<sizeof(T) == 1>* = nullptr)
            : ptr(make_copy(data,N*sizeof(T))), len(N*sizeof(T)) {}
        template <typename T, std::size_t N>
        binary(const std::array<T,N>& data,
               std::enable_if_t<sizeof(T) == 1>* = nullptr)
            : ptr(make_copy(data.data(),N*sizeof(T))), len(N*sizeof(T)) {}
        
        binary(const void* p, size_type n)
            : ptr(make_copy(p,n)), len(n) {}

        template <typename CHAR, typename TRAITS, typename ALLOC>
        explicit binary(const std::basic_string<CHAR,TRAITS,ALLOC>& p) {
            assign(p);
        }

        template <typename T, typename ALLOC>
        explicit binary(const std::vector<T,ALLOC>& p) {
            assign(p);
        }

        template <typename CHAR, typename TRAITS, typename ALLOC>
        explicit binary(std::basic_string<CHAR,TRAITS,ALLOC>&& x) {
            operator=(move(x));
        }

        template <typename T, typename ALLOC>
        binary(std::vector<T,ALLOC>&& x) {
            operator=(move(x));
        }

        binary(const binary& p, size_type pos, size_type n = size_type(-1)) 
            : ptr(p.ptr,static_cast<const char*>(p.ptr.get()) + pos),
              len(n <= p.len - pos ? n : p.len - pos) {
            assert(pos <= p.len);
        }
        
        inline void clear() {
            ptr.reset();
            len = 0;
        }

        inline void resize(size_type n) {
            assert(n <= len);
            len = n;
        }

        inline void swap(binary& other) {
            ptr.swap(other.ptr);
            size_type t = len;
            len = other.len;
            other.len = t;
        }
        //@}


        /** \name Assign
         */
        //@{
        inline void assign(const void* p, size_type n) {
            ptr = make_copy(p,n);
            len = n;
        }

        inline void assign(const binary& p, 
                           size_type pos = 0, size_type n = size_type(-1)) {
            assert(pos <= p.len);
            ptr = ptr_type(p.ptr,static_cast<const char*>(p.ptr.get()) + pos);
            len = n <= p.len - pos ? n : p.len - pos;
        }

        template <typename CHAR, typename TRAITS, typename ALLOC>
        inline void assign(const std::basic_string<CHAR,TRAITS,ALLOC>& p) {
            const auto buf = 
                std::make_shared<std::basic_string<CHAR,TRAITS,ALLOC> >(p);
            ptr = ptr_type(buf,buf->data());
            len = p.size() * sizeof(CHAR);
        }

        template <typename T, typename ALLOC>
        inline void assign(const std::vector<T,ALLOC>& p) {
            if (p.empty())
                clear();
            else {
                const auto buf = std::make_shared<std::vector<T,ALLOC> >(p);
                ptr = ptr_type(buf,buf->data());
                len = p.size() * sizeof(T);
            }
        }
        //@}


        /** \name Move
         */
        //@{
        template <typename CHAR, typename TRAITS, typename ALLOC>
        inline binary& operator=(std::basic_string<CHAR,TRAITS,ALLOC>&& x) {
            const auto buf = 
                std::make_shared<std::basic_string<CHAR,TRAITS,ALLOC> >();
            buf->swap(x);
            ptr = ptr_type(buf,buf->data());
            len = buf->size() * sizeof(CHAR);
            return *this;
        }

        template <typename T, typename ALLOC>
        inline binary& operator=(std::vector<T,ALLOC>&& x) {
            const auto buf = std::make_shared<std::vector<T,ALLOC> >();
            buf->swap(x);
            ptr = ptr_type(buf,buf->data());
            len = buf->size() * sizeof(T);
            return *this;
        }
        //@}


        /** \name Access
         */
        //@{
        inline bool empty() const {
            return len == 0;
        }
        inline size_type size() const {
            return len;
        }
        inline const void* data() const {
            return ptr.get();
        }
        template <typename CHAR>
        inline std::enable_if_t<sizeof(CHAR) == 1, const CHAR*> data() const {
            return static_cast<const CHAR*>(ptr.get());
        }
        inline const std::shared_ptr<const void>& shared_ptr() const {
            return ptr;
        }
        //@}


        /** \name Compare
         */
        //@{
        int compare(const void* other, size_type n) const noexcept {
            if (n == 0 || other == nullptr)
                return len == 0 ? 0 : 1;
            if (len == 0)
                return -1;
            if (ptr.get() == other)
                return len == n ? 0 : len < n ? -1 : 1;
            if (len < n)
               return memcmp(ptr.get(), other, len) <= 0 ? -1 : 1;
            if (n < len)
                return memcmp(ptr.get(), other, n) < 0 ? -1 : 1;
            return memcmp(ptr.get(), other, n); // len == n
        }
        inline int compare(const binary& other) const noexcept {
            return compare(other.data(), other.size());
        }
        //@}


    private:
        ptr_type ptr;
        size_type len;
    };


    /** \name Comparison
     */
    //@{
    inline bool operator==(const binary& a, const binary& b) {
        return a.compare(b) == 0;
    }
    inline bool operator!=(const binary& a, const binary& b) {
        return a.compare(b) != 0;
    }
    inline bool operator<(const binary& a, const binary& b) {
        return a.compare(b) < 0;
    }
    //@}
    

    /** \brief Swap
     */
    inline void swap(binary& a, binary& b) {
        return a.swap(b);
    }


    /** \brief Reference to binary data in memory.
     *
     * This method creates a binary object that does not own it's data.
     * Therefore the buffer will not be deallocated by the binary object's
     * destructor.
     *
     * The caller must ensure that both of the following hold:<ul>
     *   <li>the data buffer remains valid for the life of the binary object
     *       (and any copies of it)</li>
     *   <li>the data buffer is deallocated when it's no longer needed</li>
     * </ul>
     *
     * This method is ideally used with application static data (ie. assets
     * or resources).  In this case the data buffer is always valid and does
     * not require deallocation.
     *
     * \returns binary object pointing to specified data buffer
     */
    inline auto binary_ref(const void* data, std::size_t size) {
        // this shared_ptr will never attempt to delete the data
        return binary(std::shared_ptr<const void>(data,[](auto){}), size);
    }
}


