#pragma once

#include <memory>
#include <mutex>
#include <condition_variable>
#include <stdexcept>
#include <typeindex>
#include <type_traits>
#include <unordered_map>

namespace core {

    /** \brief Container for objects of various types, one instance per type.
     *
     * Types may be const, volatile or pointer, but not array or reference.
     * A const object is different from the corresponding non-const object.
     *
     * Once an object is constructed it cannot be replaced or destroyed.
     * The address of an object is fixed for the life of the container.
     *
     * If multithread is true, then the container is multi-thread safe.
     * If two threads race to construct the same object, they may both
     * construct an instance but only one will be stored. 
     * The address of the one stored will be returned to both threads.
     * Use of individual objects is not protected.
     */
    template <bool multithread>
    class object_store {
        using value_type =  std::unique_ptr<void,void(*)(void*)>;
        std::unordered_map<std::type_index, value_type> data;

        struct dummy_mutex_lock {
            constexpr void lock() {}
            constexpr void unlock() {}
        };
        struct dummy_cv {
            constexpr void notify_all() {}
            inline void wait(dummy_mutex_lock&) {
                throw std::logic_error("object constructed in mutliple threads without locking");
            }
        };
        
        using mutex_type =
            std::conditional_t<multithread, std::mutex, dummy_mutex_lock>;
        using lock_type =
            std::conditional_t<multithread, std::unique_lock<std::mutex>, dummy_mutex_lock>;
        using condition_variable_type = std::conditional_t<
            multithread, std::condition_variable, dummy_cv>;
        mutable mutex_type mutex;
        mutable condition_variable_type construction_done;

        template <typename T>
        static void deleter(void* p) {
            delete static_cast<T*>(p);
        }
        /// special marker to indicate object construction in progress
        static void constructing(void*) {}

        template <typename T, typename... ARGS>
        static inline value_type make(ARGS&&... args) {
            using U = typename std::remove_cv<T>::type;
            return { new U{std::forward<ARGS>(args)...}, &deleter<U> };
        }

        inline void* data_get(const std::type_info& t) {
            lock_type lock(mutex);
            const auto it = data.find(t);
            return it != data.end() ? it->second.get() : nullptr;
        }
        inline const void* data_get(const std::type_info& t) const {
            lock_type lock(mutex);
            const auto it = data.find(t);
            return it != data.end() ? it->second.get() : nullptr;
        }
        // look for t0 first, then if t0 not found, look for t1 
        inline const void*
        data_get(const std::type_info& t0, const std::type_info& t1) const {
            lock_type lock(mutex);
            auto it = data.find(t0);
            if (it != data.end())
                return it->second.get();
            if (t0 != t1 && (it = data.find(t1)) != data.end())
                return it->second.get();
            return nullptr;
        }

        object_store(object_store&&) = delete;
        object_store(const object_store&) = delete;
        object_store& operator=(object_store&&) = delete;
        object_store& operator=(const object_store&) = delete;

        
    public:
        object_store() = default;


        /** \brief Insert object.
         *
         * A new object is only constructed and inserted if it does not
         * already exist in the store.  
         * It is possible for a race by two or more threads to cause the
         * construction of multiple instance of the object, but only one
         * will be inserted and the address of that one will be returned
         * to all threads.
         * If the object already exists in the store, then the address
         * of the existing object is returned.
         */
        template <typename T, typename... ARGS>
        inline T& emplace(ARGS&&... args) {
            lock_type lock(mutex);
            auto& p = data.try_emplace(
                std::type_index(typeid(T)),nullptr,&deleter<T>).first->second;
            if (p.get_deleter() == &constructing) {
                // other thread is currently constructing object -- wait
                do {
                    construction_done.wait(lock);
                } while (p.get_deleter() == &constructing);
            }
            if (!p) {
                // first one here -- attempt to construct the object
                p = { nullptr, &constructing };
                lock.unlock();
                try {
                    auto r = object_store::make<T>(std::forward<ARGS>(args)...);
                    lock.lock();
                    p = move(r);
                }
                catch (...) {
                    lock.lock();
                    p = { nullptr, &deleter<T> };
                    construction_done.notify_all();
                    throw;
                }
                construction_done.notify_all();
            }
            return *static_cast<T*>(p.get());
        }

        
        /** \brief Insert object.
         *
         * Same as emplace().
         */
        template <typename T, typename... ARGS>
        inline typename std::enable_if<
            0 < sizeof...(ARGS) || std::is_default_constructible<T>::value,
            T&>::type 
        get(ARGS&&... args) {
            return emplace<T>(std::forward<ARGS>(args)...);
        }


        /** \brief Reference to object.
         *
         * An exception is thrown if object is not present.
         */
        template <typename T>
        inline typename std::enable_if<
            !std::is_default_constructible<T>::value, T&>::type
        get() {
            if (auto p = data_get(typeid(T)))
                return *static_cast<T*>(p);
            throw std::domain_error(
                std::string("object not found (") + typeid(T).name() + ")");
        }
        template <typename T>
        inline const T& get() const {
            if (auto p = data_get(typeid(T)))
                return *static_cast<const T*>(p);
            throw std::domain_error(
                std::string("object not found (") + typeid(T).name() + ")");
        }
        template <typename T>
        inline const T& cget() const {
            if (auto p = data_get(typeid(const T),typeid(T)))
                return *static_cast<const T*>(p);
            throw std::domain_error(
                std::string("object not found (") + typeid(T).name() + ")");
        }
        

        /** \brief Pointer to object.
         *
         * Returns nullptr if object is not present.
         */
        template <typename T>
        inline T* ptr() {
            return static_cast<T*>(data_get(typeid(T)));
        }
        template <typename T>
        inline const T* ptr() const {
            return static_cast<const T*>(data_get(typeid(T)));
        }
        template <typename T>
        inline const T* cptr() const {
            return static_cast<const T*>(data_get(typeid(const T),typeid(T)));
        }
    };

    template <typename T, bool B, typename... ARGS>
    inline T& emplace(object_store<B>& store, ARGS&&... args) {
        return store.template emplace<T>(std::forward<ARGS>(args)...);
    }

    template <typename T, bool B, typename... ARGS>
    inline T& get(object_store<B>& store, ARGS&&... args) {
        return store.template get<T>(std::forward<ARGS>(args)...);
    }
    template <typename T, bool B>
    inline const T& get(const object_store<B>& store) {
        return store.template get<T>();
    }
    template <typename T, bool B>
    inline const T& cget(const object_store<B>& store) {
        return store.template cget<T>();
    }

    template <typename T, bool B>
    inline T* ptr(object_store<B>& store) {
        return store.template ptr<T>();
    }
    template <typename T, bool B>
    inline const T* ptr(const object_store<B>& store) {
        return store.template ptr<T>();
    }
    template <typename T, bool B>
    inline const T* cptr(const object_store<B>& store) {
        return store.template cptr<T>();
    }
}
