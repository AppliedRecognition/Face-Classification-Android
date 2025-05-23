#pragma once

#include "types.hpp"
#include "pull_types.hpp"
#include "push_types.hpp"

#ifdef __IPHONE_OS_VERSION_MIN_REQUIRED
  #if __IPHONE_OS_VERSION_MIN_REQUIRED < 120000
    #include "visit_impl.hpp"
  #else
    namespace stdx { using std::visit; }
  #endif
#else
  namespace stdx { using std::visit; }
#endif

namespace json {
    template <typename FUNC>
    inline decltype(auto) visit(FUNC&& f, value& v) {
        auto& var = static_cast<detail::value_base&>(v);
        return stdx::visit(std::forward<FUNC>(f), var);
    }
    template <typename FUNC>
    inline decltype(auto) visit(FUNC&& f, value&& v) {
        auto& var = static_cast<detail::value_base&>(v);
        return stdx::visit(std::forward<FUNC>(f), move(var));
    }
    template <typename FUNC>
    inline decltype(auto) visit(FUNC&& f, const value& v) {
        auto& var = static_cast<const detail::value_base&>(v);
        return stdx::visit(std::forward<FUNC>(f), var);
    }

    template <typename FUNC>
    inline decltype(auto) visit(FUNC&& f, value_puller& v) {
        auto& var = static_cast<detail::value_puller_base&>(v);
        return stdx::visit(std::forward<FUNC>(f), var);
    }
    template <typename FUNC>
    inline decltype(auto) visit(FUNC&& f, value_puller&& v) {
        auto& var = static_cast<detail::value_puller_base&>(v);
        return stdx::visit(std::forward<FUNC>(f), move(var));
    }
    template <typename FUNC>
    inline decltype(auto) visit(FUNC&& f, const value_puller& v) {
        auto& var = static_cast<const detail::value_puller_base&>(v);
        return stdx::visit(std::forward<FUNC>(f), var);
    }

    template <typename FUNC>
    inline decltype(auto) visit(FUNC&& f, value_pusher& v) {
        auto& var = static_cast<detail::value_pusher_base&>(v);
        return stdx::visit(std::forward<FUNC>(f), var);
    }
    template <typename FUNC>
    inline decltype(auto) visit(FUNC&& f, value_pusher&& v) {
        auto& var = static_cast<detail::value_pusher_base&>(v);
        return stdx::visit(std::forward<FUNC>(f), move(var));
    }
    template <typename FUNC>
    inline decltype(auto) visit(FUNC&& f, const value_pusher& v) {
        auto& var = static_cast<const detail::value_pusher_base&>(v);
        return stdx::visit(std::forward<FUNC>(f), var);
    }


    template <typename HANDLER>
    void value_pusher::set_final_handler(
        HANDLER h, object::key_compare comp) {
        struct handler {
            HANDLER h;
            object::key_compare comp;
            inline void operator()(null_type) { h(null); }
            inline void operator()(boolean t) { h(t); }
            inline void operator()(integer t) { h(t); }
            inline void operator()(real t)    { h(t); }
            inline void operator()(string_pusher& t) {
                t.set_final_handler(h);
            }
            inline void operator()(binary_pusher& t) {
                t.set_final_handler(h);
            }
            inline void operator()(array_pusher& t) {
                t.set_final_handler(h);
            }
            inline void operator()(object_pusher& t) {
                t.set_final_handler(h,comp);
            }
        };
        visit(handler{h,comp}, *this);
    }

    template <typename HANDLER>
    void value_pusher::set_parent_handler(const HANDLER& h) {
        struct handler {
            HANDLER h;
            inline void operator()(null_type) { h(); }
            inline void operator()(boolean)   { h(); }
            inline void operator()(integer)   { h(); }
            inline void operator()(real)      { h(); }
            inline void operator()(string_pusher& t) {
                t.set_parent_handler(h);
            }
            inline void operator()(binary_pusher& t) {
                t.set_parent_handler(h);
            }
            inline void operator()(array_pusher& t) {
                t.set_parent_handler(h);
            }
            inline void operator()(object_pusher& t) {
                t.set_parent_handler(h);
            }
        };
        visit(handler{h}, *this);
    }
}
