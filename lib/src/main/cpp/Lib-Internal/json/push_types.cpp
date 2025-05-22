
#include "push_types.hpp"
#include "encode.hpp"
#include "visit.hpp"

#include <stdext/base64.hpp>

#include <applog/core.hpp>

#include <vector>
#include <deque>
#include <list>
#include <memory>
#include <cstddef>


using namespace json;
using namespace json::detail;


struct detail::string_pusher_ops {
    using const_buffer_type = std::vector<string>;
    static void fill_const_buffer(const_buffer_type& dest, string& val) {
        dest.resize(dest.size()+1);
        dest.back().swap(val);
    }
    static string take_final(const_buffer_type& buf) {
        AR_CHECK(buf.size() <= 1);
        string s;
        if (!buf.empty()) {
            s = move(buf.front());
            buf.clear();
        }
        return s;
    }
    static const string& final_const(const const_buffer_type& buf) {
        static const string empty_string;
        AR_CHECK(buf.size() <= 1);
        return buf.empty() ? empty_string : *buf.begin();
    }
    template <typename Iterator, typename T>
    static string final_value(Iterator begin, Iterator end, const T&) {
        if (begin == end)
            return string();
        const string first = *begin;
        if (++begin == end) 
            return first;
        // concatenate strings
        std::string::size_type total = first.size();
        for (Iterator jt = begin; jt != end; ++jt)
            total += jt->size();
        std::string buf;
        buf.reserve(total);
        buf.append(first);
        do {
            buf.append(*begin);
        } while (++begin != end);
        assert(buf.size() == total);
        return buf;
    }
    static bool is_final(const string&) {
        return true;
    }
    static std::size_t element_size(const string& x) {
        return x.size();
    }
    template <typename T, typename H>
    static void set_parent_handler(T&, H) {
        assert(!"not valid");
    }
};

struct detail::binary_pusher_ops {
    using const_buffer_type = std::vector<binary>;
    static void fill_const_buffer(const_buffer_type& dest, binary& val) {
        dest.push_back(val);
    }
    static binary take_final(const_buffer_type& buf) {
        AR_CHECK(buf.size() <= 1);
        binary s;
        if (!buf.empty()) {
            s = move(buf.front());
            buf.clear();
        }
        return s;
    }
    static const binary& final_const(const const_buffer_type& buf) {
        static const binary empty_binary;
        AR_CHECK(buf.size() <= 1);
        return buf.empty() ? empty_binary : *buf.begin();
    }
    template <typename Iterator, typename T>
    static binary final_value(Iterator begin, Iterator end, const T&) {
        if (begin == end)
            return binary();
        const binary first = *begin;
        if (++begin == end)
            return first;
        // concatenate binaries
        auto total = first.size();
        for (Iterator jt = begin; jt != end; ++jt)
            total += jt->size();
        std::vector<std::byte> buf;
        buf.reserve(total);
        auto d = first.data<std::byte>();
        buf.insert(buf.end(),d,d+first.size());
        do {
            d = begin->template data<std::byte>();
            buf.insert(buf.end(),d,d+begin->size());
        } while (++begin != end);
        assert(buf.size() == total);
        return buf;
    }
    static bool is_final(const binary&) {
        return true;
    }
    static std::size_t element_size(const binary& x) {
        return x.size();
    }
    template <typename T, typename H>
    static void set_parent_handler(T&, H) {
        assert(!"not valid");
    }
};

struct detail::array_pusher_ops {
    using const_buffer_type = array;
    static void fill_const_buffer(const_buffer_type& dest, array& val) {
        dest.swap(val);
    }
    static inline array&& take_final(const_buffer_type& buf) {
        return move(buf);
    }
    static inline const array& final_const(const const_buffer_type& buf) {
        return buf;
    }
    template <typename Iterator, typename T>
    static array final_value(Iterator begin, Iterator end, const T&) {
        array result;
        for (; begin != end; ++begin)
            result.push_back(begin->final_value());
        return result;
    }
    static bool is_final(const value_pusher& x) noexcept {
        return x.is_final();
    }
    static std::size_t element_size(const value_pusher&) {
        return 1;
    }
    template <typename T, typename H>
    static void set_parent_handler(T& x, H h) {
        x.set_parent_handler(h);
    }
};

struct detail::object_pusher_ops {
    using const_buffer_type = object;
    static void fill_const_buffer(const_buffer_type& dest, object& val) {
        dest.swap(val);
    }
    static inline object&& take_final(const_buffer_type& buf) {
        return move(buf);
    }
    static inline const object& final_const(const const_buffer_type& buf) {
        return buf;
    }
    template <typename Iterator>
    static object final_value(Iterator begin, Iterator end,
                              object::key_compare comp) {
        object result(comp);
        for (; begin != end; ++begin)
            result[begin->first] = begin->second.final_value();
        return result;
    }
    template <typename T>
    static bool is_final(const T& x) noexcept {
        return x.second.is_final();
    }
    template <typename T>
    static std::size_t element_size(const T&) {
        return 1;
    }
    template <typename T, typename H>
    static void set_parent_handler(T& x, H h) {
        x.second.set_parent_handler(h);
    }
};



/**************** class basic_pusher ****************/

template <typename T, typename FINAL, class OPS>
struct basic_pusher<T,FINAL,OPS>::internal {

    using const_buffer_type = typename OPS::const_buffer_type;
    const_buffer_type const_buffer;

    using buffer_type = std::deque<value_type>;
    buffer_type buffer;

    bool stalled;
    std::optional<typename buffer_type::iterator> stalled_pos;

    std::unique_ptr<value_handler_base> value_handler;
    std::unique_ptr<range_handler_base> range_handler;
    std::unique_ptr<final_handler_base> final_handler;
    object::key_compare comp;
    std::list<std::unique_ptr<parent_handler_base> > parent_handlers;

    std::optional<size_type> final_size;
    size_type size_thus_far;
    bool final;

    void do_push(const std::shared_ptr<internal>& ptr);
    void do_final();
    void do_parent();

    struct parent_handler {
        std::shared_ptr<internal> ptr;
        parent_handler(const std::shared_ptr<internal>& ptr) : ptr(ptr) {}
        void operator()() {
            AR_CHECK(ptr->stalled);
            ptr->stalled = false;
            ptr->stalled_pos = std::nullopt;
            ptr->do_push(ptr);
        }
    };

    internal(const std::optional<size_type>& final_size)
        : stalled(false),
          final_size(final_size),
          size_thus_far(0),
          final(false) {
    }
    ~internal() {
        if (!final || stalled) {
            if (!stalled && !final_size && size_thus_far == 0)
                FILE_LOG(logINFO) << "pusher[" << type_name<FINAL>()
                                  << "] not used";
            else
                FILE_LOG(logWARNING) << "pusher[" << type_name<FINAL>()
                                     << "] destructed before complete";
        }
    }
};

template <typename T, typename FINAL, class OPS>
void basic_pusher<T,FINAL,OPS>::internal::do_push(
    const std::shared_ptr<internal>& ptr) {
    if (stalled && !stalled_pos) 
        return;
    if (value_handler) {
        if (!const_buffer.empty()) {
            // just copy to buffer since we need to remove elements one-by-one
            buffer.insert(buffer.begin(),
                          const_buffer.begin(),const_buffer.end());
            const_buffer.clear();
        }
        std::unique_ptr<value_handler_base> h;
        h.swap(value_handler);  // handler is cleared if exception thrown
        if (stalled) {
            typename buffer_type::iterator end = *stalled_pos;
            ++end;
            do {
                (*h)(buffer.front());
                buffer.pop_front();
            } while (buffer.begin() != end);
            stalled_pos = std::nullopt;
            value_handler.swap(h);
            return;
        }
        while (!buffer.empty()) {
            (*h)(buffer.front());
            if (!OPS::is_final(buffer.front())) {
                OPS::set_parent_handler(buffer.front(),parent_handler(ptr));
                buffer.pop_front();
                stalled_pos = std::nullopt;
                stalled = true;
                value_handler.swap(h);
                return;
            }
            buffer.pop_front();
        }
        if (final) {
            try {
                (*h)();
            }
            catch (const std::exception& e) {
                FILE_LOG(logERROR) << "pusher[" << type_name<FINAL>()
                                   << "] exception during push: " << e.what();
                do_final();
                throw;
            }
            do_final();
            return;
        }
        value_handler.swap(h);
    }
    else if (range_handler) {
        std::unique_ptr<range_handler_base> h;
        h.swap(range_handler);  // handler is cleared if exception thrown
        if (!const_buffer.empty()) {
            (*h)(const_buffer.begin(),const_buffer.end());
            const_buffer.clear();
        }
        if (stalled) {
            typename buffer_type::iterator end = *stalled_pos;
            (*h)(buffer.begin(),++end);
            stalled_pos = std::nullopt;
            buffer.erase(buffer.begin(),end);
            range_handler.swap(h);
            return;
        }
        while (!buffer.empty()) {
            typename buffer_type::iterator active = buffer.end();
            typename buffer_type::iterator end = buffer.begin();
            assert(end != buffer.end());
            do {
                if (!OPS::is_final(*end)) {
                    active = end++;
                    break;
                }
            } while (++end != buffer.end());
            (*h)(buffer.begin(),end);
            if (active != buffer.end() && !OPS::is_final(*active)) {
                OPS::set_parent_handler(*active,parent_handler(ptr));
                buffer.erase(buffer.begin(),end);
                stalled_pos = std::nullopt;
                stalled = true;
                range_handler.swap(h);
                return;
            }
            buffer.erase(buffer.begin(),end);
        }
        if (final) {
            try {
                (*h)(buffer.begin(),buffer.end());
            }
            catch (const std::exception& e) {
                FILE_LOG(logERROR) << "pusher[" << type_name<FINAL>()
                                   << "] exception during push: " << e.what();
                do_final();
                throw;
            }
            do_final();
            return;
        }
        range_handler.swap(h);
    }
    else if (final && !stalled) {
        for (typename buffer_type::iterator 
                 it=buffer.begin(),end=buffer.end(); it!=end; ++it)
            if (!OPS::is_final(*it)) {
                OPS::set_parent_handler(*it,parent_handler(ptr));
                stalled_pos = it;
                stalled = true;
                return;
            }
        do_final();
    }
}

template <typename T, typename FINAL, class OPS>
void basic_pusher<T,FINAL,OPS>::internal::do_final() {
    AR_CHECK(final && !stalled);
    try {
        if (final_handler) {
            std::unique_ptr<final_handler_base> h;
            h.swap(final_handler);
            if (!buffer.empty()) {
                AR_CHECK(const_buffer.empty());
                FINAL x = OPS::final_value(buffer.begin(),buffer.end(),comp);
                buffer.clear();
                OPS::fill_const_buffer(const_buffer,x);
            }
            (*h)(OPS::final_const(const_buffer));
        }
    }
    catch (const std::exception& e) {
        FILE_LOG(logERROR) << "pusher[" << type_name<FINAL>()
                           << "] exception during final: " << e.what();
        do_parent();
        throw;
    }
    do_parent();
}

template <typename T, typename FINAL, class OPS>
void basic_pusher<T,FINAL,OPS>::internal::do_parent() {
    AR_CHECK(final && !stalled);
    while (!parent_handlers.empty()) {
        try {
            const auto handler = move(parent_handlers.front());
            parent_handlers.pop_front();
            (*handler)();
        }
        catch (const std::exception& e) {
            FILE_LOG(logERROR) << "pusher[" << type_name<FINAL>()
                               << "] exception from parent: " << e.what();
            do_parent();
            throw;
        }
    }
}

template <typename T, typename FINAL, class OPS>
basic_pusher<T,FINAL,OPS>::basic_pusher(
    const std::optional<size_type>& final_size)
    : state(std::make_shared<internal>(final_size)) {
}

template <typename T, typename FINAL, class OPS>
basic_pusher<T,FINAL,OPS>::basic_pusher(const final_type& val)
    : state(std::make_shared<internal>(val.size())) {
    final_type val_copy(val);
    OPS::fill_const_buffer(state->const_buffer,val_copy);
    state->final = true;
}
            
template <typename T, typename FINAL, class OPS>
void basic_pusher<T,FINAL,OPS>::set_final_size(size_type final_size) {
    AR_CHECK(!state->final_size || *state->final_size == final_size);
    AR_CHECK(state->size_thus_far <= final_size);
    state->final_size = final_size;
}

template <typename T, typename FINAL, class OPS>
const std::optional<typename basic_pusher<T,FINAL,OPS>::size_type>& 
basic_pusher<T,FINAL,OPS>::final_size() const {
    return state->final_size;
}

template <typename T, typename FINAL, class OPS>
bool basic_pusher<T,FINAL,OPS>::is_final() const noexcept {
    return state->final && !state->stalled;
}

template <typename T, typename FINAL, class OPS>
FINAL basic_pusher<T,FINAL,OPS>::take_final() {
    AR_CHECK(state->final && !state->stalled);
    if (!state->buffer.empty()) {
        AR_CHECK(state->const_buffer.empty());
        FINAL x = OPS::final_value(
            state->buffer.begin(),state->buffer.end(),state->comp);
        state->buffer.clear();
        OPS::fill_const_buffer(state->const_buffer,x);
    }
    return OPS::take_final(state->const_buffer);
}

template <typename T, typename FINAL, class OPS>
const FINAL& basic_pusher<T,FINAL,OPS>::final_value() const {
    AR_CHECK(state->final && !state->stalled);
    if (!state->buffer.empty()) {
        AR_CHECK(state->const_buffer.empty());
        FINAL x = OPS::final_value(
            state->buffer.begin(),state->buffer.end(),state->comp);
        state->buffer.clear();
        OPS::fill_const_buffer(state->const_buffer,x);
    }
    return OPS::final_const(state->const_buffer);
}

template <typename T, typename FINAL, class OPS>
void basic_pusher<T,FINAL,OPS>::set_value_handler_obj(
    std::unique_ptr<value_handler_base> handler) {
    AR_CHECK(!state->value_handler && !state->range_handler);
    state->value_handler = move(handler);
    state->do_push(state);
}

template <typename T, typename FINAL, class OPS>
void basic_pusher<T,FINAL,OPS>::set_range_handler_obj(
    std::unique_ptr<range_handler_base> handler) {
    AR_CHECK(!state->value_handler && !state->range_handler);
    state->range_handler = move(handler);
    state->do_push(state);
}

template <typename T, typename FINAL, class OPS>
void basic_pusher<T,FINAL,OPS>::set_final_handler_obj(
    std::unique_ptr<final_handler_base> handler, object::key_compare comp) {
    AR_CHECK(!state->final_handler);
    state->final_handler = move(handler);
    state->comp = comp;
    state->do_push(state);
}

template <typename T, typename FINAL, class OPS>
void basic_pusher<T,FINAL,OPS>::set_parent_handler_obj(
    std::unique_ptr<parent_handler_base> handler) {
    AR_CHECK(!state->final || state->stalled);
    state->parent_handlers.push_front(move(handler));
}

template <typename T, typename FINAL, class OPS>
void basic_pusher<T,FINAL,OPS>::operator()() {
    AR_CHECK(!state->final);
    if (state->final_size && *state->final_size != state->size_thus_far)
        FILE_LOG(logERROR) << "pusher[" << type_name<FINAL>()
                           << "] short stream (" << state->size_thus_far 
                           << " < " << *state->final_size << ")";
    state->final = true;
    state->final_size = state->size_thus_far;
    state->do_push(state);
}

template <typename T, typename FINAL, class OPS>
void basic_pusher<T,FINAL,OPS>::operator()(std::optional<T> val) {
    if (val) {
        AR_CHECK(!state->final);
        auto size = OPS::element_size(*val);
        AR_CHECK(!state->final_size || 
                 *state->final_size >= state->size_thus_far + size);
        state->size_thus_far += size;
        state->buffer.push_back(std::move(*val));
        state->do_push(state);
    }
    else  // end-of-stream
        operator()();
}

template <typename T, typename FINAL, class OPS>
void basic_pusher<T,FINAL,OPS>::operator()(iterator begin, iterator end) {
    if (begin == end) {  // end-of-stream 
        operator()();
        return;
    }
    AR_CHECK(!state->final);
    size_type size = 0;
    for (iterator it = begin; it != end; ++it)
        size += OPS::element_size(*it);
    AR_CHECK(!state->final_size || 
              *state->final_size >= state->size_thus_far + size);
    state->size_thus_far += size;
    state->buffer.insert(state->buffer.end(),begin,end);
    state->do_push(state);
}


/**************** class object_pusher ****************/

object object_pusher::take_final(object::key_compare comp) {
    AR_CHECK(state->final && !state->stalled);
    if (!state->buffer.empty()) {
        AR_CHECK(state->const_buffer.empty());
        object x = detail::object_pusher_ops::final_value(
            state->buffer.begin(),state->buffer.end(),comp);
        state->buffer.clear();
        state->const_buffer.swap(x);
    }
    return move(state->const_buffer);
}

const object& object_pusher::final_value(object::key_compare comp) const {
    AR_CHECK(state->final && !state->stalled);
    if (!state->buffer.empty()) {
        AR_CHECK(state->const_buffer.empty());
        object x = detail::object_pusher_ops::final_value(
            state->buffer.begin(),state->buffer.end(),comp);
        state->buffer.clear();
        state->const_buffer.swap(x);
    }
    return state->const_buffer;
}

void object_pusher::push_value(const string& key, const value& val) {
    operator()(value_type(key,val));
}


/**************** class value_pusher ****************/

namespace {
    struct pusher_from_value {
        inline value_pusher_base operator()(const null_type&) const noexcept {
            return null;
        }
        inline value_pusher_base operator()(boolean t) const noexcept {
            return t;
        }
        inline value_pusher_base operator()(integer t) const noexcept {
            return t;
        }
        inline value_pusher_base operator()(real t) const noexcept {
            return t;
        }
        inline value_pusher_base operator()(const string& t) const {
            return string_pusher(t);
        }
        inline value_pusher_base operator()(const binary& t) const {
            return binary_pusher(t);
        }
        inline value_pusher_base operator()(const array& t) const {
            return array_pusher(t);
        }
        inline value_pusher_base operator()(const object& t) const {
            return object_pusher(t);
        }
    };
}

value_pusher::value_pusher(const value& v)
    : val(visit(pusher_from_value{}, v)) {
}

const char* json::type_name(const value_pusher& v) {
    return visit(type_name_visitor{}, v);
}

namespace {
    struct is_final_visitor {
        template <typename T>
        constexpr bool operator()(T, std::enable_if_t<is_json_type<T>::value>* = nullptr) const noexcept {
            return true;
        }
        template <typename T>
        inline bool operator()(const T& x, std::enable_if_t<!is_json_type<T>::value>* = nullptr) const noexcept  {
            return x.is_final();
        }
    };
}
bool value_pusher::is_final() const noexcept {
    return visit(is_final_visitor{}, *this);
}

namespace {
    struct take_final_visitor {
        object::key_compare comp;
        template <typename T>
        inline value operator()(T x, std::enable_if_t<is_json_type<T>::value>* = nullptr) const noexcept {
            return x;
        }
        template <typename T>
        inline value operator()(T& x, std::enable_if_t<!is_json_type<T>::value>* = nullptr) const {
            static_assert(!std::is_same<T,object_pusher>::value,
                          "overload resolution failure");
            return x.take_final();
        }
        inline value operator()(object_pusher& x) const {
            return x.take_final(comp);
        }
    };
    struct final_value_visitor {
        object::key_compare comp;
        template <typename T>
        inline value operator()(T x, std::enable_if_t<is_json_type<T>::value>* = nullptr) const noexcept {
            return x;
        }
        template <typename T>
        inline value operator()(const T& x, std::enable_if_t<!is_json_type<T>::value>* = nullptr) const {
            static_assert(!std::is_same<T,object_pusher>::value,
                          "overload resolution failure");
            return x.final_value();
        }
        inline value operator()(const object_pusher& x) const {
            return x.final_value(comp);
        }
    };
}

value value_pusher::take_final(object::key_compare comp) {
    return visit(take_final_visitor{comp}, *this);
}
value value_pusher::final_value(object::key_compare comp) const {
    return visit(final_value_visitor{comp}, *this);
}


/**************** conversion ****************/

real json::make_real(const value_pusher& v) {
    return is_type<integer>(v) ? real(get<integer>(v)) : get<real>(v);
}

namespace {
    struct push_to_binary {
        binary_pusher stream;
        push_to_binary(const binary_pusher& stream) : stream(stream) {}
        void operator()(std::optional<string> str) {
            if (!str) stream();
            else stream(binary(move(*str)));
        }
    };
    
    struct base64_push_decoder {
        binary_pusher stream;
        std::string buffer;
        base64_push_decoder(const binary_pusher& stream) : stream(stream) {}

        void operator()(const std::optional<string>& str) {
            unsigned char out_buf[3];

            if (!str) {
                if (!buffer.empty()) {
                    if (buffer.size() == 1)
                        throw stdx::invalid_base64("invalid base64 string");
                    while (buffer.size() != 4)
                        buffer += '=';
                    auto out_len = stdx::base64_decode3(out_buf,buffer.data());
                    stream(binary{out_buf,out_len});
                }
                stream();
                return;
            }

            const char* src = str->data();
            size_t src_len = str->size();

            std::vector<unsigned char> result;
            result.reserve(3*((buffer.size()+src_len+3)/4));

            if (!buffer.empty()) {
                while (buffer.size() < 4) {
                    if (src_len == 0) {
                        stream(move(result));
                        return;
                    }
                    buffer += *src;
                    ++src;
                    --src_len;
                }
                const auto len = stdx::base64_decode3(out_buf,buffer.data());
                result.insert(result.end(),out_buf,out_buf+len);
                buffer.clear();
            }

            while (src_len >= 4) {
                const auto len = stdx::base64_decode3(out_buf,src);
                result.insert(result.end(),out_buf,out_buf+len);
                src += 4;
                src_len -= 4;
            }

            if (src_len > 0)
                buffer.assign(src,src_len);

            stream(move(result));
        }
    };

    struct push_to_string {
        string_pusher stream;
        push_to_string(const string_pusher& stream) : stream(stream) {}
        void operator()(const std::optional<binary>& bin) {
            if (!bin) stream();
            else {
                string str(bin->data<char>(), bin->size());
                stream(str);
            }
        }
    };

    struct base64_push_encoder {
        string_pusher stream;
        unsigned char buffer[4] = {};
        size_t nbuffer = 0;

        base64_push_encoder(const string_pusher& stream)
            : stream(stream) {}

        void operator()(const std::optional<binary>& bin) {
            char out_buf[4];

            if (!bin) {
                if (nbuffer > 0) {
                    stdx::base64_encode3(out_buf, buffer, nbuffer);
                    nbuffer = 0;
                    stream(string(out_buf,4));
                }
                stream();
                return;
            }

            auto src = bin->data<unsigned char>();
            auto src_len = bin->size();

            std::string result;
            result.reserve(4*(src_len+2)/3);

            if (nbuffer > 0) {
                while (nbuffer < 3) {
                    if (src_len == 0) {
                        stream(result);
                        return;  // wait for more input
                    }
                    buffer[nbuffer++] = *src;
                    ++src;
                    --src_len;
                }
                stdx::base64_encode3(out_buf, buffer, 3);
                result.append(out_buf,4);
                nbuffer = 0;
            }
            
            while (src_len >= 3) {
                stdx::base64_encode3(out_buf, src, 3);
                result.append(out_buf,4);
                src += 3;
                src_len -= 3;
            }

            if (src_len > 0) {
                memcpy(buffer,src,src_len);
                nbuffer = src_len;
            }
            stream(move(result));
        }
    };
}

string_pusher json::get_string_pusher(const value_pusher& val,
                                      convert_type convert) {
    if (convert != convert_none && is_type<binary_pusher>(val)) {
        string_pusher result;
        if (convert == convert_cast) {
            // todo: could set final_size in result (to bytes remaining?)
            result.set_value_handler(push_to_binary(get<binary_pusher>(val)));
        }
        else  // convert == convert_base64
            result.set_value_handler(base64_push_decoder(get<binary_pusher>(val)));
        return result;
    }
    return get<string_pusher>(val);
}

binary_pusher json::get_binary_pusher(const value_pusher& val,
                                      convert_type convert) {
    if (convert != convert_none && is_type<string_pusher>(val)) {
        binary_pusher result;
        if (convert == convert_cast) {
            // todo: could set final_size in result (to bytes remaining?)
            result.set_value_handler(push_to_string(get<string_pusher>(val)));
        }
        else  // convert == convert_base64
            result.set_value_handler(base64_push_encoder(get<string_pusher>(val)));
        return result;
    }
    return get<binary_pusher>(val);
}

array_pusher json::get_array_pusher(const value_pusher& val) {
    return get<array_pusher>(val);
}

object_pusher json::get_object_pusher(const value_pusher& val) {
    return get<object_pusher>(val);
}



namespace json { namespace detail {
    // explicit template instantiation
    template class basic_pusher<string,string,detail::string_pusher_ops>;
    template class basic_pusher<binary,binary,detail::binary_pusher_ops>;
    template class basic_pusher<value_pusher,array,detail::array_pusher_ops>;
    template class basic_pusher<std::pair<string,value_pusher>,object,
                                detail::object_pusher_ops>;
} }
