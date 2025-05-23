
#include "pull_types.hpp"
#include "encode.hpp"
#include "visit.hpp"

#include <stdext/base64.hpp>

#include <applog/core.hpp>

#include <deque>
#include <memory>
#include <cstddef>


using namespace json;
using namespace json::detail;


struct detail::string_puller_ops {
    template <typename T>
    static std::size_t size(const T& x) {
        return x.size();
    }
    template <typename Iterator>
    static string final(Iterator begin, Iterator end) {
        std::string result;
        for (; begin != end; ++begin)
            result.append(*begin);
        return result;
    }
    static void describe_handler(std::ostream& out) {
        out << "<STRING>";
    }
    template <typename Iterator>
    static void describe_buffer(std::ostream& out, 
                                const std::string& /*indent*/,
                                Iterator begin, Iterator end, bool more) {
        std::string str = *begin;
        if (++begin != end)
            more = true;
        if (detail::manip_max_string >= 0) {
            const auto max_len =
                string::size_type(out.iword(detail::manip_max_string));
            if (max_len > 0 && str.size() > max_len) {
                str.resize(max_len);
                more = true;
            }
        }
        std::string result;
        result.reserve(4 + str.size() + str.size()/10);
        detail::encode_string(result,str);
        out << '"' << result << '"';
        if (more) out << "++";
    }
};

struct detail::binary_puller_ops {
    template <typename T>
    static std::size_t size(const T& x) {
        return x.size();
    }
    template <typename Iterator>
    static binary final(Iterator begin, Iterator end) {
        std::vector<std::byte> buf;
        for (; begin != end; ++begin) {
            const auto d = begin->template data<std::byte>();
            buf.insert(buf.end(), d, d + begin->size());
        }
        return buf;
    }
    static void describe_handler(std::ostream& out) {
        out << "<BINARY>";
    }
    template <typename Iterator>
    static void describe_buffer(std::ostream& out, 
                                const std::string& /*indent*/,
                                Iterator /*begin*/, Iterator /*end*/, 
                                bool /*more*/) {
        out << "<BINARY>";
    }
};

struct detail::array_puller_ops {
    template <typename T>
    static std::size_t size(const T&) {
        return 1;
    }
    template <typename Iterator>
    static array final(Iterator begin, Iterator end) {
        array result;
        for (; begin != end; ++begin)
            result.push_back(begin->pull_final());
        return result;
    }
    static void describe_handler(std::ostream& out) {
        out << "<ARRAY>";
    }
    template <typename Iterator>
    static void describe_buffer(std::ostream& out, const std::string& base,
                                Iterator begin, Iterator end, bool more) {
        if (begin == end) {
            out << (more ? "<ARRAY>" : "[]");
            return;
        }
        long max_len =
            detail::manip_max_array>=0 ? out.iword(detail::manip_max_array) : 0;
        long count = 0;
        out << '[';
        std::string prefix(base);
        if (detail::manip_indent >= 0 && out.pword(detail::manip_indent))
            prefix += static_cast<const char*>(out.pword(detail::manip_indent));
        if (!prefix.empty())
            out << std::endl;
        out << prefix;
        begin->describe(out,prefix);
        for (++begin; begin!=end; ++begin) {
            if (max_len > 0 && ++count >= max_len) {
                more = true;
                break;
            }
            out << ',';
            if (!prefix.empty())
                out << std::endl << prefix;
            begin->describe(out,prefix);
        }
        if (more) {
            out << ',';
            if (!prefix.empty())
                out << std::endl << prefix;
            out << "...";
        }
        if (!prefix.empty())
            out << std::endl << base;
        out << ']';

    }
};

struct detail::object_puller_ops {
    template <typename T>
    static std::size_t size(const T&) {
        return 1;
    }
    template <typename Iterator>
    static object final(Iterator begin, Iterator end) {
        object result;
        for (; begin != end; ++begin)
            result[begin->first] = begin->second.pull_final();
        return result;
    }
    static void describe_handler(std::ostream& out) {
        out << "<OBJECT>";
    }
    template <typename Iterator>
    static void describe_buffer(std::ostream& out, const std::string& base,
                                Iterator begin, Iterator end, bool more) {
        if (begin == end) {
            out << (more ? "<OBJECT>" : "{}");
            return;
        }
        out << '{';
        std::string prefix(base);
        if (detail::manip_indent >= 0 && out.pword(detail::manip_indent))
            prefix += static_cast<const char*>(out.pword(detail::manip_indent));
        if (!prefix.empty())
            out << std::endl << prefix;
        encode(out,begin->first);
        out << ':';
        if (!prefix.empty())
            out << ' ';
        begin->second.describe(out,prefix);
        for (++begin; begin != end; ++begin) {
            out << ',';
            if (!prefix.empty())
                out << std::endl << prefix;
            encode(out,begin->first);
            out << ':';
            if (!prefix.empty())
                out << ' ';
            begin->second.describe(out,prefix);
        }
        if (more) {
            out << ',';
            if (!prefix.empty())
                out << std::endl << prefix;
            out << "...";
        }
        if (!prefix.empty())
            out << std::endl << base;
        out << '}';
    }
};


/**************** class basic_puller ****************/

template <typename T, typename FINAL, class OPS>
struct basic_puller<T,FINAL,OPS>::internal {

    using buffer_type = std::deque<value_type>;
    buffer_type buffer;

    std::unique_ptr<handler_base> handler;

    std::optional<size_type> final_size;
    size_type size_thus_far;
    bool final;


    internal(const std::optional<size_type>& final_size)
        : final_size(final_size),
          size_thus_far(0),
          final(false) {
    }
    void pull_all() {
        if (!handler)
            return;
        while (const auto value = (*handler)()) {
            size_type size = OPS::size(*value);
            AR_CHECK(!final_size || *final_size >= size_thus_far + size);
            buffer.push_back(*value);
            size_thus_far += size;
        }
    }
    void make_final() {
        AR_CHECK(!final_size || *final_size == size_thus_far);
        final = true;
        final_size = size_thus_far;
        handler.reset();
    }
};

template <typename T, typename FINAL, class OPS>
basic_puller<T,FINAL,OPS>::basic_puller(
    const std::optional<size_type>& final_size) 
    : state(std::make_shared<internal>(final_size)) {
}

template <typename T, typename FINAL, class OPS>
basic_puller<T,FINAL,OPS>::basic_puller(const value_type& value, const_t)
    : state(std::make_shared<internal>(OPS::size(value))) {
    state->buffer.push_back(value);
    state->size_thus_far += OPS::size(value);
    state->final = true;
}

template <typename T, typename FINAL, class OPS>
void basic_puller<T,FINAL,OPS>::set_final_size(size_type final_size) {
    AR_CHECK(!state->final_size || *state->final_size == final_size);
    AR_CHECK(state->size_thus_far <= final_size);
    state->final_size = final_size;
}
        
template <typename T, typename FINAL, class OPS>
const std::optional<typename basic_puller<T,FINAL,OPS>::size_type>& 
basic_puller<T,FINAL,OPS>::final_size() const {
    return state->final_size;
}

template <typename T, typename FINAL, class OPS>
bool basic_puller<T,FINAL,OPS>::is_final() const noexcept {
    return state->final;
}

template <typename T, typename FINAL, class OPS>
void basic_puller<T,FINAL,OPS>::set_handler_obj(
    std::unique_ptr<handler_base> handler, bool final) {
    AR_CHECK(!state->final && !state->handler);
    if (final) {
        AR_CHECK(state->final_size);
        state->final = true;
    }
    state->handler = move(handler);
}

template <typename T, typename FINAL, class OPS>
void basic_puller<T,FINAL,OPS>::push_back(
    const std::optional<value_type>& value) {
    AR_CHECK(!state->final);
    AR_CHECK(!state->handler);
    if (value) {
        size_type size = OPS::size(*value);
        AR_CHECK(!state->final_size || 
                  *state->final_size >= state->size_thus_far + size);
        state->buffer.push_back(*value);
        state->size_thus_far += size;
    }
    else
        state->make_final();
}

template <typename T, typename FINAL, class OPS>
std::optional<typename basic_puller<T,FINAL,OPS>::value_type> 
basic_puller<T,FINAL,OPS>::operator()() {
    std::optional<value_type> result;
    if (!state->buffer.empty()) {
        result = state->buffer.front();
        state->buffer.pop_front();
    }
    else if (state->handler) {
        result = (*state->handler)();
        if (result) {
            size_type size = OPS::size(*result);
            AR_CHECK(!state->final_size || 
                      *state->final_size >= state->size_thus_far + size);
            state->size_thus_far += size;
        }
        else 
            state->make_final();
    }
    else if (!state->final)
        state->make_final();
    return result;
}

template <typename T, typename FINAL, class OPS>
typename basic_puller<T,FINAL,OPS>::size_type 
basic_puller<T,FINAL,OPS>::pull_size() {
    if (!state->final) {
        state->pull_all();
        state->make_final();
    }
    return *state->final_size;
}

template <typename T, typename FINAL, class OPS>
typename basic_puller<T,FINAL,OPS>::final_type 
basic_puller<T,FINAL,OPS>::pull_final() {
    state->pull_all();
    state->make_final();
    return OPS::final(state->buffer.begin(),state->buffer.end());
}

template <typename T, typename FINAL, class OPS>
void basic_puller<T,FINAL,OPS>::handler_base::describe(
    std::ostream& out, const std::string&) const {
    OPS::describe_handler(out);
}

template <typename T, typename FINAL, class OPS>
void basic_puller<T,FINAL,OPS>::describe(std::ostream& out,
                                         const std::string& indent) const {
    if (state->buffer.empty() && state->handler) 
        state->handler->describe(out,indent);
    else
        OPS::describe_buffer(out,indent,
                             state->buffer.begin(),state->buffer.end(),
                             !state->final || state->handler);
}




/**************** class array_puller ****************/

struct array_puller::const_array final : public array_puller::handler_base {
    const array arr;
    array::const_iterator begin;
    array::const_iterator end;
    const_array(const array& a)
        : arr(a),
          begin(arr.begin()),
          end(arr.end()) {
    }
    const_array(const array& arr,
                array::const_iterator begin,
                array::const_iterator end)
        : arr(arr),
          begin(begin),
          end(end) {
    }
    std::optional<value_puller> operator()() override {
        if (begin != end)
            return value_puller(*(begin++));
        return {};
    }
    void describe(std::ostream& out, const std::string& indent) const override {
        if (detail::is_simple_array(begin,end))
            encode_array(out,begin,end);
        else
            format_array(out,begin,end,indent);
    }
};

array_puller::array_puller(const array& v)
    : base_type(v.size()) {
    set_handler_obj(std::make_unique<const_array>(v),true);
}

array_puller::array_puller(const array& v,
                           array::const_iterator begin,
                           array::const_iterator end)
    : base_type(std::size_t(end - begin)) {
    set_handler_obj(std::make_unique<const_array>(v,begin,end),true);
}

void array_puller::push_back(const value& v) {
    push_back(value_puller(v));
}


/**************** class object_puller ****************/

struct object_puller::const_object final : public object_puller::handler_base {
    const object obj;
    object::const_iterator begin;
    object::const_iterator end;
    const_object(const object& o)
        : obj(o),
          begin(obj.begin()),
          end(obj.end()) {
    }
    std::optional<value_type> operator()() override {
        if (begin != end)
            return value_type(*(begin++));
        return {};
    }
    void describe(std::ostream& out, const std::string& indent) const override {
        if (detail::is_simple_object(begin,end))
            encode_object(out,begin,end);
        else
            format_object(out,begin,end,indent);
    }
};

object_puller::object_puller(const object& v)
    : base_type(v.size()) {
    set_handler_obj(std::make_unique<const_object>(v),true);
}

void object_puller::push_back(const string& key, const value_puller& value) {
    push_back(value_type(key,value));
}



/**************** class value_puller ****************/

namespace {
    struct puller_from_value {
        inline value_puller_base operator()(const null_type&) const noexcept {
            return null;
        }
        inline value_puller_base operator()(boolean t) const noexcept {
            return t;
        }
        inline value_puller_base operator()(integer t) const noexcept {
            return t;
        }
        inline value_puller_base operator()(real t) const noexcept {
            return t;
        }
        inline value_puller_base operator()(const string& t) const {
            return string_puller(t);
        }
        inline value_puller_base operator()(const binary& t) const {
            return binary_puller(t);
        }
        inline value_puller_base operator()(const array& t) const {
            return array_puller(t);
        }
        inline value_puller_base operator()(const object& t) const {
            return object_puller(t);
        }
    };
}

value_puller::value_puller(const value& v) 
    : val(visit(puller_from_value{},v)) {
}

const char* json::type_name(const value_puller& v) {
    return visit(type_name_visitor{}, v);
}

namespace {
    struct is_final_visitor {
        template <typename T>
        constexpr bool operator()(T, std::enable_if_t<is_json_type<T>::value>* = nullptr) const noexcept {
            return true;
        }
        template <typename T>
        inline bool operator()(const T& x, std::enable_if_t<!is_json_type<T>::value>* = nullptr) const noexcept {
            return x.is_final();
        }
    };
}
bool value_puller::is_final() const noexcept {
    return visit(is_final_visitor{}, *this);
}

namespace {
    struct pull_final_visitor {
        template <typename T>
        inline value operator()(T x, std::enable_if_t<is_json_type<T>::value>* = nullptr) const noexcept {
            return x;
        }
        template <typename T>
        inline value operator()(T& x, std::enable_if_t<!is_json_type<T>::value>* = nullptr) const {
            return x.pull_final();
        }
    };
}
value value_puller::pull_final() {
    return visit(pull_final_visitor{}, *this);
}

namespace {
    struct describe_visitor {
        std::ostream& out;
        std::string indent;

        inline void operator()(const null_type&) const {
            out << "null";
        }
        inline void operator()(boolean t) const {
            out << (t ? "true" : "false");
        }
        template <typename T>
        inline void operator()(const T& t) const {
            out << t;
        }
        inline void operator()(const string_puller& t) const {
            t.describe(out,indent);
        }
        inline void operator()(const binary_puller& t) const {
            t.describe(out,indent);
        }
        inline void operator()(const array_puller& t) const {
            t.describe(out,indent);
        }
        inline void operator()(const object_puller& t) const {
            t.describe(out,indent);
        }
    };
}

void value_puller::describe(std::ostream& out,
                            const std::string& indent) const {
    visit(describe_visitor{out,indent}, *this);
}

std::ostream& json::operator<<(std::ostream& out, const value_puller& stream) {
    stream.describe(out);
    return out;
}


/**************** access and conversion ****************/

real json::make_real(const value_puller& v) {
    return is_type<integer>(v) ? real(get<integer>(v)) : get<real>(v);
}

namespace {
    struct cast_to_string {
        binary_puller stream;
        cast_to_string(const binary_puller& stream) : stream(stream) {}
        std::optional<string> operator()() {
            if (const auto bin = stream())
                return string(bin->data<char>(), bin->size());
            return {};
        }
    };

    struct base64_pull_encoder {
        binary_puller stream;
        unsigned char buffer[4];
        size_t nbuffer;
        base64_pull_encoder(const binary_puller& stream)
            : stream(stream), nbuffer(0) {}
        std::optional<string> operator()() {
            char out_buf[4];
            const auto bin = stream();
            if (!bin) {
                if (nbuffer > 0) {
                    stdx::base64_encode3(out_buf, buffer, nbuffer);
                    nbuffer = 0;
                    return string(out_buf,4);
                }
                return {};
            }
            
            auto src = bin->data<unsigned char>();
            auto src_len = bin->size();

            std::string result;
            result.reserve(4*(src_len+2)/3);

            if (nbuffer > 0) {
                while (nbuffer < 3 && src_len > 0) {
                    buffer[nbuffer++] = *src;
                    ++src;
                    --src_len;
                }
                if (nbuffer < 3)
                    return string(move(result));  // wait for more input (src_len == 0)
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
            return string(move(result));
        }
    };

    struct cast_to_binary {
        string_puller stream;
        cast_to_binary(const string_puller& stream) : stream(stream) {}
        std::optional<binary> operator()() {
            if (auto value = stream())
                return binary(move(*value));
            return {};
        }
    };

    struct base64_pull_decoder {
        string_puller stream;
        std::string buffer;
        base64_pull_decoder(const string_puller& stream) : stream(stream) {}
        std::optional<binary> operator()() {
            unsigned char out_buf[3];
            const auto str = stream();
            if (!str) {
                if (!buffer.empty()) {
                    if (buffer.size() == 1)
                        throw stdx::invalid_base64("invalid base64 string");
                    while (buffer.size() != 4)
                        buffer += '=';
                    auto out_len = stdx::base64_decode3(out_buf,buffer.data());
                    std::basic_string<unsigned char> result(out_buf,out_len);
                    return binary(move(result));
                }
                return {};
            }

            const char* src = str->data();
            size_t src_len = str->size();
            
            std::basic_string<unsigned char> result;
            result.reserve(3*((buffer.size()+src_len+3)/4));
            
            if (!buffer.empty()) {
                while (buffer.size() < 4) {
                    if (src_len == 0)
                        return binary(move(result));
                    buffer += *src;
                    ++src;
                    --src_len;
                }
                result.append(out_buf,stdx::base64_decode3(out_buf,buffer.data()));
                buffer.clear();
            }
            
            while (src_len >= 4) {
                result.append(out_buf,stdx::base64_decode3(out_buf,src));
                src += 4;
                src_len -= 4;
            }
            
            if (src_len > 0)
                buffer.assign(src,src_len);
            
            return binary(move(result));
        }
    };
}

string_puller json::pull_string(const value_puller& stream, 
                                convert_type convert) {
    if (convert != convert_none && is_type<binary_puller>(stream)) {
        string_puller result;
        if (convert == convert_cast)
            result.set_handler(cast_to_string(get<binary_puller>(stream)));
        else  // convert == convert_base64
            result.set_handler(base64_pull_encoder(get<binary_puller>(stream)));
        return result;
    }
    return get<string_puller>(stream);
}

binary_puller json::pull_binary(const value_puller& stream, 
                                convert_type convert) {
    if (convert != convert_none && is_type<string_puller>(stream)) {
        binary_puller result;
        if (convert == convert_cast)
            result.set_handler(cast_to_binary(get<string_puller>(stream)));
        else  // convert == convert_base64
            result.set_handler(base64_pull_decoder(get<string_puller>(stream)));
        return result;
    }
    return get<binary_puller>(stream);
}

array_puller json::pull_array(const value_puller& stream) {
    return get<array_puller>(stream);
}

object_puller json::pull_object(const value_puller& stream) {
    return get<object_puller>(stream);
}


// explicit template instantiation
template class detail::basic_puller<string,string,detail::string_puller_ops>;
template class detail::basic_puller<binary,binary,detail::binary_puller_ops>;
template class detail::basic_puller<value_puller,array,
                                    detail::array_puller_ops>;
template class detail::basic_puller<std::pair<string,value_puller>,object,
                                    detail::object_puller_ops>;


