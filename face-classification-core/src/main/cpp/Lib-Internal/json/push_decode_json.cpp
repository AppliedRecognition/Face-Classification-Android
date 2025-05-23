
#include "push_decode_json.hpp"

#include <applog/core.hpp>

#include <cctype>
#include <cstdlib>
#include <iomanip>
#include <memory>
#include <sstream>

using namespace json;


namespace {

    struct dummy_predicate {
        inline bool operator()(char) const { return false; }
    };
    struct comma_predicate {
        inline bool operator()(char c) const { return c == ','; }
    };
    struct comma_colon_predicate {
        inline bool operator()(char c) const { return c == ',' || c == ':'; }
    };

    template <typename PRED = dummy_predicate>
    struct whitespace_consumer {
        const PRED pred = {};
        bool comment = false;

        template <typename T>
        bool operator()(T& input) {
            for (;;) {
                if (std::isspace(*input.pos)) {
                    if (*input.pos == '\n')
                        comment = false;
                }
                else if (*input.pos == '#')
                    comment = true;
                else if (!(comment || pred(*input.pos)))
                    return false;
                if (++input.pos == input.data->end()) {
                    input.data = 0;  // all input consumed
                    return true;
                }
            };
        }
    };

    
    class stream_decoder_base {
    public:
        using input_push_type = decoder_input_type;
    
        /** \brief Parse input.
         *
         * If the input string is uninitialized, no more input is
         * available and the parser should either finish up or throw
         * an exception.
         *
         * If, on return, the string is uninitialized, all of the input
         * was consumed and more is expected.  
         * If, however, the pair contains the input string, parsing of
         * the value is complete.  Note that the iterator will point 
         * to the end of the string if all of the input was consumed.
         */
        virtual void push_input(input_push_type& input) = 0;
        
        /** \brief Is decode complete?
         */
        inline bool is_complete() const {
            return m_complete;
        }
        
        virtual ~stream_decoder_base();
        
    protected:
        template <typename U, typename V>
        inline void push(U& stream, const V& value) {
            try {
                stream(value);
            }
            catch (std::exception& e) {
                FILE_LOG(logWARNING) << "push_decode_json: " << e.what();
                if (!m_eh || !(*m_eh)(e))
                    throw;
            }
        }

        template <typename U>
        inline void eos(U& stream) {
            try {
                stream();
            }
            catch (std::exception& e) {
                FILE_LOG(logWARNING) << "push_decode_json: " << e.what();
                if (!m_eh || !(*m_eh)(e))
                    throw;
            }
        }

        stream_decoder_base(detail::exception_handler_base* eh)
            : m_eh(eh) {
        }

        bool m_complete = false;
        detail::exception_handler_base* m_eh;

    private:
        stream_decoder_base(stream_decoder_base&&) = delete;
        stream_decoder_base(const stream_decoder_base&) = delete;
        stream_decoder_base& operator=(stream_decoder_base&&) = delete;
        stream_decoder_base& operator=(const stream_decoder_base&) = delete;
    };


    class value_pusher_decoder : public stream_decoder_base {
    public:
        using value_type = value_pusher;
        using value_handler_type = std::function<void(value_type)>;

        value_pusher_decoder(value_handler_type handler,
                             detail::exception_handler_base* eh);
        virtual ~value_pusher_decoder();

        void push_input(input_push_type& input) override;

        struct push_input_fn {
            const std::shared_ptr<detail::exception_handler_base> eh;
            const std::shared_ptr<value_pusher_decoder> ptr;
            push_input_fn(value_handler_type handler,
                          std::shared_ptr<detail::exception_handler_base> eh)
                : eh(move(eh)),
                  ptr(std::make_shared<value_pusher_decoder>(handler,this->eh.get())) {
            }
            inline void operator()(input_push_type& input) const {
                return ptr->push_input(input);
            }
        };


    private:
        whitespace_consumer<> whitespace;
        std::optional<value_type> m_value;
        value_handler_type m_value_handler;
        std::unique_ptr<stream_decoder_base> m_decoder;
        std::string m_buffer;
    };
}

stream_decoder_base::~stream_decoder_base() {
}


/**************** class string_pusher_decoder ****************/

namespace {
    class string_pusher_decoder : public stream_decoder_base {
    public:
        using value_type = string_pusher;

        inline const value_type& get_value() const {
            return m_value;
        }

        void push_input(input_push_type& input) override;

        string_pusher_decoder(detail::exception_handler_base* eh)
            : stream_decoder_base(eh) {}
        virtual ~string_pusher_decoder();

    private:
        value_type m_value;
        bool m_started = false;
        std::string m_buffer;
    };
}

string_pusher_decoder::~string_pusher_decoder() {
    if (!m_complete)
        FILE_LOG(logERROR) << "push_decode_json: destructed before string complete";
}

void string_pusher_decoder::push_input(input_push_type& input) {
    if (m_complete)
        AR_CHECK(!"push_input called on complete stream");
    if (!input.data)
        throw parse_error("json string decoder failed (close quotes expected)");

    if (input.pos == input.data->end()) {
        input.data = 0;  // empty input -- consumed
        return;
    }
    
    if (!m_started) {
        if (*input.pos != '"')
            throw parse_error("json string decoder failed (open quotes expected)");
        ++input.pos;
        m_started = true;
    }

    if (m_buffer.empty() && input.pos == input.data->begin() && 
        input.data->find_first_of("\\\"") == input.data->npos) {
        // take whole string
        string str(move(*input.data));
        input.data = 0;
        push(m_value,str);
    }
    
    else {
        // decode string (including m_buffer at beginning of string)
        std::string str;
        const auto remain = input.data->end() - input.pos;
        if (remain >= 0)
            str.reserve(std::size_t(remain) + m_buffer.size());
        
        while (input.pos != input.data->end()) {

            if (!m_buffer.empty()) {
                // process m_buffer
                AR_CHECK(m_buffer[0]=='\\');
                do {
                    m_buffer += *input.pos++;
                } while (m_buffer[1]=='u' && m_buffer.size() < 6 && 
                         input.pos != input.data->end());
                
                switch (m_buffer[1]) {
                case 'u': {
                    if (m_buffer.size() != 6) {
                        AR_CHECK(input.pos == input.data->end());
                        input.data = 0;  // all input consumed
                        if (!str.empty())
                            push(m_value,str);
                        return;  // wait for more input
                    }
                    // UTF-8
                    unsigned ch;
                    std::istringstream(m_buffer.substr(2,4)) >> std::hex >> ch;
                    if (ch < 0x0080)
                        str += char(ch);
                    else if (ch < 0x0800) {
                        str += char(0xc0|(ch>>6));
                        str += char(0x80|(ch&0x3f));
                    }
                    else {
                        str += char(0xe0|(ch>>12));
                        str += char(0x80|((ch>>6)&0x3f));
                        str += char(0x80|(ch&0x3f));
                    }
                    break;
                }
                    
                case 'b':
                    str += '\b';
                    break;
                case 'f':
                    str += '\f';
                    break;
                case 'n':
                    str += '\n';
                    break;
                case 'r':
                    str += '\r';
                    break;
                case 't':
                    str += '\t';
                    break;
                    
                default:
                    str += m_buffer[1];
                }
                m_buffer.clear();
            }

            // process string to \ or "
            while (input.pos != input.data->end()) {
                if (*input.pos == '"') {
                    if (!str.empty())
                        push(m_value,move(str));
                    ++input.pos;
                    m_complete = true;
                    eos(m_value);  // end of string signal
                    return;
                }
                else if (*input.pos == '\\') {
                    m_buffer += *input.pos++;
                    break;
                }
                else
                    str += *input.pos++;
            }  
        }

        input.data = 0;  // all input consumed

        if (!str.empty())
            push(m_value,move(str));
    }
}


/**************** class array_pusher_decoder ****************/

namespace {
    class array_pusher_decoder : public stream_decoder_base {
    public:
        using value_type = array_pusher;

        inline const value_type& get_value() const {
            return m_value;
        }

        void push_input(input_push_type& input) override;


        array_pusher_decoder(detail::exception_handler_base* eh)
            : stream_decoder_base(eh) {}
        virtual ~array_pusher_decoder();

    private:
        void handle_value(const value_pusher& value);

        struct handle_value_fn {
            array_pusher_decoder* ptr;
            handle_value_fn(array_pusher_decoder* ptr) : ptr(ptr) {}
            inline void operator()(const value_pusher& value) const {
                ptr->handle_value(value);
            }
        };
        
        value_type m_value;
        bool m_started = false;
        whitespace_consumer<comma_predicate> whitespace;
        std::unique_ptr<value_pusher_decoder> m_value_decoder;
    };
}

array_pusher_decoder::~array_pusher_decoder() {
    if (!m_complete)
        FILE_LOG(logERROR) << "push_decode_json: destructed before array complete";
}

void array_pusher_decoder::push_input(input_push_type& input) {
    if (m_complete)
        AR_CHECK(!"push_input called on complete stream");
    if (!input.data)
        throw std::runtime_error("json array decoder failed (close bracket expected)");

    if (input.pos == input.data->end()) {
        input.data = 0;  // empty input -- consumed
        return;
    }
    
    if (!m_started) {
        if (*input.pos != '[')
            throw std::runtime_error("json array decoder failed (open bracket expected)");
        ++input.pos;
        m_started = true;
    }

    while (input.pos != input.data->end()) {
        if (m_value_decoder) {
            m_value_decoder->push_input(input);
            if (!input.data)
                return;
            m_value_decoder.reset();  // value decode complete
        }
        else {
            // consume whitespace, commas and comments
            if (whitespace(input))
                return;
            if (*input.pos == ']') {
                // end of array
                ++input.pos;
                m_complete = true;
                eos(m_value);  // end of array signal
                return;
            }
            m_value_decoder = std::make_unique<value_pusher_decoder>(
                handle_value_fn(this),m_eh);
        }
    }

    input.data = 0;  // all input consumed
}

void array_pusher_decoder::handle_value(const value_pusher& value) {
    push(m_value,value);
}


/**************** class object_pusher_decoder ****************/

namespace {
    class object_pusher_decoder : public stream_decoder_base {
    public:
        using value_type = object_pusher;

        inline const value_type& get_value() const {
            return m_value;
        }

        void push_input(input_push_type& input) override;


        object_pusher_decoder(detail::exception_handler_base* eh)
            : stream_decoder_base(eh) {}
        virtual ~object_pusher_decoder();

    private:
        void handle_value(const value_pusher& value);

        struct handle_value_fn {
            object_pusher_decoder* ptr;
            handle_value_fn(object_pusher_decoder* ptr) : ptr(ptr) {}
            inline void operator()(const value_pusher& value) const {
                ptr->handle_value(value);
            }
        };

        value_type m_value;
        bool m_started = false;
        whitespace_consumer<comma_colon_predicate> whitespace;
        std::unique_ptr<string_pusher_decoder> m_key_decoder;
        std::unique_ptr<value_pusher_decoder> m_value_decoder;
        bool m_key_complete = false;
    };
}

object_pusher_decoder::~object_pusher_decoder() {
    if (!m_complete)
        FILE_LOG(logERROR) << "push_decode_json: destructed before object complete";
}

void object_pusher_decoder::push_input(input_push_type& input) {
    if (m_complete)
        AR_CHECK(!"push_input called on complete stream");
    if (!input.data)
        throw std::runtime_error("json object decoder failed (close brace expected)");

    if (input.pos == input.data->end()) {
        input.data = 0;  // empty input -- consumed
        return;
    }
    
    if (!m_started) {
        if (*input.pos != '{')
            throw std::runtime_error("json object decoder failed (open brace expected)");
        ++input.pos;
        m_started = true;
    }

    while (input.pos != input.data->end()) {
        if (m_value_decoder) {
            m_value_decoder->push_input(input);
            if (!input.data)
                return;
            m_value_decoder.reset();  // key,value pair decode complete
            m_key_decoder.reset();
        }
        else if (m_key_decoder && !m_key_complete) {
            m_key_decoder->push_input(input);
            if (!input.data)
                return;
            m_key_complete = true;
        }
        else {
            // consume whitespace, commas, colons and comments
            if (whitespace(input))
                return;
            if (*input.pos == '}') {
                // end of object
                ++input.pos;
                m_complete = true;
                eos(m_value);  // end of object signal
                return;
            }
            if (!m_key_decoder) {
                m_key_complete = false;
                m_key_decoder = std::make_unique<string_pusher_decoder>(m_eh);
            }
            else {
                m_value_decoder = std::make_unique<value_pusher_decoder>(
                    handle_value_fn(this),m_eh);
            }
        }
    }

    input.data = 0;  // all input consumed
}

void object_pusher_decoder::handle_value(const value_pusher& value) {
    string key = m_key_decoder->get_value().final_value();
    push(m_value,object_pusher::value_type(key,value));
}


/**************** class value_pusher_decoder ****************/

value_pusher_decoder::value_pusher_decoder(value_handler_type handler,
                                           detail::exception_handler_base* eh)
    : stream_decoder_base(eh),
      m_value_handler(handler) {
}

value_pusher_decoder::~value_pusher_decoder() {
    if (m_value_handler)
        FILE_LOG(logERROR) << "push_decode_json: destructed before value known";
}

void value_pusher_decoder::push_input(input_push_type& input) {
    if (m_complete)
        AR_CHECK(!"push_input called on complete stream");

    if (m_decoder) {
        m_decoder->push_input(input);
        if (input.data)
            m_complete = true;
        return;
    }

    if (m_buffer.empty()) {
        if (!input.data)
            throw std::runtime_error("json value decoder failed (premature end)");
        if (input.pos == input.data->end()) {
            input.data = 0;  // empty input -- consumed
            return;
        }

        // consume initial whitespace and single line comments
        if (whitespace(input))
            return;

        switch (*input.pos) {
        case '"': {
            auto dec = std::make_unique<string_pusher_decoder>(m_eh);
            dec->push_input(input);
            m_value = value_pusher(dec->get_value());
            m_decoder = move(dec);
            if (m_value_handler) {
                m_value_handler(*m_value);
                m_value_handler = nullptr;
            }
            if (input.data)
                m_complete = true;
            return;
        }
        case '[': {
            auto dec = std::make_unique<array_pusher_decoder>(m_eh);
            dec->push_input(input);
            m_value = value_pusher(dec->get_value());
            m_decoder = move(dec);
            if (m_value_handler) {
                m_value_handler(*m_value);
                m_value_handler = nullptr;
            }
            if (input.data)
                m_complete = true;
            return;
        }
        case '{': {
            auto dec = std::make_unique<object_pusher_decoder>(m_eh);
            dec->push_input(input);
            m_value = value_pusher(dec->get_value());
            m_decoder = move(dec);
            if (input.data)
                m_complete = true;
            if (m_value_handler) {
                m_value_handler(*m_value);
                m_value_handler = nullptr;
            }
            return;
        }
        default: // null, false, true or number (parsed below)
            break;
        }
    }

    // copy to buffer
    if (input.data) {
        while (input.pos != input.data->end() &&
               *input.pos != ',' &&
               *input.pos != ']' &&
               *input.pos != '}' &&
               !std::isspace(*input.pos))
            m_buffer += *input.pos++;
        if (input.pos == input.data->end()) {
            input.data = 0;
            return;  // wait for more
        }
    }

    if (m_buffer.empty())
        throw parse_error("json value decoder failed (premature end)");

    // parse value as null, true, false, int or real
    if (m_buffer == "null")
        m_value = value_pusher();
    else if (m_buffer == "false")
        m_value = value_pusher(false);
    else if (m_buffer == "true")
        m_value = value_pusher(true);
    else if (m_buffer[0]=='-' || (m_buffer[0]>='0' && m_buffer[0]<='9')) {
        if (m_buffer.find_first_of(".eE") == m_buffer.npos)
            m_value = value_pusher(std::atol(m_buffer.c_str()));
        else
            m_value = std::atof(m_buffer.c_str());
    }
    else {
        if (m_buffer.size() > 16) 
            m_buffer.resize(16);
        std::stringstream ss;
        for (std::string::const_iterator 
                 it=m_buffer.begin(),end=m_buffer.end(); it!=end; ++it)
            ss << ' ' << std::hex << std::setfill('0') << std::setw(2)
               << unsigned(*it);
        FILE_LOG(logWARNING) << "json: invalid data:" << ss.str();
        throw parse_error("json value decoder failed (invalid json value)");
    }
    
    m_complete = true;
    if (m_value_handler) {
        m_value_handler(*m_value);
        m_value_handler = nullptr;
    }
}


/**************** decode_pusher method ****************/

decoder_input_fn json::detail::push_decode_json(
    decoder_output_fn fn, std::shared_ptr<exception_handler_base> eh) {
    return value_pusher_decoder::push_input_fn(fn,move(eh));
}


