
#include <cmath>
#include <iomanip>
#include <memory>
#include <functional>
#include <sstream>

#include <stdext/bswap.hpp>

#include <applog/core.hpp>

#include "amf3_helpers.hpp"
#include "push_decode_amf3.hpp"


using namespace json;


namespace {

    enum object_type {
        object_static,
        object_dynamic,
        object_externalizable
    };

    struct object_traits {
        using members_type = std::vector<string>;
        string class_name;
        members_type sealed_members;
        object_type type;
        object_traits(object_type type) noexcept : type(type) {}
    };

    struct dec_history {
        std::vector<string> strings;

        std::vector<std::shared_ptr<object_traits> > traits;

        using complex_object = value;  // string, binary, array or object
        std::vector<complex_object> objects;
        const bool store_objects;

        detail::exception_handler_base* eh;

        dec_history(bool store_objects,
                    detail::exception_handler_base* eh)
            : store_objects(store_objects), eh(eh) {}

    private:
        dec_history(const dec_history&);
        dec_history& operator=(const dec_history&);
    };
    
    class decoder_base {
    public:
        using input_push_type = decoder_input_type;

        inline bool is_complete() const {
            return m_complete;
        }
    
        virtual void push_input(input_push_type& input) = 0;

        virtual value_pusher get_value_pusher() const = 0;

        virtual value get_complete_value() const = 0;

        decoder_base(dec_history& state) : m_state(state) {}
        virtual ~decoder_base() = default;
        
    protected:
        template <typename U, typename IT>
        inline void push(U& stream, const IT& begin, const IT& end) {
            try {
                stream(begin,end);
            }
            catch (std::exception& e) {
                FILE_LOG(logWARNING) << "push_decode_amf3: " << e.what();
                if (!m_state.eh || !(*m_state.eh)(e))
                    throw;
            }
        }
        
        template <typename U, typename V>
        inline void push(U& stream, const V& value) {
            try {
                stream(value);
            }
            catch (std::exception& e) {
                FILE_LOG(logWARNING) << "push_decode_amf3: " << e.what();
                if (!m_state.eh || !(*m_state.eh)(e))
                    throw;
            }
        }

        template <typename U>
        inline void eos(U& stream) {
            try {
                stream();
            }
            catch (std::exception& e) {
                FILE_LOG(logWARNING) << "push_decode_amf3: " << e.what();
                if (!m_state.eh || !(*m_state.eh)(e))
                    throw;
            }
        }

        decoder_base(const decoder_base&) = delete;

        dec_history& m_state;
        bool m_complete = false;
    };

    class uint_decoder {
    public:
        using input_push_type = decoder_input_type;
        using value_type = unsigned;

        inline bool is_complete() const {
            return m_complete;
        }

        inline value_type get_unsigned() const {
            return m_value;
        }
        
        void push_input(input_push_type& input);

        uint_decoder() : m_value(0), m_bytes(0) {}

    private:
        bool m_complete = false;
        unsigned m_value;
        unsigned m_bytes;
    };

    class value_decoder : public decoder_base {
    public:
        using value_type = value_pusher;
        using value_handler_type = std::function<void(value_type)>;

        value_pusher get_value_pusher() const override {
            return *m_value;
        }

        value get_complete_value() const override;

        void push_input(input_push_type& input) override;

        void push_number(input_push_type& input);

        value_decoder(value_handler_type handler,
                      dec_history& state)
            : decoder_base(state),
              m_value_handler(handler) {}
        virtual ~value_decoder();

    private:
        std::optional<value_type> m_value;
        value_handler_type m_value_handler;
        std::unique_ptr<decoder_base> m_decoder;
        std::string m_number;
    };

    class top_level_value_decoder : public value_decoder {
    public:
        top_level_value_decoder(value_handler_type handler,
                                bool allow_object_refs,
                                detail::exception_handler_base* eh)
            : value_decoder(handler,m_state), 
              m_state(allow_object_refs,eh) {}

        struct push_input_fn {
            const std::shared_ptr<detail::exception_handler_base> eh;
            const std::shared_ptr<value_decoder> ptr;
            push_input_fn(value_handler_type handler,
                          bool allow_object_refs,
                          std::shared_ptr<detail::exception_handler_base> eh)
                : eh(move(eh)),
                  ptr(std::make_shared<top_level_value_decoder>(
                          handler,allow_object_refs,this->eh.get())) {
            }
            inline void operator()(input_push_type& input) const {
                return ptr->push_input(input);
            }
        };

    private:
        dec_history m_state;
    };
}


/**************** class uint_decoder ****************/

void uint_decoder::push_input(input_push_type& input) {
    if (m_complete)
        AR_CHECK(!"push_input called on complete stream");
    if (!input.data)
        throw parse_error("uint decoder failed (too few bytes)");

    while (input.pos != input.data->end()) {
        if (m_bytes < 3) {
            m_value = (m_value<<7) + (*input.pos & 0x7f);
            if (*input.pos & 0x80) {
                ++m_bytes;
                ++input.pos;
                continue;
            }
        }
        else   // m_bytes == 3
            m_value = (m_value<<8) + (*input.pos & 0xff);

        m_complete = true;
        ++input.pos;
        return;
    }

    input.data = 0;  // all input consumed
}


/**************** class string_decoder ****************/

namespace {
    class string_decoder : public decoder_base {
    public:
        using value_type = string_pusher;
        using complete_handler_type = std::function<void(const string&)>;
        
        value_pusher get_value_pusher() const override {
            return *m_value;
        }

        value get_complete_value() const override {
            AR_CHECK(m_complete);
            return m_complete_string;
        }

        inline const string& get_complete_string() const {
            AR_CHECK(m_complete);
            return m_complete_string;
        }

        void push_input(input_push_type& input) override;

        inline void set_complete_handler(complete_handler_type handler) {
            m_complete_handler = handler;
        }

        string_decoder(dec_history& history, bool construct_value = true)
            : decoder_base(history), m_length(0) {
            if (construct_value)
                m_value = string_pusher();
        }
        virtual ~string_decoder();

    private:
        complete_handler_type m_complete_handler;
        std::optional<value_type> m_value;
        uint_decoder m_length_decoder;
        unsigned m_length;
        std::string m_pending;
        string m_complete_string;
    };
}

string_decoder::~string_decoder() {
    if (!m_complete && m_value)
        FILE_LOG(logERROR) << "push_decode_amf3: destructed before string complete";
}

void string_decoder::push_input(input_push_type& input) {
    if (m_complete)
        AR_CHECK(!"push_input called on complete stream");
    if (!input.data)
        throw parse_error("string decoder failed (unexpected end of stream)");

    if (input.pos == input.data->end()) {
        input.data = 0;  // empty input -- consumed
        return;
    }

    if (!m_length_decoder.is_complete()) {
        m_length_decoder.push_input(input);
        if (!input.data) {
            assert(!m_length_decoder.is_complete());
            return;
        }
        assert(m_length_decoder.is_complete());
        m_length = m_length_decoder.get_unsigned();
        if ((m_length&1) == 0) {
            // reference to previous string
            m_length >>= 1;
            if (m_length >= m_state.strings.size())
                throw parse_error("invalid string reference");
            m_complete_string = m_state.strings[m_length];
            assert(!m_complete_string.empty());
            m_complete = true;
            if (m_value) {
                m_value->set_final_size(m_complete_string.size());
                push(*m_value,m_complete_string);
                eos(*m_value);  // end of string signal
            }
            if (m_complete_handler)
                m_complete_handler(m_complete_string);
            return;
        }

        m_length >>= 1;
        if (m_length == 0) {
            // empty string
            m_complete = true;
            if (m_value)
                eos(*m_value);  // end of string signal
            if (m_complete_handler)
                m_complete_handler(string());
            return;
        }
        else if (m_value)
            m_value->set_final_size(m_length);

        if (input.pos == input.data->end()) {
            input.data = 0;  // input consumed
            return;
        }
    }

    std::string str;
    if (input.pos == input.data->begin() && 
        input.data->size() <= m_length) {
        // take whole string
        str.swap(*input.data);
        input.pos = input.data->end();
    }
    else if (unsigned(input.data->end() - input.pos) <= m_length) {
        // copy remainder of input
        str.assign(input.pos, input.data->end());
        input.pos = input.data->end();
    }
    else {
        // copy part of input
        str.assign(input.pos, input.pos + m_length);
        input.pos += m_length;
    }
    
    assert(!str.empty() && str.size() <= m_length);
    m_length -= unsigned(str.size());

    if (m_pending.empty())
        m_pending = str;
    else
        m_pending.append(str);
    if (m_value)
        push(*m_value,move(str));
    
    if (m_length == 0) {
        m_complete_string = move(m_pending);
        assert(!m_complete_string.empty());
        m_state.strings.push_back(m_complete_string);
        m_complete = true;
        if (m_value)
            eos(*m_value);  // end of string signal
        if (m_complete_handler)
            m_complete_handler(m_complete_string);
    }
    else
        input.data = 0;  // input consumed
}


/**************** class binary_decoder ****************/

namespace {
    class binary_decoder : public decoder_base {
    public:
        using value_type = binary_pusher;
        
        value_pusher get_value_pusher() const override {
            return m_value;
        }

        value get_complete_value() const override {
            AR_CHECK(m_complete && m_state.store_objects);
            return m_complete_binary;
        }

        void push_input(input_push_type& input) override;

        binary_decoder(dec_history& history)
            : decoder_base(history), m_length(0) {}
        virtual ~binary_decoder();

    private:
        value_type m_value;
        uint_decoder m_length_decoder;
        unsigned m_length;
        std::string m_pending;
        binary m_complete_binary;
    };
}

binary_decoder::~binary_decoder() {
    if (!m_complete)
        FILE_LOG(logERROR) << "push_decode_amf3: destructed before binary complete";
}

void binary_decoder::push_input(input_push_type& input) {
    if (m_complete)
        AR_CHECK(!"push_input called on complete stream");
    if (!input.data)
        throw parse_error("binary decoder failed (unexpected end of stream)");

    if (input.pos == input.data->end()) {
        input.data = 0;  // empty input -- consumed
        return;
    }

    if (!m_length_decoder.is_complete()) {
        m_length_decoder.push_input(input);
        if (!input.data) {
            assert(!m_length_decoder.is_complete());
            return;
        }
        assert(m_length_decoder.is_complete());
        m_length = m_length_decoder.get_unsigned();
        if ((m_length&1) == 0) {
            // reference to previous binary
            m_length >>= 1;
            if (m_length >= m_state.objects.size())
                throw parse_error("invalid binary reference");
            if (!is_type<binary>(m_state.objects[m_length]))
                throw parse_error("referenced object has invalid type (expected binary)");
            m_complete_binary = get_binary(m_state.objects[m_length]);
            m_value.set_final_size(m_complete_binary.size());
            push(m_value,m_complete_binary);
            m_complete = true;
            eos(m_value);  // end of binary signal
            return;
        }
        else if (m_length == 1) { // empty binary
            if (m_state.store_objects)
                m_state.objects.push_back(m_complete_binary);
            else
                m_state.objects.push_back(json::null);
            m_value.set_final_size(0);
            push(m_value,m_complete_binary);
            m_complete = true;
            eos(m_value);  // end of binary signal
            return;
        }
        m_length >>= 1;
        m_value.set_final_size(m_length);
        //FILE_LOG(logTRACE) << "amf3_binary_decoder: length " << m_length;

        if (input.pos == input.data->end()) {
            input.data = 0;  // input consumed
            return;
        }
    }

    if (m_length > 0) {
        std::string str;
        if (input.pos == input.data->begin() && 
            input.data->size() <= m_length) {
            // take whole string
            str.swap(*input.data);
            input.pos = input.data->end();
        }
        else if (unsigned(input.data->end() - input.pos) <= m_length) {
            // copy remainder of input
            str.assign(input.pos, input.data->end());
            input.pos = input.data->end();
        }
        else {
            // copy part of input
            str.assign(input.pos, input.pos + m_length);
            input.pos += m_length;
        }

        assert(!str.empty() && str.size() <= m_length);
        m_length -= unsigned(str.size());

        if (m_state.store_objects) {
            if (m_pending.empty())
                m_pending = str;
            else
                m_pending.append(str);
        }
        push(m_value,binary(move(str)));
    }

    if (m_length == 0) {
        m_complete_binary = move(m_pending);
        if (m_state.store_objects)
            m_state.objects.push_back(m_complete_binary);
        else
            m_state.objects.push_back(json::null);
        m_complete = true;
        eos(m_value);  // end of binary signal
    }
    else
        input.data = 0;  // input consumed
}


/**************** class array_decoder ****************/

namespace {
    class array_decoder : public decoder_base {
    public:
        using value_type = array_pusher;

        value_pusher get_value_pusher() const override {
            return m_value;
        }

        value get_complete_value() const override {
            AR_CHECK(m_complete && m_state.store_objects);
            return m_complete_array;
        }

        void push_input(input_push_type& input) override;

        array_decoder(dec_history& state);
        virtual ~array_decoder();

    private:
        void handle_value(const value_pusher& value);

        struct handle_value_fn {
            array_decoder* ptr;
            handle_value_fn(array_decoder* ptr) : ptr(ptr) {}
            inline void operator()(const value_pusher& value) const {
                ptr->handle_value(value);
            }
        };

        const std::size_t m_object_index;
        
        value_type m_value;
        std::unique_ptr<value_decoder> m_value_decoder;

        uint_decoder m_length_decoder;
        unsigned m_length;
        bool m_dense_started = false;
        array m_complete_array;
    };
}

array_decoder::array_decoder(dec_history& state)
    : decoder_base(state),
      m_object_index(state.objects.size()) {
    state.objects.push_back(json::null);
}

array_decoder::~array_decoder() {
    if (!m_complete)
        FILE_LOG(logERROR) << "push_decode_amf3: destructed before array complete";
}

void array_decoder::push_input(input_push_type& input) {
    if (m_complete)
        AR_CHECK(!"push_input called on complete stream");
    if (!input.data)
        throw parse_error("array decoder failed (unexpected end of stream)");
    
    if (input.pos == input.data->end()) {
        input.data = 0;  // empty input -- consumed
        return;
    }

    if (!m_length_decoder.is_complete()) {
        m_length_decoder.push_input(input);
        if (!input.data) {
            assert(!m_length_decoder.is_complete());
            return;
        }
        assert(m_length_decoder.is_complete());
        m_length = m_length_decoder.get_unsigned();
        if ((m_length&1) == 0) {
            // reference to previous array
            m_length >>= 1;
            if (m_length >= m_state.objects.size())
                throw parse_error("invalid array reference");
            if (!is_type<array>(m_state.objects[m_length]))
                throw parse_error("referenced object has invalid type (expected array)");
            const array arr = get_array(m_state.objects[m_length]);
            m_value.set_final_size(arr.size());
            if (!arr.empty())
                push(m_value,arr.begin(),arr.end());
            m_complete = true;
            eos(m_value);  // end of array signal
            return;
        }
        m_length >>= 1;
        m_value.set_final_size(m_length);
        if (input.pos == input.data->end()) {
            input.data = 0;  // input consumed
            return;
        }
    }

    if (!m_dense_started) {
        if (*input.pos != 1) 
            throw parse_error("amf3 array decoder only supports dense arrays");
        m_dense_started = true;
        if (++input.pos == input.data->end()) {
            if (m_length > 0) {
                input.data = 0;  // input consumed
                return;
            }
            else {
                // end of array (empty array)
                if (m_state.store_objects)
                    m_state.objects[m_object_index] = m_complete_array;
                m_complete = true;
                eos(m_value);  // end of array signal
                return;
            }
        }
    }

    while (input.pos != input.data->end()) {
        if (m_value_decoder) {
            m_value_decoder->push_input(input);
            if (!input.data)
                return;
            if (m_state.store_objects)
                m_complete_array.push_back(m_value_decoder->get_complete_value());
            m_value_decoder.reset();  // value decode complete
        }
        if (m_length > 0) {
            m_value_decoder =
                std::make_unique<value_decoder>(handle_value_fn(this),m_state);
            --m_length;
        }
        else {
            // end of array
            if (m_state.store_objects)
                m_state.objects[m_object_index] = m_complete_array;
            m_complete = true;
            eos(m_value);  // end of array signal
            return;
        }
    }

    input.data = 0;  // all input consumed
}

void array_decoder::handle_value(const value_pusher& value) {
    push(m_value,value);
}


/**************** class traits_decoder ****************/

namespace {
    class traits_decoder : public decoder_base {
    public:
        using value_type = object_traits;

        // get referenced complex object (if we found an object reference)
        inline const value& get_referenced_value() const {
            return m_referenced_object;
        }

        inline std::shared_ptr<const value_type> get_traits() const {
            return m_traits;
        }

        void push_input(input_push_type& input) override;

        traits_decoder(dec_history& history)
            : decoder_base(history), m_length(0) {}
        virtual ~traits_decoder() = default;
        
    private:
        void handle_string(const string& str);

        struct handle_string_fn {
            traits_decoder* ptr;
            handle_string_fn(traits_decoder* ptr) : ptr(ptr) {}
            inline void operator()(const string& str) const {
                ptr->handle_string(str);
            }
        };


        value_pusher get_value_pusher() const override {
            AR_CHECK(!"object traits are not a json value");
            return json::null;
        }

        value get_complete_value() const override {
            AR_CHECK(!"object traits are not a json value");
            return json::null;
        }

        uint_decoder m_length_decoder;
        unsigned m_length;
        value m_referenced_object;
        std::shared_ptr<value_type> m_traits;

        std::unique_ptr<string_decoder> m_string_decoder;
        bool m_class_name_decoded = false;
    };
}

void traits_decoder::push_input(input_push_type& input) {
    if (m_complete)
        AR_CHECK(!"push_input called on complete stream");
    if (!input.data)
        throw parse_error("object decoder failed (unexpected end of stream)");

    if (input.pos == input.data->end()) {
        input.data = 0;  // empty input -- consumed
        return;
    }

    if (!m_length_decoder.is_complete()) {
        m_length_decoder.push_input(input);
        if (!input.data) {
            assert(!m_length_decoder.is_complete());
            return;
        }
        assert(m_length_decoder.is_complete());
        m_length = m_length_decoder.get_unsigned();
        
        if (m_length & 1) {
            m_length >>= 1;
            if (m_length & 1) {
                m_length >>= 1;
                if (m_length & 1) {
                    // externalizable
                    m_traits = std::make_shared<object_traits>(
                        object_externalizable);
                    m_state.traits.push_back(m_traits);
                    m_length = 0;  // no sealed members
                    // class name follows
                }
                else {
                    // anonymous, typed or dynamic
                    m_length >>= 1;
                    m_traits = std::make_shared<object_traits>(
                        m_length & 1 ? object_dynamic : object_static);
                    m_state.traits.push_back(m_traits);
                    m_length >>= 1;
                    // class name and sealed member names follow
                }
            }
            else {
                // is traits reference
                m_length >>= 1;
                if (m_length >= m_state.traits.size())
                    throw parse_error("invalid traits reference");
                m_traits = m_state.traits[m_length];
                assert(m_traits);
                m_complete = true;
                return;
            }
        } 
        else {
            // is object reference
            m_length >>= 1;
            if (m_length >= m_state.objects.size()) {
                FILE_LOG(logWARNING) << "object reference " << m_length
                                     << " beyond " << m_state.objects.size();
                throw parse_error("invalid object reference");
            }
            m_referenced_object = m_state.objects[m_length];
            if (m_referenced_object == null)
                throw parse_error("invalid referenced object (possible recursive reference)");
            m_complete = true;
            return;
        }
    }

    // traits to be decoded
    while (input.pos != input.data->end()) {
        if (m_string_decoder) {
            m_string_decoder->push_input(input);
            if (!input.data)
                return;
            m_string_decoder.reset();  // string decode complete
        }
        if (!m_class_name_decoded || m_length > 0) {
            m_string_decoder = std::make_unique<string_decoder>(m_state,false);
            m_string_decoder->set_complete_handler(handle_string_fn(this));
        }
        else {
            // end of traits
            m_complete = true;
            return;
        }
    }

    input.data = 0;  // all input consumed
}

void traits_decoder::handle_string(const string& str) {
    if (!m_class_name_decoded) {
        m_traits->class_name = str;
        m_class_name_decoded = true;
    }
    else {
        m_traits->sealed_members.push_back(str);
        assert(m_length > 0);
        --m_length;
    }
}


/**************** class object_decoder ****************/

namespace {
    class object_decoder : public decoder_base {
    public:

        value_pusher get_value_pusher() const override {
            return m_value_pusher;
        }

        value get_complete_value() const override {
            AR_CHECK(m_complete && m_state.store_objects);
            return m_complete_value;
        }

        void push_input(input_push_type& input) override;

        object_decoder(dec_history& state);
        virtual ~object_decoder();

    private:

        // an incoming amf3 object may be one of:
        //   string stream (externalizable "json.stream.string" object)
        //   binary stream (externalizable "json.stream.binary" object)
        //   array stream (externalizable "json.stream.array" object)
        //   object (standard amf3 anonymous, typed or dynamic object)

        struct internal_base {
            virtual ~internal_base() = default;
            virtual void push_input(
                object_decoder& decoder, input_push_type& input) = 0;
        };

        struct internal_string : public internal_base {
            string_pusher m_value;
            std::string m_pending;
            unsigned m_length;
            std::unique_ptr<uint_decoder> m_length_decoder;

            void push_input(
                object_decoder& decoder, input_push_type& input) override;

            internal_string() : m_length(0) {}
        };

        struct internal_binary : public internal_base {
            binary_pusher m_value;
            std::string m_pending;
            unsigned m_length;
            std::unique_ptr<uint_decoder> m_length_decoder;

            void push_input(
                object_decoder& decoder, input_push_type& input) override;

            internal_binary() : m_length(0) {}
        };

        struct internal_array : public internal_base {
            array_pusher m_value;
            array m_pending;

            std::unique_ptr<value_decoder> m_value_decoder;

            void push_input(
                object_decoder& decoder, input_push_type& input) override;

            void handle_value(
                object_decoder& decoder, const value_pusher& value);
            
            struct handle_value_fn {
                object_decoder& decoder;
                internal_array* ptr;
                handle_value_fn(object_decoder& decoder, 
                                internal_array* ptr) 
                    : decoder(decoder),
                      ptr(ptr) {}
                inline void operator()(const value_pusher& value) const {
                    ptr->handle_value(decoder,value);
                }
            };
        };

        struct internal_object : public internal_base {
            const std::shared_ptr<const object_traits> m_traits;
            object_traits::members_type::const_iterator m_member_iterator;
            
            object_pusher m_value;
            object m_pending;
            
            std::unique_ptr<string_decoder> m_key_decoder;
            std::unique_ptr<value_decoder> m_value_decoder;

            internal_object(const std::shared_ptr<const object_traits> &traits)
                : m_traits(traits) {
                if (m_traits->type == object_static)
                    m_value.set_final_size(m_traits->sealed_members.size());
                // we can't predict the size of a dynamic object
                m_member_iterator = m_traits->sealed_members.begin();
            }

            void push_input(
                object_decoder& decoder, input_push_type& input) override;

            void handle_value(
                object_decoder& decoder, const value_pusher& value);
            
            struct handle_value_fn {
                object_decoder& decoder;
                internal_object* ptr;
                handle_value_fn(object_decoder& decoder, 
                                internal_object* ptr) 
                    : decoder(decoder),
                      ptr(ptr) {}
                inline void operator()(const value_pusher& value) const {
                    ptr->handle_value(decoder,value);
                }
            };
        };

        const std::size_t m_object_index;

        traits_decoder m_traits_decoder;

        value_pusher m_value_pusher;        
        value m_complete_value;

        std::unique_ptr<internal_base> m_internal;
    };
}

object_decoder::object_decoder(dec_history& state)
    : decoder_base(state),
      m_object_index(state.objects.size()),
      m_traits_decoder(state) {
    state.objects.push_back(json::null);
}

object_decoder::~object_decoder() {
    if (!m_complete)
        FILE_LOG(logERROR) << "push_decode_amf3: destructed before object complete";
}

void object_decoder::push_input(input_push_type& input) {
    if (m_complete)
        AR_CHECK(!"push_input called on complete stream");
    if (!input.data)
        throw parse_error("object decoder failed (unexpected end of stream)");

    if (input.pos == input.data->end()) {
        input.data = 0;  // empty input -- consumed
        return;
    }

    if (!m_traits_decoder.is_complete()) {
        m_traits_decoder.push_input(input);
        if (!input.data) {
            assert(!m_traits_decoder.is_complete());
            return;
        }
        assert(m_traits_decoder.is_complete());

        const json::value& referenced_value = 
            m_traits_decoder.get_referenced_value();
        if (referenced_value != null) {
            // reference to saved string, binary, array or object
            m_value_pusher = m_complete_value = referenced_value;
            m_complete = true;
            return;
        }

        const auto traits = m_traits_decoder.get_traits();
        assert(traits);

        if (traits->type == object_externalizable) {
            FILE_LOG(logDETAIL) << "amf3 externalizable object: "
                                << traits->class_name;
            if (traits->class_name == amf3_stream_string) {
                auto obj = std::make_unique<internal_string>();
                m_value_pusher = obj->m_value;
                m_internal = move(obj);
            }
            else if (traits->class_name == amf3_stream_binary) {
                auto obj = std::make_unique<internal_binary>();
                m_value_pusher = obj->m_value;
                m_internal = move(obj);
            }
            else if (traits->class_name == amf3_stream_array) {
                auto obj = std::make_unique<internal_array>();
                m_value_pusher = obj->m_value;
                m_internal = move(obj);
            }
            else
                throw parse_error("unrecognized externalizable object");
        }
        else {
            auto obj = std::make_unique<internal_object>(traits);
            m_value_pusher = obj->m_value;
            m_internal = move(obj);
        }
    }

    assert(m_internal);
    m_internal->push_input(*this,input);
}

void object_decoder::internal_string::push_input(
    object_decoder& decoder, input_push_type& input) {

    while (input.pos != input.data->end()) {
        if (m_length > 0) {
            std::string str;
            if (input.pos == input.data->begin() && 
                input.data->size() <= m_length) {
                // take whole string
                str.swap(*input.data);
                input.pos = input.data->end();
            }
            else if (unsigned(input.data->end() - input.pos) <= m_length) {
                // copy remainder of input
                str.assign(input.pos, input.data->end());
                input.pos = input.data->end();
            }
            else {
                // copy part of input
                str.assign(input.pos, input.pos + m_length);
                input.pos += m_length;
            }

            assert(!str.empty() && str.size() <= m_length);
            m_length -= unsigned(str.size());

            if (decoder.m_state.store_objects) {
                if (m_pending.empty())
                    m_pending = str;
                else
                    m_pending.append(str);
            }
            decoder.push(m_value,str);
        }

        else if (m_length_decoder) {
            m_length_decoder->push_input(input);
            if (!input.data) {
                assert(!m_length_decoder->is_complete());
                return;
            }
            assert(m_length_decoder->is_complete());
            m_length = m_length_decoder->get_unsigned();
            assert(m_length > 0);
            m_length_decoder.reset();
        }
        
        else if (*input.pos)
            m_length_decoder = std::make_unique<uint_decoder>();

        else {
            // end of string
            ++input.pos;
            if (decoder.m_state.store_objects)
                decoder.m_state.objects[decoder.m_object_index] = 
                    decoder.m_complete_value = string(move(m_pending));
            decoder.m_complete = true;
            decoder.eos(m_value);  // end of object signal
        }
    }

    input.data = 0;  // all input consumed
}

void object_decoder::internal_binary::push_input(
    object_decoder& decoder, input_push_type& input) {

    while (input.pos != input.data->end()) {
        if (m_length > 0) {
            std::string str;
            if (input.pos == input.data->begin() && 
                input.data->size() <= m_length) {
                // take whole string
                str.swap(*input.data);
                input.pos = input.data->end();
            }
            else if (unsigned(input.data->end() - input.pos) <= m_length) {
                // copy remainder of input
                str.assign(input.pos, input.data->end());
                input.pos = input.data->end();
            }
            else {
                // copy part of input
                str.assign(input.pos, input.pos + m_length);
                input.pos += m_length;
            }

            assert(!str.empty() && str.size() <= m_length);
            m_length -= unsigned(str.size());

            if (decoder.m_state.store_objects) {
                if (m_pending.empty())
                    m_pending = str;
                else
                    m_pending.append(str);
            }
            decoder.push(m_value,binary(move(str)));
        }

        else if (m_length_decoder) {
            m_length_decoder->push_input(input);
            if (!input.data) {
                assert(!m_length_decoder->is_complete());
                return;
            }
            assert(m_length_decoder->is_complete());
            m_length = m_length_decoder->get_unsigned();
            assert(m_length > 0);
            m_length_decoder.reset();
        }
        
        else if (*input.pos)
            m_length_decoder = std::make_unique<uint_decoder>();

        else {
            // end of string
            ++input.pos;
            if (decoder.m_state.store_objects)
                decoder.m_state.objects[decoder.m_object_index] = 
                    decoder.m_complete_value = binary(move(m_pending));
            decoder.m_complete = true;
            decoder.eos(m_value);  // end of object signal
        }
    }

    input.data = 0;  // all input consumed
}

void object_decoder::internal_array::push_input(
    object_decoder& decoder, input_push_type& input) {

    while (input.pos != input.data->end()) {
        if (m_value_decoder) {
            m_value_decoder->push_input(input);
            if (!input.data)
                return;
            // value decode complete
            if (decoder.m_state.store_objects)
                m_pending.push_back(m_value_decoder->get_complete_value());
            m_value_decoder.reset();
        }

        else if (*input.pos) {
            // start next value
            m_value_decoder = std::make_unique<value_decoder>(
                handle_value_fn(decoder,this),decoder.m_state);
        }

        else {
            // end of array
            ++input.pos;
            if (decoder.m_state.store_objects)
                decoder.m_state.objects[decoder.m_object_index] = 
                    decoder.m_complete_value = m_pending;
            decoder.m_complete = true;
            decoder.eos(m_value);  // end of object signal
            return;
        }
    }    

    input.data = 0;  // all input consumed
}

void object_decoder::internal_array::handle_value(
    object_decoder& decoder, const value_pusher& value) {
    decoder.push(m_value,value);
}

void object_decoder::internal_object::push_input(
    object_decoder& decoder, input_push_type& input) {

    while (input.pos != input.data->end()) {
        if (m_value_decoder) {
            m_value_decoder->push_input(input);
            if (!input.data)
                return;
            // value decode complete
            if (decoder.m_state.store_objects) {
                if (m_key_decoder)
                    m_pending.insert(
                        std::make_pair(m_key_decoder->get_complete_string(),
                                       m_value_decoder->get_complete_value()));
                else
                    m_pending.insert(
                        std::make_pair(*m_member_iterator,
                                       m_value_decoder->get_complete_value()));
            }
            if (!m_key_decoder)
                ++m_member_iterator;
            m_value_decoder.reset();
            m_key_decoder.reset();
        }

        else if (m_key_decoder) {
            m_key_decoder->push_input(input);
            if (!input.data)
                return;
            // key decode complete, start value
            m_value_decoder = std::make_unique<value_decoder>(
                handle_value_fn(decoder,this),decoder.m_state);
        }

        else if (m_member_iterator != m_traits->sealed_members.end()) {
            m_value_decoder = std::make_unique<value_decoder>(
                handle_value_fn(decoder,this),decoder.m_state);
        }

        else if (m_traits->type == object_dynamic) {
            if (*input.pos != 1)
                m_key_decoder =
                    std::make_unique<string_decoder>(decoder.m_state,false);
            else {
                // end of object
                ++input.pos;
                if (decoder.m_state.store_objects)
                    decoder.m_state.objects[decoder.m_object_index] = 
                        decoder.m_complete_value = m_pending;
                decoder.m_complete = true;
                decoder.eos(m_value);  // end of object signal
                return;
            }
        }

        else {
            // end of object
            if (decoder.m_state.store_objects)
                decoder.m_state.objects[decoder.m_object_index] =
                    decoder.m_complete_value = m_pending;
            decoder.m_complete = true;
            decoder.eos(m_value);  // end of object signal
            return;
        }
    }    

    if (m_traits->type == object_dynamic || 
        m_member_iterator != m_traits->sealed_members.end())
        input.data = 0;  // all input consumed

    else {
        // end of object
        if (decoder.m_state.store_objects)
            decoder.m_state.objects[decoder.m_object_index] =
                decoder.m_complete_value = m_pending;
        decoder.m_complete = true;
        decoder.eos(m_value);  // end of object signal
    }
}

void object_decoder::internal_object::handle_value(
    object_decoder& decoder, const value_pusher& value) {
    string key = m_key_decoder ? 
        m_key_decoder->get_complete_string() : *m_member_iterator;
    decoder.push(m_value,object_pusher::value_type(key,value));
}


/**************** class value_decoder ****************/

value_decoder::~value_decoder() {
    if (m_value_handler)
        FILE_LOG(logWARNING) << "value_decoder: destructed before value known";
}

value value_decoder::get_complete_value() const {
    if (m_decoder)
        return m_decoder->get_complete_value();
    AR_CHECK(m_complete && m_value);
    return m_value->final_value();
}

void value_decoder::push_number(input_push_type& input) {
    AR_CHECK(input.data && input.pos != input.data->end());
    
    // integer or double
    do {
        char c = *input.pos++;
        m_number += c;

        if (*m_number.begin() == 4) {
            if (m_number.size() >= 5 ||
                (m_number.size() > 1 && (c&0x80) == 0))
                break;  // integer is complete
        }
        else if (m_number.size() >= 9)
            break;  // double is complete

        if (input.pos == input.data->end()) {
            input.data = 0;
            return;  // wait for more
        }
    } while (true);

    // integer or double is complete
    auto it = m_number.begin();
    if (*it == 4) {
        ++it;
        AR_CHECK(it != m_number.end());
        integer i = *it & 0x7f;
        if (m_number.size() == 5) {
            ++it;
            i = (i<<7) | (*it & 0x7f);
            ++it;
            i = (i<<7) | (*it & 0x7f);
            ++it;
            i = (i<<8) | (*it & 0xff);
            AR_CHECK(++it == m_number.end());
        }
        else {
            const auto end = m_number.end();
            for (++it; it != end; ++it)
                i = (i<<7) | (*it & 0x7f);
        }
        if (i >> 28)
            i = -(i^0x1fffffff)-1;  // sign-extend
        m_value = i;
    }

    else {  // double
        assert(sizeof(double) == 8);
        AR_CHECK(m_number.size() == 9);
        union { double d; char buf[8]; } v;
        stdx::copy_be(m_number.data()+1, m_number.data()+9, v.buf);
        
        // check if double is actually a large integer
        m_value = v.d;
        if (v.d >= (1<<28) || v.d < -(1<<28)) {
            const auto i = integer(v.d);
            if (std::fpclassify(double(i) - v.d) == FP_ZERO)
                m_value = i;
        }
    }
    
    m_complete = true;
    if (m_value_handler) {
        m_value_handler(*m_value);
        m_value_handler = nullptr;
    }
}

void value_decoder::push_input(input_push_type& input) {
    if (m_complete)
        AR_CHECK(!"push_input called on complete stream");

    if (m_decoder) {
        m_decoder->push_input(input);
        if (input.data)
            m_complete = true;
        if (m_value_handler) {
            value_pusher pusher = m_decoder->get_value_pusher();
            if (!is_null(pusher)) {
                m_value = pusher;
                m_value_handler(pusher);
                m_value_handler = nullptr;
            }
        }
        return;
    }

    if (!input.data)
        throw parse_error("AMF3 decoder failed (unexpected end of stream)");
    
    if (input.pos == input.data->end()) {
        input.data = 0;  // empty input -- consumed
        return;
    }

    if (!m_number.empty()) {
        push_number(input);
        return;
    }

    switch (*input.pos) {
    case 0x00:  // undefined
    case 0x01:  // null
        m_value = value_pusher();
        break;
        
    case 0x02:  // false
        m_value = value_pusher(false);
        break;
        
    case 0x03:  // true
        m_value = value_pusher(true);
        break;
        
    case 0x04:  // integer
    case 0x05:  // double
        push_number(input);
        return;

    case 0x06:  // string
        m_decoder = std::make_unique<string_decoder>(m_state);
        break;
        
    case 0x07:  // xml-doc
        throw parse_error("AMF3 decoder failed (xml-doc not supported)");
        
    case 0x08:  // date
        throw parse_error("AMF3 decoder failed (date not supported)");
        
    case 0x09:  // array
        m_decoder = std::make_unique<array_decoder>(m_state);
        break;
        
    case 0x0A:  // object
        m_decoder = std::make_unique<object_decoder>(m_state);
        break;
        
    case 0x0B:  // xml
        throw parse_error("AMF3 decoder failed (xml not supported)");
        
    case 0x0C:  // byte-array
        m_decoder = std::make_unique<binary_decoder>(m_state);
        break;
        
    default: {
        std::stringstream ss;
        unsigned n = 0;
        for (string::const_iterator 
                 it=input.pos; it!=input.data->end() && n<16; ++it,++n)
            ss << ' ' << std::hex << std::setfill('0') << std::setw(2)
               << unsigned(*it);
        FILE_LOG(logWARNING) << "amf3: invalid data:" << ss.str();
        throw parse_error("AMF3 decoder failed (invalid value)");
    }
    }

    ++input.pos;

    if (m_decoder) {
        m_decoder->push_input(input);
        value_pusher pusher = m_decoder->get_value_pusher();
        if (!is_null(pusher))
            m_value = pusher;
    }

    if (input.data)
        m_complete = true;

    if (m_value_handler && m_value) {
        m_value_handler(*m_value);
        m_value_handler = nullptr;
    }
}


/**************** decode_stream method ****************/

decoder_input_fn json::detail::push_decode_amf3(
    decoder_output_fn fn, bool allow_object_refs, 
    std::shared_ptr<exception_handler_base> eh) {
    return top_level_value_decoder::push_input_fn(fn,allow_object_refs,move(eh));
}


