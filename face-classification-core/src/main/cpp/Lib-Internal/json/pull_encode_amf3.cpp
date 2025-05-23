
#include <memory>

#include <applog/core.hpp>

#include "pull_encode_amf3.hpp"
#include "amf3_encode_helpers.hpp"
#include "visit.hpp"


using namespace json;


/**************** struct amf3_pull_encoder ****************/

namespace {

    struct amf3_pull_base {

        amf3_pull_base() : m_complete(false) {}
        virtual ~amf3_pull_base() = default;

        amf3_pull_base(amf3_pull_base&&) = delete;
        amf3_pull_base(const amf3_pull_base&) = delete;
        amf3_pull_base& operator=(amf3_pull_base&&) = delete;
        amf3_pull_base& operator=(const amf3_pull_base&) = delete;
        
        /** \brief Pull encoded data.
         *
         * Encoded data is either placed within the remaining capacity of
         * the dest string, as indicated by dest_size, and/or returned
         * as a separate binary object.
         * In the case that both output modes are used, the returned binary
         * is to come after the content of dest in the output stream.
         *
         * To avoid reallocation, dest should have at least 9 bytes of
         * available capacity when pull is called for the first time.
         */
        virtual binary pull(std::string& dest, unsigned dest_size, 
                            unsigned copy_threshold) = 0;
        
        inline bool complete() const {
            return m_complete;
        }

    protected:
        bool m_complete;
    };

    // allocate amf3_pull_base object to encode value
    std::unique_ptr<amf3_pull_base> make_pull(
        const value_puller& value, enc_history& state);

    struct amf3_pull_null : public amf3_pull_base {
        amf3_pull_null() {}
        binary pull(std::string& dest, unsigned, unsigned) override {
            dest += char(1);  // null-marker
            m_complete = true;
            return binary();
        }
    };

    struct amf3_pull_boolean : public amf3_pull_base {
        amf3_pull_boolean(boolean b) : m_bool(b) {}
        binary pull(std::string& dest, unsigned, unsigned) override {
            dest += m_bool 
                ? char(3)   // true-marker
                : char(2);  // false-marker
            m_complete = true;
            return binary();
        }
    private:
        bool m_bool;
    };

    struct amf3_pull_integer : public amf3_pull_base {
        amf3_pull_integer(integer i) : m_int(i) {}
        binary pull(std::string& dest, unsigned, unsigned) override {
            if (const auto x = m_int >> 28) {
                if (x == -1) {
                    dest += char(4);  // integer-marker
                    encode_unsigned(dest, unsigned(m_int) & 0x1fffffff);
                }
                else
                    encode_double(dest, double(m_int));
            }
            else {
                dest += char(4);  // integer-marker
                encode_unsigned(dest, unsigned(m_int));
            }
            m_complete = true;
            return binary();
        }
    private:
        integer m_int;
    };

    struct amf3_pull_real : public amf3_pull_base {
        amf3_pull_real(real n) : m_real(n) {}
        binary pull(std::string& dest, unsigned, unsigned) override {
            encode_double(dest, m_real);
            m_complete = true;
            return binary();
        }
    private:
        double m_real;
    };

    struct amf3_pull_string : public amf3_pull_base {
        amf3_pull_string(const string_puller& stream, enc_history& state)
            : m_stream(stream), m_state(state), m_started(false),
              m_bytes_remaining(0) {
        }
        binary pull(std::string& dest, unsigned dest_size, 
                    unsigned copy_threshold) override;
    private:
        string_puller m_stream;
        enc_history& m_state;
        bool m_started;
        std::size_t m_bytes_remaining;
        string m_buffer;
    };
}

binary amf3_pull_string::pull(std::string& dest, unsigned dest_size,
                              unsigned copy_threshold) {
    AR_CHECK(!m_complete);
    if (!m_started) {
        if (auto size = m_stream.final_size())
            m_bytes_remaining = *size;
        else {
            FILE_LOG(logTRACE) << "amf3_pull_string: pulling to get length";
            m_bytes_remaining = m_stream.pull_size();
        }
        dest += char(6);  // string-marker
        encode_unsigned(dest, (m_bytes_remaining<<1) + 1);
        if (m_bytes_remaining > 0)
            ++m_state.num_strings;
        m_started = true;
    }
    else if (!m_buffer.empty()) {
        AR_CHECK(dest.size() + m_buffer.size() <= dest_size);
        dest += m_buffer;
        m_buffer.clear();
    }
    while (auto str = m_stream()) {
        if (str->size() > m_bytes_remaining) {
            FILE_LOG(logERROR)
                << "amf3_pull_string: too many characters (found " 
                << str->size() << " but expected " << m_bytes_remaining << ")";
            str = str->substr(0,m_bytes_remaining);
        }
        m_bytes_remaining -= str->size();
        if (str->size() >= copy_threshold)
            return binary(move(*str));
        if (dest.size() + str->size() > dest_size) {
            m_buffer.swap(*str);
            return binary();  // dest too small for buffer
        }
        dest += *str;
    }
    AR_CHECK(m_bytes_remaining == 0);
    AR_CHECK(m_buffer.empty());
    m_complete = true;
    return binary();
}

namespace {
    struct amf3_pull_binary : public amf3_pull_base {
        amf3_pull_binary(const binary_puller& stream, enc_history& state)
            : m_stream(stream), m_state(state), m_started(false),
              m_bytes_remaining(0) {
        }
        binary pull(std::string& dest, unsigned dest_size, 
                    unsigned copy_threshold) override;
    private:
        binary_puller m_stream;
        enc_history& m_state;
        bool m_started;
        std::size_t m_bytes_remaining;
        binary m_buffer;
    };
}

binary amf3_pull_binary::pull(std::string& dest, unsigned dest_size,
                              unsigned copy_threshold) {
    AR_CHECK(!m_complete);
    binary result;
    if (!m_started) {
        if (auto size = m_stream.final_size())
            m_bytes_remaining = *size;
        else {
            FILE_LOG(logTRACE) << "amf3_pull_binary: pulling to get length";
            m_bytes_remaining = m_stream.pull_size();
        }
        FILE_LOG(logDETAIL) << "amf3_pull_binary: sending binary of length "
                           << m_bytes_remaining;
        dest += char(0x0c);  // byte-array-marker
        encode_unsigned(dest, (m_bytes_remaining<<1) + 1);
        m_started = true;
    }
    else if (m_buffer.size() != 0) {
        AR_CHECK(dest.size() + m_buffer.size() <= dest_size);
        dest.append(m_buffer.data<char>(), m_buffer.size());
        m_buffer.clear();
    }
    while (auto bin = m_stream()) {
        if (bin->size() > m_bytes_remaining) {
            FILE_LOG(logERROR)
                << "amf3_pull_binary: too many bytes (found " 
                << bin->size() << " but expected " 
                << m_bytes_remaining << ")";
            bin->resize(m_bytes_remaining);
        }
        m_bytes_remaining -= bin->size();
        if (bin->size() >= copy_threshold)
            return *bin;
        if (dest.size() + bin->size() > dest_size) {
            m_buffer.swap(*bin);
            return binary();  // dest too small for buffer
        }
        dest.append(bin->data<char>(), bin->size());
    }
    AR_CHECK(m_bytes_remaining == 0);
    AR_CHECK(m_buffer.size() == 0);
    m_complete = true;
    return binary();
}

namespace {
    struct amf3_pull_array : public amf3_pull_base {
        amf3_pull_array(const array_puller& stream, enc_history& state)
            : m_stream(stream), m_state(state), m_started(false),
              m_elements_remaining(0) {
        }
        binary pull(std::string& dest, unsigned dest_size, 
                    unsigned copy_threshold) override;
    private:
        array_puller m_stream;
        enc_history& m_state;
        bool m_started;
        std::size_t m_elements_remaining;
        std::unique_ptr<amf3_pull_base> m_value_encoder;
    };
}

binary amf3_pull_array::pull(std::string& dest, unsigned dest_size,
                             unsigned copy_threshold) {
    AR_CHECK(!m_complete);
    if (!m_started) {
        size_t n;
        if (auto size = m_stream.final_size())
            n = m_elements_remaining = *size;
        else {
            FILE_LOG(logTRACE) << "amf3_pull_array: pulling to get length";
            n = m_elements_remaining = m_stream.pull_size();
        }
        if (auto val = m_stream()) {
            if (m_elements_remaining > 0) {
                m_value_encoder = make_pull(*val,m_state);
                --m_elements_remaining;
            }
            else 
                FILE_LOG(logERROR) << "amf3_pull_array: too many elements";
        }
        dest += char(9);  // array-marker
        encode_unsigned(dest, (n<<1) + 1);
        dest += char(1);  // empty string (no key/value pairs)
        m_started = true;
    }
    while (m_value_encoder) {
        if (!m_value_encoder->complete()) {
            if (dest.size() + 9 > dest_size)
                return binary();  // not enough room for pull from value
            binary result =
                m_value_encoder->pull(dest,dest_size,copy_threshold);
            if (result.size() > 0 || !m_value_encoder->complete())
                return result;
        }
        m_value_encoder.reset();
        if (auto val = m_stream()) {
            if (m_elements_remaining > 0) {
                m_value_encoder = make_pull(*val,m_state);
                --m_elements_remaining;
            }
            else
                FILE_LOG(logERROR) << "amf3_pull_array: too many elements";
        }
    }
    AR_CHECK(m_elements_remaining == 0);
    m_complete = true;
    return binary();
}

namespace {
    struct amf3_pull_object : public amf3_pull_base {
        amf3_pull_object(const object_puller& stream, enc_history& state)
            : m_stream(stream), m_state(state), m_started(false) {
        }
        binary pull(std::string& dest, unsigned dest_size, 
                    unsigned copy_threshold) override;
    private:
        object_puller m_stream;
        enc_history& m_state;
        bool m_started;
        std::optional<string> m_key;
        std::unique_ptr<amf3_pull_base> m_value_encoder;
    };
}

binary amf3_pull_object::pull(std::string& dest, unsigned dest_size,
                              unsigned copy_threshold) {
    AR_CHECK(!m_complete);
    if (!m_started) {
        if (auto val = m_stream()) {
            AR_CHECK(!val->first.empty() && val->first.size() < (1<<28));
            m_key = val->first;
            m_value_encoder = make_pull(val->second,m_state);
        }
        dest += char(0x0a);  // object-marker
        if (!m_state.base_traits_sent) {
            dest += char(0x0b);  // object traits (dynamic object)
            dest += char(0x01);  // empty string (class-name)
            m_state.base_traits_sent = true;
        }
        else
            dest += char(0x01);  // reference to base traits
        m_started = true;
        if (m_key) {
            enc_history::string_map_type::const_iterator it = 
                m_state.string_map.find(*m_key);
            if (it != m_state.string_map.end()) {
                // encode string as reference to previous string
                encode_unsigned(dest,it->second<<1);
                m_key = std::nullopt;
            }
            else {
                encode_unsigned(dest, (m_key->size()<<1) + 1);
                m_state.string_map[*m_key] = m_state.num_strings++;
            }
        }
    }
    while (m_value_encoder) {
        if (m_key) {
            if (m_key->size() >= copy_threshold) {
                auto result = binary(move(*m_key));
                m_key = std::nullopt;
                return result;
            }
            if (dest.size() + m_key->size() > dest_size) 
                return binary();
            dest += *m_key;
            m_key = std::nullopt;
        }
        if (!m_value_encoder->complete()) {
            if (dest.size() + 9 > dest_size)
                return binary();  // not enough room for pull from value
            binary result =
                m_value_encoder->pull(dest,dest_size,copy_threshold);
            if (result.size() > 0 || !m_value_encoder->complete())
                return result;
        }
        if (dest.size() + 4 > dest_size)
            return binary();  // not enough room for key ref/size
        if (auto val = m_stream()) {
            AR_CHECK(!val->first.empty() && val->first.size() < (1<<28));
            enc_history::string_map_type::const_iterator it = 
                m_state.string_map.find(val->first);
            if (it != m_state.string_map.end()) {
                // encode string as reference to previous string
                encode_unsigned(dest,it->second<<1);
                m_key = std::nullopt;
            }
            else {
                encode_unsigned(dest, (val->first.size()<<1) + 1);
                m_state.string_map[val->first] = m_state.num_strings++;
                m_key = val->first;
            }
            m_value_encoder = make_pull(val->second,m_state);
        }
        else
            break;
    }

    dest += char(0x01);  // empty string (end of object)
    m_complete = true;
    return binary();
}

namespace {
    struct amf3_pull_visitor {
        enc_history& state;

        using result_type = std::unique_ptr<amf3_pull_base>;
        inline result_type operator()(const null_type&) const {
            return std::make_unique<amf3_pull_null>();
        }
        inline result_type operator()(boolean val) const {
            return std::make_unique<amf3_pull_boolean>(val);
        }
        inline result_type operator()(integer val) const {
            return std::make_unique<amf3_pull_integer>(val);
        }
        inline result_type operator()(real val) const {
            return std::make_unique<amf3_pull_real>(val);
        }
        inline result_type operator()(const string_puller& val) const {
            return std::make_unique<amf3_pull_string>(val,state);
        }
        inline result_type operator()(const binary_puller& val) const {
            return std::make_unique<amf3_pull_binary>(val,state);
        }
        inline result_type operator()(const array_puller& val) const {
            return std::make_unique<amf3_pull_array>(val,state);
        }
        inline result_type operator()(const object_puller& val) const {
            return std::make_unique<amf3_pull_object>(val,state);
        }
    };

    std::unique_ptr<amf3_pull_base>
    make_pull(const value_puller& value, enc_history& state) {
        return visit(amf3_pull_visitor{state}, value);
    }

    struct amf3_pull_encoder {

        static const unsigned C_64K = 65536;
        
        amf3_pull_encoder(const value_puller& stream,
                          unsigned buffer_size,
                          unsigned copy_threshold,
                          bool chunk)
            : m_state(std::make_shared<enc_history>()),
              m_value_encoder(make_pull(stream,*m_state)),
              m_buffer_size(buffer_size),
              m_copy_threshold(copy_threshold),
              m_chunk(chunk) {
            AR_CHECK(m_buffer_size >= 16);
            AR_CHECK(!m_chunk || m_buffer_size <= C_64K);
            AR_CHECK(m_copy_threshold <= m_buffer_size);
        }

        std::optional<binary> operator()();
        
        const std::shared_ptr<enc_history> m_state;
        const std::shared_ptr<amf3_pull_base> m_value_encoder;
        const unsigned m_buffer_size;
        const unsigned m_copy_threshold;
        const bool m_chunk;

        binary m_extra;
    };

}

std::optional<binary> amf3_pull_encoder::operator()() {
    if (m_extra.size() > 0) {
        binary result;
        result.swap(m_extra);
        return result;
    }
    if (m_value_encoder->complete())
        return {};

    std::string buffer;
    buffer.reserve(m_buffer_size);
    if (m_chunk)
        buffer.append("\0\0",2);  // reserved chunk size

    m_extra = m_value_encoder->pull(buffer,m_buffer_size,m_copy_threshold);
    AR_CHECK(buffer.size() <= m_buffer_size);

    if (m_chunk) {
        auto len = buffer.size() + m_extra.size() - 2;
        if (len < C_64K) {
            if (len > 0) {
                buffer[0] = char(len >> 8);
                buffer[1] = char(len);
                if (m_value_encoder->complete()) {
                    AR_CHECK(m_extra.size() == 0);
                    buffer.append("\0\0",2);  // end of encode marker
                }
            }
            else
                AR_CHECK(m_value_encoder->complete());
        }
        else {
            AR_CHECK(!m_value_encoder->complete());
            len = buffer.size() - 2;
            AR_CHECK(len < C_64K);
            buffer[0] = char(len >> 8);
            buffer[1] = char(len);
            if (m_extra.size() < C_64K) {
                len = m_extra.size();
                buffer += char(len >> 8);
                buffer += char(len);
            }
            else {
                // have to split m_extra into chunks by copying it
                auto data = m_extra.data<char>();
                auto datalen = m_extra.size();
                std::string dest;
                dest.reserve(datalen + (datalen>>15) + 16);
                do {
                    len = datalen < C_64K ? datalen : (C_64K-1);
                    dest += char(len >> 8);
                    dest += char(len & 0xff);
                    dest.append(data,len);
                    data += len;
                    datalen -= len;
                } while (datalen > 0);
                m_extra = move(dest);
            }
        }
    }
    else if (buffer.empty()) {
        binary result;
        result.swap(m_extra);
        return result;
    }
    return binary(move(buffer));
}


binary_puller json::pull_encode_amf3(const value_puller& stream,
                                     unsigned buffer_size,
                                     unsigned threshold,
                                     bool chunk) {
    binary_puller result;
    result.set_handler(amf3_pull_encoder(stream,buffer_size,threshold,chunk));
    return result;
}
