#include <iomanip>
#include <sstream>

#include <memory>

#include <applog/core.hpp>

#include "pull_encode_json.hpp"
#include "encode.hpp"
#include "encode_json_helpers.hpp"
#include "visit.hpp"


using namespace json;





/**************** class pull_encode_json ****************/

namespace {

    struct pull_base {

        pull_base() : m_complete(false) {}
        virtual ~pull_base() = default;

        pull_base(pull_base&&) = delete;
        pull_base(const pull_base&) = delete;
        pull_base& operator=(pull_base&&) = delete;
        pull_base& operator=(const pull_base&) = delete;

        /** \brief Pull encoded data.
         *
         * Encoded data is either placed within the remaining capacity of
         * the dest string, as indicated by dest_size, and/or returned
         * as a separate string.
         * In the case that both output modes are used, the returned string
         * is to come after the content of dest in the output stream.
         *
         * To avoid reallocation, dest should have at least 24 bytes of
         * available capacity when pull is called for the first time.
         */
        virtual string pull(std::string& dest, unsigned dest_size, 
                            unsigned copy_threshold) = 0;

        inline bool complete() const {
            return m_complete;
        }

    protected:
        bool m_complete;
    };

    // allocate pull_base object to encode value
    std::unique_ptr<pull_base> make_pull(const value_puller& value);

    struct pull_null : public pull_base {
        pull_null() {}
        string pull(std::string& dest, unsigned, unsigned) override {
            dest += "null";
            m_complete = true;
            return string();
        }
    };

    struct pull_boolean : public pull_base {
        pull_boolean(boolean b) : m_bool(b) {}
        string pull(std::string& dest, unsigned, unsigned) override {
            dest += m_bool ? "true" : "false";
            m_complete = true;
            return string();
        }
    private:
        bool m_bool;
    };

    struct pull_integer : public pull_base {
        pull_integer(integer i) : m_int(i) {}
        string pull(std::string& dest, unsigned, unsigned) override {
            std::stringstream ss;
            ss << m_int;
            dest += ss.str();
            m_complete = true;
            return string();
        }
    private:
        integer m_int;
    };

    struct pull_real : public pull_base {
        pull_real(real n) : m_real(n) {}
        string pull(std::string& dest, unsigned, unsigned) override {
            // in addition to 12 digits, there is a sign, '.', 'e', sign, and
            // 3 exponent digits for a maximum of 19 characters
            if (std::isnan(m_real))
                dest += "null";
            else if (std::isinf(m_real))
                dest += std::signbit(m_real) ? "-1e9999" : "1e9999";
            else {
                std::stringstream ss;
                ss << std::setprecision(12) << m_real;
                dest += ss.str();
            }
            m_complete = true;
            return string();
        }
    private:
        double m_real;
    };


    struct pull_string : public pull_base {
        pull_string(const string_puller& stream)
            : m_stream(stream), m_started(false) {
        }
        string pull(std::string& dest, unsigned dest_size, 
                    unsigned copy_threshold) override;

    private:
        string_puller m_stream;
        bool m_started;
        string m_buffer;
    };
}

string pull_string::pull(std::string& dest, unsigned dest_size,
                         unsigned copy_threshold) {
    AR_CHECK(!m_complete);
    if (!m_started) {
        dest += '"';
        m_started = true;
    }
    else if (!m_buffer.empty()) {
        AR_CHECK(dest.size() + m_buffer.size() <= dest_size);
        dest += m_buffer;
        m_buffer.clear();
        if (dest.size() == dest_size)
            return string();  // no room for end quote
    }
    while (auto raw_str = m_stream()) {
        string str = encode_string(*raw_str);
        if (str.size() >= copy_threshold)
            return str;
        if (dest.size() + str.size() > dest_size) {
            m_buffer.swap(str);
            return string();  // no room for m_buffer in dest
        }
        dest += str;
        if (dest.size() == dest_size)
            return string();  // no room for end quote
    }
    AR_CHECK(m_buffer.empty());
    AR_CHECK(dest.size() < dest_size);
    dest += '"';
    m_complete = true;
    return string();
}

namespace {
    struct pull_binary : public pull_base {
        pull_binary(const binary_puller& stream)
            : m_stream(stream), m_started(false) {
        }
        string pull(std::string& dest, unsigned dest_size, 
                    unsigned copy_threshold) override;

    private:
        binary_puller m_stream;
        bool m_started;
        string m_buffer;
        std::basic_string<unsigned char> m_pre_input;
    };
}

string pull_binary::pull(std::string& dest, unsigned dest_size,
                         unsigned copy_threshold) {
    AR_CHECK(!m_complete);
    if (!m_started) {
        dest += '"';
        m_started = true;
    }
    else if (!m_buffer.empty()) {
        AR_CHECK(dest.size() + m_buffer.size() <= dest_size);
        dest += m_buffer;
        m_buffer.clear();
        if (dest.size() + 5 > dest_size)
            return string();  // no room for end data and quote
    }
    while (auto raw_bin = m_stream()) {
        string str = encode_binary(*raw_bin,m_pre_input);
        if (str.size() >= copy_threshold)
            return str;
        if (dest.size() + str.size() > dest_size) {
            m_buffer.swap(str);
            return string();  // no room for m_buffer in dest
        }
        dest += str;
        if (dest.size() + 5 > dest_size)
            return string();  // no room for end data and quote
    }
    AR_CHECK(m_buffer.empty());
    AR_CHECK(dest.size() + 5 <= dest_size);
    dest += finish_binary(m_pre_input);
    dest += '"';
    m_complete = true;
    return string();
}

namespace {
    struct pull_array : public pull_base {
        pull_array(const array_puller& stream)
            : m_stream(stream), m_started(false) {
        }
        string pull(std::string& dest, unsigned dest_size, 
                    unsigned copy_threshold) override;

    private:
        array_puller m_stream;
        bool m_started;
        std::unique_ptr<pull_base> m_value_encoder;
    };
}

string pull_array::pull(std::string& dest, unsigned dest_size,
                        unsigned copy_threshold) {
    AR_CHECK(!m_complete);
    if (!m_started) {
        if (auto val = m_stream())
            m_value_encoder = make_pull(*val);
        dest += '[';
        m_started = true;
    }
    while (m_value_encoder) {
        if (!m_value_encoder->complete()) {
            if (dest.size() + 24 > dest_size)
                return string();  // not enough room for pull from value
            string result = 
                m_value_encoder->pull(dest,dest_size,copy_threshold);
            if (!result.empty() || !m_value_encoder->complete())
                return result;
        }
        if (dest.size() >= dest_size)
            return string();  // not enough room for , or end bracket
        if (auto val = m_stream()) {
            m_value_encoder = make_pull(*val);
            dest += ',';
        }
        else
            break;
    }
    dest += ']';
    m_complete = true;
    return string();
}

namespace {
    struct pull_object : public pull_base {
        pull_object(const object_puller& stream)
            : m_stream(stream), m_started(false) {
        }
        string pull(std::string& dest, unsigned dest_size, 
                    unsigned copy_threshold) override;

    private:
        object_puller m_stream;
        bool m_started;
        std::optional<string> m_key;
        std::unique_ptr<pull_base> m_value_encoder;
    };
}

string pull_object::pull(std::string& dest, unsigned dest_size,
                         unsigned copy_threshold) {
    AR_CHECK(!m_complete);
    if (!m_started) {
        if (auto val = m_stream()) {
            dest += "{\"";
            m_key = encode_string(val->first);
            m_value_encoder = make_pull(val->second);
        }
        else
            dest += '{';
        m_started = true;
    }
    while (m_value_encoder) {
        if (m_key) {
            if (m_key->size() >= copy_threshold) {
                string result;
                result.swap(*m_key);
                return result;
            }
            if (dest.size() + m_key->size() > dest_size)
                return string();
            dest += *m_key;
            if (dest.size() + 2 > dest_size) {
                m_key->clear();
                return string();
            }
            dest += "\":";
            m_key = std::nullopt;
        }
        if (!m_value_encoder->complete()) {
            if (dest.size() + 24 > dest_size)
                return string();  // not enough room for pull from value
            string result = 
                m_value_encoder->pull(dest,dest_size,copy_threshold);
            if (!result.empty() || !m_value_encoder->complete())
                return result;
        }
        if (dest.size() + 2 > dest_size)
            return string();  // not enough room for ," or end bracket
        if (auto val = m_stream()) {
            dest += ",\"";
            m_key = encode_string(val->first);
            m_value_encoder = make_pull(val->second);
        }
        else
            break;
    }
    dest += '}';
    m_complete = true;
    return string();
}

namespace {
    struct pull_visitor {
        using result_type = std::unique_ptr<pull_base>;
        inline result_type operator()(const null_type&) const {
            return std::make_unique<pull_null>();
        }
        inline result_type operator()(boolean val) const {
            return std::make_unique<pull_boolean>(val);
        }
        inline result_type operator()(integer val) const {
            return std::make_unique<pull_integer>(val);
        }
        inline result_type operator()(real val) const {
            return std::make_unique<pull_real>(val);
        }
        inline result_type operator()(const string_puller& val) const {
            return std::make_unique<pull_string>(val);
        }
        inline result_type operator()(const binary_puller& val) const {
            return std::make_unique<pull_binary>(val);
        }
        inline result_type operator()(const array_puller& val) const {
            return std::make_unique<pull_array>(val);
        }
        inline result_type operator()(const object_puller& val) const {
            return std::make_unique<pull_object>(val);
        }
    };

    std::unique_ptr<pull_base> make_pull(const value_puller& value) {
        return visit(pull_visitor{}, value);
    }

    struct pull_encoder {
        
        pull_encoder(const value_puller& stream,
                     unsigned buffer_size,
                     unsigned copy_threshold)
            : m_value_encoder(make_pull(stream)),
              m_buffer_size(buffer_size),
              m_copy_threshold(copy_threshold) {
            AR_CHECK(m_buffer_size >= 32);
            AR_CHECK(m_copy_threshold <= m_buffer_size);
        }

        std::optional<string> operator()();
        
        const std::shared_ptr<pull_base> m_value_encoder;
        const unsigned m_buffer_size;
        const unsigned m_copy_threshold;

        string m_extra;
    };

}

std::optional<string> pull_encoder::operator()() {
    string result;
    if (!m_extra.empty()) {
        result.swap(m_extra);
        return result;
    }
    if (m_value_encoder->complete())
        return {};

    std::string buf;
    buf.reserve(m_buffer_size);
    result = m_value_encoder->pull(buf,m_buffer_size,m_copy_threshold);
    if (!buf.empty()) {
        AR_CHECK(buf.size() <= m_buffer_size);
        m_extra.swap(result);
        result = move(buf);
    }

    return result;
}


string_puller json::pull_encode_json(const value_puller& stream,
                                     unsigned buffer_size,
                                     unsigned copy_threshold) {
    string_puller result;
    result.set_handler(pull_encoder(stream,buffer_size,copy_threshold));
    return result;
}



