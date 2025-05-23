
#include <memory>

#include <applog/core.hpp>

#include "pull_encode_cbor.hpp"
#include "visit.hpp"
#include "zlib.hpp"  // for is_compressed()

#include <stdext/bswap.hpp>


using namespace json;


// number of bytes required to encode integer
template <typename T>
static unsigned encoded_size(T x) {
    static_assert(std::is_unsigned_v<T>);
    if (x < 24)
        return 1;
    if (x < 0x100)
        return 2;
    if (x < 0x10000)
        return 3;
    if (x < (uint64_t(1)<<32))
        return 5;
    return 9;
}

// max output is 9 bytes
template <typename T>
static void encode_unsigned(std::string& dest, uint8_t header, T x) {
    static_assert(std::is_unsigned_v<T>);
    if (x < 24)
        dest += char(header + x);
    else if (x < 0x100) {
        dest += char(header + 24);
        dest += char(x);
    }
    else if (x < 0x10000) {
        dest += char(header + 25);
        dest += char(x>>8);
        dest += char(x);
    }
    else if (x < (uint64_t(1)<<32)) {
        dest += char(header + 26);
        for (unsigned i = 4; i > 0; )
            dest += char(x>>(8*--i));
    }
    else { // 8-byte encode
        dest += char(header + 27);
        for (unsigned i = 8; i > 0; )
            dest += char(x >> (8*--i));
    }
}

namespace {

    struct cbor_pull_base {

        cbor_pull_base() : m_complete(false) {}
        virtual ~cbor_pull_base() = default;

        cbor_pull_base(cbor_pull_base&&) = delete;
        cbor_pull_base(const cbor_pull_base&) = delete;
        cbor_pull_base& operator=(cbor_pull_base&&) = delete;
        cbor_pull_base& operator=(const cbor_pull_base&) = delete;
        
        /** \brief Pull encoded data.
         *
         * Encoded data is either placed within the remaining capacity of
         * the dest string, as indicated by dest_size, and/or returned
         * as a separate binary object.
         * In the case that both output modes are used, the returned binary
         * is to come after the content of dest in the output stream.
         *
         * To avoid reallocation, dest must have at least 9 bytes of
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

    // allocate cbor_pull_base object to encode value
    std::unique_ptr<cbor_pull_base> make_pull(const value_puller& value);

    struct cbor_pull_null : public cbor_pull_base {
        cbor_pull_null() {}
        binary pull(std::string& dest, unsigned, unsigned) override {
            dest += char(0xf6); // null special value
            m_complete = true;
            return binary();
        }
    };

    struct cbor_pull_boolean : public cbor_pull_base {
        cbor_pull_boolean(boolean b) : m_bool(b) {}
        binary pull(std::string& dest, unsigned, unsigned) override {
            dest += m_bool 
                ? char(0xf5)  // true special value
                : char(0xf4); // false special value
            m_complete = true;
            return binary();
        }
    private:
        bool m_bool;
    };

    struct cbor_pull_integer : public cbor_pull_base {
        cbor_pull_integer(integer i) : m_int(i) {}
        binary pull(std::string& dest, unsigned, unsigned) override {
            if (0 <= m_int)
                encode_unsigned(dest, 0, uint64_t(m_int));
            else // negative integer
                encode_unsigned(dest, 0x20, uint64_t(-(1+m_int)));
            m_complete = true;
            return binary();
        }
    private:
        integer m_int;
    };

    struct cbor_pull_real : public cbor_pull_base {
        cbor_pull_real(real n) : f64(n), f32(float(n)) {}
        binary pull(std::string& dest, unsigned, unsigned) override {
            if (f32 <= f64 && f64 <= f32) {
                dest += char(0xfa); // float-marker
                static_assert(sizeof(f32) == 4);
                auto* p = reinterpret_cast<const char*>(&f32);
                stdx::copy_be(p, p+4, std::back_inserter(dest));
            }
            else {
                dest += char(0xfb); // double-marker
                static_assert(sizeof(f64) == 8);
                auto* p = reinterpret_cast<const char*>(&f64);
                stdx::copy_be(p, p+8, std::back_inserter(dest));
            }
            m_complete = true;
            return binary();
        }
    private:
        double f64;
        float f32;
    };

    template <typename PULLER>
    struct cbor_pull_bytes : public cbor_pull_base {
        static constexpr auto header =
            std::is_same_v<PULLER,string_puller> ? 0x60 : 0x40;
        cbor_pull_bytes(const PULLER& stream)
            : m_stream(stream), m_started(false), m_bytes_remaining(0) {
        }
        binary pull(std::string& dest, unsigned dest_size, 
                    unsigned copy_threshold) override;
    private:
        PULLER m_stream;
        bool m_started;
        std::optional<std::size_t> m_bytes_remaining;
        typename PULLER::value_type m_buffer;
    };
    using cbor_pull_string = cbor_pull_bytes<string_puller>;
    using cbor_pull_binary = cbor_pull_bytes<binary_puller>;
}

template <typename PULLER>
binary cbor_pull_bytes<PULLER>::pull(std::string& dest, unsigned dest_size,
                                     unsigned copy_threshold) {
    AR_CHECK(!m_complete);
    if (!m_started) {
        m_bytes_remaining = m_stream.final_size();
        if (m_bytes_remaining)
            encode_unsigned(dest, header, *m_bytes_remaining);
        else
            dest += char(header + 0x1f); // indefinite length
        m_started = true;
    }
    else if (!m_buffer.empty()) {
        AR_CHECK(dest.size() + m_buffer.size() < dest_size);
        if (!m_bytes_remaining) // indefinite length -- need length encoded
            encode_unsigned(dest, header, m_buffer.size());
        dest.append(static_cast<const char*>(m_buffer.data()), m_buffer.size());
        m_buffer.clear();
    }
        
    if (m_bytes_remaining) {
        while (auto data = m_stream()) {
            if (data->size() > *m_bytes_remaining) {
                FILE_LOG(logERROR)
                    << "cbor_pull_bytes: too many bytes (found " 
                    << data->size() << " but expected "
                    << *m_bytes_remaining << ")";
                data->resize(*m_bytes_remaining);
            }
            *m_bytes_remaining -= data->size();
            if (data->size() >= copy_threshold)
                return binary(move(*data));
            if (dest.size() + data->size() > dest_size) {
                m_buffer.swap(*data);
                return {}; // dest too small for data
            }
            dest.append(static_cast<const char*>(data->data()), data->size());
        }
        AR_CHECK(*m_bytes_remaining == 0);
    }

    else { // indefinite length
        while (auto data = m_stream()) {
            const auto n = encoded_size(data->size());
            if (data->size() >= copy_threshold ||
                data->size() + n > dest_size) {
                encode_unsigned(dest, header, data->size());
                return binary(move(*data));
            }
            if (dest.size() + data->size() + n > dest_size) {
                m_buffer.swap(*data);
                return binary();  // dest too small for buffer
            }
            encode_unsigned(dest, header, data->size());
            dest.append(static_cast<const char*>(data->data()), data->size());
        }
        dest += char(0xff); // end tag
    }

    AR_CHECK(m_buffer.empty());
    m_complete = true;
    return {};
}

namespace {
    struct cbor_pull_array : public cbor_pull_base {
        cbor_pull_array(const array_puller& stream)
            : m_stream(stream), m_started(false), m_elements_remaining(0) {
        }
        binary pull(std::string& dest, unsigned dest_size, 
                    unsigned copy_threshold) override;
    private:
        array_puller m_stream;
        bool m_started;
        std::optional<std::size_t> m_elements_remaining;
        std::unique_ptr<cbor_pull_base> m_value_encoder;
    };
}

binary cbor_pull_array::pull(std::string& dest, unsigned dest_size,
                             unsigned copy_threshold) {
    AR_CHECK(!m_complete);
    if (!m_started) {
        if (auto val = m_stream()) {
            m_elements_remaining = m_stream.final_size();
            if (m_elements_remaining) {
                AR_CHECK(0 < *m_elements_remaining);
                encode_unsigned(dest, 0x80, *m_elements_remaining);
                --*m_elements_remaining;
            }
            else
                dest += char(0x9f); // indefinite length array
            m_value_encoder = make_pull(*val);
        }
        else {
            dest += char(0x80); // empty array
            m_elements_remaining = 0;
        }
        m_started = true;
    }
    while (m_value_encoder) {
        if (!m_value_encoder->complete()) {
            if (dest.size() + 9 > dest_size)
                return binary();  // not enough room for pull from value
            auto result = m_value_encoder->pull(dest,dest_size,copy_threshold);
            if (!result.empty() || !m_value_encoder->complete())
                return result;
        }
        m_value_encoder.reset();
        if (auto val = m_stream()) {
            if (m_elements_remaining) {
                if (*m_elements_remaining == 0) {
                    FILE_LOG(logERROR) << "cbor_pull_array: too many elements";
                    break;
                }
                --*m_elements_remaining;
            }
            m_value_encoder = make_pull(*val);
        }
        else if (m_elements_remaining)
            AR_CHECK(*m_elements_remaining == 0);
        else // indefinite length array
            dest += char(0xff); // end of array tag
    }
    m_complete = true;
    return binary();
}

namespace {
    struct cbor_pull_object : public cbor_pull_base {
        cbor_pull_object(const object_puller& stream)
            : m_stream(stream), m_started(false) {
        }
        binary pull(std::string& dest, unsigned dest_size, 
                    unsigned copy_threshold) override;
    private:
        object_puller m_stream;
        bool m_started;
        std::optional<std::size_t> m_elements_remaining;
        std::optional<string> m_key;
        std::unique_ptr<cbor_pull_base> m_value_encoder;
    };
}

binary cbor_pull_object::pull(std::string& dest, unsigned dest_size,
                              unsigned copy_threshold) {
    AR_CHECK(!m_complete);
    if (!m_started) {
        if (auto val = m_stream()) {
            m_elements_remaining = m_stream.final_size();
            if (m_elements_remaining) {
                AR_CHECK(0 < *m_elements_remaining);
                encode_unsigned(dest, 0xa0, *m_elements_remaining);
                --*m_elements_remaining;
            }
            else
                dest += char(0xbf); // indefinite length object
            m_key = move(val->first);
            m_value_encoder = make_pull(val->second);
        }
        else {
            dest += char(0xa0); // empty object
            m_elements_remaining = 0;
        }
        m_started = true;
    }
    while (m_value_encoder) {
        if (m_key) {
            const auto n = encoded_size(m_key->size());
            if (m_key->size() >= copy_threshold ||
                m_key->size() + n > dest_size) {
                encode_unsigned(dest, 0x60, m_key->size());
                auto result = binary(move(*m_key));
                m_key = std::nullopt;
                return result;
            }
            if (dest.size() + m_key->size() + n > dest_size) 
                return binary();
            encode_unsigned(dest, 0x60, m_key->size());
            dest += *m_key;
            m_key = std::nullopt;
        }
        if (!m_value_encoder->complete()) {
            if (dest.size() + 9 > dest_size)
                return binary();  // not enough room for pull from value
            auto result = m_value_encoder->pull(dest,dest_size,copy_threshold);
            if (!result.empty() || !m_value_encoder->complete())
                return result;
        }
        m_value_encoder.reset();
        if (auto val = m_stream()) {
            if (m_elements_remaining) {
                if (*m_elements_remaining == 0) {
                    FILE_LOG(logERROR) << "cbor_pull_object: too many members";
                    break;
                }
                --*m_elements_remaining;
            }
            m_key = move(val->first);
            m_value_encoder = make_pull(val->second);
        }
        else if (m_elements_remaining)
            AR_CHECK(*m_elements_remaining == 0);
        else // indefinite length object
            dest += char(0xff); // end of object tag
    }
    m_complete = true;
    return binary();
}

namespace {
    struct cbor_pull_visitor {
        using result_type = std::unique_ptr<cbor_pull_base>;
        inline result_type operator()(const null_type&) const {
            return std::make_unique<cbor_pull_null>();
        }
        inline result_type operator()(boolean val) const {
            return std::make_unique<cbor_pull_boolean>(val);
        }
        inline result_type operator()(integer val) const {
            return std::make_unique<cbor_pull_integer>(val);
        }
        inline result_type operator()(real val) const {
            return std::make_unique<cbor_pull_real>(val);
        }
        inline result_type operator()(const string_puller& val) const {
            return std::make_unique<cbor_pull_string>(val);
        }
        inline result_type operator()(const binary_puller& val) const {
            return std::make_unique<cbor_pull_binary>(val);
        }
        inline result_type operator()(const array_puller& val) const {
            return std::make_unique<cbor_pull_array>(val);
        }
        inline result_type operator()(const object_puller& val) const {
            return std::make_unique<cbor_pull_object>(val);
        }
    };

    std::unique_ptr<cbor_pull_base>
    make_pull(const value_puller& value) {
        return visit(cbor_pull_visitor{}, value);
    }

    struct cbor_pull_encoder {
        cbor_pull_encoder(const value_puller& stream,
                          unsigned buffer_size,
                          unsigned copy_threshold)
            : m_value_encoder(make_pull(stream)),
              m_buffer_size(buffer_size),
              m_copy_threshold(copy_threshold) {
            AR_CHECK(16 <= m_buffer_size);
            AR_CHECK(m_copy_threshold <= m_buffer_size);
        }

        std::optional<binary> operator()();
        
        const std::shared_ptr<cbor_pull_base> m_value_encoder;
        const unsigned m_buffer_size;
        const unsigned m_copy_threshold;
        binary m_extra;
        bool m_started = false;
    };
}

// no-op tag indicating cbor data follows
// if necessary, prepend this value to ensure our cbor encoding
// is not confused with amf3, json or deflate encoding
static constexpr const char cbor_magic[] = {
    char(0xd9), char(0xd9), char(0xf7)
};

std::optional<binary> cbor_pull_encoder::operator()() {
    binary result;
    if (!m_extra.empty()) {
        result.swap(m_extra);
        return result;
    }
    if (m_value_encoder->complete())
        return {};
    std::string buf;
    buf.reserve(m_buffer_size);
    result = m_value_encoder->pull(
        buf, m_buffer_size - (m_started ? 0 : 3), m_copy_threshold);
    AR_CHECK(buf.size() <= m_buffer_size);
    if (!m_started) {
        AR_CHECK(!buf.empty());
        if (buf.size() < 2 || uint8_t(buf.front()) < 128 ||
            is_compressed(buf.data()))
            buf.insert(
                buf.begin(), std::begin(cbor_magic), std::end(cbor_magic));
        m_extra.swap(result);
        result = move(buf);
        m_started = true;
    }
    else if (!buf.empty()) {
        m_extra.swap(result);
        result = move(buf);
    }
    return result;
}


binary_puller json::pull_encode_cbor(const value_puller& stream,
                                     unsigned buffer_size,
                                     unsigned threshold) {
    binary_puller result;
    result.set_handler(cbor_pull_encoder(stream,buffer_size,threshold));
    return result;
}
