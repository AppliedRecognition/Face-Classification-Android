
#include "push_decode_cbor.hpp"

#include <applog/core.hpp>

#include <stdext/bswap.hpp>

#include <cctype>
#include <cstdlib>
#include <iomanip>
#include <memory>
#include <sstream>

using namespace json;


// number of bytes required to decode token
static unsigned token_size(uint8_t header) {
    header &= 0x1f;
    if (header < 24) return 1;
    switch (header) {
    case 24: return 2;
    case 25: return 3;
    case 26: return 5;
    case 27: return 9;
    case 31: return 1;
    }
    throw std::runtime_error("invalid cbor token");
}

// decode uint64_t from token
static uint64_t token_unsigned(uint8_t const* data) {
    auto len = unsigned(*data) & 0x1fu;
    if (len < 24) return len;
    switch (len) {
    case 24: return data[1];
    case 25: len = 2; break;
    case 26: len = 4; break;
    case 27: len = 8; break;
    default: throw std::runtime_error("invalid cbor token");
    }
    uint64_t x = 0;
    for (unsigned i = 1; i <= len; ++i)
        x = (x<<8) + data[i];
    return x;
}

// decode element count from token
// returns a non-negative count or -1 for indefinite
static int64_t element_count(uint8_t const* data) {
    if ((*data & 0x1fu) == 0x1f)
        return -1;
    const auto n = int64_t(token_unsigned(data));
    if (n < 0)
        throw std::runtime_error(
            "cbor element count too large (does not fit int64_t");
    return n;
}

namespace {
    class decoder_base {
        decoder_base(decoder_base&&) = delete;
        decoder_base(const decoder_base&) = delete;
        decoder_base& operator=(decoder_base&&) = delete;
        decoder_base& operator=(const decoder_base&) = delete;

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

        virtual ~decoder_base() = default;

    protected:
        template <typename U, typename V>
        inline void push(U& stream, V&& value) {
            try {
                stream(std::forward<V>(value));
            }
            catch (std::exception& e) {
                FILE_LOG(logWARNING) << "push_decode_cbor: " << e.what();
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
                FILE_LOG(logWARNING) << "push_decode_cbor: " << e.what();
                if (!m_eh || !(*m_eh)(e))
                    throw;
            }
        }

        decoder_base(detail::exception_handler_base* eh) : m_eh(eh) {}

        bool m_complete = false;
        detail::exception_handler_base* m_eh;
    };


    class value_decoder : public decoder_base {
    public:
        using value_type = value_pusher;
        using value_handler_type = std::function<void(value_type)>;

        value_decoder(value_handler_type handler,
                      detail::exception_handler_base* eh)
            : decoder_base(eh),
              m_value_handler(handler) {
            m_buffer.reserve(9);
        }

        ~value_decoder() {
            if (m_value_handler)
                FILE_LOG(logERROR) << "push_decode_cbor: destructed before value known";
        }

        void push_input(input_push_type& input) override;

        inline auto take_final() {
            if (!m_value)
                throw std::runtime_error("cbor_value_decoder: value not known");
            return m_value->take_final();
        }

    private:
        std::optional<value_type> m_value;
        value_handler_type m_value_handler;
        std::unique_ptr<decoder_base> m_decoder;
        std::vector<uint8_t> m_buffer;
    };
}


/**************** class bytes_decoder ****************/

namespace {
    // for definite length string or binary
    class bytes_decoder : public decoder_base {
    public:
        bytes_decoder(string_pusher pusher, uint64_t length,
                      detail::exception_handler_base* eh,
                      bool send_eos = true)
            : decoder_base(eh), m_remaining(length),
              m_pusher(move(pusher)), m_send_eos(send_eos) {}
        ~bytes_decoder() {
            if (!m_complete)
                FILE_LOG(logERROR) << "push_decode_cbor: destructed before string complete";
        }

        void push_input(input_push_type& input) override;

    private:
        uint64_t m_remaining;
        string_pusher m_pusher;
        bool m_send_eos;
    };
}

void bytes_decoder::push_input(input_push_type& input) {
    if (m_complete)
        AR_CHECK(!"push_input called on complete stream");
    if (m_remaining == 0) {
        m_complete = true;
        if (m_send_eos)
            eos(m_pusher); // end of string signal
        return;
    }
    if (!input.data)
        throw std::runtime_error(
            "cbor bytes decoder failed (more input expected)");
    const auto avail = std::size_t(input.data->end() - input.pos);
    if (avail <= m_remaining) {
        if (avail == 0) {
            input.data = nullptr;
            return;
        }
        // move input string
        if (input.pos != input.data->begin())
            input.data->erase(input.data->begin(), input.pos);
        m_remaining -= input.data->size();
        push(m_pusher,move(*input.data));
        if (m_remaining) {
            input.data = nullptr; // input consumed, need more
            return;
        }
        *input.data = {};
        input.pos = input.data->end(); // input consumed, decode done
    }
    else { // m_remaining < avail
        const auto start = input.pos;
        input.pos += int64_t(m_remaining);
        m_remaining = 0;
        push(m_pusher,std::string(start,input.pos));
    }
    m_complete = true;
    if (m_send_eos)
        eos(m_pusher); // end of string signal
}


/**************** class chunk_decoder ****************/

namespace {
    // for indefinite length string or binary
    class chunk_decoder : public decoder_base {
    public:
        chunk_decoder(string_pusher pusher, detail::exception_handler_base* eh)
            : decoder_base(eh), m_pusher(std::move(pusher)) {
            m_buffer.reserve(9);
        }
        ~chunk_decoder() {
            if (!m_complete)
                FILE_LOG(logERROR) << "push_decode_cbor: destructed before string complete";
        }

        void push_input(input_push_type& input) override;

    private:
        string_pusher m_pusher;
        std::unique_ptr<bytes_decoder> m_decoder;
        std::vector<uint8_t> m_buffer;
    };
}

void chunk_decoder::push_input(input_push_type& input) {
    if (m_complete)
        AR_CHECK(!"push_input called on complete stream");
    if (!input.data)
        throw std::runtime_error(
            "cbor bytes decoder failed (chunk data expected)");
    while (input.pos != input.data->end()) {
        if (m_decoder) {
            m_decoder->push_input(input);
            if (!input.data)
                return;
            m_decoder.reset(); // chunk decode complete
        }
        else if (uint8_t(*input.pos) == 0xffu) {
            // end of indefinite string
            ++input.pos;
            m_complete = true;
            eos(m_pusher);  // end of string signal
            return;
        }
        else { // decoding length for next chunk
            if (m_buffer.empty()) {
                m_buffer.push_back(uint8_t(*input.pos));
                ++input.pos;
            }
            const auto needed = token_size(m_buffer.front());
            while (m_buffer.size() < needed) {
                if (input.pos == input.data->end()) {
                    input.data = nullptr; // wait for more input
                    return;
                }
                m_buffer.push_back(uint8_t(*input.pos));
                ++input.pos;
            }
            const auto type = unsigned(m_buffer.front()) >> 5;
            if (type < 2 || 3 < type)
                throw std::runtime_error("expected cbor chunk");
            const auto len = token_unsigned(m_buffer.data());
            m_decoder = std::make_unique<bytes_decoder>(
                m_pusher, len, m_eh, false);
            m_buffer.clear();
        }
    }
    input.data = nullptr;  // all input consumed
}


/**************** class array_decoder ****************/

namespace {
    class array_decoder : public decoder_base {
    public:
        array_decoder(int64_t length, detail::exception_handler_base* eh)
            : decoder_base(eh), m_remaining(length) {}
        ~array_decoder() {
            if (!m_complete)
                FILE_LOG(logERROR) << "cbor_array_decoder: destructed before array complete";
        }

        inline const auto& pusher() const {
            return m_pusher;
        }

        void push_input(input_push_type& input) override;

    private:
        int64_t m_remaining; // -1 for indefinite
        array_pusher m_pusher;
        std::unique_ptr<value_decoder> m_decoder;
    };
}

void array_decoder::push_input(input_push_type& input) {
    if (m_complete)
        AR_CHECK(!"push_input called on complete stream");
    if (m_remaining == 0 && !m_decoder) {
        m_complete = true;
        eos(m_pusher);  // end of array signal
        return;
    }
    if (!input.data)
        throw std::runtime_error(
            "cbor array decoder failed (value expected)");

    while (input.pos != input.data->end()) {
        if (m_decoder) {
            m_decoder->push_input(input);
            if (!input.data)
                return;
            m_decoder.reset(); // element decode complete
            if (0 == m_remaining) {
                m_complete = true;
                eos(m_pusher);  // end of array signal
                return;
            }
        }
        else if (0 < m_remaining) {
            --m_remaining;
            m_decoder = std::make_unique<value_decoder>(
                [this](const auto& value) { push(m_pusher,value); },
                m_eh);
        }
        else if (0 == m_remaining) {
            m_complete = true;
            eos(m_pusher);  // end of array signal
            return;
        }
        else if (uint8_t(*input.pos) == 0xffu) {
            // end of indefinite array
            ++input.pos;
            m_complete = true;
            eos(m_pusher);  // end of array signal
            return;
        }
        else // next element in indefinite array
            m_decoder = std::make_unique<value_decoder>(
                [this](const auto& value) { push(m_pusher,value); },
                m_eh);
    }

    input.data = nullptr;  // all input consumed
}


/**************** class object_decoder ****************/

namespace {
    class object_decoder : public decoder_base {
    public:
        object_decoder(int64_t length, detail::exception_handler_base* eh)
            : decoder_base(eh), m_remaining(length) {}
        ~object_decoder() {
            if (!m_complete)
                FILE_LOG(logERROR) << "push_decode_cbor: destructed before object complete";
        }

        inline const auto& pusher() const {
            return m_pusher;
        }

        void push_input(input_push_type& input) override;

    private:
        int64_t m_remaining; // -1 for indefinite
        object_pusher m_pusher;
        std::unique_ptr<value_decoder> m_key_decoder;
        std::unique_ptr<value_decoder> m_value_decoder;
    };
}

void object_decoder::push_input(input_push_type& input) {
    if (m_complete)
        AR_CHECK(!"push_input called on complete stream");
    if (m_remaining == 0 && !m_key_decoder) {
        m_complete = true;
        eos(m_pusher);  // end of object signal
        return;
    }
    if (!input.data)
        throw std::runtime_error(
            "cbor object decoder failed (key/value expected)");
    while (input.pos != input.data->end()) {
        if (m_value_decoder) {
            m_value_decoder->push_input(input);
            if (!input.data)
                return;
            m_value_decoder.reset(); // key and value decode complete
            m_key_decoder.reset();
            if (0 == m_remaining) {
                m_complete = true;
                eos(m_pusher);  // end of object signal
                return;
            }
        }
        else if (m_key_decoder) {
            m_key_decoder->push_input(input);
            if (!input.data)
                return;
            auto key = m_key_decoder->take_final();
            m_value_decoder = std::make_unique<value_decoder>(
                [this, k = move(get_string(key))](const auto& value) {
                    push(m_pusher, object_pusher::value_type(k,value));
                },
                m_eh);
        }
        else if (0 < m_remaining) {
            --m_remaining;
            m_key_decoder = std::make_unique<value_decoder>(
                [](const auto&) noexcept {}, m_eh);
        }
        else if (0 == m_remaining) {
            m_complete = true;
            eos(m_pusher);  // end of object signal
            return;
        }
        else if (uint8_t(*input.pos) == 0xffu) {
            // end of indefinite array
            ++input.pos;
            m_complete = true;
            eos(m_pusher);  // end of object signal
            return;
        }
        else // next element in indefinite object
            m_key_decoder = std::make_unique<value_decoder>(
                [](const auto&) noexcept {}, m_eh);
    }

    input.data = nullptr;  // all input consumed
}


/**************** class value_decoder ****************/

void value_decoder::push_input(input_push_type& input) {
    if (m_complete)
        AR_CHECK(!"push_input called on complete stream");

    if (m_decoder) {
        m_decoder->push_input(input);
        if (input.data)
            m_complete = true;
        return;
    }

    if (!input.data)
        throw parse_error("CBOR decoder failed (unexpected end of stream)");

    while (input.pos != input.data->end()) {
        if (m_buffer.empty()) {
            m_buffer.push_back(uint8_t(*input.pos));
            ++input.pos;
        }
        const auto needed = token_size(m_buffer.front());
        while (m_buffer.size() < needed) {
            if (input.pos == input.data->end()) {
                input.data = nullptr;  // empty input -- consumed
                return;
            }
            m_buffer.push_back(uint8_t(*input.pos));
            ++input.pos;
        }

        const auto type = unsigned(m_buffer.front()) >> 5;
        switch (type) {

        case 6: // tagged data item -- ignore the tag
            m_buffer.clear();
            continue;

        case 0: // unsigned integer
            m_value = json::integer(token_unsigned(m_buffer.data()));
            break;
        case 1: // negative integer
            m_value = -1 - json::integer(token_unsigned(m_buffer.data()));
            break;

        case 7: { // simple, special or floating point value
            const auto arg = unsigned(m_buffer.front()) & 0x1fu;
            switch (arg) {
            case 20: m_value = false; break;
            case 21: m_value = true;  break;
            case 22: m_value = json::null; break;
            case 23: m_value = json::null; break;

            case 25: throw std::runtime_error(
                "support for cbor float16 not implemented");

            case 26: {
                float f;
                stdx::copy_be(m_buffer.data() + 1, m_buffer.data() + 5,
                              reinterpret_cast<uint8_t*>(&f));
                m_value = f;
                break;
            }
            case 27: {
                double f;
                stdx::copy_be(m_buffer.data() + 1, m_buffer.data() + 9,
                              reinterpret_cast<uint8_t*>(&f));
                m_value = f;
                break;
            }
            default:
                throw std::runtime_error("unknown cbor simple value");
            }
            break;
        }

        case 2: // binary
        case 3: { // or string
            if (type == 2)
                m_value = binary_pusher{};
            else
                m_value = string_pusher{};
            if ((m_buffer.front() & 0x1fu) != 0x1fu) {
                auto dec = std::make_unique<bytes_decoder>(
                    get_string_pusher(*m_value, convert_cast),
                    token_unsigned(m_buffer.data()), m_eh);
                dec->push_input(input);
                m_decoder = move(dec);
            }
            else { // indefinite length
                auto dec = std::make_unique<chunk_decoder>(
                    get_string_pusher(*m_value, convert_cast), m_eh);
                dec->push_input(input);
                m_decoder = move(dec);
            }
            break;
        }

        case 4: { // array
            auto dec = std::make_unique<array_decoder>(
                element_count(m_buffer.data()), m_eh);
            dec->push_input(input);
            m_value = dec->pusher();
            m_decoder = move(dec);
            break;
        }

        case 5: { // map
            auto dec = std::make_unique<object_decoder>(
                element_count(m_buffer.data()), m_eh);
            dec->push_input(input);
            m_value = dec->pusher();
            m_decoder = move(dec);
            break;
        }

        } // switch

        if (m_value_handler) {
            m_value_handler(*m_value);
            m_value_handler = nullptr;
        }
        if (input.data)
            m_complete = true;
        return;
    }

    input.data = nullptr;  // empty input -- consumed
}


/**************** decode_pusher method ****************/

decoder_input_fn json::detail::push_decode_cbor(
    decoder_output_fn fn, std::shared_ptr<exception_handler_base> eh) {
    return [
        eh, ptr = std::make_shared<value_decoder>(fn, eh.get())
        ](auto& input) {
        return ptr->push_input(input);
    };
}
