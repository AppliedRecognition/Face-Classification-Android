
#include "zlib.hpp"

#include <applog/core.hpp>

#include <stdext/uninitialized_resize.hpp>
#include <stdext/convert.hpp>

#include <zlib.h>


using namespace json;


binary json::inflate(const void* data, std::size_t size) {
    z_stream strm;
    memset(&strm,0,sizeof(z_stream));
    int r = inflateInit(&strm);
    if (r != Z_OK) {
        FILE_LOG(logERROR) << "zlib inflate init error " << r;
        throw std::runtime_error("zlib inflate init failed");
    }
    strm.next_in = static_cast<Bytef*>(const_cast<void*>(data));
    strm.avail_in = stdx::convert_from(size);
    
    std::vector<std::basic_string<Bytef> > out;
    std::size_t total = 0;
    for (;;) {
        out.emplace_back();
        auto& buf = out.back();
        stdx::uninitialized_resize(buf,size);
        strm.next_out = &*buf.begin();
        strm.avail_out = stdx::convert_from(buf.size());

        const auto r = inflate(&strm,Z_SYNC_FLUSH);
        if (r == Z_STREAM_END) {
            if (strm.avail_in != 0)
                FILE_LOG(logWARNING) << "zlib inflate done with " 
                                     << strm.avail_in 
                                     << " input bytes remaining";
            buf.resize(buf.size() - strm.avail_out);
            total += buf.size();
            break;
        }
        if (r != Z_OK) {
            FILE_LOG(logWARNING) << "zlib inflate error " << r;
            throw std::runtime_error("zlib inflate failed");
        }
        if (strm.avail_out != 0) {
            FILE_LOG(logWARNING) << "zlib inflate error (premature end)";
            throw std::runtime_error("zlib inflate failed");
        }
        total += buf.size();
    }

    if (out.size() > 1) {
        out.front().reserve(total);
        for (std::size_t i = 1; i < out.size(); ++i)
            out.front().append(out[i].begin(), out[i].end());
    }
    return stdx::binary(move(out.front()));
}


/**** pull ****/

namespace {
    struct pull_deflate_obj {

        std::optional<binary> pull();

        pull_deflate_obj(const binary_puller& input, unsigned buffer_size)
            : m_input(input), m_buffer_size(buffer_size), m_done(false) {
            memset(&strm,0,sizeof(z_stream));
            int r = deflateInit(&strm,Z_DEFAULT_COMPRESSION);
            if (r != Z_OK) {
                FILE_LOG(logERROR) << "zlib deflate init error " << r;
                throw std::runtime_error("zlib deflate init failed");
            }
        }
        ~pull_deflate_obj() {
            FILE_LOG(logDETAIL) << "zlib_deflate: total " 
                               << strm.total_in << " in "
                               << strm.total_out << " out";
            deflateEnd(&strm);
        }

    private:
        std::optional<binary_puller> m_input;
        binary m_pending_input;
        const unsigned m_buffer_size;
        bool m_done;
        z_stream strm;
    };

    struct pull_inflate_obj {

        std::optional<string> pull();

        pull_inflate_obj(const binary_puller& input, unsigned buffer_size)
            : m_input(input), m_buffer_size(buffer_size), m_done(false) {
            memset(&strm,0,sizeof(z_stream));
            int r = inflateInit(&strm);
            if (r != Z_OK) {
                FILE_LOG(logERROR) << "zlib inflate init error " << r;
                throw std::runtime_error("zlib inflate init failed");
            }
        }
        ~pull_inflate_obj() {
            FILE_LOG(logDETAIL) << "zlib_inflate: total " 
                               << strm.total_in << " in " 
                               << strm.total_out << " out";
            inflateEnd(&strm);
        }

    private:
        std::optional<binary_puller> m_input;
        binary m_pending_input;
        const unsigned m_buffer_size;
        bool m_done;
        z_stream strm;
    };
}

std::optional<binary> pull_deflate_obj::pull() {
    if (!m_input && m_done)
        return {};

    auto out = stdx::uninitialized_buffer<Bytef>(m_buffer_size);
    strm.next_out = out.get();
    strm.avail_out = m_buffer_size;

    do {
        if (!m_input) {
            int r = deflate(&strm,Z_FINISH);
            if (r == Z_OK) {
                // more output data to come later
                assert(strm.avail_out == 0);
                return binary(move(out),m_buffer_size);
            }
            if (r == Z_STREAM_END) {
                assert(strm.avail_in == 0);
                m_done = true;
                if (strm.avail_out < m_buffer_size)
                    return binary(move(out),m_buffer_size-strm.avail_out);
                return {};
            }
            FILE_LOG(logWARNING) << "zlib deflate error " << r;
            throw std::runtime_error("zlib deflate failed");
        }

        if (m_pending_input.empty()) {
            if (auto bin = (*m_input)()) {
                if (bin->empty())
                    continue;
                m_pending_input = *bin;
                strm.next_in =
                    const_cast<Bytef*>(m_pending_input.data<Bytef>());
                strm.avail_in = stdx::convert_from(m_pending_input.size());
            }
            else {
                m_input = std::nullopt;  // end of input stream
                continue;
            }
        }

        int r = deflate(&strm,Z_NO_FLUSH);
        if (r != Z_OK) {
            FILE_LOG(logWARNING) << "zlib deflate error " << r;
            throw std::runtime_error("zlib deflate failed");
        }

        if (strm.avail_in == 0)
            m_pending_input.clear();
        
        if (strm.avail_out == 0)
            return binary(move(out),m_buffer_size);

        assert(m_pending_input.empty());

    } while (true);
}

std::optional<string> pull_inflate_obj::pull() {
    if (!m_input && m_done)
        return {};
    
    std::string out;
    stdx::uninitialized_resize(out,m_buffer_size);
    strm.next_out = reinterpret_cast<Bytef*>(&*out.begin());
    strm.avail_out = stdx::convert_from(out.size());

    do {
        if (!m_input) {
            int r = inflate(&strm,Z_SYNC_FLUSH);
            if (r == Z_OK) {
                // more output data to come later
                if (strm.avail_out != 0) {
                    FILE_LOG(logWARNING) << "zlib inflate error (premature end)";
                    throw std::runtime_error("zlib inflate failed");
                }
                return string(move(out));
            }
            if (r == Z_STREAM_END) {
                if (strm.avail_in != 0)
                    FILE_LOG(logWARNING) << "zlib inflate done with " 
                                         << strm.avail_in 
                                         << " input bytes remaining";
                m_done = true;
                out.resize(m_buffer_size - strm.avail_out);
                if (!out.empty())
                    return string(move(out));
                return {};
            }
            FILE_LOG(logWARNING) << "zlib inflate error " << r;
            throw std::runtime_error("zlib inflate failed");
        }

        if (m_pending_input.empty()) {
            if (auto bin = (*m_input)()) {
                if (bin->empty())
                    continue;
                m_pending_input = *bin;
                strm.next_in =
                    const_cast<Bytef*>(m_pending_input.data<Bytef>());
                strm.avail_in = stdx::convert_from(m_pending_input.size());
            }
            else {
                m_input = std::nullopt;  // end of input stream
                continue;
            }
        }

        int r = inflate(&strm,Z_SYNC_FLUSH);
        if (r != Z_OK) {
            if (r == Z_STREAM_END) {
                if (strm.avail_in != 0)
                    FILE_LOG(logWARNING) << "zlib inflate done with " 
                                         << strm.avail_in 
                                         << " input bytes remaining";
                else if (auto bin = (*m_input)())
                    FILE_LOG(logWARNING) << "zlib inflate done with " 
                                         << bin->size()
                                         << " input bytes remaining";
                m_input = std::nullopt;
                m_done = true;
                out.resize(m_buffer_size - strm.avail_out);
                if (!out.empty())
                    return string(move(out));
                return {};
            }
            FILE_LOG(logWARNING) << "zlib inflate error " << r;
            throw std::runtime_error("zlib inflate failed");
        }

        if (strm.avail_in == 0)
            m_pending_input.clear();
        
        if (strm.avail_out == 0)
            return string(move(out));

        assert(m_pending_input.empty());

    } while (true);
}

binary_puller json::pull_deflate(
    const binary_puller& input, unsigned buffer_size) {
    auto obj = std::make_shared<pull_deflate_obj>(input,buffer_size);
    binary_puller result;
    result.set_handler(std::bind(&pull_deflate_obj::pull,obj));
    return result;
}

string_puller json::pull_inflate_string(
    const binary_puller& input, unsigned buffer_size) {
    auto obj = std::make_shared<pull_inflate_obj>(input,buffer_size);
    string_puller result;
    result.set_handler(std::bind(&pull_inflate_obj::pull,obj));
    return result;
}


/**** push ****/

namespace {
    struct push_inflate_obj {

        void push(const std::optional<binary>&);

        push_inflate_obj(const string_pusher& output, unsigned buffer_size)
            : m_output(output), m_buffer_size(buffer_size), m_done(false) {
            memset(&strm,0,sizeof(z_stream));
            int r = inflateInit(&strm);
            if (r != Z_OK) {
                FILE_LOG(logERROR) << "zlib inflate init error " << r;
                throw std::runtime_error("zlib inflate init failed");
            }
        }
        ~push_inflate_obj() {
            FILE_LOG(logDETAIL) << "zlib_inflate: total " 
                               << strm.total_in << " in " 
                               << strm.total_out << " out";
            inflateEnd(&strm);
        }

    private:
        string_pusher m_output;
        const unsigned m_buffer_size;
        std::string m_buffer;
        bool m_done;
        z_stream strm;
    };
}

void push_inflate_obj::push(const std::optional<binary>& bin) {
    assert(strm.avail_in == 0);
    if (m_done) {
        if (bin && !bin->empty())
            FILE_LOG(logWARNING) << "zlib inflate done with " 
                                 << bin->size() 
                                 << " input bytes remaining";
        return;
    }
    
    if (bin) {
        if (bin->empty()) return;
        strm.next_in = const_cast<Bytef*>(bin->data<Bytef>());
        strm.avail_in = stdx::convert_from(bin->size());
    }

    do {
        if (strm.avail_out == 0) {
            stdx::uninitialized_resize(m_buffer,m_buffer_size);
            strm.next_out = reinterpret_cast<Bytef*>(&*m_buffer.begin());
            strm.avail_out = stdx::convert_from(m_buffer.size());
        }

        int r = inflate(&strm, Z_SYNC_FLUSH);
        if (r != Z_OK) {
            if (r == Z_STREAM_END) {
                if (strm.avail_in != 0)
                    FILE_LOG(logWARNING) << "zlib inflate done with " 
                                         << strm.avail_in 
                                         << " input bytes remaining";
                m_buffer.resize(m_buffer_size - strm.avail_out);
                if (!m_buffer.empty())
                    m_output(move(m_buffer));
                m_output();
                m_done = true;
                return;
            }
            FILE_LOG(logWARNING) << "zlib inflate error " << r;
            throw std::runtime_error("zlib inflate failed");
        }
        
        if (strm.avail_out == 0) {
            m_output(move(m_buffer));
            if (!bin) continue;
        }

        else if (!bin) {
            FILE_LOG(logWARNING) << "zlib inflate error (premature end)";
            throw std::runtime_error("zlib inflate failed");
        }
        
        if (strm.avail_in == 0)
            return;
        
        assert(strm.avail_out == 0);
    }
    while (true);
}

binary_pusher json::push_inflate(
    const string_pusher& output, unsigned buffer_size) {
    auto obj = std::make_shared<push_inflate_obj>(output,buffer_size);
    binary_pusher result;
    result.set_value_handler(
        std::bind(&push_inflate_obj::push,obj,std::placeholders::_1));
    return result;
}


/* note: do not write to log from the push deflate methods since
 * comm::pgp_sink uses this.
 */

namespace {
    struct push_deflate_obj {

        void push(const std::optional<binary>&);

        push_deflate_obj(binary_push_function output, 
                         unsigned buffer_size, bool sync)
            : m_output(output), 
              m_buffer_size(buffer_size), 
              m_sync(sync), 
              m_done(false) {
            memset(&strm,0,sizeof(z_stream));
            int r = deflateInit(&strm,Z_DEFAULT_COMPRESSION);
            if (r != Z_OK) 
                throw std::runtime_error("zlib deflate init failed");
        }
        ~push_deflate_obj() {
            deflateEnd(&strm);
        }

    private:
        binary_push_function m_output;
        const unsigned m_buffer_size;
        const bool m_sync;
        stdx::uninitialized_buffer<Bytef> m_buffer;
        bool m_done;
        z_stream strm;
    };
}

void push_deflate_obj::push(const std::optional<binary>& bin) {
    assert(strm.avail_in == 0);
    assert(!m_done);

    // note: no logging in this method because comm::pgp_sink uses it
    
    if (bin) {
        if (bin->empty()) return;
        strm.next_in = const_cast<Bytef*>(bin->data<Bytef>());
        strm.avail_in = stdx::convert_from(bin->size());
    }

    do {
        if (strm.avail_out == 0) {
            m_buffer = stdx::uninitialized_buffer<Bytef>(m_buffer_size);
            strm.next_out = m_buffer.get();
            strm.avail_out = m_buffer_size;
        }

        if (!bin) {
            // end of input
            int r = deflate(&strm,Z_FINISH);
            if (r == Z_OK) {
                // more output data to come
                assert(strm.avail_out == 0);
                m_output(binary(move(m_buffer),m_buffer_size));
                continue;
            }
            if (r == Z_STREAM_END) {
                if (strm.avail_out < m_buffer_size)
                    m_output(
                        binary(move(m_buffer),m_buffer_size-strm.avail_out));
                m_output(std::nullopt);
                m_done = true;
                return;
            }
            throw std::runtime_error("zlib deflate failed");
        }
        
        else {
            int r = deflate(&strm, m_sync ? Z_PARTIAL_FLUSH : Z_NO_FLUSH);
            if (r != Z_OK)
                throw std::runtime_error("zlib deflate failed");

            if (strm.avail_out == 0)
                m_output(binary(move(m_buffer),m_buffer_size));

            if (strm.avail_in == 0)
                return;
            
            assert(strm.avail_out == 0);
        }
    }
    while (true);
}

binary_push_function json::push_deflate_fn(
    binary_push_function output, unsigned buffer_size, bool sync) {
    auto obj = std::make_shared<push_deflate_obj>(output,buffer_size,sync);
    return std::bind(&push_deflate_obj::push,obj,std::placeholders::_1);
}

binary_pusher json::push_deflate(
    const binary_pusher& output, unsigned buffer_size) {
    auto obj = std::make_shared<push_deflate_obj>(output,buffer_size,false);
    binary_pusher result;
    result.set_value_handler(
        std::bind(&push_deflate_obj::push,obj,std::placeholders::_1));
    return result;
}
