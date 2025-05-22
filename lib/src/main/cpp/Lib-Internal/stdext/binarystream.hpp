#pragma once

#include "binary.hpp"
#include <istream>
#include <streambuf>

namespace stdx {

    /** \brief An std::streambuf to read from an stdx::binary object.
     */
    template <typename CharT>
    class basic_binarybuf : public std::basic_streambuf<CharT> {
        binary m_bin;
    public:
        basic_binarybuf(stdx::binary b) : m_bin(std::move(b)) {
            const auto p = const_cast<CharT*>(m_bin.data<CharT>());
            this->setg(p, p, p + m_bin.size());
        }
    };

    /** \brief An std::istream to read from an stdx::binary object.
     */
    template <typename CharT>
    class basic_binarystream : public std::basic_istream<CharT> {
        basic_binarybuf<CharT> m_buf;
    public:
        basic_binarystream(stdx::binary b)
            : std::istream(&m_buf),
              m_buf(std::move(b)) {
            this->rdbuf(&m_buf);
        }
    };
    using binarystream = basic_binarystream<char>;
}
