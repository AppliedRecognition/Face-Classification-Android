#pragma once

#include "push_decode.hpp"

namespace json {
    namespace detail {
        decoder_input_fn push_decode_cbor(
            decoder_output_fn fn, std::shared_ptr<exception_handler_base> eh);
    }

    /** \brief Decode cbor byte stream to value stream.
     *
     * The following rules apply to the argument passed to the returned input
     * function:<ul>
     *  <li>If the input string pointer is NULL, no more input is
     *      available and the parser will either finish up or throw
     *      an exception.</li>
     *  <li>If, after the input function returns, the string pointer is NULL,
     *      all of the input was consumed and more is expected.</li>
     *  <li>If, on the other hand, the input string pointer remains non-NULL,
     *      parsing of the value is complete.  Note that the iterator will
     *      point to the end of the string if all of the input was
     *      consumed.</li>
     * </ul>
     *
     * Note that in the case of an error during parsing, the output function
     * may never be called.
     *
     * \param[in] fn function to call when output value_pusher is ready
     * \return function to call to provide input data
     */
    inline decoder_input_fn push_decode_cbor(decoder_output_fn fn) {
        return detail::push_decode_cbor(fn,nullptr);
    }

    /** \brief Decode cbor byte stream to value stream.
     *
     * The following rules apply to the argument passed to the returned input
     * function:<ul>
     *  <li>If the input string pointer is NULL, no more input is
     *      available and the parser will either finish up or throw
     *      an exception.</li>
     *  <li>If, after the input function returns, the string pointer is NULL,
     *      all of the input was consumed and more is expected.</li>
     *  <li>If, on the other hand, the input string pointer remains non-NULL,
     *      parsing of the value is complete.  Note that the iterator will
     *      point to the end of the string if all of the input was
     *      consumed.</li>
     * </ul>
     *
     * Note that in the case of an error during parsing, the output function
     * may never be called.
     *
     * If an exception of type std::exception is caught while trying to push
     * a value into a stream, the provided exception handler is called.
     * It is not called when an error occurs in the decoding of the input
     * data, nor is it used when calling the output function.
     * The handler must be capable of accepting an argument which is a
     * reference to either of these exception types.
     *
     * The handler function should return true to indicate that the exception
     * was handled and false to have the exception re-thrown.
     *
     * \param[in] fn function to call when output value_pusher is ready
     * \param[in] h function to handle exceptions when pushing
     * \return function to call to provide input data
     */
    template <typename HANDLER>
    inline decoder_input_fn push_decode_cbor(decoder_output_fn fn, HANDLER h) {
        return detail::push_decode_cbor(
            fn, std::make_shared<detail::exception_hander_obj<HANDLER> >(h));
    }
}

