#pragma once

#include "pull_types.hpp"

namespace json {
    
    /** \brief Streamed json pull encode.
     *
     * During the encoding process, an internal buffer is filled up to, but
     * not exceeding, buffer_size; 
     * however, if an encoded string or binary fragment exceeds copy_threshold,
     * that string will be returned as is (without copy).  
     * Therefore, it is possible that a returned string may exceed buffer_size.
     *
     * An exception thrown while attempting to pull from the stream will be
     * propagated through without corrupting the internal state of the encoder
     * (or ending the stream which threw the exception).
     * Therefore, if there is reason to believe that continuing to pull output
     * after such an exception will yield an appropriate result, encoding may
     * continue. 
     *
     * \param[in] stream value stream to encode
     * \param[in] buffer_size optimial size of output strings
     * \param[in] copy_threshold threshold beyond which attempts are made
     *            to not copy data
     * \return stream to pull output from
     */
    string_puller pull_encode_json(const value_puller& stream,
                                   unsigned buffer_size,
                                   unsigned copy_threshold);
    

}
