#pragma once

#include "pull_types.hpp"

namespace json {
    
    /** \brief Streamed cbor pull encode.
     *
     * During the encoding process, an internal buffer is filled up to, but
     * not exceeding, buffer_size; 
     * however, if an string or binary fragment exceeds copy_threshold,
     * that data will be returned as is (without copy).  
     * Therefore, it is possible that a returned binary may exceed buffer_size.
     *
     * An exception thrown while attempting to pull from the stream will be
     * propagated through.
     * This will most likely corrupt the internal state of the encoder so one
     * must not continue with the encode.
     *
     * \param[in] stream value stream to encode
     * \param[in] buffer_size optimial size of output strings
     * \param[in] copy_threshold threshold beyond which attempts are made
     *            to not copy data
     * \return stream to pull output from
     */
    binary_puller pull_encode_cbor(const value_puller& stream,
                                   unsigned buffer_size,
                                   unsigned copy_threshold);
    

}
