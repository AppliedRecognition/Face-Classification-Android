#pragma once

#include "amf3_stream_type.hpp"
#include "pull_types.hpp"


namespace json {
    
    /** \brief Streamed amf3 pull encode.
     *
     * During the encoding process, an internal buffer is filled up to, but
     * not exceeding, buffer_size; 
     * however, if an string or binary fragment exceeds copy_threshold,
     * that data will be returned as is (without copy).  
     * Therefore, it is possible that a returned binary may exceed buffer_size.
     *
     * With chunk encoding, the encoded value is split into chunks with
     * each chunk having a non-zero 2-byte (short) big-endian unsigned
     * integer prefix indicating its length.
     * The last chunk of the data will be followed by the 2-byte zero integer.
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
     * \param[in] chunk use chucked encoding
     * \return stream to pull output from
     */
    binary_puller pull_encode_amf3(
        const value_puller& stream,
        unsigned buffer_size,
        unsigned copy_threshold,
        bool chunk = false);
    

}
