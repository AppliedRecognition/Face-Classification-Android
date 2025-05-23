#pragma once

namespace json {

    /** \brief Type of streaming allowed by encoder.
     *
     * This option controls how attempts to do open-ended (total length not
     * pre-determined) streaming of string, binary or array. 
     */
    enum amf3_stream_type {
        amf3_stream_buffer,      ///< buffer streamed data
        amf3_stream_extension,   ///< use stream extensions
        amf3_stream_none         ///< throw exception on attempt to stream
    };
}
