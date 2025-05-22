#pragma once

#include "types.hpp"


namespace {

    /** \name Class name for open-ended string stream.
     *
     * An open-ended (no predetermined finite length) string stream is
     * sent as an externalizable object with the specified class name.
     * The data is described as:
     *   <code>*(U29 *(UTF8-char)) U29-zero</code>,
     * where <code>U29</code> is the length of the following block and
     * <code>U29-zero</code> is a single zero byte.
     */
    const json::string amf3_stream_string("amf3.stream.string");

    /** \name Class name for open-ended binary stream.
     *
     * An open-ended (no predetermined finite length) binary stream is
     * sent as an externalizable object with the specified class name.
     * The data is described as:
     *   <code>*(U29 *(U8)) U29-zero</code>,
     * where <code>U29</code> is the length of the following block and
     * <code>U29-zero</code> is a single zero byte.
     */
    const json::string amf3_stream_binary("amf3.stream.binary");

    /** \name Class name for open-ended array stream.
     *
     * An open-ended (no predetermined finite length) array stream is
     * sent as an externalizable object with the specified class name.
     * The data is described as:
     *   <code>*(value-type) end-of-array-marker</code>,
     * where <code>end-of-array-marker</code> is a single zero byte
     * (same as an undefined value).
     */
    const json::string amf3_stream_array("amf3.stream.array");
}

