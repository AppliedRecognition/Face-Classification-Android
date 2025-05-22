#pragma once

#include <stdext/binary.hpp>
#include <stdext/options_tuple.hpp>

namespace json {
    class value;
}
namespace rec {
    /** \brief Binary serialization format.
     */
    enum class serialize_type {
        def = 0, raw = 1, json = 2, amf3 = 3, cbor = 4
    };
    static constexpr auto raw = serialize_type::raw;
    static constexpr auto cbor = serialize_type::cbor;
    static constexpr auto amf3 = serialize_type::amf3;
    static constexpr auto json = serialize_type::json;

    /** \brief Binary serialization compression.
     */
    enum class compression_type { def = 0, uncompressed = 1, deflate = 2 };
    static constexpr auto uncompressed = compression_type::uncompressed;
    static constexpr auto deflate = compression_type::deflate;

    /** \brief Serialize to binary.
     *
     * See the object specific to_binary_with_opts() to find the
     * default options.
     */
    template <typename T, typename... Opts>
    inline auto to_binary(const T& obj, Opts&&... opts) {
        return to_binary_with_opts(obj, { std::forward<Opts>(opts)... });
    }


    /** \brief Serialize json value to binary.
     *
     * Don't use this method directly.
     * Use to_binary() instead (above).
     *
     * Default is deflate compressed amf3.
     * Note that raw is the same as amf3.
     */
    stdx::binary to_binary_with_opts(
        const json::value& val,
        const stdx::options_tuple<serialize_type,compression_type>& opts);
}
