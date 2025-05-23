#pragma once

#include "types.hpp"
#include <stdext/options_tuple.hpp>
#include <stdext/stdio.hpp>

namespace json {
    /** \brief Output format selection.
     */
    struct cbor_tag;
    using cbor_option = stdx::option_bool<cbor_tag>;
    const cbor_option cbor{true};

    struct amf3_tag;
    using amf3_option = stdx::option_bool<amf3_tag>;
    const amf3_option amf3{true};

    struct json_tag;
    using json_option = stdx::option_bool<json_tag>;
    const json_option json{true};

    struct deflate_tag;
    using deflate_option = stdx::option_bool<deflate_tag>;
    const deflate_option deflate{true};


    namespace internal {
        /* These are the internal implementation methods and not meant
         * to be called directly.  Use the versions outside the internal
         * namespace below.
         */
        value load(stdx::file_ptr file, const std::string& path);
        void save(const value&, FILE* outfile, const std::string& path,
                  const stdx::options_tuple<cbor_option,amf3_option,json_option,deflate_option>& opts);
    }


    /** \brief Load value from file.
     *
     * The format of the file is auto detected.
     */
    template <typename PATH>
    inline std::enable_if_t<stdx::is_fopen_path_v<PATH>, value>
    load(const PATH& path) {
        return internal::load(stdx::fopen_rb(path),stdx::generic_string(path));
    }


    /** \brief Write value to file.
     *
     * If no options are specified, then an attempt is made to determine
     * the desired format from the path extension.  If this fails, the
     * default format is cbor without compression.  Recognized extensions
     * are: ".cbor", ".cbor.gz", ".json", ".json.gz", ".amf3" and ".amf3.gz".
     *
     * An exception is throw if more than one of
     * cbor, amf3 or json is requested.
     */
    template <typename PATH, typename... Opts>
    inline std::enable_if_t<stdx::is_fopen_path_v<PATH> >
    save(const value& v, const PATH& path, Opts&&... opts) {
        internal::save(v, stdx::fopen_wb(path).get(),
                       stdx::generic_string(path),
                       { std::forward<Opts>(opts)... });
    }
}
