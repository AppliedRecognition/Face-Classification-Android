#pragma once

#include <iomanip>
#include <ostream>

namespace json {

    namespace detail {
        extern int manip_max_string;
        extern int manip_binary_subst;
        extern int manip_indent;
        extern int manip_max_array;
    }

    /** \name IO Manipulators
     */
    //@{
    /// set the maximum length of string to encode
    class max_string {
    public:
        max_string(long length) : length(length) {}
        long length;
    };
    std::ostream& set_max_string(std::ostream& strm, long length = 0);
    inline std::ostream& operator<<(std::ostream& out, const max_string& obj) {
        return set_max_string(out,obj.length);
    }

    /// set a substitution for binary data when encoding
    /// if the subst string contains the substring "###", then this
    /// substring is replaced with the length of the binary data in bytes
    class binary_subst {
    public:
        binary_subst(const char* subst) : subst(subst) {}
        const char* subst;
    };
    std::ostream& set_binary_subst(std::ostream& strm, const char* subst = nullptr);
    inline std::ostream& operator<<(std::ostream& out, const binary_subst& obj) {
        return set_binary_subst(out,obj.subst);
    }

    /// set the maximum number of array elements to encode
    class max_array {
    public:
        max_array(long length) : length(length) {}
        long length;
    };
    std::ostream& set_max_array(std::ostream& strm, long length = 0);
    inline std::ostream& operator<<(std::ostream& out, const max_array& obj) {
        return set_max_array(out,obj.length);
    }

    /// set the string to use for indenting values
    class indent {
    public:
        indent(const char* indent_str) : indent_str(indent_str) {}
        const char* indent_str;
    };
    std::ostream& set_indent(std::ostream& strm, const char* indent_str = nullptr);
    inline std::ostream& operator<<(std::ostream& out, const indent& obj) {
        return set_indent(out,obj.indent_str);
    }
    //@}
}

