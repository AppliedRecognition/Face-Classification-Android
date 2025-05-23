
#include "io_manip.hpp"


using namespace json;


int json::detail::manip_max_string = -1;
int json::detail::manip_binary_subst = -1;
int json::detail::manip_max_array = -1;
int json::detail::manip_indent = -1;


std::ostream& json::set_max_string(std::ostream& strm, long length) {
    if (detail::manip_max_string<0) 
        detail::manip_max_string = std::ios_base::xalloc();
    strm.iword(detail::manip_max_string) = length;
    return strm;
}
std::ostream& json::set_binary_subst(std::ostream& strm, const char* subst) {
    if (detail::manip_binary_subst<0) 
        detail::manip_binary_subst = std::ios_base::xalloc();
    strm.pword(detail::manip_binary_subst) = const_cast<char*>(subst);
    return strm;
}
std::ostream& json::set_max_array(std::ostream& strm, long length) {
    if (detail::manip_max_array<0) 
        detail::manip_max_array = std::ios_base::xalloc();
    strm.iword(detail::manip_max_array) = length;
    return strm;
}
std::ostream& json::set_indent(std::ostream& strm, const char* indent_str) {
    if (detail::manip_indent<0) 
        detail::manip_indent = std::ios_base::xalloc();
    strm.pword(detail::manip_indent) = const_cast<char*>(indent_str);
    return strm;
}

