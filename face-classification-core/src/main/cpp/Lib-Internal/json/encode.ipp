
#include "io_manip.hpp"


template<typename ITER>
std::ostream& json::format_array(std::ostream& out, ITER begin, ITER end,
                                 const string& base) {
    long max_len = 
        detail::manip_max_array>=0 ? out.iword(detail::manip_max_array) : 0;
    long count = 0;
    out << '[';
    const string prefix = 
        detail::manip_indent>=0 && out.pword(detail::manip_indent) ?
        string(base + static_cast<const char*>(out.pword(detail::manip_indent)) ):
        base;
    if (!prefix.empty())
        out << std::endl;
    if (begin != end) {
        out << prefix;
        encode(out,*begin,prefix);
        for (++begin; begin!=end; ++begin) {
            out << ',';
            if (!prefix.empty())
                out << std::endl << prefix;
            if (max_len > 0 && ++count >= max_len) {
                out << "...";
                break;
            }
            encode(out,*begin,prefix);
        }
    }
    if (!prefix.empty())
        out << std::endl << base;
    return out << ']';
}

template<typename ITER>
void json::encode_array(std::string& out, ITER begin, ITER end) {
    out += '[';
    if (begin != end) {
        encode(out,*begin);
        for (++begin; begin!=end; ++begin) {
            out += ',';
            encode(out,*begin);
        }
    }
    out += ']';
}

template<typename ITER>
std::ostream& json::encode_array(std::ostream& out, ITER begin, ITER end) {
    long max_len = 
        detail::manip_max_array>=0 ? out.iword(detail::manip_max_array) : 0;
    long count = 0;
    out << '[';
    if (begin != end) {
        encode(out,*begin);
        for (++begin; begin!=end; ++begin) {
            out << ',';
            if (max_len > 0 && ++count >= max_len) {
                out << "...";
                break;
            }
            encode(out,*begin);
        }
    }
    return out << ']';
}


template<typename ITER>
std::ostream& json::format_object(std::ostream& out, ITER begin, ITER end,
                                  const string& base) {
    long max_len = 
        detail::manip_max_array>=0 ? out.iword(detail::manip_max_array) : 0;
    long num_complex = 0;
    out << '{';
    const string prefix = 
        detail::manip_indent>=0 && out.pword(detail::manip_indent) ?
        string(base + static_cast<const char*>(out.pword(detail::manip_indent)) ):
        base;
    if (begin != end) {
        if (!prefix.empty())
            out << std::endl << prefix;
        encode(out,begin->first);  
        out << ':';  
        if (!prefix.empty())
            out << ' ';
        if (begin->first.substr(0,5) == "x-fb-" && 
            is_type<string>(begin->second))
            out << "<STRING>";
        else
            encode(out,begin->second,prefix);
        if (max_len > 0 && !is_simple(begin->second))
            ++num_complex;
        for (++begin; begin != end; ++begin) {
            if (max_len == 0 || is_simple(begin->second) ||
                ++num_complex <= max_len) {
                out << ',';
                if (!prefix.empty())
                    out << std::endl << prefix;
                encode(out,begin->first); 
                out << ':'; 
                if (!prefix.empty())
                    out << ' ';
                if (begin->first.substr(0,5) == "x-fb-" && 
                    is_type<string>(begin->second))
                    out << "<STRING>";
                else
                    encode(out,begin->second,prefix);
            }
        }
        if (max_len > 0 && num_complex > max_len) {
            out << ',';
            if (!prefix.empty())
                out << std::endl << prefix;
            out << "..."; 
        }
        if (!prefix.empty())
            out << std::endl << base;
    }
    return out << '}';
}

template<typename ITER>
void json::encode_object(std::string& out, ITER begin, ITER end) {
    out += '{';
    if (begin != end) {
        encode(out,begin->first);  
        out += ':';  
        encode(out,begin->second);
        for (++begin; begin != end; ++begin) {
            out += ',';
            encode(out,begin->first); 
            out += ':'; 
            encode(out,begin->second);
        }
    }
    out += '}';
}

template<typename ITER>
std::ostream& json::encode_object(std::ostream& out, ITER begin, ITER end) {
    out << '{';
    if (begin != end) {
        encode(out,begin->first);  
        out << ':';  
        encode(out,begin->second);
        for (++begin; begin != end; ++begin) {
            out << ',';
            encode(out,begin->first); 
            out << ':'; 
            encode(out,begin->second);
        }
    }
    return out << '}';
}
