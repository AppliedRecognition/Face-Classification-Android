#pragma once

#include <cmath>
#include <sstream>
#include <iomanip>

#include <stdext/type_traits.hpp>

#include "types.hpp"
#include "is_simple.hpp"


namespace json {

    namespace detail {
        void encode_string(std::string& out, const std::string_view& sv);
    }



    /** \name Encode Null
     */
    //@{
    inline string encode(const null_type&) {
        static const string str_null("null");
        return str_null;
    }
    inline void encode(std::string& out, const null_type&) {
        out += encode(null_type());
    }
    inline std::ostream& encode(std::ostream& out, const null_type&,
                                const string& = {}) {
        return out << encode(null_type());
    }
    inline std::ostream& operator<<(std::ostream& out, const null_type&) {
        return out << encode(null_type());
    }
    //@}


    /** \name Encode Boolean
     */
    //@{
    template <typename T>
    inline std::enable_if_t<stdx::is_bool<T>::value,string>
    encode(T b) {
        static const string str_true("true");
        static const string str_false("false");
        return b ? str_true : str_false;
    }
    template <typename T>
    inline std::enable_if_t<stdx::is_bool<T>::value>
    encode(std::string& out, T b) {
        out += encode(b);
    }
    template <typename T>
    inline std::enable_if_t<stdx::is_bool<T>::value,std::ostream&>
    encode(std::ostream& out, T b, const string& = {}) {
        out << encode(b);
        return out;
    }
    //@}


    /** \name Encode Numeric Types
     */
    //@{
    template <typename T>
    inline std::enable_if_t<stdx::is_pure_integral<T>::value>
    encode(std::string& out, const T& i) {
        std::stringstream o;
        o << i;
        out += o.str();
    }
    template <typename T>
    inline std::enable_if_t<std::is_floating_point<T>::value>
    encode(std::string& out, const T& i) {
        if (std::isnan(i))
            out += "null";
        else if (std::isinf(i))
            out += std::signbit(i) ? "-1e9999" : "1e9999";
        else {
            std::stringstream o;
            o << std::setprecision(12) << i;
            out += o.str();
        }
    }
    template <typename T>
    inline std::enable_if_t<stdx::is_pure_arithmetic<T>::value,string>
    encode(const T& i) {
        std::string result;
        encode(result,i);
        return result;
    }
    template <typename T>
    inline std::enable_if_t<stdx::is_pure_arithmetic<T>::value,std::ostream&>
    encode(std::ostream& out, const T& i, const string& = {}) {
        out << encode(i);
        return out;
    }
    template <typename T>
    inline std::enable_if_t<std::is_enum<T>::value>
    encode(std::string& out, const T& i) {
        using U = typename std::underlying_type<T>::type;
        std::stringstream o;
        o << U(i);
        out += o.str();
    }
    template <typename T>
    inline std::enable_if_t<std::is_enum<T>::value,string>
    encode(const T& i) {
        std::string result;
        encode(result,i);
        return result;
    }
    template <typename T>
    inline std::enable_if_t<std::is_enum<T>::value,std::ostream&>
    encode(std::ostream& out, const T& i, const string& = {}) {
        out << encode(i);
        return out;
    }
    //@}


    /** \name Encode String
     */
    //@{
    void encode(std::string& out, std::string_view sv);
    std::ostream& encode(std::ostream& out, std::string_view sv,
                         const string& indent = {});
    inline string encode(std::string_view sv) {
        std::string result;
        encode(result,sv);
        return result;
    }
    //@}


    /** \name Encode Binary
     */
    //@{
    void encode(std::string& out, const binary& b);
    std::ostream& encode(std::ostream& out, const binary& b,
                         const string& indent = {});
    inline string encode(const binary& b) {
        std::string result;
        encode(result,b);
        return result;
    }
    //@}


    /** \name Encode Array
     */
    //@{
    template<typename ITER>
    std::ostream& format_array(std::ostream& out, ITER begin, ITER end,
                               const string& indent = {});
    
    template<typename ITER>
    void encode_array(std::string& out, ITER begin, ITER end);
    template<typename ITER>
    std::ostream& encode_array(std::ostream& out, ITER begin, ITER end);
    template<typename ITER>
    inline string encode_array(ITER begin, ITER end) {
        std::string result;
        encode_array(result,begin,end);
        return result;
    }

    template<typename T>
    inline typename
    std::enable_if<detail::is_array_type<T>::value, string>::type
    encode(const T& c) {
        return encode_array(c.begin(),c.end());
    }
    template<typename T>
    inline typename std::enable_if<detail::is_array_type<T>::value>::type
    encode(std::string& out, const T& c) {
        encode_array(out,c.begin(),c.end());
    }
    template<typename T>
    inline typename
    std::enable_if<detail::is_array_type<T>::value, std::ostream&>::type
    encode(std::ostream& out, const T& c, const string& indent = {}) {
        return is_simple(c) ?
            encode_array(out,c.begin(),c.end()) : 
            format_array(out,c.begin(),c.end(),indent);
    }
    //@}

    
    /** \name Encode Object
     */
    //@{
    template<typename ITER>
    std::ostream& format_object(std::ostream& out, ITER begin, ITER end,
                                const string& indent = {});

    template<typename ITER>
    void encode_object(std::string& out, ITER begin, ITER end);
    template<typename ITER>
    std::ostream& encode_object(std::ostream& out, ITER begin, ITER end);
    template<typename ITER>
    inline string encode_object(ITER begin, ITER end) {
        std::string result;
        encode_object(result,begin,end);
        return result;
    }
  
    template<typename T>
    inline typename
    std::enable_if<detail::is_object_type<T>::value,string>::type
    encode(const T& c) {
        return encode_object(c.begin(),c.end());
    }
    template<typename T>
    inline typename
    std::enable_if<detail::is_object_type<T>::value>::type
    encode(std::string& out, const T& c) {
        encode_object(out,c.begin(),c.end());
    }

    template<typename T>
    inline typename
    std::enable_if<detail::is_object_type<T>::value,std::ostream&>::type
    encode(std::ostream& out, const T& c, const string& indent = {}) {
        return is_simple(c) ?
            encode_object(out,c.begin(),c.end()) : 
            format_object(out,c.begin(),c.end(),indent);
    }
    //@}


    /** \name Encode Value
     */
    //@{
    template <typename T>
    std::enable_if_t<std::is_same_v<T,value> >
    encode(std::string& out, const T& v);

    template <typename T>
    std::enable_if_t<std::is_same_v<T,value>, std::ostream&>
    encode(std::ostream& out, const T& v, const string& indent = {});

    template <typename T>
    inline std::enable_if_t<std::is_same_v<T,value>, string>
    encode(const T& v) {
        std::string result;
        encode(result,v);
        return result;
    }
    //@}
}


#include "encode.ipp"


