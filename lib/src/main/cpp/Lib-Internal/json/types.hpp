#pragma once

#include <functional>
#include <map>
#include <optional>
#include <stdexcept>
#include <string_view>
#include <type_traits>
#include <variant>

#include <stdext/type_traits.hpp>
#include <stdext/convert.hpp>
#include <stdext/rounding.hpp>
#include <stdext/binary.hpp>


namespace json {

    using std::move;

    /// runtime error for failed parse
    struct parse_error : public std::runtime_error {
        parse_error() : runtime_error("json::parse failed") {}
        parse_error(const std::string& msg) : runtime_error(msg) {}
    };

    /// attempt to get value of wrong type
    class bad_get : public std::exception {
        std::string msg;
    public:
        bad_get(const char* expected, const char* but_found);
        const char* what() const noexcept override { return msg.c_str(); }
    };

    /** \brief Binary predicate for map that defaults to less<T> but can
     * be overridden at runtime on a per object basis.
     */
    template <typename T>
    class binary_predicate {
        std::function<bool(const T&, const T&)> f;
    public:
        binary_predicate() : f(std::less<T>()) {}

        binary_predicate(const binary_predicate& other) = default;
        binary_predicate& operator=(const binary_predicate& other) = default;

        binary_predicate(binary_predicate&& other) : f(move(other.f)) {
            other.f = std::less<T>{};
        }
        inline binary_predicate& operator=(binary_predicate&& other) {
            if (this != &other) {
                f = move(other.f);
                other.f = std::less<T>{};
            }
            return *this;
        }

        template <typename F, typename = std::enable_if_t<!std::is_same<std::decay_t<F>,binary_predicate>::value> >
        binary_predicate(F&& f)
            : f(std::forward<F>(f)) {}

        inline bool operator()(const T& a, const T& b) const {
            return f(a,b);
        }
    };


    /** \brief Conversion options for string to or from binary.
     */
    enum convert_type {
        convert_none,
        convert_cast,
        convert_base64
    };


    using null_type = std::monostate;
    using boolean = bool;
    using integer = long long;
    using real = double;
    using string = std::string;
    using binary = stdx::binary;

    class value;
    using array = std::vector<value>;

    /** \brief Custom std::map for json::object.
     */
    class object : public std::map<string,value,binary_predicate<string> > {
        using base = std::map<string,value,binary_predicate<string> >;
        //class object : public std::map<string,value> {
        //using base = std::map<string,value>;
    public:
        using base::base;
        object(object&&) = default;
        object(const object&) = default;
        object& operator=(object&&) = default;
        object& operator=(const object&) = default;

        /** \brief Reference to value for key, or null value if key
         * doesn't exist in object.
         */
        const value& operator[](const string& key) const;
        using base::operator[];
    };


    // forward declarations (full declaration in pull_types.hpp)
    class string_puller;
    class binary_puller;
    class array_puller;
    class object_puller;
    class value_puller;


    // forward declarations (full declaration in push_types.hpp)
    class string_pusher;
    class binary_pusher;
    class array_pusher;
    class object_pusher;
    class value_pusher;


    // pretty names
    template <typename T> constexpr auto type_name()  { return "unknown"; }
    template <> constexpr auto type_name<null_type>() { return "null"; }
    template <> constexpr auto type_name<boolean>()   { return "boolean"; }
    template <> constexpr auto type_name<integer>()   { return "integer"; }
    template <> constexpr auto type_name<real>()      { return "real"; }
    template <> constexpr auto type_name<string>()    { return "string"; }
    template <> constexpr auto type_name<binary>()    { return "binary"; }
    template <> constexpr auto type_name<array>()     { return "array"; }
    template <> constexpr auto type_name<object>()    { return "object"; }
    template <> constexpr auto type_name<string_puller>() { return "string_puller"; }
    template <> constexpr auto type_name<binary_puller>() { return "binary_puller"; }
    template <> constexpr auto type_name<array_puller>()  { return "array_puller"; }
    template <> constexpr auto type_name<object_puller>() { return "object_puller"; }
    template <> constexpr auto type_name<string_pusher>() { return "string_pusher"; }
    template <> constexpr auto type_name<binary_pusher>() { return "binary_pusher"; }
    template <> constexpr auto type_name<array_pusher>()  { return "array_pusher"; }
    template <> constexpr auto type_name<object_pusher>() { return "object_pusher"; }
    struct type_name_visitor {
        template <typename T>
        constexpr const char* operator()(const T&) const noexcept {
            return type_name<T>();
        }
    };


    namespace detail {
        using value_base =
            std::variant<null_type, boolean, integer, real,
                         string, binary, array, object>;

        template <typename T, std::size_t I = 0>
        constexpr std::enable_if_t<std::is_same_v<T,std::variant_alternative_t<I,value_base> >, std::size_t> index_of() {
            return I;
        }
        template <typename T, std::size_t I = 0>
        constexpr std::enable_if_t<!std::is_same_v<T,std::variant_alternative_t<I,value_base> >, std::size_t> index_of() {
            return index_of<T,I+1>();
        }

        template <typename T>
        using is_json_type = stdx::is_one_of<
            T,
            null_type, boolean, integer, real,
            string, binary, array, object>;

        template <typename T, typename R>
        using json_type_ret = std::enable_if_t<is_json_type<T>::value,R>;

        // like std::underlying_type, but is defined when T is not enum
        template <typename T, typename = void>
        struct underlying_type { using type = T; };
        template <typename T>
        struct underlying_type<T, std::enable_if_t<std::is_enum<T>::value> > {
            using type = std::underlying_type_t<T>;
        };

        template <typename T>
        constexpr auto numeric_cast(T v) {
            using U = typename underlying_type<T>::type;
            using V = std::conditional_t<
                stdx::is_bool<U>::value, boolean,
                std::conditional_t<
                    std::is_floating_point<U>::value, real,
                    integer> >;
            return stdx::convert_to<V>(U(v));
        }

        // container with value_type that will convert to value
        // care is taken to avoid infinitely nested containers
        // (e.g. std::filesystem::path)
        template <typename T, typename = void>
        struct is_array_type : std::false_type {};
        template <typename T>
        struct is_array_type<T, std::enable_if_t<stdx::is_range<T>::value && (stdx::range_depth<T>::value < stdx::range_depth_limit)> >
            : std::is_constructible<value, stdx::range_value_type<T> > {};

        // container with value_type that will convert to object::value_type
        template <typename T, typename = void>
        struct is_object_type : std::false_type {};
        template <typename T>
        struct is_object_type<T, std::enable_if_t<stdx::is_range<T>::value> >
            : std::is_constructible<object::value_type, stdx::range_value_type<T> > {};
    }


    /// null value
    const null_type null = null_type();


    /// json value
    class value : public detail::value_base {
        using val = detail::value_base;

    public:
        /** \name Constructor
         */
        //@{
        // default / null
        constexpr value() : val(null) {}
        constexpr value(const null_type&) : val(null) {}

        // numeric (including bool and enum, but not char types)
        // note: a simple constructor for bool gets chosen too easily,
        // so we need a template
        template <typename T>
        constexpr value(T v,
              std::enable_if_t<
              stdx::is_bool<T>::value ||
              std::is_enum<T>::value ||
              stdx::is_pure_integral<T>::value ||
              std::is_floating_point<T>::value>* = nullptr)
            : val(detail::numeric_cast(v)) {}

        // string
        value(const char* v)
            : val(std::in_place_type<string>, v) {}
        value(std::string_view v)
            : val(std::in_place_type<string>, v.data(), v.size()) {}
        value(string v)
            : val(move(v)) {}

        // binary
        value(const void* data, size_t length)
            : val(std::in_place_type<binary>, data, length) {}
        template <std::size_t N>
        value(const std::array<std::byte,N>& data)
            : val(std::in_place_type<binary>, data.data(), N) {}
        value(binary v)
            : val(move(v)) {}

        // array
        template <typename T>
        value(const T& v,
              std::enable_if_t<detail::is_array_type<const T>::value>* = nullptr)
            : val(std::in_place_type<array>, std::begin(v), std::end(v)) {}
        value(std::initializer_list<value> ilist)
            : val(std::in_place_type<array>, ilist) {}
        value(array v)
            : val(move(v)) {}

        // object
        template <typename T>
        value(const T& v,
              std::enable_if_t<detail::is_object_type<const T>::value>* = nullptr)
            : val(std::in_place_type<object>, std::begin(v), std::end(v)) {}
        value(object v)  // preserves ordering
            : val(move(v)) {}

        // optional
        template <typename T>
        explicit value(std::optional<T> v)
            : val(v ? val(move(*v)) : val()) {}

        // move / copy
        value(value&& other) noexcept : val(move(other)) {}
        value(const value&) = default;
        //@}


        /** \name Assignment
         */
        //@{
        value& operator=(value&&) = default;
        value& operator=(const value&) = default;
        template <typename T, typename... Args>
        inline std::enable_if_t<detail::is_json_type<T>::value,T&>
        emplace(Args&&... args) {
            return std::get<T>(*this = T(std::forward<Args>(args)...));
        }
        inline void swap(value& other) {
            static_cast<val*>(this)->swap(other);
        }
        //@}


        /** \name Comparison
         */
        //@{
        constexpr int compare(null_type) const noexcept {
            return index() == 0 ? 0 : 1;  // all non-null > null
        }

        // boolean
        template <typename T>
        constexpr std::enable_if_t<stdx::is_bool<T>::value, int>
        compare(T v) const noexcept {
            if (auto* p = std::get_if<boolean>(this))
                return int(*p) - int(v);
            return index() < detail::index_of<boolean>() ? -1 : 1;
        }

        // integer
        template <typename T>
        constexpr std::enable_if_t<
            std::is_enum<T>::value || stdx::is_pure_integral<T>::value, int>
        compare(T v) const noexcept {
            if (auto* p = std::get_if<integer>(this)) {
                using ull = unsigned long long;
                using I = typename detail::underlying_type<T>::type;
                if (0 < I(v) &&
                    ull(std::numeric_limits<integer>::max()) < ull(v))
                    return -1;
                if (*p < integer(v)) return -1;
                if (integer(v) < *p) return 1;
                return 0;
            }
            return index() < detail::index_of<integer>() ? -1 : 1;
        }

        // real
        template <typename T>
        std::enable_if_t<std::is_floating_point<T>::value, int>
        compare(T v) const noexcept {
            if (auto* p = std::get_if<real>(this)) {
                //if (isnan(*p)) return isnan(v) ? 0 : -1;
                //if (isnan(v)) return 1;
                if (*p < real(v)) return -1;
                if (real(v) < *p) return 1;
                return 0;
            }
            return index() < detail::index_of<real>() ? -1 : 1;
        }

        // string
        int compare(const std::string_view& sv) const noexcept {
            if (auto* p = std::get_if<string>(this))
                return p->compare(sv);
            return index() < detail::index_of<string>() ? -1 : 1;
        }

        // binary
        int compare(const void* data, size_t length) const noexcept {
            if (auto* p = std::get_if<binary>(this))
                return p->compare(data,length);
            return index() < detail::index_of<binary>() ? -1 : 1;
        }
        template <std::size_t N>
        inline int compare(const std::array<std::byte,N>& data) const noexcept {
            return compare(data.data(), data.size());
        }
        inline int compare(const binary& data) const noexcept {
            return compare(data.data(), data.size());
        }

        // array
        template <typename T>
        std::enable_if_t<detail::is_array_type<const T>::value, int>
        compare(const T& v) const noexcept {
            if (auto* p = std::get_if<array>(this)) {
                const auto end = std::end(v);
                auto jt = std::begin(v);
                for (auto it = p->begin(); ; ++it, ++jt) {
                    if (jt == end) return it == p->end() ? 0 : 1;
                    if (it == p->end()) return -1;
                    if (auto k = it->compare(*jt))
                        return k;
                }
            }
            return index() < detail::index_of<array>() ? -1 : 1;
        }

        // object
        template <typename T>
        std::enable_if_t<detail::is_object_type<const T>::value, int>
        compare(const T& v) const noexcept {
            if (auto* p = std::get_if<object>(this)) {
                const auto end = std::end(v);
                auto jt = std::begin(v);
                for (auto it = p->begin(); ; ++it, ++jt) {
                    if (jt == end) return it == p->end() ? 0 : 1;
                    if (it == p->end()) return -1;
                    if (auto k = it->first.compare(jt->first))
                        return k;
                    if (auto k = it->second.compare(jt->second))
                        return k;
                }
            }
            return index() < detail::index_of<object>() ? -1 : 1;
        }

        // optional
        template <typename T>
        inline int compare(const std::optional<T>& v) const noexcept {
            return v ? compare(*v) : compare(null);
        }

        // value
        template <typename T>
        std::enable_if_t<std::is_same_v<T,value>, int>
        compare(const T& v) const noexcept;
        //@}
    };


    /** \name Comparison
     */
    //@{
    template <typename U, typename V>
    inline std::enable_if_t<std::is_same_v<U,value>, int>
    compare(const U& a, const V& b) {
        return a.compare(b);
    }
    template <typename U, typename V>
    inline std::enable_if_t<
        !std::is_same_v<U,value> && std::is_same_v<V,value>, int>
    compare(const U& a, const V& b) {
        return -b.compare(a);
    }

    template <typename U, typename V>
    inline std::enable_if_t<
        std::is_same_v<U,value> || std::is_same_v<V,value>, bool>
    operator==(const U& a, const V& b) {
        return compare(a,b) == 0;
    }
    template <typename U, typename V>
    inline std::enable_if_t<
        std::is_same_v<U,value> || std::is_same_v<V,value>, bool>
    operator!=(const U& a, const V& b) {
        return compare(a,b) != 0;
    }
    template <typename U, typename V>
    inline std::enable_if_t<
        std::is_same_v<U,value> || std::is_same_v<V,value>, bool>
    operator<(const U& a, const V& b) {
        return compare(a,b) < 0;
    }
    template <typename U, typename V>
    inline std::enable_if_t<
        std::is_same_v<U,value> || std::is_same_v<V,value>, bool>
    operator<=(const U& a, const V& b) {
        return compare(a,b) <= 0;
    }
    template <typename U, typename V>
    inline std::enable_if_t<
        std::is_same_v<U,value> || std::is_same_v<V,value>, bool>
    operator>(const U& a, const V& b) {
        return compare(a,b) > 0;
    }
    template <typename U, typename V>
    inline std::enable_if_t<
        std::is_same_v<U,value> || std::is_same_v<V,value>, bool>
    operator>=(const U& a, const V& b) {
        return compare(a,b) >= 0;
    }
    //@}


    /** \name Access Methods
     */
    //@{
    /// test for null
    inline bool is_null(const value& v) { return v == null; }

    /// test for specific json type
    template <typename T>
    inline detail::json_type_ret<T,bool>
    is_type(const value& v) { return std::holds_alternative<T>(v); }

    /// pretty name for type
    const char* type_name(const value& v);

    /// same as std::get but only works with value object and
    /// throws json::bad_get with a descriptive message in case of failure
    template <typename T>
    detail::json_type_ret<T,T&> get(value& v) {
        if (auto p = std::get_if<T>(&v))
            return *p;
        throw bad_get(type_name<T>(), type_name(v));
    }
    template <typename T>
    detail::json_type_ret<T,const T&> get(const value& v) {
        if (auto p = std::get_if<T>(&v))
            return *p;
        throw bad_get(type_name<T>(), type_name(v));
    }
    template <typename T>
    detail::json_type_ret<T,T&&> get(value&& v) {
        if (auto p = std::get_if<T>(&v))
            return move(*p);
        throw bad_get(type_name<T>(), type_name(v));
    }

    /// same as get but returns default value if value has incorrect type
    template <typename T>
    inline detail::json_type_ret<T,T>
    get_safe(const value& v, const T& defval = T()) {
        if (auto p = std::get_if<T>(&v))
            return *p;
        return defval;
    }

    /// same as get but returns a std::optional<T> which is not set if
    /// the value is null
    template <typename T>
    inline detail::json_type_ret<T,std::optional<T> >
    get_optional(const value& v) {
        if (auto p = std::get_if<T>(&v))
            return *p;
        if (v == null)
            return {};
        throw bad_get(type_name<T>(), type_name(v));
    }

    /// same as get but returns a std::optional<T> which is not set if
    /// the value does not have the correct type
    template <typename T>
    inline detail::json_type_ret<T,std::optional<T> >
    get_optional_safe(const value& v) {
        if (auto p = std::get_if<T>(&v))
            return *p;
        return {};
    }
    //@}


    /** \name Access Methods (boolean)
     */
    //@{
    inline boolean get_boolean(const value& v) {
        return get<boolean>(v);
    }
    inline boolean get_boolean_safe(const value& v, boolean defval = false) {
        return get_safe<boolean>(v,defval);
    }
    /** \brief Make boolean from any value type.
     *
     * Similar to php, this method will return false if value is any of the
     * following:<ul>
     *   <li>null</li>
     *   <li>false</li>
     *   <li>0 (integer)</li>
     *   <li>0.0 (real)</li>
     *   <li>empty string or the string "0"</li>
     *   <li>zero length binary</li>
     *   <li>empty array</li>
     *   <li>empty object</li>
     * </ul>
     * This method returns true in all other cases.
     */
    boolean make_boolean(const value& v);
    //@}


    /** \name Access Methods (integer)
     */
    //@{
    inline integer get_integer(const value& v) {
        return get<integer>(v);
    }
    inline integer get_integer_safe(const value& v, integer defval = 0) {
        return get_safe<integer>(v,defval);
    }
    /// make integer (converting from string if necessary)
    integer make_integer(const value& v);
    //@}


    /** \name Access Methods (real)
     */
    //@{
    inline real get_real(const value& v) {
        return get<real>(v);
    }
    inline real get_real_safe(const value& v, real defval = 0.0) {
        return get_safe<real>(v,defval);
    }
    /// make real (converting from integer or string if necessary)
    real make_real(const value& v);
    //@}


    /** \name Access Methods (any numeric)
     */
    //@{
    template <typename T>
    inline std::enable_if_t<std::is_integral_v<T>,T>
    make_number(const value& v) {
        if (auto p = std::get_if<integer>(&v))
            return stdx::convert_to<T>(*p);
        throw bad_get(type_name<integer>(), type_name(v));
    }
    template <typename T>
    inline std::enable_if_t<!std::is_integral_v<T>,T>
    make_number(const value& v) {
        if (auto p = std::get_if<integer>(&v))
            return stdx::convert_to<T>(*p);
        if (auto p = std::get_if<real>(&v))
            return stdx::convert_to<T>(*p);
        throw bad_get(type_name<real>(), type_name(v));
    }
    template <typename T>
    inline T make_number(const value& v, const T& defval) {
        return v != null ? make_number<T>(v) : defval;
    }
    struct number_proxy {
        const value v;
        template <typename T,
                  typename = std::enable_if_t<std::is_arithmetic_v<T> > >
        inline operator T() const { return make_number<T>(v); }
    };
    inline number_proxy make_number(const value& v) { return {v}; }

    /// rounding
    template <typename T>
    inline T round_to(const value& v) {
        if (auto p = std::get_if<integer>(&v))
            return stdx::round_to<T>(*p);
        if (auto p = std::get_if<real>(&v))
            return stdx::round_to<T>(*p);
        throw bad_get(type_name<real>(), type_name(v));
    }
    template <typename T>
    inline T round_to(const value& v, const T& defval) {
        return v != null ? round_to<T>(v) : defval;
    }
    struct round_proxy {
        const value v;
        template <typename T,
                  typename = std::enable_if_t<std::is_arithmetic_v<T> > >
        inline operator T() const { return round_to<T>(v); }
    };
    inline round_proxy round_from(const value& v) { return {v}; }
    //@}


    /** \name Access Methods (string)
     */
    //@{
    inline string& get_string(value& v) {
        return get<string>(v);
    }
    inline const string& get_string(const value& v) {
        return get<string>(v);
    }
    inline string get_string(const value&& v) {
        return get<string>(v);
    }
    inline string get_string_safe(const value& v,
                                  const string& defval = string()) {
        return get_safe<string>(v,defval);
    }
    /// make string (base64 encode binary is necessary)
    string make_string(const value& v);
    //@}


    /** \name Access Methods (binary)
     */
    //@{
    inline binary& get_binary(value& v) {
        return get<binary>(v);
    }
    inline const binary& get_binary(const value& v) {
        return get<binary>(v);
    }
    inline binary get_binary(const value&& v) {
        return get<binary>(v);
    }
    inline binary get_binary_safe(const value& v,
                                  const binary& defval = binary()) {
        return get_safe<binary>(v,defval);
    }
    /// make binary (base64 decode string if necessary)
    binary make_binary(const value& v);
    //@}


    /** \name Access Methods (array)
     */
    //@{
    inline array& get_array(value& v) {
        return get<array>(v);
    }
    inline const array& get_array(const value& v) {
        return get<array>(v);
    }
    inline array get_array(const value&& v) {
        return get<array>(v);
    }
    inline array get_array_safe(const value& v,
                                const array& defval = array()) {
        return get_safe<array>(v,defval);
    }
    //@}


    /** \name Access Methods (object)
     */
    //@{
    inline object& get_object(value& v) {
        return get<object>(v);
    }
    inline const object& get_object(const value& v) {
        return get<object>(v);
    }
    inline object get_object(const value&& v) {
        return get<object>(v);
    }
    inline object get_object_safe(const value& v,
                                  const object& defval = object()) {
        return get_safe<object>(v,defval);
    }
    //@}


    /** \name Array Helpers
     */
    //@{
    /// transform iterator (get on array iterator dereference)
    template <typename T>
    struct transform_iterator : array::const_iterator {
        using value_type = const T;
        using reference = const T&;
        using pointer = const T*;
        explicit transform_iterator(array::const_iterator iter)
            : array::const_iterator(move(iter)) {}
        reference operator*() const {
            return get<T>(**static_cast<const array::const_iterator*>(this));
        }
        pointer operator->() const {
            return &get<T>(**static_cast<const array::const_iterator*>(this));
        }
    };
    template <typename T>
    inline detail::json_type_ret<T,transform_iterator<T> >
    transform_to(json::array::const_iterator iter) {
        return transform_iterator<T>(move(iter));
    }

    /// pseudo-container to do get<T>() on array values in for loop
    template <typename T>
    struct array_ref_to {
        const array* a;
        using iterator = transform_iterator<T>;
        iterator begin() const { return transform_to<T>(a->begin()); }
        iterator end() const   { return transform_to<T>(a->end());   }
        array::size_type size() const { return a->size(); }
        bool empty() const { return a->empty(); }
    };
    template <typename T>
    inline detail::json_type_ret<T,array_ref_to<T> >
    array_to(const array& a) { return { &a }; }
    template <typename T>
    void array_to(const array&&) = delete;
    template <typename T>
    inline detail::json_type_ret<T,array_ref_to<T> >
    array_to(const value& v) { return { &get_array(v) }; }
    template <typename T>
    void array_to(const value&&) = delete;

    /// boolean
    template <typename U>
    inline array_ref_to<boolean>
    boolean_from_array(U&& x) {
        return array_to<boolean>(std::forward<U>(x));
    }

    /// integer
    template <typename U>
    inline array_ref_to<integer>
    integer_from_array(U&& x) {
        return array_to<integer>(std::forward<U>(x));
    }

    /// real
    template <typename U>
    inline array_ref_to<real>
    real_from_array(U&& x) {
        return array_to<real>(std::forward<U>(x));
    }

    /// string
    template <typename U>
    inline array_ref_to<string>
    string_from_array(U&& x) {
        return array_to<string>(std::forward<U>(x));
    }

    /// binary
    template <typename U>
    inline array_ref_to<binary>
    binary_from_array(U&& x) {
        return array_to<binary>(std::forward<U>(x));
    }

    /// array
    template <typename U>
    inline array_ref_to<array>
    array_from_array(U&& x) {
        return array_to<array>(std::forward<U>(x));
    }

    /// object
    template <typename U>
    inline array_ref_to<object>
    object_from_array(U&& x) {
        return array_to<object>(std::forward<U>(x));
    }

    /// helper to get or make a value
    template <typename T>
    inline std::enable_if_t<std::is_same_v<T,bool>,T>
    get_or_make(const value& v) { return make_boolean(v); }
    template <typename T>
    inline std::enable_if_t<std::is_arithmetic_v<T>,T>
    get_or_make(const value& v) { return make_number<T>(v); }
    template <typename T>
    inline std::enable_if_t<std::is_same_v<T,string>,T>
    get_or_make(const value& v) { return make_string(v); }
    template <typename T>
    inline std::enable_if_t<std::is_same_v<T,binary>,T>
    get_or_make(const value& v) { return make_binary(v); }
    template <typename T>
    inline std::enable_if_t<std::is_same_v<T,array>,const T&>
    get_or_make(const value& v) { return get_array(v); }
    template <typename T>
    inline std::enable_if_t<std::is_same_v<T,object>,const T&>
    get_or_make(const value& v) { return get_object(v); }

    /// json::array  ->  std::vector<T>
    template <typename T>
    auto make_vector(const value& v) {
        auto& arr = get_array(v);
        std::vector<T> vec;
        vec.reserve(arr.size());
        for (auto& v : arr)
            vec.emplace_back(get_or_make<T>(v));
        return vec;
    }

    /// json::array  ->  std::array<T,N>
    template <typename T, std::size_t N>
    auto make_array(const value& v) {
        auto& arr = get_array(v);
        if (arr.size() != N)
            throw std::range_error("json array has wrong size");
        std::array<T,N> r;
        auto dest = r.begin();
        for (auto& v : arr)
            *dest++ = get_or_make<T>(v);
        return r;
    }
    //@}


    /** \name JSON Encode / Decode
     */
    //@{
    void encode_json(std::string& out, const value& v);
    inline string encode_json(const value& v) {
        std::string out;  encode_json(out,v);  return out;
    }
    value decode_json(const char* data, std::size_t size);
    value decode_json(const std::string& in);
    //@}


    /** \name AMF3 Encode / Decode
     */
    //@{
    binary encode_amf3(const value& v);
    value decode_amf3(const void* data, std::size_t size);
    value decode_amf3(const binary& in);
    //@}


    /** \name CBOR (bag) Encode / Decode
     */
    //@{
    binary encode_cbor(const value& v);
    value decode_cbor(const void* data, std::size_t size);
    value decode_cbor(const binary& in);
    //@}


    /** \name Decode any of AMF3, CBOR or JSON
     *
     * Selection is made based on the value of the first byte.
     * If less than 32, then AMF3 is assumed.
     * If greater or equal to 128, then CBOR is assumed.
     * Otherwise, JSON decode is attempted.
     *
     * A CBOR integer, binary or string will only be recognized if it
     * is tagged.  In particular, the 3 bytes 0xd9 0xd9 0xf7 can be used
     * as a magic prefix to indicate that what follows is CBOR data.
     * CBOR array and map values will be recognized in their "naked" form.
     */
    //@{
    value decode_any(const void* data, std::size_t size);
    value decode_any(const binary& in);
    inline value decode_amf3_or_json(const void* data, std::size_t size) {
        return decode_any(data,size);
    }
    inline value decode_amf3_or_json(const binary& in) {
        return decode_any(in);
    }
    //@}


    /** \name JSON Input and Decode
     */
    //@{
    std::istream& operator>>(std::istream&, array&);
    std::istream& operator>>(std::istream&, object&);
    std::istream& operator>>(std::istream&, value&);
    //@}


    /** \name Describe Value
     */
    //@{
    std::ostream& operator<<(std::ostream&, const array&);
    std::ostream& operator<<(std::ostream&, const object&);
    std::ostream& operator<<(std::ostream&, const value&);
    //@}
}


