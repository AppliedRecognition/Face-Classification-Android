#include <boost/test/unit_test.hpp>

#include <applog/core.hpp>
#include <json/json.hpp>
#include <stdext/typeinfo.hpp>

#include <filesystem>


BOOST_AUTO_TEST_SUITE(json)


// parse string, check type, then re-encode and compare to original string
template <typename T>
static bool parse_encode_test(const std::string& msg) {
    json::value v;
    std::string e;
    try {
        v = json::decode_json(msg);
        e = encode(v);
        if (json::is_type<T>(v) && e==msg)
            return true;
    }
    catch (const std::exception& e) {
        FILE_LOG(logERROR) << e.what();
    }
    FILE_LOG(logINFO) << std::endl
                      << "test:\t" << type_name<T>() << std::endl
                      << "found:\t" << type_name(v) << std::endl
                      << "input:\t" << msg << std::endl
                      << "output:\t" << e;
    return false;
}

static bool parse_encode_test_binary(const std::string& msg) {
    json::value v;
    std::string e;
    try {
        v = make_binary(json::decode_json(msg));
        e = encode(v);
        if (json::is_type<json::binary>(v) && e==msg)
            return true;
    }
    catch (const std::exception& e) {
        FILE_LOG(logERROR) << e.what();
    }
    FILE_LOG(logINFO) << std::endl
                      << "test:\t" << type_name<json::binary>() << std::endl
                      << "found:\t" << type_name(v) << std::endl
                      << "input:\t" << msg << std::endl
                      << "output:\t" << e;
    return false;
}

BOOST_AUTO_TEST_CASE(stdext_typeinfo) {
    BOOST_REQUIRE_EQUAL(stdx::typeptr<int>, stdx::typeptr<int>);
    BOOST_REQUIRE_NE(stdx::typeptr<int>, stdx::typeptr<long>);
}

BOOST_AUTO_TEST_CASE(object_copy_move) {
    const auto obj = object{ {"a",1}, {"b",array{1,2,3}} };

    auto copy1 = obj;
    BOOST_CHECK_EQUAL(copy1.size(), 2);

    object copy2;
    copy2 = obj;
    BOOST_CHECK_EQUAL(copy2.size(), 2);

    auto move1 = move(copy1);
    BOOST_CHECK_EQUAL(move1.size(), 2);
    BOOST_CHECK(copy1.empty());

    object move2;
    move2 = move(copy2);
    BOOST_CHECK_EQUAL(move2.size(), 2);
    BOOST_CHECK(copy2.empty());
}

BOOST_AUTO_TEST_CASE(parse_encode_types) {
    BOOST_CHECK(parse_encode_test<json::null_type>("null"));

    BOOST_CHECK(parse_encode_test<json::boolean>("true"));
    BOOST_CHECK(parse_encode_test<json::boolean>("false"));

    BOOST_CHECK(parse_encode_test<json::integer>("0"));
    BOOST_CHECK(parse_encode_test<json::integer>("-54242"));
    BOOST_CHECK(parse_encode_test<json::integer>("75785939482983857"));

    BOOST_CHECK(parse_encode_test<json::real>("3.14"));
    BOOST_CHECK(parse_encode_test<json::real>("-2.56"));
    BOOST_CHECK(parse_encode_test<json::real>("0.543"));

    BOOST_CHECK(parse_encode_test<json::string>("\"test \\\"more\\\" \\\\ \\b \\f \\n \\r \\t done\""));

    BOOST_CHECK(parse_encode_test<json::array>("[null,true,false,0,1,3.14,\"hello\",[1,2,3],{\"a\":null,\"b\":3.14}]"));

    BOOST_CHECK(parse_encode_test<json::object>("{\"A\":null,\"B\":true,\"C\":false,\"D\":0,\"E\":1,\"F\":3.14,\"G\":\"hello\",\"H\":[1,2,3],\"I\":{\"a\":null,\"b\":3.14}}"));
}

BOOST_AUTO_TEST_CASE(parse_encode_binary) {
    BOOST_CHECK(parse_encode_test_binary("\"C/nua5+XCmLY\""));
    BOOST_CHECK(parse_encode_test_binary("\"C/nua5+XCmLYPQ==\""));
    BOOST_CHECK(parse_encode_test_binary("\"C/nua5+XCmLY8kM=\""));
}

BOOST_AUTO_TEST_CASE(binary_compare_test) {
    const stdx::binary empty;
    const stdx::binary as0("as",2);
    const stdx::binary asdf("asdf",4);
    const stdx::binary as1(asdf,0,2);
    const stdx::binary qw0("qw",2);
    const stdx::binary qwer("qwer",4);
    const stdx::binary qw1(qwer,0,2);

    const auto compare_equal =
        [](const binary& a, const binary& b) {
            return a.compare(b.data(), b.size()) == 0 &&
                b.compare(a.data(), a.size()) == 0;
        };

    const auto compare_less =
        [](const binary& a, const binary& b) {
            return a.compare(b.data(), b.size()) < 0 &&
                b.compare(a.data(), a.size()) > 0;
        };

    BOOST_CHECK(compare_equal(empty, empty));
    BOOST_CHECK(compare_equal(as0, as1));
    BOOST_CHECK(compare_equal(qw0, qw1));

    BOOST_CHECK(compare_less(empty, as0));
    BOOST_CHECK(compare_less(empty, asdf));
    BOOST_CHECK(compare_less(as0, asdf));
    BOOST_CHECK(compare_less(as1, asdf));

    BOOST_CHECK(compare_less(as0, qw0));
    BOOST_CHECK(compare_less(asdf, qwer));
    BOOST_CHECK(compare_less(asdf, qw1));

    BOOST_CHECK(compare_less(empty, qw0));
    BOOST_CHECK(compare_less(empty, qwer));
    BOOST_CHECK(compare_less(qw0, qwer));
    BOOST_CHECK(compare_less(qw1, qwer));
}

namespace {
    struct compare_base {
        virtual ~compare_base() = default;
        virtual json::value get() const = 0;
        virtual int compare(const json::value&) const = 0;
    };
    template <typename T>
    struct compare_t : compare_base {
        T other;

        template <typename... Args>
        compare_t(Args&&... args)
            : other(std::forward<Args>(args)...) {}

        json::value get() const override {
            return other;
        }
        int compare(const json::value& v) const override {
            BOOST_CHECK((v == other) != (other != v));
            BOOST_CHECK((v < other) != (other <= v));
            BOOST_CHECK((v > other) != (other >= v));
            if (auto k = v.compare(other)) {
                BOOST_CHECK(v != other);
                BOOST_CHECK(other != v);
                BOOST_CHECK((v < other) == (k < 0));
                BOOST_CHECK((other < v) == (k > 0));
                return -k;
            }
            BOOST_CHECK(v == other);
            BOOST_CHECK(other == v);
            BOOST_CHECK(other <= v);
            BOOST_CHECK(other >= v);
            BOOST_CHECK(v <= other);
            BOOST_CHECK(v >= other);
            return 0;
        }
    };

    template <typename T, typename... Args>
    auto make_compare(Args&&... args) {
        return std::make_unique<compare_t<T> >(std::forward<Args>(args)...);
    }
    template <typename T>
    auto make_compare(T&& x) {
        return std::make_unique<compare_t<std::decay_t<T> > >(std::forward<T>(x));
    }
}

BOOST_AUTO_TEST_CASE(value_compare_test) {
    // values in the order they should compare as
    enum e0 { a, b, c };
    enum class e1 { a, b, c };
    const std::unique_ptr<compare_base> values[] = {
        make_compare(null),
        make_compare(false),
        make_compare(true),
        make_compare(std::numeric_limits<integer>::min()),
        make_compare(-5),
        make_compare(0),
        make_compare<e0>(b),
        make_compare(e1::c),
        make_compare<unsigned>(4u),
        make_compare<int>(5),
        make_compare(static_cast<unsigned long long>(std::numeric_limits<integer>::max())),
        make_compare<double>(-INFINITY),
        make_compare<float>(-1.25f),
        make_compare<float>(0.0f),
        make_compare<double>(M_PI),
        make_compare<float>(INFINITY),
        make_compare<std::string_view>(),
        make_compare<const char*>("hell"),
        make_compare<const char*>("hello"),
        make_compare<std::string_view>("hello world"),
        make_compare<std::string>("worl"),
        make_compare<std::string>("world"),
        make_compare<binary>(),
        make_compare<binary>("hello"),
        make_compare<array>(),
        make_compare(array{1}),
        make_compare(array{1,2}),
        make_compare(array{1,2,3}),
        make_compare(array{1,3}),
        make_compare(array{2}),
        make_compare<object>(),
    };

    BOOST_CHECK_LT(json::value(std::numeric_limits<integer>().max()).compare(
                       std::numeric_limits<unsigned long long>().max()), 0);

    for (auto it = std::begin(values), end = std::end(values); it != end; ++it)
        for (auto jt = next(it); jt != end; ++jt) {
            auto& a = **it;
            const auto av = a.get();
            auto& b = **jt;
            const auto bv = b.get();
            BOOST_CHECK_EQUAL(a.compare(av),0);
            BOOST_CHECK_EQUAL(b.compare(bv),0);
            //FILE_LOG(logINFO) << av << ' ' << bv;
            BOOST_CHECK_LT(a.compare(bv),0);
            BOOST_CHECK_GT(b.compare(av),0);
        }
}

BOOST_AUTO_TEST_CASE(enum_tests) {
    enum e0 { a, b, c };
    const auto v0 = json::value(b);
    BOOST_CHECK(is_type<json::integer>(v0));
    BOOST_CHECK(v0 == b);
    enum class e1 { a, b, c };
    const auto v1 = json::value(e1::c);
    BOOST_CHECK(is_type<json::integer>(v1));
    BOOST_CHECK(v1 == e1::c);
}

BOOST_AUTO_TEST_CASE(string_tests) {
    const json::value v0 = "hello";
    const json::string s0 = std::string("hello");
    const auto v1 = json::value(std::string("hello"));
    const json::value v2 = std::string("hello");
    const json::value v3 = json::string("hello");
}

static_assert(stdx::range_depth<std::filesystem::path>::value == stdx::range_depth_limit, "container_depth failure");
static_assert(!json::detail::is_array_type<std::filesystem::path>::value, "is_array_type failure");

BOOST_AUTO_TEST_CASE(array_tests) {
    std::vector<int> c0;
    const auto v0 = json::value(c0);
    BOOST_CHECK(is_type<json::array>(v0));

    std::vector<std::vector<int> > c1;
    const auto v1 = json::value(c1);
    BOOST_CHECK(is_type<json::array>(v1));

    std::list<std::string> c2;
    const json::value v2 = c2;
    BOOST_CHECK(is_type<json::array>(v2));

    static const bool c3[] = { true, false };
    const auto v3 = json::value(c3);
    BOOST_CHECK(is_type<json::array>(v3));

    const auto c4 = { 3.14, 8.92 };
    const auto v4 = json::value(c4);
    BOOST_CHECK(is_type<json::array>(v4));

    const json::array v5 = { 1, false, "hello", { v3, v4 } };
    BOOST_CHECK(!v5.empty());

    const json::value v6 = { 1, false, "hello", { v3, v4 } };
    BOOST_CHECK(is_type<json::array>(v6));

    static const auto K_c = json::string("c");
    static const auto K_d = std::string("d");
    const json::value v7 = {   // note: not an object
        { "a", 1 },
        { "b", false },
        { "c", {1,2} },
        { K_c, K_d },
    };
    BOOST_CHECK(is_type<json::array>(v7));

    BOOST_CHECK(v6 == v5);
    BOOST_CHECK(v5 == v6);
}

BOOST_AUTO_TEST_CASE(object_tests) {
    std::map<std::string, int> c0;
    const auto v0 = json::value(c0);
    BOOST_CHECK(is_type<json::object>(v0));

    std::vector<std::pair<std::string, int> > c1;
    const json::value v1 = c1;
    BOOST_CHECK(is_type<json::object>(v1));

    static const auto K_c = json::string("c");
    static const auto K_d = std::string("d");
    const json::object v2 = {
        { "a", 1 },
        { "b", false },
        { K_c, v0 },
        { K_d, {1,2} },
        { "e", json::array{1,2} },
    };
    BOOST_CHECK(!v2.empty());
}

BOOST_AUTO_TEST_CASE(is_simple_tests) {
    BOOST_CHECK(json::is_simple(false));
    BOOST_CHECK(json::is_simple(3));
    BOOST_CHECK(json::is_simple(3.14));
    BOOST_CHECK(json::is_simple("hello"));
    BOOST_CHECK(json::is_simple(json::array{1,2,3}));
    BOOST_CHECK(json::is_simple(json::object{{"a",1}}));

    const auto ns = json::object{{"a",1},{"b",2}};
    BOOST_CHECK(!json::is_simple(ns));
    BOOST_CHECK(!json::is_simple(json::value(ns)));
    BOOST_CHECK(!json::is_simple(json::array{1,2,ns}));
    BOOST_CHECK(!json::is_simple(json::object{{"a",ns}}));
}

/*
static void parse_encode(const std::string& msg) {
    json::value v;
    std::string e;
    try {
        v = json::decode_json(msg);
        e = encode(v);
    }
    catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
    }
    std::cerr << "found:\t" << v.type().name() << std::endl
              << "input:\t" << msg << std::endl
              << "output:\t" << e << std::endl << std::endl;
}
*/

/*
int main(int argc, char*argv[]) {

    std::cerr << std::endl;

    for (--argc, ++argv; argc; --argc, ++argv)
        parse_encode(*argv);

    json::value v;
    std::cout<<v.type().name()<<std::endl;

    v = true;
    std::cout<<v.type().name()<<std::endl;

    v = 10;
    std::cout<<v.type().name()<<'\t'<<get_integer(v)<<std::endl;

    json::get<json::integer>(v);

    v = 3.2;
    std::cout<<v.type().name()<<'\t'<<get_real(v)<<std::endl;

    v = "test";
    std::cout<<v.type().name()<<std::endl;


    json::object o;
    json::array a;
    a.push_back(v);
    a.push_back(7);
    o["key"] = a;
    o["bool"] = false;
    
    std::string encode;
    json::encode(encode,o);

    v = json::decode_json(encode);
    std::cout<<v.type().name()<<std::endl;

    std::string s2;
    json::encode(s2,v);
    std::cout<<s2<<std::endl;

    std::vector<int> vec;
    vec.push_back(1);
    vec.push_back(2);
    vec.push_back(3);
    json::value vec_value = vec;
    v = vec;
    s2.clear();
    json::encode(s2,v);
    std::cout<<s2<<std::endl;

    std::map<std::string,int> amap;
    amap["a"]=1;
    amap["b"]=2;
    amap["c"]=3;
    v = amap;
    s2.clear();
    json::encode(s2,v);
    std::cout<<s2<<std::endl;

    std::cout<<json::encode(json::decode_json("[{\"personid\":2}]"))<<std::endl;

    return 0;
}
*/


BOOST_AUTO_TEST_SUITE_END()
