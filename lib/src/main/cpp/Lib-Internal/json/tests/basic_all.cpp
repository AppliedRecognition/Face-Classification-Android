#include <boost/test/unit_test.hpp>

#include <json/types.hpp>
#include <json/push_decode_cbor.hpp>
#include <json/zlib.hpp>

#include <applog/core.hpp>

#include <iomanip>


static const json::value for_all[] = {
    json::null,
    true,
    false,
    0, 1, -1, 2, -2,

    // integers that look like deflate header in cbor
    25, 242, -49, -235,

    // amf3 integer extemes
    (1l<<28) - 1,
    -(1l<<28) + 1,
    -(1l<<28),

    // too large for amf3 integer (encodes as double instead)
    (1l << 28),
    -(1l<<28) - 1,

    // int32 extremes (also fit in double)
    2147483647,
    -2147483647,
    -2147483648,

    // int in double extremes
    (1ll << 53) - 1,
    (1ll << 53),
    -(1ll << 53) + 1,
    -(1ll << 53),

    // int64 extremes that will fit in double
    (1ll << 62),
    -9223372036854775807ll-1,  // -2^63

    // floats that fit in cbor half precision
    1.0f / (1<<14),
    0.9375,
    1.0625,
    199.375,
    1023.5,
    -1.0f / (1<<14),
    -0.9375,
    -1.0625,
    -199.375,
    -1023.5,

    // infinity
    INFINITY,
    -INFINITY,

    // strings
    "",
    "Hello",
    "Hello 'World' \t \\ \"quotes\"!",
    "$1234567", ///< looks like deflate in CBOR
    "C1234567",
    "b1234567",

    // array
    json::array{},
    json::array{ json::null, true, false, 0, 1, -2, 3.25f },
    json::array{ "abc", json::array{1,2,3}, json::object{{"a",false}} },
    json::array(24,0), ///< appears deflated

    // object
    json::object{},
    json::object{ {"z", 0}, {"x",json::array{1,2}}, {"o",json::object{}} },
    json::object{
        {"00000000000000000",0},{"1",1},{"2",2},{"3",3},
        {"4",4},{"5",5},{"6",6},{"7",7}
    }, ///< appears deflated
};

static const json::value not_amf3[] = {
    // int64 extremes that don't fit in double (amf3)
    9223372036854775807ll,   // 2^63-1
    -9223372036854775807ll,  // -2^63+1
    json::object{ {"","empty key"} }  ///< why is this?
};

static const json::value not_json[] = {
    stdx::binary{},    // 0 bytes
    stdx::binary{""},  // 1 byte (the null terminator)
    stdx::binary{"X"}, // 2 bytes
    stdx::binary{"Hello World!"},  // 13 bytes with null terminator
};

static json::value decode_cbor2(const void* data, std::size_t size) {
    using namespace json;
    std::string in_str(static_cast<const char*>(data),size);
    decoder_input_type in_pair(in_str);
    value_pusher out;
    decoder_input_fn in_fn = push_decode_cbor([&](auto v) { out = v; });
    in_fn(in_pair);
    if (!in_pair.data) {
        FILE_LOG(logDETAIL) << "decode_cbor: sending end-of-stream";
        in_fn(in_pair);  // decoder must finish or throw exception
    }
    return move_value(out);
}
static json::value decode_cbor2(const json::binary& in) {
    return decode_cbor2(in.data(),in.size());
}

template <bool test_amf3 = true, bool test_json = true>
static auto do_test(const json::value& v_orig) {
    FILE_LOG(logDETAIL) << v_orig;
    const auto cbor = encode_cbor(v_orig);
    if (2 <= cbor.size() && json::is_compressed(cbor.data()))
        FILE_LOG(logWARNING) << "CBOR appears deflated: " << json::value(cbor)
                             << ' ' << v_orig;
    const auto v_cbor = json::decode_cbor(cbor);
    if (v_orig != v_cbor) {
        FILE_LOG(logERROR) << "CBOR: " << json::value(cbor);
        BOOST_CHECK_EQUAL(v_orig, v_cbor);
    }
    const auto v_cbor2 = decode_cbor2(cbor);
    if (v_orig != v_cbor2) {
        FILE_LOG(logERROR) << "CBOR: " << json::value(cbor);
        BOOST_CHECK_EQUAL(v_orig, v_cbor2);
    }

    if (test_amf3) {
        const auto amf3 = encode_amf3(v_orig);
        if (2 <= amf3.size() && json::is_compressed(amf3.data()))
            FILE_LOG(logWARNING) << "AMF3 appears deflated: "
                                 << json::value(amf3);
        const auto v_amf3 = json::decode_amf3(amf3);
        if (v_orig != v_amf3) {
            FILE_LOG(logERROR) << "AMF3: " << json::value(amf3);
            BOOST_CHECK_EQUAL(v_orig, v_amf3);
        }
    }

    if (test_json) {
        const auto json = encode_json(v_orig);
        if (2 <= json.size() && json::is_compressed(json.data()))
            FILE_LOG(logWARNING) << "JSON appears deflated: " << json;
        const auto v_json = json::decode_json(json);
        if (v_orig != v_json) {
            FILE_LOG(logERROR) << "JSON: " << json;
            BOOST_CHECK_EQUAL(v_orig, v_json);
        }
    }
}


BOOST_AUTO_TEST_SUITE(json)

BOOST_AUTO_TEST_CASE(basic_all) {
    for (auto const& v : for_all)
        do_test(v);
    for (auto const& v : not_amf3)
        do_test<false,true>(v);
    for (auto const& v : not_json)
        do_test<true,false>(v);
}

static auto pretty_hex(const stdx::binary& bin) {
    std::stringstream ss;
    ss << std::setfill('0') << std::hex;
    auto data = bin.data<uint8_t>();
    for (auto size = bin.size(); size > 0; --size, ++data)
        ss << ' ' << std::setw(2) << unsigned(*data);
    return ss.str();
}

BOOST_AUTO_TEST_CASE(empty_binary) {
    const json::binary empty_binary;

    {
        auto enc = encode_amf3(empty_binary);
        BOOST_CHECK_EQUAL(enc.size(), 2);
        FILE_LOG(logINFO) << "amf3 empty binary:" << pretty_hex(enc);
        auto dec1 = decode_amf3(enc);
        auto dec2 = decode_any(enc);
        BOOST_REQUIRE(is_type<json::binary>(dec1));
        BOOST_CHECK(get_binary(dec1).empty());
        BOOST_REQUIRE(is_type<json::binary>(dec2));
        BOOST_CHECK(get_binary(dec2).empty());
    }

    {
        auto enc = encode_cbor(empty_binary);
        BOOST_CHECK_EQUAL(enc.size(), 4);
        FILE_LOG(logINFO) << "cbor empty binary:" << pretty_hex(enc);
        auto dec1 = decode_cbor(enc);
        auto dec2 = decode_any(enc);
        BOOST_REQUIRE(is_type<json::binary>(dec1));
        BOOST_CHECK(get_binary(dec1).empty());
        BOOST_REQUIRE(is_type<json::binary>(dec2));
        BOOST_CHECK(get_binary(dec2).empty());
    }
}

BOOST_AUTO_TEST_CASE(empty_string) {
    const json::string empty_string;

    {
        auto enc = encode_amf3(empty_string);
        BOOST_CHECK_EQUAL(enc.size(), 2);
        FILE_LOG(logINFO) << "amf3 empty string:" << pretty_hex(enc);
        auto dec1 = decode_amf3(enc);
        auto dec2 = decode_any(enc);
        BOOST_REQUIRE(is_type<json::string>(dec1));
        BOOST_CHECK(get_string(dec1).empty());
        BOOST_REQUIRE(is_type<json::string>(dec2));
        BOOST_CHECK(get_string(dec2).empty());
    }

    {
        auto enc = encode_cbor(empty_string);
        BOOST_CHECK_EQUAL(enc.size(), 4);
        FILE_LOG(logINFO) << "cbor empty string:" << pretty_hex(enc);
        auto dec1 = decode_cbor(enc);
        auto dec2 = decode_any(enc);
        BOOST_REQUIRE(is_type<json::string>(dec1));
        BOOST_CHECK(get_string(dec1).empty());
        BOOST_REQUIRE(is_type<json::string>(dec2));
        BOOST_CHECK(get_string(dec2).empty());
    }
}

BOOST_AUTO_TEST_CASE(empty_array) {
    const json::array empty_array;

    {
        auto enc = encode_amf3(empty_array);
        BOOST_CHECK_EQUAL(enc.size(), 3);
        FILE_LOG(logINFO) << "amf3 empty array: " << pretty_hex(enc);
        auto dec1 = decode_amf3(enc);
        auto dec2 = decode_any(enc);
        BOOST_REQUIRE(is_type<json::array>(dec1));
        BOOST_CHECK(get_array(dec1).empty());
        BOOST_REQUIRE(is_type<json::array>(dec2));
        BOOST_CHECK(get_array(dec2).empty());
    }

    {
        auto enc = encode_cbor(empty_array);
        BOOST_CHECK_EQUAL(enc.size(), 4);
        FILE_LOG(logINFO) << "cbor empty array: " << pretty_hex(enc);
        auto dec1 = decode_cbor(enc);
        auto dec2 = decode_any(enc);
        BOOST_REQUIRE(is_type<json::array>(dec1));
        BOOST_CHECK(get_array(dec1).empty());
        BOOST_REQUIRE(is_type<json::array>(dec2));
        BOOST_CHECK(get_array(dec2).empty());
    }

    {
        const json::array array1 { json::null };
        auto enc = encode_cbor(array1);
        BOOST_CHECK_EQUAL(enc.size(), 2);
        FILE_LOG(logINFO) << "cbor array [null]:" << pretty_hex(enc);
    }
}

BOOST_AUTO_TEST_CASE(empty_object) {
    const json::object empty_object;

    {
        auto enc = encode_amf3(empty_object);
        BOOST_CHECK_EQUAL(enc.size(), 4);
        FILE_LOG(logINFO) << "amf3 empty object:" << pretty_hex(enc);
        auto dec1 = decode_amf3(enc);
        auto dec2 = decode_any(enc);
        BOOST_REQUIRE(is_type<json::object>(dec1));
        BOOST_CHECK(get_object(dec1).empty());
        BOOST_REQUIRE(is_type<json::object>(dec2));
        BOOST_CHECK(get_object(dec2).empty());
    }

    {
        auto enc = encode_cbor(empty_object);
        BOOST_CHECK_EQUAL(enc.size(), 4);
        FILE_LOG(logINFO) << "cbor empty object:" << pretty_hex(enc);
        auto dec1 = decode_cbor(enc);
        auto dec2 = decode_any(enc);
        BOOST_REQUIRE(is_type<json::object>(dec1));
        BOOST_CHECK(get_object(dec1).empty());
        BOOST_REQUIRE(is_type<json::object>(dec2));
        BOOST_CHECK(get_object(dec2).empty());
    }

    {
        const json::object object1 { { "", 0 } };
        auto enc = encode_cbor(object1);
        BOOST_CHECK_EQUAL(enc.size(), 3);
        FILE_LOG(logINFO) << "cbor object{\"\":0}:" << pretty_hex(enc);
    }
}

BOOST_AUTO_TEST_CASE(cbor_magic_tag) {
    // tagged encoding of integer zero
    uint8_t const cbor[] = { 0xd9, 0xd9, 0xf7, 0 };
    const auto v = json::decode_any(cbor, 4);
    BOOST_REQUIRE_EQUAL(v, 0);
    BOOST_REQUIRE(!json::is_compressed(cbor));

    // search for cbor encodings that have first 2 bytes matching
    // deflate header
    if (0) {
        for (int i = 24; i < 256; ++i)
            for (auto j : { i, -i-1 }) {
                const auto cbor = encode_cbor(j);
                BOOST_REQUIRE_EQUAL(cbor.size(), 2);
                if (json::is_compressed(cbor.data()))
                    FILE_LOG(logWARNING) << "CBOR encoding of " << j
                                         << " looks like deflate header";
            }
    }
    if (0) {
        std::string s = "01234567";
        for (char c = ' '; c < 127; ++c) {
            s.front() = c;
            const auto cbor = encode_cbor(s);
            BOOST_REQUIRE_EQUAL(cbor.size(), 9);
            if (json::is_compressed(cbor.data()))
                FILE_LOG(logWARNING) << "CBOR encoding of '" << s
                                     << "' looks like deflate header";
        }
    }
    if (0) {
        for (unsigned n = 24; n < 256; ++n) {
            json::array a(n,0);
            const auto cbor = encode_cbor(a);
            BOOST_REQUIRE_EQUAL(cbor.size(), 2+n);
            if (json::is_compressed(cbor.data()))
                FILE_LOG(logWARNING) << "CBOR encoding of "
                                     << json::value(a) << ' ' << n
                                     << " looks like deflate header";
        }
    }
    if (0) {
        json::object o;
        for (unsigned i = 0; i < 8; ++i)
            o.emplace(std::to_string(i),0);
        for (unsigned n = 0; n < 24; ++n) {
            o.erase(o.begin());
            o.emplace(std::string(n,'0'),0);
            BOOST_REQUIRE_EQUAL(o.size(), 8);
            const auto cbor = encode_cbor(o);
            //BOOST_REQUIRE_EQUAL(cbor.size(), 2+3*n);
            if (json::is_compressed(cbor.data()))
                FILE_LOG(logWARNING) << "CBOR encoding of "
                                     << json::value(o) << ' ' << n
                                     << " looks like deflate header";
        }
    }
}

BOOST_AUTO_TEST_SUITE_END()
