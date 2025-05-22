#include <boost/test/unit_test.hpp>

#include <json/push_decode_cbor.hpp>

#include <applog/core.hpp>

BOOST_AUTO_TEST_SUITE(json)

BOOST_AUTO_TEST_CASE(cbor_float) {
    // these should all encode as float32
    for (double val : { 0.0, 1.25, -2.0, -3.125, double(INFINITY) }) {
        auto enc = encode_cbor(val);
        BOOST_CHECK_EQUAL(enc.size(),5);
    }
    // these should encode as double
    for (double val : { M_PI, double(NAN), double(-NAN) }) {
        auto enc = encode_cbor(val);
        BOOST_CHECK_EQUAL(enc.size(),9);
    }
}

BOOST_AUTO_TEST_CASE(cbor_indefinite_string) {
    json::value_pusher out;
    auto in_fn = push_decode_cbor([&](auto v) { out = v; });

    std::string orig; // string that was encoded
    std::string enc;  // complete encoding

    {   
        std::string header;
        header += char(0x7f); // indefinite string header
        enc += header;
        json::decoder_input_type in_pair(header);
        in_fn(in_pair);
        BOOST_REQUIRE(in_pair.data == nullptr);
    }

    {
        std::string empty(1,char(0x60)); // encoding of empty string
        enc += empty;
        json::decoder_input_type in_pair(empty);
        in_fn(in_pair);
        BOOST_REQUIRE(in_pair.data == nullptr);
    }
    
    for (unsigned i = 0; i < 150; ++i) {
        auto s = std::to_string(i);
        orig += s;
        s.insert(0, 1, char(0x60 + s.size()));
        enc += s;
        json::decoder_input_type in_pair(s);
        in_fn(in_pair);
        BOOST_REQUIRE(in_pair.data == nullptr);
    }

    {
        std::string empty(1,char(0x60)); // encoding of empty string
        enc += empty;
        json::decoder_input_type in_pair(empty);
        in_fn(in_pair);
        BOOST_REQUIRE(in_pair.data == nullptr);
    }
    
    {   
        std::string term;
        term += char(0xff); // indefinite terminator
        enc += term;
        json::decoder_input_type in_pair(term);
        in_fn(in_pair);
        BOOST_REQUIRE(in_pair.data != nullptr &&
                      in_pair.pos == in_pair.data->end());
    }

    const auto val0 = out.final_value();
    BOOST_REQUIRE(json::is_type<json::string>(val0));
    BOOST_CHECK_EQUAL(orig, get_string(val0));

    const auto val1 = decode_cbor(stdx::binary(move(enc)));
    BOOST_REQUIRE(json::is_type<json::string>(val1));
    BOOST_CHECK_EQUAL(orig, get_string(val1));
}

BOOST_AUTO_TEST_CASE(cbor_indefinite_array) {
    json::value_pusher out;
    auto in_fn = push_decode_cbor([&](auto v) { out = v; });

    json::array orig; // array that was encoded
    std::string enc;  // complete encoding

    {   
        std::string header;
        header += char(0x9f); // indefinite array header
        enc += header;
        json::decoder_input_type in_pair(header);
        in_fn(in_pair);
        BOOST_REQUIRE(in_pair.data == nullptr);
    }
    
    for (unsigned i = 10; i < 300; ++i) {
        orig.push_back(i);
        std::string s;
        if (i < 24) s += char(i);
        else if (i < 256) {
            s += char(24);
            s += char(i);
        }
        else {
            s += char(25);
            s += char(i>>8);
            s += char(i);
        }
        enc += s;
        json::decoder_input_type in_pair(s);
        in_fn(in_pair);
        BOOST_REQUIRE(in_pair.data == nullptr);
    }

    {   
        std::string term;
        term += char(0xff); // indefinite terminator
        enc += term;
        json::decoder_input_type in_pair(term);
        in_fn(in_pair);
        BOOST_REQUIRE(in_pair.data != nullptr &&
                      in_pair.pos == in_pair.data->end());
    }

    const auto val0 = out.final_value();
    BOOST_REQUIRE(json::is_type<json::array>(val0));
    BOOST_CHECK_EQUAL(orig, get_array(val0));

    const auto val1 = decode_cbor(stdx::binary(move(enc)));
    BOOST_REQUIRE(json::is_type<json::array>(val1));
    BOOST_CHECK_EQUAL(orig, get_array(val1));
}

BOOST_AUTO_TEST_CASE(cbor_indefinite_object) {
    json::value_pusher out;
    auto in_fn = push_decode_cbor([&](auto v) { out = v; });

    json::object orig; // object that was encoded
    std::string enc;   // complete encoding

    {   
        std::string header;
        header += char(0xbf); // indefinite object header
        enc += header;
        json::decoder_input_type in_pair(header);
        in_fn(in_pair);
        BOOST_REQUIRE(in_pair.data == nullptr);
    }
    
    for (unsigned i = 10; i < 300; ++i) {
        auto s0 = std::to_string(i);
        orig.emplace(s0,i);
        s0.insert(0, 1, char(0x60 + s0.size()));
        enc += s0;
        std::string s1;
        if (i < 24) s1 += char(i);
        else if (i < 256) {
            s1 += char(24);
            s1 += char(i);
        }
        else {
            s1 += char(25);
            s1 += char(i>>8);
            s1 += char(i);
        }
        enc += s1;
        json::decoder_input_type in_pair(s0);
        in_fn(in_pair);
        BOOST_REQUIRE(in_pair.data == nullptr);
        in_pair = s1;
        in_fn(in_pair);
        BOOST_REQUIRE(in_pair.data == nullptr);
    }

    {   
        std::string term;
        term += char(0xff); // indefinite terminator
        enc += term;
        json::decoder_input_type in_pair(term);
        in_fn(in_pair);
        BOOST_REQUIRE(in_pair.data != nullptr &&
                      in_pair.pos == in_pair.data->end());
    }

    const auto val0 = out.final_value();
    BOOST_REQUIRE(json::is_type<json::object>(val0));
    BOOST_CHECK_EQUAL(orig, get_object(val0));

    const auto val1 = decode_cbor(stdx::binary(move(enc)));
    BOOST_REQUIRE(json::is_type<json::object>(val1));
    BOOST_CHECK_EQUAL(orig, get_object(val1));
}

BOOST_AUTO_TEST_SUITE_END()
