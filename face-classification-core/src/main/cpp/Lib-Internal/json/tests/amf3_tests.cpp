#include <fstream>

#include <boost/test/unit_test.hpp>
#include <boost/bind/bind.hpp>

#include <applog/core.hpp>
#include <applog/base_directory.hpp>

#include <json/json.hpp>


BOOST_AUTO_TEST_SUITE(json)

using namespace json;

namespace {
    struct value_pusher_handler {
        void operator()(value_pusher strm) {
            stream = strm;
        }
        value_pusher_handler(value_pusher& stream) : stream(stream) {}
        value_pusher& stream;
    };
}

static value decode_json_file(const std::filesystem::path& filename) {
    std::ifstream in(filename, std::ios_base::in|std::ios_base::binary);
    BOOST_CHECK(in.good());
    in.exceptions(std::ifstream::badbit);

    value_pusher result;
    value_pusher_handler handler(result);
    decoder_input_fn in_fn = push_decode_json(handler);

    bool done = false;

    char buf[4096];
    while (!in.eof()) {
        in.read(buf,sizeof(buf));
        std::streamsize n = in.gcount();

        if (done) {
            // check that nothing but whitespace is in buffer
            const char* p = buf;
            while (n > 0) {
                BOOST_CHECK(std::isspace(*p++));
                --n;
            }
            continue;
        }

        assert(n >= 0);
        std::string str(buf,std::size_t(n));
        decoder_input_type in_str(str);

        in_fn(in_str);
        if (in_str.data) {
            done = true;
            // check that nothing but whitespace remains
            while (in_str.pos != in_str.data->end()) {
                BOOST_CHECK(std::isspace(*in_str.pos));
                ++in_str.pos;
            }
        }
    }

    BOOST_CHECK(done);

    return result.final_value();
}

static value decode_json_vector(const std::vector<string>& input) {
    value_pusher result;
    value_pusher_handler handler(result);
    decoder_input_fn in_fn = push_decode_json(handler);
    for (std::vector<string>::const_iterator
             it=input.begin(),end=input.end(); it!=end; ++it) {
        std::string s = *it;
        decoder_input_type in_str(s);
        in_str.pos = in_str.data->begin();
        in_fn(in_str);
        if (in_str.data) {
            BOOST_CHECK(in_str.pos == in_str.data->end());
            BOOST_CHECK(++it == end);
            break;
        }
    }
    return result.final_value();
}

static value decode_amf3_file(const std::filesystem::path& filename) {
    std::ifstream in(filename, std::ios_base::in|std::ios_base::binary);
    BOOST_CHECK(in.good());
    in.exceptions(std::ifstream::badbit);

    value_pusher result;
    value_pusher_handler handler(result);
    decoder_input_fn in_fn = push_decode_amf3(handler);

    bool done = false;

    char buf[4096];
    while (!in.eof()) {
        in.read(buf,sizeof(buf));
        std::streamsize n = in.gcount();

        if (done) {
            BOOST_CHECK(n == 0);
            continue;
        }

        assert(n >= 0);
        std::string str(buf,std::size_t(n));
        decoder_input_type in_str(str);

        in_fn(in_str);
        if (in_str.data) {
            done = true;
            BOOST_CHECK(in_str.pos == in_str.data->end());
        }
    }

    BOOST_CHECK(done);

    return result.final_value();
}

static value decode_amf3_vector(const std::vector<string>& input) {
    value_pusher result;
    value_pusher_handler handler(result);
    decoder_input_fn in_fn = push_decode_amf3(handler);
    for (std::vector<string>::const_iterator
             it=input.begin(),end=input.end(); it!=end; ++it) {
        std::string s = *it;
        decoder_input_type in_str(s);
        in_fn(in_str);
        if (in_str.data) {
            BOOST_CHECK(in_str.pos == in_str.data->end());
            BOOST_CHECK(++it == end);
            break;
        }
    }
    return result.final_value();
}

static void pull_encode_json(std::vector<string>& dest, const value& val,
                             unsigned buffer_size,
                             unsigned copy_threshold) {
    value_puller stream(val);
    string_puller out_fn = pull_encode_json(stream,buffer_size,copy_threshold);
    for (auto buf = out_fn(); buf; buf = out_fn())
        dest.push_back(*buf);
}

namespace {
    template<typename T>
    struct vector_push_back {
        std::vector<T>& dest;
        vector_push_back(std::vector<T>& dest) : dest(dest) {}
        void operator()(const std::optional<T>& val) {
            if (val && !val->empty())
                dest.push_back(*val);
        }
    };
}

static void pull_encode_amf3(std::vector<binary>& dest, const value& val,
                             unsigned buffer_size,
                             unsigned copy_threshold) {
    value_puller stream(val);
    binary_puller out_fn = 
        pull_encode_amf3(stream,buffer_size,copy_threshold);
    for (auto buf = out_fn(); buf; buf = out_fn())
        dest.push_back(*buf);
}

static void binary_to_string(std::vector<string>& dest, 
                             const std::vector<binary>& src) {
    for (std::vector<binary>::const_iterator
             it=src.begin(),end=src.end(); it!=end; ++it) {
        string str(static_cast<const char*>(it->data()),it->size());
        dest.push_back(str);
    }
}

BOOST_AUTO_TEST_CASE(amf3_decode) {
    const auto test_dir = base_directory("lib-internal") / "json" / "tests";

    const auto json_obj_fn = test_dir / "sample_object.json";
    const auto amf3_obj1_fn = test_dir / "sample_object_1.amf3";
    const auto amf3_obj2_fn = test_dir / "sample_object_2.amf3";
    value json_obj = decode_json_file(json_obj_fn);
    value amf3_obj1 = decode_amf3_file(amf3_obj1_fn);
    value amf3_obj2 = decode_amf3_file(amf3_obj2_fn);
    BOOST_CHECK(json_obj == amf3_obj1);
    BOOST_CHECK(json_obj == amf3_obj2);

    std::vector<binary> amf3_obj_enc_bin;
    pull_encode_amf3(amf3_obj_enc_bin, json_obj, 4096, 1024);
    std::vector<string> amf3_obj_enc_str;
    binary_to_string(amf3_obj_enc_str,amf3_obj_enc_bin);
    value amf3_obj_dec = decode_amf3_vector(amf3_obj_enc_str);
    BOOST_CHECK(json_obj == amf3_obj_dec);

    //std::ofstream("dump.amf3") << *amf3_obj_enc_str.begin();


    value json_obj_ref = decode_json("[{\"obj\":\"A\",\"child\":{\"obj\":\"B\"}},{\"obj\":\"B\"}]");
    const auto amf3_obj_ref_fn = test_dir / "sample_object_ref.amf3";
    value amf3_obj_ref = decode_amf3_file(amf3_obj_ref_fn);
    BOOST_CHECK(json_obj_ref == amf3_obj_ref);


    const auto amf3_obj_cyc_fn = test_dir / "sample_object_cyc.amf3";
    FILE_LOG(logWARNING) << "======== ERRORS EXPECTED START ========";
    BOOST_CHECK_THROW(decode_amf3_file(amf3_obj_cyc_fn), std::runtime_error);
    FILE_LOG(logWARNING) << "======== ERRORS EXPECTED END ========";

    
    const auto json_arr_fn = test_dir / "sample_array.json";
    const auto amf3_arr_fn = test_dir / "sample_array.amf3";
    FILE_LOG(logDETAIL) << "reading " << json_arr_fn;
    value json_arr = decode_json_file(json_arr_fn);
    FILE_LOG(logDETAIL) << "reading " << amf3_arr_fn;
    value amf3_arr = decode_amf3_file(amf3_arr_fn);
    FILE_LOG(logDETAIL) << "reading done";
    BOOST_CHECK(json_arr == amf3_arr);

    // amf3
    std::vector<binary> amf3_arr_enc_bin;
    FILE_LOG(logDETAIL) << "amf3 encoding array (pull)";
    pull_encode_amf3(amf3_arr_enc_bin, amf3_arr, 4096, 1024);
    FILE_LOG(logDETAIL) << "encode done";
    std::vector<string> amf3_arr_enc_str;
    binary_to_string(amf3_arr_enc_str,amf3_arr_enc_bin);
    value amf3_arr_dec = decode_amf3_vector(amf3_arr_enc_str);
    BOOST_CHECK(amf3_arr == amf3_arr_dec);

    size_t total = 0;
    for (std::vector<binary>::const_iterator
             it=amf3_arr_enc_bin.begin(),
             end=amf3_arr_enc_bin.end(); it!=end; ++it) {
        total += it->size();
        //FILE_LOG(logDETAIL) << it->size();
    }
    FILE_LOG(logDETAIL) << "amf3 encode size: " << total;

    // json
    std::vector<string> json_arr_enc_str;
    FILE_LOG(logDETAIL) << "json encoding array (pull)";
    pull_encode_json(json_arr_enc_str, json_arr, 4096, 1024);
    FILE_LOG(logDETAIL) << "encode done";
    value json_arr_dec = decode_json_vector(json_arr_enc_str);
    BOOST_CHECK(json_arr == json_arr_dec);

    total = 0;
    for (std::vector<string>::const_iterator
             it=json_arr_enc_str.begin(),
             end=json_arr_enc_str.end(); it!=end; ++it) {
        total += it->size();
        //FILE_LOG(logDETAIL) << it->size();
    }
    FILE_LOG(logDETAIL) << "json encode size: " << total;
}

BOOST_AUTO_TEST_CASE(amf3_integer_test) {
    const auto test = [](integer x) {
        const auto y = decode_amf3(encode_amf3(x));
        BOOST_CHECK_EQUAL(x, get_integer(y));
    };
    for (auto shift : { 5, 12, 19, 26, 30, 38 })
        for (integer base = 0; base < 64; ++base) {
            const auto x = base << shift;
            test(x-1);
            test(x);
            test(x+1);
            test(-x-1);
            test(-x);
            test(-x+1);
        }
}

BOOST_AUTO_TEST_CASE(amf3_albums_test) {
    string json_enc("{\"type\":2,\"id\":[1,[1,3]],\"code\":0,\"data\":[{\"AlbumId\":-4,\"Name\":\"Photos with unidentified faces\",\"Query\":\"\",\"Type\":\"system\"},{\"AlbumId\":-3,\"Name\":\"Photos not in an album\",\"Query\":\"\",\"Type\":\"system\"},{\"AlbumId\":-2,\"Name\":\"Recently added photos\",\"Query\":\"\",\"Type\":\"system\"},{\"AlbumId\":1,\"HiddenCount\":0,\"ImageCount\":3,\"Name\":\"Halloween\",\"Query\":null,\"ThumbnailAuto\":true,\"ThumbnailId\":9,\"Type\":\"fixed\"},{\"AlbumId\":2,\"HiddenCount\":0,\"ImageCount\":0,\"Name\":\"new\",\"Query\":null,\"ThumbnailAuto\":true,\"ThumbnailId\":null,\"Type\":\"fixed\"}]}");
    std::vector<string> json_enc_vec;
    json_enc_vec.push_back(json_enc);
    value obj = decode_json_vector(json_enc_vec);

    std::vector<binary> amf3_enc_bin;
    pull_encode_amf3(amf3_enc_bin, obj, 4096, 1024);
    std::vector<string> amf3_enc_str;
    binary_to_string(amf3_enc_str,amf3_enc_bin);

    //std::string s = *amf3_enc_str.begin();
    //std::ofstream of("dump.amf3");
    //of << s;

    value obj2 = decode_amf3_vector(amf3_enc_str);

    BOOST_CHECK(obj == obj2);
}

/*
template <typename U, typename V>
static inline void repeat_push(U& stream, const V& value, unsigned n) {
    for (; n > 0; --n)
        stream(value);
}

static void push_binary_to_string(decoder_input_fn fn, 
                                  std::optional<binary> bin) {
    if (bin) {
        std::string s(static_cast<const char*>(bin->data()),bin->size());
        decoder_input_type in(s);
        fn(in);
        if (!in.data) {
            //FILE_LOG(logDETAIL) << "input consumed";
        }
        else if (in.pos == in.data->end())
            FILE_LOG(logDETAIL) << "input consumed (decode complete)";
        else
            FILE_LOG(logDETAIL) << "input remaining";
    }
    else
        FILE_LOG(logDETAIL) << "end of input";
}

static size_t value_size(const json::value& v) {
    if (json::is_type<json::string>(v))
        return get_string(v).size();
    if (json::is_type<json::binary>(v))
        return get_binary(v).size();
    if (json::is_type<json::array>(v))
        return get_array(v).size();
    if (json::is_type<json::object>(v))
        return get_object(v).size();
    return 1;
}

template <typename PUSHER, typename VALUE>
static void test_push_stream(const VALUE& v, amf3_stream_type stream_type) {
    FILE_LOG(logINFO) << "test_push_stream: " << typeid(PUSHER).name();

    using value_type = typename PUSHER::value_type;

    PUSHER expected_output;
    for (unsigned i=0; i<1024; ++i)
        expected_output(value_type(v));
    expected_output();
    value expected_value = expected_output.final_value();
    FILE_LOG(logDETAIL) << "expected_value: "
                       << max_string(8)
                       << binary_subst("BINARY")
                       << max_array(2)
                       << expected_value
                       << " [" << value_size(expected_value) << "]";

    PUSHER input;
    binary_pusher decompressed;
    binary_pusher compressed = push_inflate(decompressed,16);
    binary_pusher encoded = push_deflate(compressed,16);
    push_encode_amf3(encoded,input,64,32,stream_type);
    
    value_pusher val_output;
    decoder_input_fn input_fn = 
        push_decode_amf3(value_pusher_handler(val_output));
    decompressed.set_value_handler(
        bind(&push_binary_to_string,input_fn,boost::placeholders::_1));

    // let er rip
    FILE_LOG(logDETAIL) << "pushing 1024 values";
    for (unsigned i=0; i<1024; ++i)
        input(value_type(v));
    FILE_LOG(logDETAIL) << "pushing end-of-stream";
    input();
    FILE_LOG(logDETAIL) << "done push";

    value final_value = val_output.final_value();
    FILE_LOG(logDETAIL) << "final_value: "
                       << max_string(8)
                       << binary_subst("BINARY")
                       << max_array(2)
                       << final_value
                       << " [" << value_size(final_value) << "]";
    
    BOOST_CHECK(expected_value == final_value);

    FILE_LOG(logINFO) << "test_push_stream: " << typeid(PUSHER).name()
                      << " (done)";
}

BOOST_AUTO_TEST_CASE(amf3_push_stream_test) {
    string s(64,'A');
    test_push_stream<string_pusher>(s,amf3_stream_buffer);
    test_push_stream<string_pusher>(s,amf3_stream_extension);

    binary b(s);
    test_push_stream<binary_pusher>(b,amf3_stream_buffer);
    test_push_stream<binary_pusher>(b,amf3_stream_extension);
    
    test_push_stream<array_pusher>(s,amf3_stream_extension);
}
*/


BOOST_AUTO_TEST_SUITE_END()
