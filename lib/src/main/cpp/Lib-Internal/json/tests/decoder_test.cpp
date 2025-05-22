#include <boost/test/unit_test.hpp>
#include <boost/bind/bind.hpp>

#include <applog/core.hpp>
#include <applog/base_directory.hpp>

#include <json/json.hpp>


BOOST_AUTO_TEST_SUITE(json)

using namespace json;


static void pull_encode_amf3(std::vector<binary>& dest, const value& val,
                             unsigned buffer_size,
                             unsigned copy_threshold) {
    value_puller stream(val);
    binary_puller out_fn =
        pull_encode_amf3(stream,buffer_size,copy_threshold);
    for (auto buf = out_fn(); buf; buf = out_fn())
        dest.push_back(*buf);
}

namespace {
    struct obj_push {
        template <typename ITER>
        void operator()(ITER begin, ITER end) {
            if (begin == end) {
                FILE_LOG(logTRACE) << "object done";
                obj();
            }
            else {
                for (ITER it = begin; it != end; ++it)
                    FILE_LOG(logTRACE) << "object value: " << (*it).first;
                obj(begin,end);
            }
        }
        object_pusher obj;
    };
}


static void decoder_output(value_pusher val) {
    object_pusher obj = get<object_pusher>(val);
    obj.set_range_handler(obj_push());
}

BOOST_AUTO_TEST_CASE(amf3_decode_test) {
    //static const std::string test_json_str = "{\"args\":{\"DateBins\":2,\"Query\":[{\"hidden\":false}]},\"id\":\"12526077139490.05563201801851392\",\"method\":\"photoSummary\",\"type\":1}";
    static const std::string test_json_str = "{\"xargs\":{\"DateBins\":2,\"Query\":[{\"hidden\":false}]},\"id\":\"12526077139490.05563201801851392\",\"method\":\"photoSummary\",\"type\":1}";
    const value test_obj = decode_json(test_json_str);

    std::vector<binary> enc_bin;
    pull_encode_amf3(enc_bin,test_obj,1024,1024);
    assert(enc_bin.size() == 1);
    const string enc_str(static_cast<const char*>(enc_bin[0].data()),
                         enc_bin[0].size());
    
    for (size_t i=1; i<enc_str.size(); ++i) {
        std::string s0 = enc_str.substr(0,i);
        std::string s1 = enc_str.substr(i);
        decoder_input_fn infn = push_decode_amf3(&decoder_output);

        FILE_LOG(logDETAIL) << "input: " << s0.size() << " bytes";
        decoder_input_type in0(s0);
        infn(in0);
        assert(!in0.data);

        FILE_LOG(logDETAIL) << "input: " << s1.size() << " bytes";
        decoder_input_type in1(s1);
        infn(in1);
        assert(in1.data && in1.pos == in1.data->end());
    }
}

BOOST_AUTO_TEST_SUITE_END()
