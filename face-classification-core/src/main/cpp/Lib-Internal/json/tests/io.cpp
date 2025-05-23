#include <boost/test/unit_test.hpp>

#include <json/json.hpp>

#include <applog/core.hpp>
#include <applog/base_directory.hpp>

#include <filesystem>
#include <fstream>


BOOST_AUTO_TEST_SUITE(json)

BOOST_AUTO_TEST_CASE(save_load_tests) {
    const auto test_dir = base_directory("lib-internal") / "json" / "tests";
    const auto sample_array_json = test_dir / "sample_array.json";
    const auto sample_array_amf3 = test_dir / "sample_array.amf3";

    const auto sa_json = load(sample_array_json);
    const auto sa_amf3 = load(sample_array_amf3);
    value sa_json_ios;
    std::ifstream(sample_array_json) >> sa_json_ios;
    BOOST_REQUIRE_EQUAL(sa_json, sa_json_ios);
    BOOST_REQUIRE_EQUAL(sa_amf3, sa_json_ios);

    const auto test_json = test_dir / "test_json.bin";
    const auto test_amf3 = test_dir / "test_amf3.bin";
    const auto test_json_deflate = test_dir / "test_jz.bin";
    const auto test_amf3_deflate = test_dir / "test_az.bin";

    save(sa_json, test_json, json);
    save(sa_json, test_amf3, amf3);
    save(sa_json, test_json_deflate, json, deflate);
    save(sa_json, test_amf3_deflate, amf3, deflate);

    for (auto p : {
            test_json, test_json_deflate, test_amf3, test_amf3_deflate }) {
        auto a = load(p);
        BOOST_CHECK_EQUAL(a, sa_amf3);
    }
}

BOOST_AUTO_TEST_SUITE_END()
