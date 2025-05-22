#include <boost/test/unit_test.hpp>

#include <cmath>
#include <applog/core.hpp>
#include <json/json.hpp>

BOOST_AUTO_TEST_SUITE(json)

BOOST_AUTO_TEST_CASE(float_special) {
    const json::real nan = NAN;
    const json::real nnan = -NAN;

    const auto nan_enc = json::encode_json(nan);
    const auto nnan_enc = json::encode_json(nnan);
    FILE_LOG(logINFO) << "json nan encoding: " << nan_enc << '\t' << nnan_enc;

    const auto nan_dec = json::decode_json(nan_enc);
    const auto nnan_dec = json::decode_json(nnan_enc);
    BOOST_CHECK(is_null(nan_dec));
    BOOST_CHECK(is_null(nnan_dec));


    const json::real inf = INFINITY;
    const json::real ninf = -INFINITY;

    const auto inf_enc = json::encode_json(inf);
    const auto ninf_enc = json::encode_json(ninf);
    FILE_LOG(logINFO) << "json inf encoding: " << inf_enc << '\t' << ninf_enc;

    const auto inf_dec = json::decode_json(inf_enc);
    const auto ninf_dec = json::decode_json(ninf_enc);
    BOOST_CHECK(inf_dec == inf);
    BOOST_CHECK(ninf_dec == ninf);
}

BOOST_AUTO_TEST_SUITE_END()
