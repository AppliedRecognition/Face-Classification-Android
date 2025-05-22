#include <boost/test/unit_test.hpp>
#include <applog/base_directory.hpp>
#include <applog/core.hpp>

#include <raw_image/input_extractor_retina.hpp>
#include <dlibx/input_extractor_facechip.hpp>

BOOST_AUTO_TEST_SUITE(dlibx)

BOOST_AUTO_TEST_CASE(facechip_test) {
    FILE_LOG(logINFO) << "--";

    using lv = std::pair<raw_image::pixel_layout,const char*>;
    for (const auto& layout : {
            lv { raw_image::pixel::rgb24, "rgb"  },
            lv { raw_image::pixel::yuv,   "yuv"  },
            lv { raw_image::pixel::gray8, "gray" } }) {

        for (auto size : { 1, 15, 160, 224 }) {

            using pv = std::pair<float,const char*>;
            for (const auto& pad : {
                    pv { 0.0f,  "+0" },
                    pv { 0.25f, "+025" },
                    pv { 0.25f, "+0.25" },
                    pv { 1.25f, "+1.25" },
                    pv { 1.0f,  "+1" },
                    pv { 10.0f, "+10" },
                    pv { 12.0f, "+12" },
                    pv { -0.0f,   "-0" },
                    pv { -0.125f, "-0125" },
                    pv { -0.125f, "-0.125" } }) {

                std::stringstream ss;
                ss << "facechip" << size << pad.second << layout.second;
                const auto name = ss.str();
                const auto t = facechip_decode(name);
                BOOST_CHECK_EQUAL(std::get<0>(t), size);
                BOOST_CHECK_EQUAL(std::get<1>(t), pad.first);
                BOOST_CHECK_EQUAL(std::get<2>(t), layout.first);
            }
        }
    }

    FILE_LOG(logINFO) << "facechip: done";
}

BOOST_AUTO_TEST_CASE(retina_test) {
    FILE_LOG(logINFO) << "--";

    using lv = std::pair<raw_image::pixel_layout,const char*>;
    for (const auto& layout : {
            lv { raw_image::pixel::rgb24, "rgb"  },
            lv { raw_image::pixel::yuv,   "yuv"  },
            lv { raw_image::pixel::gray8, "gray" } }) {
        for (auto size : { 1, 15, 112, 160, 224 })
            for (auto scale = 0.1f; scale < 4; scale += 0.7f)
                for (auto yofs = -0.25f; yofs < 0.5f; yofs += 0.1f) {
                    std::stringstream ss;
                    ss << "retina" << size << '*' << scale
                       << std::showpos << yofs << layout.second;
                    const auto name = ss.str();
                    FILE_LOG(logTRACE) << name;
                    const auto t = raw_image::retina_decode(name);
                    const auto serr = std::abs(std::get<1>(t) - scale);
                    const auto cerr = std::abs(std::get<2>(t) - yofs);
                    BOOST_CHECK_EQUAL(std::get<0>(t), size);
                    BOOST_CHECK_LT(serr, 1e-5);
                    BOOST_CHECK_LT(cerr, 1e-5);
                    BOOST_CHECK_EQUAL(std::get<3>(t), layout.first);
                }
    }

    FILE_LOG(logINFO) << "retina: done";
}

BOOST_AUTO_TEST_SUITE_END()
