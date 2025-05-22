
#include <rec_dlib/engine.hpp>
#include <rec/prototype.hpp>

#include <boost/test/unit_test.hpp>
#include <applog/base_directory.hpp>
#include <applog/core.hpp>

using namespace rec::dlib;

BOOST_AUTO_TEST_SUITE(rec_dlib)

BOOST_AUTO_TEST_CASE(rec_model_init) {
    const auto models_path = base_directory("lib-internal") / "models";

    core::context_settings cs;
    auto c = core::context::construct(cs);
    initialize(c, models_path);

    rec::prototype::load_model(c, 16);
}

BOOST_AUTO_TEST_SUITE_END()
