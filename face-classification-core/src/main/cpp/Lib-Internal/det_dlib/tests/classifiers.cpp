#include <boost/test/unit_test.hpp>
#include <applog/base_directory.hpp>
#include <applog/core.hpp>

#include <raw_image_io/io.hpp>

#include <det_dlib/init.hpp>
#include <det/image.hpp>
#include <det/classifiers.hpp>

#include <core/thread_data.hpp>
#include <core/job_queue.hpp>

using namespace det;

BOOST_AUTO_TEST_SUITE(det)

BOOST_AUTO_TEST_CASE(det_classifiers) {
    const auto models_path = base_directory("lib-internal") / "models";
    const auto base_path = base_directory("lib-internal") / "det" / "tests";
    const auto img_path = base_path / "image_077.jpg";

    FILE_LOG(logINFO) << "classifiers: start";
    core::context_settings cs;
    cs.max_threads = 4; // auto detect to maximum of 4
    auto c = core::context::construct(cs);
    set_models_path(c, models_path);
    det::dlib::init(c);

    detection_settings s;
    s.detector_version = 3;
    s.confidence_threshold = 0.0f;
    s.size_range = 5;
    s.landmark_detection = lm::dlib5 + lm::dlib68;

    const auto img_raw = raw_image::load(img_path);
    auto classifiers = det::apply_classifiers(img_raw, {}, {});
    const auto image = use_pixels(c, s, *img_raw);
    auto detected = detect_faces(c, s, image, std::move(classifiers));

    BOOST_REQUIRE(!detected.empty());
    for (auto&& face : detected) {
        for (auto&& dc : face)
            FILE_LOG(logINFO) << to_string(dc.type) << ' ' << dc.confidence;
        BOOST_REQUIRE(face.size() > 1);
    }
    
    
    FILE_LOG(logINFO) << "classifiers: done";
}

BOOST_AUTO_TEST_SUITE_END()
