#include <boost/test/unit_test.hpp>
#include <applog/base_directory.hpp>
#include <applog/core.hpp>

#include <raw_image_io/io.hpp>
#include <raw_image/transform.hpp>
#include <raw_image/drawing.hpp>
#include <raw_image/points.hpp>

#include <det_tflite/init.hpp>
#include <det/image.hpp>
#include <det/drawing.hpp>
#include <det/pose.hpp>

#include <core/thread_data.hpp>
#include <core/job_queue.hpp>

#include <json/types.hpp>
#include <json/io_manip.hpp>

using namespace det;

static constexpr auto write_test_images = true;

static void plot(const raw_image::plane& dest, const face_coordinates& fc) {
    static constexpr raw_image::pixel_color colors[] = {
        raw_image::color_black,
        raw_image::color_white,
        raw_image::color_red,
        raw_image::color_blue,
        raw_image::color_green
    };
    auto* c = colors;
    for (auto& det : fc)
        draw_lines(dest, det, 1, *c++);
}

BOOST_AUTO_TEST_SUITE(det)

static float
diff_coords(const detected_coordinates& a, const detected_coordinates& b) {
    BOOST_REQUIRE_EQUAL(int(a.type), int(b.type));
    BOOST_REQUIRE_EQUAL(a.landmarks.size(), b.landmarks.size());
    if (a.landmarks.empty()) {
        const auto dlx = a.eye_left.x - b.eye_left.x;
        const auto dly = a.eye_left.y - b.eye_left.y;
        const auto drx = a.eye_right.x - b.eye_right.x;
        const auto dry = a.eye_right.y - b.eye_right.y;
        return std::sqrt(float(dlx*dlx + dly*dly + drx*drx + dry*drx) / 2);
    }
    else {
        float e = 0;
        auto jt = b.landmarks.begin();
        for (auto& it : a.landmarks) {
            const auto dx = it.x - jt->x;
            const auto dy = it.y - jt->y;
            e += dx*dx;
            e += dy*dy;
            ++jt;
        }
        return std::sqrt(e / float(a.landmarks.size()));
    }
}

static float diff_face(const face_coordinates& a, const face_coordinates& b,
                       bool mirror) {
    BOOST_REQUIRE_EQUAL(a.size(), b.size());
    float r = 0;
    auto jt = b.begin();
    for (auto& it : a) {
        const auto e = diff_coords(it, *jt);
        if (e > 0) {
            if (r < e) r = e;
            FILE_LOG(logDETAIL) << to_string(it.type) << '\t' << e;
            switch (it.type) {
            case dt::v3_dlib:   BOOST_CHECK_LT(e, mirror?6:1); break;
            case dt::v4_dlib:   BOOST_CHECK_LT(e, mirror?48:1); break;
            case dt::v6_rfb320: BOOST_CHECK_LT(e, mirror?32:9); break;
            case dt::v7_retina: BOOST_CHECK_LT(e, mirror?32:9); break;
            case dt::v8_blaze:  BOOST_CHECK_LT(e, mirror?32:9); break;
            case dt::haar_eyes: BOOST_CHECK_LT(e, 1); break;
            case dt::dlib5 :    BOOST_CHECK_LT(e, 2.5); break;
            case dt::dlib68 :   BOOST_CHECK_LT(e, 2.5); break;
            default:
                FILE_LOG(logWARNING) << "unknown detection type: "
                                     << to_string(it.type) << '\t' << e;
            }
        }
        ++jt;
    }
    return r;
}

/*
static face_coordinates
detect_v6(core::context& c, const raw_image::plane& img, bool in_place) {
    detection_settings s;
    s.detector_version = 6;
    s.confidence_threshold = 0.0f;
    s.size_range = 0.13f;
    s.landmark_detection = lm::none;

    auto& q = c.threads();

    const auto image = in_place ? share_pixels(c,s,img) : copy_image(c,s,img);
    const auto faces = q.run([&]{return detect_faces(c,s,image);});
    FILE_LOG(logDETAIL) << faces.size() << " faces found";
    BOOST_REQUIRE(!faces.empty());

    auto it = std::max_element(
        faces.begin(), faces.end(),
        [](const face_coordinates& a,
           const face_coordinates& b) {
            return a.size() <= b.size() && (a.size() < b.size() || a.back().confidence < b.back().confidence);
        });

    BOOST_REQUIRE(it->size() >= 1);
    BOOST_CHECK(it->front().type != dt::dlib68);
    BOOST_CHECK_LT(it->front().confidence, 2);
    //BOOST_CHECK(it->back().type == dt::dlib68);
    //BOOST_CHECK_GT(it->back().confidence, 9);

    for (auto const& dc : *it)
        FILE_LOG(logDETAIL) << '\t' << to_string(dc.type)
                            << '\t' << dc.landmarks.size()
                            << '\t' << dc.confidence;

    // test serialize / deserialize
    const auto serial = to_json(*it);
    const auto recover = face_coordinates(serial);
    BOOST_REQUIRE(it->size() == recover.size());
    for (unsigned i = 0; i < it->size(); ++i) {
        auto const& a = (*it)[i];
        auto const& b = recover[i];
        BOOST_CHECK_LT(std::fabs(a.confidence - b.confidence), 1e-5);
        BOOST_CHECK_LT(diff_coords(a,b), 0.1875);
    }

    return *it;
}
*/

static face_coordinates
detect_v8(core::context& c, const raw_image::plane& img, bool in_place) {
    detection_settings s;
    s.detector_version = 8;
    s.confidence_threshold = 0.5f;
    s.landmark_detection = lm::mesh478;

    auto& q = c.threads();

    const auto image = in_place ? share_pixels(c,s,img) : copy_image(c,s,img);
    const auto faces = q.run([&]{return detect_faces(c,s,image);});
    FILE_LOG(logDETAIL) << faces.size() << " faces found";
    BOOST_REQUIRE(!faces.empty());

    auto it = std::max_element(
        faces.begin(), faces.end(),
        [](const face_coordinates& a,
           const face_coordinates& b) {
            return a.size() <= b.size() && (a.size() < b.size() || a.back().confidence < b.back().confidence);
        });

    BOOST_REQUIRE(it->size() >= 1);
    BOOST_CHECK(it->front().type != dt::dlib68);
    BOOST_CHECK_LT(it->front().confidence, 2);

    BOOST_CHECK(it->back().type == dt::mesh478);
    //BOOST_CHECK_GT(it->back().confidence, 9);

    for (auto const& dc : *it)
        FILE_LOG(logDETAIL) << '\t' << to_string(dc.type)
                            << '\t' << dc.landmarks.size()
                            << '\t' << dc.confidence;

    // test serialize / deserialize
    const auto serial = to_json(*it);
    const auto recover = face_coordinates(serial);
    BOOST_REQUIRE(it->size() == recover.size());
    for (unsigned i = 0; i < it->size(); ++i) {
        auto const& a = (*it)[i];
        auto const& b = recover[i];
        BOOST_CHECK_LT(std::fabs(a.confidence - b.confidence), 1e-5);
        BOOST_CHECK_LT(diff_coords(a,b), 0.125);
    }

    return *it;
}

BOOST_AUTO_TEST_CASE(det_detection) {
    const auto lib_internal = base_directory("lib-internal");
    const auto models_path = lib_internal / "models";
    const auto base_path = lib_internal / "det_tflite" / "tests";
    const auto det_path = lib_internal / "det" / "tests";
    //const auto img_path = det_path / "image_077.jpg";
    const auto img_path = base_path / "085-12.jpg";

    FILE_LOG(logINFO) << "detection: start";
    core::context_settings cs;
    cs.max_threads = 4; // auto detect to maximum of 4
    auto c = core::context::construct(cs);
    set_models_path(c, models_path);
    det::tflite::init(c);

    const auto img_raw = raw_image::load(img_path);

    FILE_LOG(logINFO) << "-- v8 (copied)";
    std::vector<face_coordinates> v8a_list;
    for (unsigned r = 0; r < 8; ++r) {
        const auto img_rot = copy_rotate(*img_raw, r);
        auto face = detect_v8(*c, *img_rot, false);
        for (unsigned i = 0; i < v8a_list.size(); ++i) {
            FILE_LOG(logDETAIL) << r << ' ' << i;
            diff_face(v8a_list[i], face, (r==4||i==4)&&r!=i);
        }

        if (write_test_images) {
            const auto img = copy(*img_raw);
            plot(*img, face);
            std::stringstream ss;
            ss << "test_v8a_" << r << ".jpg";
            save(*img, base_path / ss.str());
        }

        v8a_list.emplace_back(move(face));
    }

    FILE_LOG(logINFO) << "-- v8 (in place)";
    std::vector<face_coordinates> v8b_list;
    for (unsigned r = 0; r < 8; ++r) {
        const auto img_rot = copy_rotate(*img_raw, r);
        auto face = detect_v8(*c, *img_rot, true);
        for (unsigned i = 0; i < v8b_list.size(); ++i) {
            FILE_LOG(logDETAIL) << r << ' ' << i;
            diff_face(v8b_list[i], face, (r==4||i==4)&&r!=i);
        }

        if (write_test_images) {
            const auto img = copy(*img_raw);
            plot(*img, face);
            std::stringstream ss;
            ss << "test_v8b_" << r << ".jpg";
            save(*img, base_path / ss.str());
        }

        v8b_list.emplace_back(move(face));
    }
    
    FILE_LOG(logINFO) << "detection: done";
}

BOOST_AUTO_TEST_SUITE_END()
