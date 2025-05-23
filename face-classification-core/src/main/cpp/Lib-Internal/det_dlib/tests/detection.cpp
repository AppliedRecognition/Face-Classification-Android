#include <boost/test/unit_test.hpp>
#include <applog/base_directory.hpp>
#include <applog/core.hpp>

#include <raw_image_io/io.hpp>
#include <raw_image/transform.hpp>
#include <raw_image/drawing.hpp>
#include <raw_image/points.hpp>

#include <det_dlib/init.hpp>
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
    BOOST_REQUIRE(a.type == b.type);
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

static face_coordinates
detect_v3(core::context& c, const raw_image::plane& img, bool in_place) {
    detection_settings s;
    s.detector_version = 3;
    s.confidence_threshold = 0.0f;
    s.size_range = 5;
    s.landmark_detection = lm::dlib5 + lm::dlib68;
    static bool first = true;
    if (first) {
        first = false;
        auto sv = to_json(s);
        FILE_LOG(logINFO) << "detection_settings: "
                          << json::indent("\t") << sv;
        const auto s2 = detection_settings(sv);
        FILE_LOG(logINFO) << "detection_settings: "
                          << json::indent("\t") << to_json(s2);
    }

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

    BOOST_REQUIRE(it->size() > 1);
    BOOST_CHECK(it->front().type != dt::dlib68);
    BOOST_CHECK(it->front().confidence < 2);
    BOOST_CHECK(it->back().type == dt::dlib68);
    BOOST_CHECK(it->back().confidence > 9);
    for (auto const& dc : *it)
        FILE_LOG(logDETAIL) << '\t' << to_string(dc.type)
                            << '\t' << dc.landmarks.size()
                            << '\t' << dc.confidence;

    {
        auto p0 = compute_pose<pose_method::nose_tip>(*it);
        auto p1 = compute_pose<pose_method::simplex>(*it);
        BOOST_CHECK(std::abs(p0.roll - p1.roll) < 1e-5);
        FILE_LOG(logDETAIL) << "yaw:   " << p0.yaw << ' ' << p1.yaw;
        FILE_LOG(logDETAIL) << "pitch: " << p0.pitch << ' ' << p1.pitch;
    }
    
    return *it;
}

static face_coordinates
detect_v4(core::context& c, const raw_image::plane& img, bool in_place) {
    detection_settings s;
    s.detector_version = 4;
    s.confidence_threshold = 0.0f;
    s.size_range = 5;
    s.landmark_detection = lm::dlib68;

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

    BOOST_REQUIRE(it->size() > 1);
    BOOST_CHECK(it->front().type != dt::dlib68);
    BOOST_CHECK(it->front().confidence < 2);
    BOOST_CHECK(it->back().type == dt::dlib68);
    BOOST_CHECK(it->back().confidence > 9);
    
    return *it;
}

static face_coordinates
detect_v5(core::context& c, const raw_image::plane& img, bool in_place) {
    detection_settings s;
    s.detector_version = 5;
    s.confidence_threshold = -2.0f;
    s.size_range = 5;
    s.landmark_detection = lm::dlib68;

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

    BOOST_REQUIRE_GT(it->size(), 1);
    BOOST_CHECK(it->front().type != dt::dlib68);
    BOOST_CHECK_LT(it->front().confidence, 2);
    BOOST_CHECK(it->back().type == dt::dlib68);
    BOOST_CHECK_GT(it->back().confidence, 7);
    for (auto const& dc : *it)
        FILE_LOG(logDETAIL) << '\t' << to_string(dc.type)
                            << '\t' << dc.landmarks.size()
                            << '\t' << dc.confidence;

    return *it;
}

static face_coordinates
detect_v6(core::context& c, const raw_image::plane& img, bool in_place) {
    detection_settings s;
    s.detector_version = 6;
    s.confidence_threshold = 0.0f;
    s.size_range = 0.13f;
    s.landmark_detection = lm::dlib68;

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

    BOOST_REQUIRE(it->size() > 1);
    BOOST_CHECK(it->front().type != dt::dlib68);
    BOOST_CHECK_LT(it->front().confidence, 2);
    BOOST_CHECK(it->back().type == dt::dlib68);
    BOOST_CHECK_GT(it->back().confidence, 9);
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

static face_coordinates
detect_v7(core::context& c, const raw_image::plane& img, bool in_place) {
    detection_settings s;
    s.detector_version = 7;
    s.confidence_threshold = 0.0f;
    s.size_range = 1;
    s.landmark_detection = lm::dlib68;

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

    BOOST_REQUIRE(it->size() > 1);
    BOOST_CHECK(it->front().type != dt::dlib68);
    BOOST_CHECK_LT(it->front().confidence, 2);
    BOOST_CHECK(it->back().type == dt::dlib68);
    BOOST_CHECK_GT(it->back().confidence, 9);
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

static face_coordinates
detect_lm(core::context& c, const raw_image::plane& img,
          detected_coordinates start, bool in_place) {

    detection_settings s;
    s.detector_version = 0;
    s.landmark_detection = lm::dlib68;

    start.landmarks.clear();
    const auto image = in_place ? share_pixels(c,s,img) : copy_image(c,s,img);
    const auto faces =
        detect_landmarks(c,s.landmark_detection,image,&start,&start+1);
    BOOST_REQUIRE_EQUAL(faces.size(),1);
    BOOST_CHECK(faces.front().back().type == dt::dlib68);
    BOOST_CHECK(faces.front().back().confidence > 9);
    return faces.front();
}

BOOST_AUTO_TEST_CASE(det_detection) {
    const auto models_path = base_directory("lib-internal") / "models";
    const auto base_path = base_directory("lib-internal") / "det" / "tests";
    const auto img_path = base_path / "image_077.jpg";

    FILE_LOG(logINFO) << "detection: start";
    core::context_settings cs;
    cs.max_threads = 4; // auto detect to maximum of 4
    auto c = core::context::construct(cs);
    set_models_path(c, models_path);
    det::dlib::init(c);

    const auto img_raw = raw_image::load(img_path);
    
    FILE_LOG(logINFO) << "-- v3 (copied)";
    std::vector<face_coordinates> v3a_list;
    for (unsigned r = 0; r < 8; ++r) {
        const auto img_rot = copy_rotate(*img_raw, r);
        auto face = detect_v3(*c, *img_rot, false);
        for (unsigned i = 0; i < v3a_list.size(); ++i) {
            FILE_LOG(logDETAIL) << r << ' ' << i;
            diff_face(v3a_list[i], face, (r==4||i==4)&&r!=i);
        }

        if (write_test_images) {
            const auto img = copy(*img_rot);
            plot(*img, face);
            std::stringstream ss;
            ss << "test_v3a_" << r << ".jpg";
            save(*img, base_path / ss.str());
        }

        v3a_list.emplace_back(move(face));
    }

    FILE_LOG(logINFO) << "-- v3 (in place)";
    std::vector<face_coordinates> v3b_list;
    for (unsigned r = 0; r < 8; ++r) {
        const auto img_rot = copy_rotate(*img_raw, r);
        auto face = detect_v3(*c, *img_rot, true);
        for (unsigned i = 0; i < v3b_list.size(); ++i) {
            FILE_LOG(logDETAIL) << r << ' ' << i;
            diff_face(v3b_list[i], face, (r==4||i==4)&&r!=i);
        }

        if (write_test_images) {
            const auto img = copy(*img_raw);
            plot(*img, face);
            std::stringstream ss;
            ss << "test_v3b_" << r << ".jpg";
            save(*img, base_path / ss.str());
        }

        v3b_list.emplace_back(move(face));
    }

    FILE_LOG(logINFO) << "-- v4 (copied)";
    std::vector<face_coordinates> v4a_list;
    for (unsigned r = 0; r < 8; ++r) {
        const auto img_rot = copy_rotate(*img_raw, r);
        auto face = detect_v4(*c, *img_rot, false);
        for (unsigned i = 0; i < v4a_list.size(); ++i) {
            FILE_LOG(logDETAIL) << r << ' ' << i;
            diff_face(v4a_list[i], face, (r==4||i==4)&&r!=i);
        }

        if (write_test_images) {
            const auto img = copy(*img_raw);
            plot(*img, face);
            std::stringstream ss;
            ss << "test_v4a_" << r << ".jpg";
            save(*img, base_path / ss.str());
        }

        v4a_list.emplace_back(move(face));
    }

    FILE_LOG(logINFO) << "-- v4 (in place)";
    std::vector<face_coordinates> v4b_list;
    for (unsigned r = 0; r < 8; ++r) {
        const auto img_rot = copy_rotate(*img_raw, r);
        auto face = detect_v4(*c, *img_rot, true);
        for (unsigned i = 0; i < v4b_list.size(); ++i) {
            FILE_LOG(logDETAIL) << r << ' ' << i;
            diff_face(v4b_list[i], face, (r==4||i==4)&&r!=i);
        }

        if (write_test_images) {
            const auto img = copy(*img_raw);
            plot(*img, face);
            std::stringstream ss;
            ss << "test_v4b_" << r << ".jpg";
            save(*img, base_path / ss.str());
        }

        v4b_list.emplace_back(move(face));
    }

    if (0) {
        // padded image so face is a little smaller
        const auto img_padded =
            create(img_raw->width*3/2, img_raw->height*3/2, img_raw->layout);
        fill(img_padded, raw_image::pixel_color{0x00808080});
        copy_pixels(img_raw,
                    crop(img_padded,
                         (img_padded->width-img_raw->width)/2,
                         (img_padded->height-img_raw->height)/2,
                         img_raw->width, img_raw->height));
        if (write_test_images)
            save(img_padded, base_path / "test_padded.jpg");

        FILE_LOG(logINFO) << "-- v5 (copied)";
        std::vector<face_coordinates> v5a_list;
        for (unsigned r = 0; r < 8; ++r) {
            const auto img_rot = copy_rotate(*img_padded, r);
            auto face = detect_v5(*c, *img_rot, false);
            for (unsigned i = 0; i < v5a_list.size(); ++i) {
                FILE_LOG(logDETAIL) << r << ' ' << i;
                diff_face(v5a_list[i], face, (r==4||i==4)&&r!=i);
            }

            if (write_test_images) {
                const auto img = copy(*img_padded);
                plot(*img, face);
                std::stringstream ss;
                ss << "test_v5a_" << r << ".jpg";
                save(*img, base_path / ss.str());
            }

            v5a_list.emplace_back(move(face));
        }
    }

    FILE_LOG(logINFO) << "-- v6 (copied)";
    std::vector<face_coordinates> v6a_list;
    for (unsigned r = 0; r < 8; ++r) {
        const auto img_rot = copy_rotate(*img_raw, r);
        auto face = detect_v6(*c, *img_rot, false);
        for (unsigned i = 0; i < v6a_list.size(); ++i) {
            FILE_LOG(logDETAIL) << r << ' ' << i;
            diff_face(v6a_list[i], face, (r==4||i==4)&&r!=i);
        }

        if (write_test_images) {
            const auto img = copy(*img_raw);
            plot(*img, face);
            std::stringstream ss;
            ss << "test_v6a_" << r << ".jpg";
            save(*img, base_path / ss.str());
        }

        v6a_list.emplace_back(move(face));
    }

    FILE_LOG(logINFO) << "-- v6 (in place)";
    std::vector<face_coordinates> v6b_list;
    for (unsigned r = 0; r < 8; ++r) {
        const auto img_rot = copy_rotate(*img_raw, r);
        auto face = detect_v6(*c, *img_rot, true);
        for (unsigned i = 0; i < v6b_list.size(); ++i) {
            FILE_LOG(logDETAIL) << r << ' ' << i;
            diff_face(v6b_list[i], face, (r==4||i==4)&&r!=i);
        }

        if (write_test_images) {
            const auto img = copy(*img_raw);
            plot(*img, face);
            std::stringstream ss;
            ss << "test_v6b_" << r << ".jpg";
            save(*img, base_path / ss.str());
        }

        v6b_list.emplace_back(move(face));
    }

    FILE_LOG(logINFO) << "-- v7 (copied)";
    std::vector<face_coordinates> v7a_list;
    for (unsigned r = 0; r < 8; ++r) {
        const auto img_rot = copy_rotate(*img_raw, r);
        auto face = detect_v7(*c, *img_rot, false);
        for (unsigned i = 0; i < v7a_list.size(); ++i) {
            FILE_LOG(logDETAIL) << r << ' ' << i;
            diff_face(v7a_list[i], face, (r==4||i==4)&&r!=i);
        }

        if (write_test_images) {
            const auto img = copy(*img_raw);
            plot(*img, face);
            std::stringstream ss;
            ss << "test_v7a_" << r << ".jpg";
            save(*img, base_path / ss.str());
        }

        v7a_list.emplace_back(move(face));
    }

    FILE_LOG(logINFO) << "-- v7 (in place)";
    std::vector<face_coordinates> v7b_list;
    for (unsigned r = 0; r < 8; ++r) {
        const auto img_rot = copy_rotate(*img_raw, r);
        auto face = detect_v7(*c, *img_rot, true);
        for (unsigned i = 0; i < v7b_list.size(); ++i) {
            FILE_LOG(logDETAIL) << r << ' ' << i;
            diff_face(v7b_list[i], face, (r==4||i==4)&&r!=i);
        }

        if (write_test_images) {
            const auto img = copy(*img_raw);
            plot(*img, face);
            std::stringstream ss;
            ss << "test_v7b_" << r << ".jpg";
            save(*img, base_path / ss.str());
        }

        v7b_list.emplace_back(move(face));
    }

    FILE_LOG(logINFO) << "-- landmarks (copied)";
    std::vector<face_coordinates> lma_list;
    for (unsigned r = 0; r < 8; ++r) {
        const auto img_rot = copy_rotate(*img_raw, r);
        auto face = detect_lm(*c, *img_rot, v3a_list.front(), false);
        for (unsigned i = 0; i < lma_list.size(); ++i) {
            FILE_LOG(logDETAIL) << r << ' ' << i;
            diff_face(lma_list[i], face, (r==4||i==4)&&r!=i);
        }

        if (write_test_images) {
            const auto img = copy(*img_raw);
            plot(*img, face);
            std::stringstream ss;
            ss << "test_lma_" << r << ".jpg";
            save(*img, base_path / ss.str());
        }

        lma_list.emplace_back(move(face));
    }

    FILE_LOG(logINFO) << "-- landmarks (in place)";
    std::vector<face_coordinates> lmb_list;
    for (unsigned r = 0; r < 8; ++r) {
        const auto img_rot = copy_rotate(*img_raw, r);
        auto face = detect_lm(*c, *img_rot, v3a_list.front(), true);
        for (unsigned i = 0; i < lmb_list.size(); ++i) {
            FILE_LOG(logDETAIL) << r << ' ' << i;
            diff_face(lmb_list[i], face, (r==4||i==4)&&r!=i);
        }

        if (write_test_images) {
            const auto img = copy(*img_raw);
            plot(*img, face);
            std::stringstream ss;
            ss << "test_lmb_" << r << ".jpg";
            save(*img, base_path / ss.str());
        }

        lmb_list.emplace_back(move(face));
    }

        
    /*
    BOOST_REQUIRE_EQUAL(faces.size(),1);
    for (auto& face : faces) {
    }
    */
    
    FILE_LOG(logINFO) << "detection: done";
}

BOOST_AUTO_TEST_SUITE_END()
