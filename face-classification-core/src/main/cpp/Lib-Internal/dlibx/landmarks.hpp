#pragma once

#include <dlib/image_processing/full_object_detection.h>
#include <raw_image/points.hpp>
#include <iterator>

namespace dlibx {
    template <typename T>
    std::vector<T> dlib5_from_68(const std::vector<T>& pts) {
        assert(pts.size() == 68);
        return { pts[45], pts[42], pts[36], pts[39], pts[33] };
    }

    template <typename T>
    void symmetry_swap_dlib5(std::vector<T>& pts) {
        static constexpr unsigned symmetry_map[] = {
            2, 3, 0, 1,  // eye corners
            4            // base of nose
        };
        static_assert(std::size(symmetry_map) == 5, "!!");
        if (pts.size() != 5)
            throw std::invalid_argument("invalid number of landmarks");
        unsigned i = 0;
        for (auto j : symmetry_map) {
            if (i < j) {
                assert(j < pts.size());
                using namespace std;
                swap(pts[i], pts[j]);
            }
            ++i;
        }
    }

    template <typename T>
    void symmetry_swap_dlib68(std::vector<T>& pts) {
        static constexpr unsigned symmetry_map[] = {
            // jaw
            16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0,
            // eyebrows
            26, 25, 24, 23, 22, 21, 20, 19, 18, 17,
            // nose
            27, 28, 29, 30, 35, 34, 33, 32, 31,
            // eyes
            45, 44, 43, 42, 47, 46,
            39, 38, 37, 36, 41, 40,
            // mouth (outer)
            54, 53, 52, 51, 50, 49, 48, 59, 58, 57, 56, 55,
            // mouth (inner)
            64, 63, 62, 61, 60, 67, 66, 65
        };
        static_assert(std::size(symmetry_map) == 68, "!!");
        if (pts.size() != 68)
            throw std::invalid_argument("invalid number of landmarks");
        unsigned i = 0;
        for (auto j : symmetry_map) {
            if (i < j) {
                assert(j < pts.size());
                using namespace std;
                swap(pts[i], pts[j]);
            }
            ++i;
        }
    }

    template <typename T>
    void symmetry_swap(std::vector<T>& pts) {
        switch (pts.size()) {
        case 5:  symmetry_swap_dlib5(pts);  break;
        case 68: symmetry_swap_dlib68(pts); break;
        default:
            throw std::invalid_argument("invalid number of landmarks");
        }
    }

    template <typename PT>
    dlib::full_object_detection
    image_full_object_detection_from_points(
        const raw_image::plane& image,
        const std::vector<PT>& pts) {

        // translate coordinates to match image
        std::vector<dlib::point> parts;
        parts.reserve(pts.size());
        for (auto& p : pts)
            parts.push_back(
                raw_image::round_from(
                    raw_image::to_image_point(
                        raw_image::round_to<dlib::dpoint>(p), image)));
        
        // mirror if necessary
        if (image.rotate & 4) 
            symmetry_swap(parts);
    
        // bounding rectangle
        dlib::rectangle rect(parts.front());
        for (const auto& p : parts)
            rect += p;
        // todo: if dlib5, maybe expand rectangle

        return { rect, parts };
    }

}
