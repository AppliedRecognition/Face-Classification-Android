
#include "drawing.hpp"

#include <stdext/rounding.hpp>

#include <applog/core.hpp>

#include <cassert>


using namespace det;


static const std::initializer_list<unsigned> retina7_lines[] = {
    { 0,1,2,3,4 }  // eyes, nose, mouth, tl, br (ignore last 2)
};

static const std::initializer_list<unsigned> blaze8_lines[] = {
    { 4,0,2}, {3,1,5} // left, nose, mouth, right (ignore last 2)
};

static const std::initializer_list<unsigned> dlib5_lines[] = {
    { 2,3,4,1,0 }  // left eye, base of nose, right eye
};

static const std::initializer_list<unsigned> dlib68_lines[] = {
    { 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16 },  // outline

    { 17,18,19,20,21 },  // left eyebrow
    { 22,23,24,25,26 },  // right eyebrow

    { 27,28,29,30,31,32,33,34,35 },  // nose

    { 36,37,38,39,40,41 },  // left eye
    { 42,43,44,45,46,47 },  // right eye

    { 48,49,50,51,52,53,54,55,56,57,58,59,48 },  // outer mouth
    { 60,61,62,63,64,65,66,67,60 }  // inner mouth
};

static const std::initializer_list<unsigned> mesh478_lines[] = {
    // outline
    { 127, 234, 93, 58, 172, 136, 149, 148,
      152,
      377, 378, 365, 397, 288, 323, 454, 356,
      389, 251, 284, 332, 297, 338,
      10,
      109, 67, 103, 54, 21, 162
    },
    // eyebrows
    { 70, 63, 105, 66, 107 },
    { 336, 296, 334, 293, 300 },
    // nose
    { 168, 197, 195, 4, 240, 97, 2, 326, 460 },
    // eyes
    { 33, 160, 158, 155, 153, 144 },
    { 382, 385, 387, 263, 373, 380 },
    // mouth (outer)
    { 61, 39, 37, 0, 267, 269, 291, 405, 314, 17, 84, 181 },
    // mouth (inner)
    { 78, 82, 13, 312, 308, 317, 14, 87 }

};

static const std::initializer_list<unsigned> stasm77_lines[] = {
    { 0,1,2,3,4,5,6,7,8,9,10,11,12 },  // jaw outline
    { 13,14,15 },  // forehead

    { 18,17,16,21 },
    { 19,20,29 },
    { 36,37,30,31,32,33,34,35,36,38 },  // left eye

    { 22,23,24,25 },
    { 26,27,28 },
    { 46,47,40,41,42,43,44,45,46,39 },  // right eye

    { 56,51,57,58,50,49,48,54,55,53,56,52 },

    { 59,60,61,62,63,64,65,72,73,74,75,76,59 },
    { 66,67,68,69,70,71,66 }
};

template <typename C>
static constexpr bool is_zero(const C& pt) {
    return pt.x <= 0 && 0 <= pt.x &&
        pt.y <= 0 && 0 <= pt.y;
}

template <typename C, typename U>
static auto fill(const C& lm, U&& lines) {
    std::vector<std::vector<coordinate_type> > r;
    r.resize(std::size(lines));
    auto line = r.begin();
    for (auto& arc : lines) {
        bool nonzero = false;
        for (auto i : arc) {
            assert(i < lm.size());
            if (!is_zero(lm[i]))
                nonzero = true;
        }
        if (nonzero) {
            line->reserve(arc.size());
            for (auto i : arc)
                line->emplace_back(lm[i]);
        }
        ++line;
    }
    return r;
}

std::vector<std::vector<coordinate_type> >
det::to_lines(const detected_coordinates& dc) {
    if (dc.landmarks.size() >= 5 &&
        (dc.landmarks.size() == 5 || dc.type == dt::dlib5))
        return fill(dc.landmarks, dlib5_lines);
    else if (dc.landmarks.size() >= 7 &&
        (dc.landmarks.size() == 7 || dc.type == dt::v7_retina))
        return fill(dc.landmarks, retina7_lines);
    else if (dc.landmarks.size() >= 8 &&
        (dc.landmarks.size() == 8 || dc.type == dt::v8_blaze))
        return fill(dc.landmarks, blaze8_lines);
    else if (dc.landmarks.size() >= 68 &&
        (dc.landmarks.size() == 68 ||
         dc.type == dt::dlib68 || dc.type == dt::mesh68))
        return fill(dc.landmarks, dlib68_lines);
    else if (dc.landmarks.size() >= 77 &&
        (dc.landmarks.size() == 77 || dc.type == dt::stasm77))
        return fill(dc.landmarks, stasm77_lines);
    else if (dc.landmarks.size() >= 478 &&
        (dc.landmarks.size() == 478 || dc.type == dt::mesh478))
        return fill(dc.landmarks, mesh478_lines);
    else if (dc.landmarks.size() <= 2) {
        std::vector<std::vector<coordinate_type> > r;
        if (dc.landmarks.size() == 2)
            r.push_back({dc.landmarks[0],dc.landmarks[1]});
        else
            r.push_back({dc.eye_left,dc.eye_right});
        return r;
    }
    FILE_LOG(logWARNING) << "det::to_lines: don't know what to do with "
                         << dc.landmarks.size() << " landmarks";
    return {};
}

std::vector<std::vector<coordinate_type> >
det::to_lines(const stdx::span<const coordinate_type>& lm) {
    switch (lm.size()) {
    case 2:  return { { lm.begin(), lm.end() } };
    case 5:  return fill(lm, dlib5_lines);
    case 7:  return fill(lm, retina7_lines);
    case 68: return fill(lm, dlib68_lines);
    case 77: return fill(lm, stasm77_lines);
    }
    FILE_LOG(logWARNING) << "det::to_lines: don't know what to do with "
                         << lm.size() << " landmarks";
    return {};
}

void det::draw_lines(stdx::arg<const raw_image::plane> dest,
                     const std::vector<std::vector<coordinate_type> >& lines,
                     int line_size, raw_image::pixel_color line_color,
                     int circle_size, raw_image::pixel_color circle_color) {
    if (line_size > 0) {
        for (auto& line : lines)
            for (std::size_t i = 1; i < line.size(); ++i)
                raw_image::line(
                    dest,
                    stdx::round_from(line[i-1].x),stdx::round_from(line[i-1].y),
                    stdx::round_from(line[i].x),stdx::round_from(line[i].y),
                    line_color,stdx::round_from(line_size));
    }
    if (circle_size != 0) {
        for (auto& line : lines)
            for (auto& p : line)
                raw_image::circle(
                    dest,stdx::round_from(p.x),stdx::round_from(p.y),
                    circle_color,circle_size);
    }
}
