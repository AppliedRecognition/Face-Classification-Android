
#include "face_landmarks.hpp"

#include <applog/core.hpp>

using namespace raw_image;

static constexpr unsigned mirror_map_eyes[] = {
    1, 0   // eyes
};
static_assert(std::size(mirror_map_eyes) == 2);

static constexpr unsigned mirror_map_retina[] = {
    1, 0,  // eyes
    2,     // nose
    4, 3   // mouth
};
static_assert(std::size(mirror_map_retina) == 5);

static constexpr unsigned mirror_map_blaze[] = {
    1, 0,  // eyes
    2,     // nose
    3,     // mouth
    5, 4   // tragion
};
static_assert(std::size(mirror_map_blaze) == 6);

static constexpr unsigned mirror_map_dlib5[] = {
    2, 3, 0, 1,  // eye corners
    4            // base of nose
};
static_assert(std::size(mirror_map_dlib5) == 5);

static constexpr unsigned mirror_map_dlib68[] = {
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
static_assert(std::size(mirror_map_dlib68) == 68);

static constexpr auto mirror_map_mesh478 = []() {
    std::array<unsigned,478> map = {};

    const auto offset_range =
        [&](unsigned ofs, unsigned first, unsigned last) {
            for (unsigned i = first; i <= last; ++i) {
                const auto j = ofs + i;
                map[i] = j;
                map[j] = i;
            }
        };
    offset_range(230,  20,  93);
    offset_range(229,  95, 150);
    offset_range(227, 153, 163);
    offset_range(226, 165, 167);
    offset_range(225, 169, 174);
    offset_range(224, 176, 194);
    offset_range(223, 196, 196);
    offset_range(222, 198, 198);
    offset_range(220, 201, 247);

    const auto fill_pair = [&](unsigned i, unsigned j) {
        map[i] = j;
        map[j] = i;
    };
    fill_pair(3,248);  // upper nose
    fill_pair(7,249);  // this one and 5 below are within eyes
    fill_pair(468,473);
    fill_pair(469,476);
    fill_pair(470,475);
    fill_pair(471,474);
    fill_pair(472,477);

    // center line
    for (auto i : {
            10, 151, 9, 8,          // forehead
            168,                    // root (or bridge) of nose
            6, 197, 195, 5,         // dorsum
            4,                      // tip of nose
            1, 19, 94, 2, 164,
            0, 11, 12, 13,          // upper lip
            14, 15, 16, 17,         // lower lip
            18, 200, 199, 175, 152  // chin
        })
        map[unsigned(i)] = unsigned(i);

    return map;
}();

using dt = raw_image::detection_type;

stdx::span<const unsigned>
raw_image::mirrored_pairs(detection_type type) {
    switch (type) {
    case dt::v7_retina:
        return mirror_map_retina;
    case dt::v8_blaze:
        return mirror_map_blaze;
    case dt::haar_eyes:
        return mirror_map_eyes;
    case dt::dlib5:
        return mirror_map_dlib5;
    case dt::dlib68:
    case dt::mesh68:
        return mirror_map_dlib68;
    case dt::mesh478:
        return mirror_map_mesh478;
    default:
        break;
    }
    throw std::runtime_error("unsupported landmark type");
}


static constexpr unsigned subset_eyes[] = { 0, 1 };
static_assert(std::size(subset_eyes) == 2);

static constexpr unsigned subset_dlib68_to_dlib5[] = {
    45, 42, 36, 39, // eye corners
    33              // base of nose
};
static_assert(std::size(subset_dlib68_to_dlib5) == 5);

static constexpr unsigned subset_mesh478_to_dlib68[] = {
    // jaw
    127, 234, 93, 58, 172, 136, 149, 148,
    152,
    377, 378, 365, 397, 288, 323, 454, 356,
    // eyebrows
    70, 63, 105, 66, 107, 336, 296, 334, 293, 300,
    // nose
    168, 197, 195, 4, 240, 97, 2, 326, 460,
    // eyes
    33, 160, 158, 155, 153, 144, 382, 385, 387, 263, 373, 380,
    // mouth (outer)
    61, 39, 37, 0, 267, 269, 291, 405, 314, 17, 84, 181,
    // mouth (inner)
    78, 82, 13, 312, 308, 317, 14, 87
};
static_assert(std::size(subset_mesh478_to_dlib68) == 68);

static constexpr unsigned subset_mesh478_to_dlib5[] = {
    subset_mesh478_to_dlib68[subset_dlib68_to_dlib5[0]],
    subset_mesh478_to_dlib68[subset_dlib68_to_dlib5[1]],
    subset_mesh478_to_dlib68[subset_dlib68_to_dlib5[2]],
    subset_mesh478_to_dlib68[subset_dlib68_to_dlib5[3]],
    subset_mesh478_to_dlib68[subset_dlib68_to_dlib5[4]],
};
static_assert(std::size(subset_mesh478_to_dlib5) == 5);

static constexpr unsigned subset_mesh478_to_blaze[] = {
    468, 473,  // eyes (centers)
    4,         // nose (tip)
    13,        // mouth (bottom of upper lip)
    127, 356   // tragion
};
static_assert(std::size(subset_mesh478_to_blaze) == 6);

static constexpr unsigned subset_mesh478_to_retina[] = {
    468, 473,  // eyes (centers)
    4,         // nose (tip)
    61, 291    // mouth (corners)
};
static_assert(std::size(subset_mesh478_to_retina) == 5);

static constexpr unsigned subset_mesh478_to_eyes[] = {
    468, 473
};
static_assert(std::size(subset_mesh478_to_eyes) == 2);


stdx::span<const unsigned>
raw_image::landmark_subset(detection_type from, detection_type to) {
    switch (from) {
    case dt::mesh478:
        switch (to) {
        case dt::dlib68:
        case dt::mesh68:
            return subset_mesh478_to_dlib68;
        case dt::dlib5:
            return subset_mesh478_to_dlib5;
        case dt::v8_blaze:
            return subset_mesh478_to_blaze;
        case dt::v7_retina:
            return subset_mesh478_to_retina;
        case dt::haar_eyes:
            return subset_mesh478_to_eyes;
        default: break;
        }
        break;

    case dt::dlib68:
    case dt::mesh68:
        if (to == dt::dlib5)
            return subset_dlib68_to_dlib5;
        break;

    case dt::v8_blaze:
        if (to == dt::haar_eyes)
            return subset_eyes;
        break;

    case dt::v7_retina:
        if (to == dt::haar_eyes)
            return subset_eyes;
        break;

    default: break;
    }
    throw std::runtime_error("unsupported landmark subset");
}

eye_coordinates raw_image::eyes_subset(const landmark_coordinates& from) {
    eye_coordinates ec;
    auto& lm = from.landmarks;
    const auto from_corners =
        [&](float x_scale, float y_ofs) {
            if (lm.size() != 2)
                throw std::runtime_error("expected 2 landmarks (bounding box)");
            const auto cx = 0.5f * (lm.front().x + lm.back().x);
            const auto cy = 0.5f * (lm.front().y + lm.back().y);
            const auto dx = x_scale * std::abs(lm.back().x - lm.front().x);
            const auto dy = y_ofs   * std::abs(lm.back().y - lm.front().y);
            ec.eye_left.x  = cx - dx;
            ec.eye_right.x = cx + dx;
            ec.eye_left.y = ec.eye_right.y = cy - dy;
        };

    switch (from.type) {
    case dt::v3_dlib:   from_corners(0.2f,    0.2f); break;
    case dt::v4_dlib:   from_corners(0.2f,    0.1f); break;
    case dt::v5_fapi:   from_corners(0.15f,   0.2f); break;
    case dt::v6_rfb320: from_corners(0.2338f, 0.1465f); break;

    case dt::v7_retina:
        if (lm.size() != 5 && lm.size() != 7)
            std::runtime_error("expected 5 or 7 landmarks (RetinaFace)");
        ec.eye_left  = lm[0];
        ec.eye_right = lm[1];
        break;

    case dt::v8_blaze:
        if (lm.size() != 6 && lm.size() != 8)
            std::runtime_error("expected 6 or 8 landmarks (BlazeFace)");
        ec.eye_left  = lm[0];
        ec.eye_right = lm[1];
        break;

    case dt::haar_eyes:
        if (lm.size() != 2)
            std::runtime_error("expected 2 landmarks (eyes)");
        ec.eye_left  = lm[0];
        ec.eye_right = lm[1];
        break;

    case dt::stasm77:
        if (lm.size() != 77)
            std::runtime_error("expected 77 landmarks (stasm)");
        ec.eye_left  = 0.5 * (lm[30] + lm[34]);
        ec.eye_right = 0.5 * (lm[40] + lm[44]);
        break;

    case dt::dlib5:
        if (lm.size() != 5)
            std::runtime_error("expected 5 landmarks");
        ec.eye_left  = 0.5 * (lm[2] + lm[3]);
        ec.eye_right = 0.5 * (lm[0] + lm[1]);
        break;

    case dt::dlib68:
    case dt::mesh68:
        if (lm.size() != 68)
            std::runtime_error("expected 68 landmarks");
        ec.eye_left  = 0.25f * (lm[37] + lm[38] + lm[40] + lm[41]);
        ec.eye_right = 0.25f * (lm[43] + lm[44] + lm[46] + lm[47]);
        break;

    case dt::mesh478:
        if (lm.size() != 478)
            std::runtime_error("expected 478 landmarks");
        ec.eye_left  = lm[468];
        ec.eye_right = lm[473];
        break;

    default:
        throw std::runtime_error("unsupported detection type");
    }
    return ec;
}

static std::array<point2f,2> bbox(const landmark_coordinates& from) {
    std::array<point2f,2> bb = {};
    switch (from.type) {
    case dt::v3_dlib:
    case dt::v4_dlib:
    case dt::v5_fapi:
    case dt::v6_rfb320:
        if (from.landmarks.size() != 2)
            std::runtime_error("expected 2 landmarks (corners)");
        [[ fallthrough ]];
    case dt::v7_retina:
    case dt::v8_blaze:
        if (from.landmarks.size() < 2)
            std::runtime_error("expected corner landmarks");
        return {
            from.landmarks[from.landmarks.size()-2],
            from.landmarks[from.landmarks.size()-1]
        };

        /*
    case dt::haar_eyes:
        if (lm.size() != 2)
            std::runtime_error("expected 2 landmarks (eyes)");
        ec.eye_left  = lm[0];
        ec.eye_right = lm[1];
        break;
    case dt::dlib5:
        if (lm.size() != 5)
            std::runtime_error("expected 5 landmarks");
        ec.eye_left  = 0.5 * (lm[2] + lm[3]);
        ec.eye_right = 0.5 * (lm[0] + lm[1]);
        break;
        */

    case dt::dlib68:
    case dt::mesh68:
    case dt::stasm77:
    case dt::mesh478:
        if (from.landmarks.size() < 68)
            std::runtime_error("insufficient landmarks");
        bb[0] = bb[1] = from.landmarks.front();
        for (const auto& p : from.landmarks) {
            bb[0].x = std::min(bb[0].x, p.x);
            bb[0].y = std::min(bb[0].y, p.y);
            bb[1].x = std::max(bb[1].x, p.x);
            bb[1].y = std::max(bb[1].y, p.y);
        }
        break;

    default:
        break;
    }
    return bb;
}

void raw_image::landmark_subset(
    const landmark_coordinates& from,
    detection_type to, landmark_coordinates& dest) {

    if (to == dt::haar_eyes) {
        auto ec = eyes_subset(from);
        dest.type = dt::haar_eyes;
        dest.landmarks = { ec.eye_left, ec.eye_right };
        dest.set_eye_coordinates_from_landmarks();
        return;
    }

    const auto copy_subset = [&](stdx::span<const unsigned> map) {
        dest.type = to;
        dest.landmarks.clear();
        dest.landmarks.reserve(map.size());
        for (auto i : map)
            dest.landmarks.emplace_back(from.landmarks[i]);
        dest.set_eye_coordinates_from_landmarks();
    };

    switch (from.type) {
    case dt::mesh478:
        switch (to) {
        case dt::dlib68:
        case dt::mesh68:
            return copy_subset(subset_mesh478_to_dlib68);
        case dt::dlib5:
            return copy_subset(subset_mesh478_to_dlib5);
        case dt::v8_blaze: {
            copy_subset(subset_mesh478_to_blaze);
            const auto bb = bbox(from);
            dest.landmarks.emplace_back(bb[0]);
            dest.landmarks.emplace_back(bb[1]);
            return;
        }
        case dt::v7_retina: {
            copy_subset(subset_mesh478_to_retina);
            const auto bb = bbox(from);
            dest.landmarks.emplace_back(bb[0]);
            dest.landmarks.emplace_back(bb[1]);
            return;
        }
        default: break;
        }
        break;

    case dt::dlib68:
    case dt::mesh68:
        switch (to) {
        case dt::dlib5:
            return copy_subset(subset_dlib68_to_dlib5);

        case dt::v7_retina: {
            dest.type = dt::v7_retina;
            static_cast<eye_coordinates&>(dest) = eyes_subset(from);
            const auto bb = bbox(from);
            dest.landmarks = {
                dest.eye_left, dest.eye_right,
                from.landmarks[30], // nose (tip)
                from.landmarks[48], from.landmarks[54], // mouth corners
                bb[0], bb[1]
            };
            return;
        }

        case dt::v8_blaze: {
            dest.type = dt::v8_blaze;
            static_cast<eye_coordinates&>(dest) = eyes_subset(from);
            const auto bb = bbox(from);
            dest.landmarks = {
                dest.eye_left, dest.eye_right,
                from.landmarks[30], // nose (tip)
                from.landmarks[62], // mouth (center)
                from.landmarks[0], from.landmarks[16], // tragion
                bb[0], bb[1]
            };
            return;
        }

        default: break;
        }
        break;

    default: break;
    }
    throw std::runtime_error("unsupported landmark subset");
}


// left half of face
static constexpr unsigned short mesh478_triangles[][3] = {
    {0,37,11},{0,164,37},{1,4,44},{1,44,19},{2,94,141},
    {2,141,97},{2,97,167},{2,167,164},{3,51,195},{3,236,51},
    {3,196,174},{3,174,236},{3,195,197},{3,197,196},{4,5,51},
    {4,45,44},{4,51,45},{5,195,51},{6,168,122},{6,122,196},
    {6,196,197},{7,33,25},{7,25,110},{7,110,163},{8,9,55},
    {8,55,193},{8,193,168},{9,107,55},{9,108,107},{9,151,108},
    {10,109,151},{11,72,12},{11,37,72},{12,38,13},{12,72,38},
    {13,38,82},{14,86,15},{14,87,86},{15,85,16},{15,86,85},
    {16,85,17},{17,83,18},{17,84,83},{17,85,84},{18,83,201},
    {18,201,200},{19,44,125},{19,141,94},{19,125,141},{20,60,99},
    {20,166,60},{20,79,166},{20,238,79},{20,99,242},{20,241,238},
    {20,242,241},{21,68,54},{21,71,68},{21,162,71},{22,145,23},
    {22,23,230},{22,26,154},{22,231,26},{22,153,145},{22,154,153},
    {22,230,231},{23,144,24},{23,24,230},{23,145,144},{24,144,110},
    {24,110,229},{24,229,230},{25,226,31},{25,31,228},{25,33,130},
    {25,228,110},{25,130,226},{26,112,155},{26,232,112},{26,155,154},
    {26,231,232},{27,159,28},{27,28,222},{27,29,160},{27,223,29},
    {27,160,159},{27,222,223},{28,157,56},{28,56,221},{28,158,157},
    {28,159,158},{28,221,222},{29,30,160},{29,224,30},{29,223,224},
    {30,161,160},{30,246,161},{30,224,225},{30,225,247},{30,247,246},
    {31,111,117},{31,226,111},{31,117,228},{32,140,171},{32,211,140},
    {32,171,208},{32,201,194},{32,194,211},{32,208,201},{33,247,130},
    {33,246,247},{34,116,143},{34,227,116},{34,139,127},{34,127,234},
    {34,156,139},{34,143,156},{34,234,227},{35,143,111},{35,111,226},
    {35,113,124},{35,226,113},{35,124,143},{36,100,101},{36,142,100},
    {36,101,205},{36,203,142},{36,206,203},{36,205,206},{37,39,72},
    {37,167,39},{37,164,167},{38,72,41},{38,41,81},{38,81,82},
    {39,40,73},{39,92,40},{39,73,72},{39,165,92},{39,167,165},
    {40,74,73},{40,185,74},{40,92,186},{40,186,185},{41,74,42},
    {41,42,81},{41,72,73},{41,73,74},{42,74,184},{42,80,81},
    {42,183,80},{42,184,183},{43,61,57},{43,57,202},{43,146,61},
    {43,106,146},{43,204,106},{43,202,204},{44,45,220},{44,241,125},
    {44,220,237},{44,237,241},{45,51,134},{45,134,220},{46,53,63},
    {46,225,53},{46,63,70},{46,70,156},{46,124,113},{46,113,225},
    {46,156,124},{47,120,100},{47,100,126},{47,114,121},{47,217,114},
    {47,121,120},{47,126,217},{48,49,102},{48,115,49},{48,102,64},
    {48,64,219},{48,219,115},{49,129,102},{49,115,209},{49,142,129},
    {49,209,142},{50,101,118},{50,205,101},{50,118,117},{50,117,123},
    {50,123,187},{50,187,205},{51,236,134},{52,63,53},{52,53,223},
    {52,105,63},{52,65,66},{52,222,65},{52,66,105},{52,223,222},
    {53,224,223},{53,225,224},{54,68,104},{54,104,103},{55,107,65},
    {55,65,221},{55,189,193},{55,221,189},{56,157,173},{56,173,190},
    {56,190,221},{57,61,185},{57,185,186},{57,186,212},{57,212,202},
    {58,215,132},{58,172,138},{58,138,215},{59,75,60},{59,60,166},
    {59,235,75},{59,166,219},{59,219,235},{60,75,240},{60,240,99},
    {61,146,76},{61,76,184},{61,184,185},{62,76,77},{62,77,96},
    {62,96,78},{63,68,71},{63,104,68},{63,71,70},{63,105,104},
    {64,129,98},{64,98,240},{64,102,129},{64,235,219},{64,240,235},
    {65,107,66},{65,222,221},{66,69,105},{66,107,69},{67,104,69},
    {67,69,108},{67,103,104},{67,108,109},{69,104,105},{69,107,108},
    {70,71,139},{70,139,156},{71,162,139},{74,185,184},{75,235,240},
    {76,146,77},{76,78,183},{76,183,184},{77,91,90},{77,90,96},
    {77,146,91},{78,96,95},{78,191,183},{79,218,166},{79,237,218},
    {79,239,237},{79,238,239},{80,183,191},{83,84,181},{83,181,182},
    {83,182,201},{84,85,180},{84,180,181},{85,86,179},{85,179,180},
    {86,87,178},{86,178,179},{88,95,89},{88,89,179},{88,179,178},
    {89,96,90},{89,90,180},{89,95,96},{89,180,179},{90,91,181},
    {90,181,180},{91,146,106},{91,106,182},{91,182,181},{92,165,206},
    {92,216,186},{92,206,216},{93,132,177},{93,147,137},{93,137,234},
    {93,177,147},{97,99,98},{97,98,167},{97,242,99},{97,141,242},
    {98,99,240},{98,129,203},{98,165,167},{98,203,165},{100,119,101},
    {100,120,119},{100,142,126},{101,119,118},{106,194,182},{106,204,194},
    {108,151,109},{110,144,163},{110,228,229},{111,116,117},{111,143,116},
    {112,133,155},{112,243,133},{112,232,233},{112,233,244},{112,244,243},
    {113,247,225},{113,226,247},{114,128,121},{114,188,128},{114,174,188},
    {114,217,174},{115,131,209},{115,220,131},{115,219,218},{115,218,220},
    {116,123,117},{116,227,123},{117,118,228},{118,119,230},{118,229,228},
    {118,230,229},{119,120,230},{120,121,232},{120,231,230},{120,232,231},
    {121,128,232},{122,168,193},{122,188,196},{122,245,188},{122,193,245},
    {123,137,147},{123,227,137},{123,147,187},{124,156,143},{125,241,141},
    {126,142,209},{126,209,198},{126,198,217},{127,139,162},{128,188,245},
    {128,233,232},{128,245,233},{129,142,203},{130,247,226},{131,134,198},
    {131,220,134},{131,198,209},{132,213,177},{132,215,213},{133,190,173},
    {133,243,190},{134,236,198},{135,136,169},{135,172,136},{135,138,172},
    {135,214,138},{135,169,210},{135,210,214},{136,150,169},{137,227,234},
    {138,214,192},{138,192,215},{140,148,171},{140,176,148},{140,170,149},
    {140,149,176},{140,211,170},{141,241,242},{147,177,213},{147,213,187},
    {148,152,175},{148,175,171},{149,170,150},{150,170,169},{165,203,206},
    {166,218,219},{169,170,211},{169,211,210},{171,175,199},{171,199,208},
    {174,196,188},{174,217,236},{182,194,201},{186,216,212},{187,213,192},
    {187,192,214},{187,207,205},{187,214,207},{189,221,190},{189,190,243},
    {189,244,193},{189,243,244},{192,213,215},{193,244,245},{194,204,211},
    {198,236,217},{199,200,208},{200,201,208},{202,210,204},{202,214,210},
    {202,212,214},{204,210,211},{205,216,206},{205,207,216},{207,214,212},
    {207,212,216},{218,237,220},{233,245,244},{237,239,241},{238,241,239},
};

std::vector<std::array<unsigned short, 3> >
raw_image::triangles(detection_type dt) {
    if (dt != detection_type::mesh478)
        throw std::runtime_error(
            "unsupported landmarks type for triangulation");
    std::vector<std::array<unsigned short, 3> > vec;
    vec.reserve(2*std::size(mesh478_triangles));
    for (auto& t : mesh478_triangles) {
        vec.push_back({t[0],t[1],t[2]});
        using ushort = unsigned short;
        vec.push_back({
                ushort(mirror_map_mesh478[t[2]]),
                ushort(mirror_map_mesh478[t[1]]),
                ushort(mirror_map_mesh478[t[0]]),
            });
    }
    return vec;
}
