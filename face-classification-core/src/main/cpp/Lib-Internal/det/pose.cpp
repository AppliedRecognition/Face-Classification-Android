
#include "pose.hpp"
#include "simplex_method.hpp"
#include "math.hpp"

#include "landmark_standardize.hpp"

#include <raw_image/point_rounding.hpp>

#include <applog/core.hpp>

using namespace det;

namespace {
    inline float clamp(float a, float limit) {
        if (a < -limit) return -limit;
        if (a > limit)  return limit;
        return a;
    }

    struct angle_params {
        float tip_y, tip_z, base_y;
    };
    const auto param_nose_base    = angle_params { 0.69f, 0.24f, 0.79f };
    const auto param_mouth_center = angle_params { 0.62f, 0.59f, 1.08f };

    struct angle_fit {
        const coordinate_type tip, base;
        const angle_params& param;
        
        float operator()(simplex::vertex_type& vert) const {
            assert(vert.size() == 2);
            for (auto& a : vert)
                a = clamp(a,80);

            const auto yaw   = raddeg(vert[0]);
            const auto cyaw = std::cos(yaw);
            const auto tyaw = std::tan(yaw);

            const auto pitch = raddeg(vert[1]);
            const auto cpitch = std::cos(pitch);
            const auto spitch = std::sin(pitch);

            const auto xy =  spitch * tyaw;
            const auto xz =  cpitch * tyaw;
            const auto yy =  cpitch / cyaw;
            const auto yz = -spitch / cyaw;

            const auto tx = xy*param.tip_y + xz*param.tip_z;
            const auto ty = yy*param.tip_y + yz*param.tip_z;

            const auto bx = xy*param.base_y;
            const auto by = yy*param.base_y;

            return sqr(tip.x-tx) + sqr(tip.y-ty)
                + sqr(base.x-bx) + sqr(base.y-by);
        }
    };
}

template <>
face_pose_type det::compute_pose<pose_method::simplex>(
    const coordinate_type& tip,
    const coordinate_type& base, base_landmark_type type) {

    // error function
    angle_fit errfn {
        tip, base,
        type == nose ? param_nose_base : param_mouth_center
    };
    
    // rough estimate
    const auto iyaw   = clamp( 90 * tip.x, 70);
    const auto ipitch = clamp(133 * (base.y - tip.y), 70);
    simplex::state s({iyaw,ipitch}, {10,10}, errfn);

    // refine with maximum 25 steps
    simplex::step_until(s, errfn, simplex::spread_all(1.0f), 25);
    const auto& best = s.best()->second;
    assert(best.size() == 2);

    face_pose_type dest = {};
    dest.yaw = best[0];
    dest.pitch = 25 - best[1];
    return dest;
}

template <>
face_pose_type det::compute_pose<pose_method::nose_tip>(
    const coordinate_type& tip, const coordinate_type&, base_landmark_type) {

    face_pose_type dest = {};
    dest.yaw = clamp(45 * tip.x, 90);

    // the -0.3125 is center of face relative to center of eyes
    // median nose_tip.y from facelocate db is 0.225775 
    dest.pitch = clamp(45 * (tip.y - (0.3125f + 0.225775f)), 90);

    return dest;
}

template <pose_method method>
face_pose_type det::compute_pose(
    const coordinate_type& eye_left, const coordinate_type& eye_right,
    const coordinate_type& nose_tip,
    const coordinate_type& base, base_landmark_type type) {
    const auto ls = landmark_standardize<>(eye_left, eye_right);
    auto r = compute_pose<method>(ls(nose_tip),ls(base),type);
    r.roll = float(atan2deg(ls.eye_vec.y, ls.eye_vec.x));
    return r;
}

template <pose_method method>
face_pose_type det::compute_pose(const std::vector<coordinate_type>& pts) {
    if (pts.size() == 68)
        return compute_pose<method>(pts[30],pts[33],nose);
    if (pts.size() == 7)
        return compute_pose<method>(pts[2],0.5f*(pts[3]+pts[4]),mouth);
    throw std::invalid_argument(
        "compute_pose requires retina_v7 or dlib68 landmarks");
}

template <pose_method method>
face_pose_type det::compute_pose(const face_coordinates& face) {
    if (face.empty())
        throw std::invalid_argument("empty face");
    auto& dc = face.back();
    auto& pts = dc.landmarks;
    if (dc.type == dt::dlib68 && pts.size() == 68)
        return compute_pose<method>(dc.eye_left,dc.eye_right,
                                    pts[30],pts[33],nose);
    if (dc.type == dt::v7_retina && pts.size() == 7)
        return compute_pose<method>(dc.eye_left,dc.eye_right,
                                    pts[2],0.5f*(pts[3]+pts[4]),mouth);
    throw std::invalid_argument(
        "compute_pose requires retina_v7 or dlib68 landmarks");
}

// explicit template instantiation
template face_pose_type det::compute_pose<pose_method::nose_tip>(const face_coordinates&);
template face_pose_type det::compute_pose<pose_method::simplex>(const face_coordinates&);
template face_pose_type det::compute_pose<pose_method::nose_tip>(const std::vector<coordinate_type>&);
template face_pose_type det::compute_pose<pose_method::simplex>(const std::vector<coordinate_type>&);
