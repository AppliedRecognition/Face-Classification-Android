#pragma once

#include "point3.hpp"

#include <raw_image/types.hpp>

#include <json/types.hpp>

namespace raw_image {

    /****************  NEW METHODS  ****************/

    using point2h = raw_image::point2<int16_t>;
    // using matrix3x4f = std::array<std::array<float,4>,3>;

    enum class distortion {
        none = 0,               ///< rectilinear
        modified_brown_conrady, ///< like Brown-Conrady, except tangential distortion applied to radially distorted points
        inverse_brown_conrady,  ///< like Brown-Conrady, except undistorts instead
        ftheta,                 ///< F-Theta fish-eye distortion model
        brown_conrady,          ///< Unmodified Brown-Conrady distortion model
        kannala_brandt4         ///< Four parameter Kannala Brandt distortion model
    };

    struct intrinsics {
        int   width;     ///< width in pixels
        int   height;    ///< weight in pixels

        point2f center;  ///< location of principal point
        point2f flen;    ///< focal length as multiple of width & height

        distortion model;        ///< distortion model
        std::array<float,5> coeffs; ///< distortion coefficients.

        intrinsics()
            : width(0), height(0),
              center{0,0}, flen{0,0},
              model(distortion::none),
              coeffs{0,0,0,0,0} {
        }

        explicit intrinsics(const json::object& obj)
            : width(make_number(obj["width"])),
              height(make_number(obj["height"])),
              model(distortion::none),
              coeffs(json::make_array<float,5>(obj["dcoeff"])) {
            auto center = json::make_array<float,2>(obj["center"]);
            this->center = { center[0], center[1] };
            auto flen = json::make_array<float,2>(obj["flen"]);
            this->flen = { flen[0], flen[1] };
            for (auto x : coeffs)
                assert(x <= 0 && 0 <= x); // ie. == 0
        }

        constexpr bool is_valid() const {
            return 0 < width && 0 < height &&
                0 < center.x && center.x < float(width) &&
                0 < center.y && center.y < float(height) &&
                0 < flen.x && 0 < flen.y;
        }

        point2f project(point3f pt) const {
            assert(model == distortion::none);
            return center + point2f {
                flen.x * pt.x / pt.z,
                flen.y * pt.y / pt.z
            };
        }

        point3f deproject(point2i pixel, float depth) const;
    };

    struct extrinsics {
        matrix3x3f rotation;  ///< rotation matrix
        point3f translation;  ///< translation vector in meters

        extrinsics() : rotation(I3x3f), translation{0,0,0} {}

        explicit extrinsics(const json::object& obj) {
            auto r = json::make_array<float,9>(obj["rotation"]);
            static_assert(sizeof(r) == sizeof(rotation));
            std::memcpy(&rotation, &r, sizeof(rotation));
            auto t = json::make_array<float,3>(obj["translation"]);
            translation = { t[0], t[1], t[2] };
            translation *= 1000; // convert metres to mm
        }

        bool is_identity() const {
            if (!(length_squared(translation) <= 0))
                return false;
            auto z = rotation - I3x3f;
            for (auto& r : z.rows)
                if (!(length_squared(r) <= 0))
                    return false;
            return true;
        }
    };


    struct point3f_point2h {
        point3f real = {0,0,0};
        point2h color = {0,0};

        inline bool is_zero() const {
            return color.x == 0 && color.y == 0 &&
                std::abs(real.x) + std::abs(real.y) + std::abs(real.z) <= 0;
        }
    };
    static_assert(sizeof(point3f_point2h) == 4 * sizeof(float));

    // tag to select Realsense SR300 style parameters
    struct sr300_t {};
    constexpr auto sr300 = sr300_t{};

    struct metadata {
        intrinsics color, depth;
        extrinsics translate;

        metadata() = default;

        /** \brief Construct from standard parameters.
         */
        explicit metadata(const json::object& obj)
            : color(get_object(obj["color"])),
              depth(get_object(obj["depth"])),
              translate(get_object(obj["translate"])) {
        }

        /** \brief Construct from Realsense SR300 (Datatang) style parameters.
         */
        explicit metadata(const json::object& obj, const sr300_t&);


        /** \brief Create point cloud from depth image pixels.
         *
         * Each returned real-world 3d point relative to the depth camera
         * also contains the coordinates of the matching color pixel.
         *
         * The rot_var (rotation variant) parameter is as follows:
         *   <0 reverse rotation,
         *    0 no rotation, or
         *   >0 assumed correct rotation.
         */
        std::vector<point3f_point2h>
        map_depth(const raw_image::plane& dimg, int rot_var = 1) const;
    };


    /****************  OLD METHODS  ****************/

    struct point_result {
        point2i cpx, dpx;   ///< pixel location in color and depth images
        point3f cloc, dloc; ///< world location rel to color and depth cameras
    };

    /** \brief Translate x,y pixel in depth image to x,y pixel in color image.
     */
    class camera_registration {
        point2f color_center, color_flen;
        point2f depth_center, depth_flen;
        point3f translate;
        matrix3x3f rotate;

    public:
        camera_registration() : color_flen{0,0}, depth_flen{0,0} {}
        explicit camera_registration(const json::object& params);

        explicit operator bool() const {
            return color_flen.x > 0;
        }

        // test if transformation is essentially the identity function
        // ie. the depth image is already aligned with the color image
        bool is_identity() const;

        // translate pixel in depth image to color image
        point_result
        operator()(point2i depth_pixel, unsigned depth_value) const;

        // align depth image to color
        void align(const raw_image::plane& depth_src,
                   const raw_image::plane& aligned_dest,
                   unsigned max_depth = 4000) const;

        // align depth image to color returning new depth image
        raw_image::plane_ptr align(const raw_image::plane& depth_src,
                                   unsigned max_depth = 4000) const;

        /*
        std::vector<rgbxyz> merge(const raw_image::plane& depth,
                                  const raw_image::plane& color,
                                  unsigned max_depth = 4000) const;
        */
    };
}
