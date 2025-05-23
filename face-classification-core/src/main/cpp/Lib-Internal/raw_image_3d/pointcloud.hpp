#pragma once

#include "polygons.hpp"
#include <raw_image/types.hpp>
#include <stdext/binary.hpp>
#include <stdext/rounding.hpp>
#include <cstdint>
#include <vector>

namespace raw_image {

    // forward declare
    class rgbpoint64;
    bool reduce_cloud(std::vector<rgbpoint64>& vec, float thres = 1);


    /** \brief Expanded 128-bits per point record.
     */
    using rgbpoint128 = point3f_rgbf;
    static_assert(sizeof(rgbpoint128) == 128/8);


    /** \brief RGB and XYZ point in real space packed into 64 bits.
     *
     *  15 bits for X -> [-512,512) mm (1/32mm resolution)
     *  15 bits for Y
     *  10 bits for Z -> [0,1023] mm
     *  8 bits per RGB
     *
     * The least significant 2 bits of Z are stored in the bottom bit
     * of X and Y, leaving 15 significant bits for each of X and Y.
     */
    class rgbpoint64 {
        int16_t m_x, m_y;
        uint8_t m_z;

    public:
        uint8_t r, g, b;

        template <typename T>
        inline void set_rgb(T r, T g, T b) {
            this->r = stdx::round_from(r);
            this->g = stdx::round_from(g);
            this->b = stdx::round_from(b);
        }

        void set_xyz(float x, float y, float z);

        constexpr float x() const {
            return m_x / 64.0f;
        }
        constexpr float y() const {
            return m_y / 64.0f;
        }
        constexpr float z() const {
            return float((((m_z<<1) + (m_y&1u)) << 1) + (m_x&1u));
        }

        inline operator rgbpoint128() const {
            rgbpoint128 pt;
            pt.x = x(), pt.y = y(), pt.z = z();
            pt.r = r, pt.g = g, pt.b = b, pt.flag = 0;
            return pt;
        }

        friend float
        length_squared(const rgbpoint64& a, const rgbpoint64& b) {
            const auto dx = (a.m_x - b.m_x) / 64.0f;
            const auto dy = (a.m_y - b.m_y) / 64.0f;
            const auto dz = a.z() - b.z();
            return dx*dx + dy*dy + dz*dz;
        }

        // returns true if something was done (ie. vec has shrunk)
        friend bool reduce_cloud(std::vector<rgbpoint64>& vec, float thres);
    };
    static_assert(sizeof(rgbpoint64) == 64/8);

    /** \brief Expand from rgbpoint64.
     */
    std::vector<rgbpoint128>
    to_rgbpoint128(const std::vector<rgbpoint64>& vec);

    /** \brief Decode from saved rgbpoint64 binary.
     */
    std::vector<rgbpoint128>
    to_rgbpoint128(const stdx::binary& bin);

    /** \brief Create RGB or RGBD image from point cloud.
     *
     * The xy_scale indicates the size of the pixels in mm.
     * That is, e.g. a value of 1.75 indicates each pixel is 1.75mm x 1.75mm.
     *
     * z_nearest is the Z-value that will map to a D-value of 255.
     * From there pixels with higher Z-value map to lower D-value.
     * On the D scale, 0 is furthest from the camera and 255 is closest.
     * Any points with z-value too low (close to the camera) such that
     * they would have D-value greater than 255 are not used.
     *
     * Note that some pixels might be left uninitialized so the caller
     * must initialize all pixels to some specific value before making
     * this call.
     *
     * Only rgb24 and rgba32 output pixel layouts are currently supported.
     */
    void fill_rgbd(const std::vector<rgbpoint128>& cloud,
                   const plane& to, float xy_scale, float z_nearest);
    [[deprecated("Use fill_rgbd() instead.")]]
    void make_rgbd(const std::vector<rgbpoint128>&, const plane&, float);

    /** \brief Create RGBD image from point cloud.
     *
     * Creates a width by height image with rgba32 pixel layout.
     *
     * The median Z-value of the points that fit within this image is
     * computed and returned.
     * This median Z-value is used to set z_nearest in the call to fill_rgbd()
     * such that this median Z-value will be mapped to median_target D-value
     * in the final image.
     *
     * The in_place_fill_bytes() method is also called with a border of
     * all-zero pixels to complete the image.
     */
    std::pair<plane_ptr, float>
    make_rgbd(const std::vector<rgbpoint128>& cloud,
              unsigned width, unsigned height,
              float xy_scale, unsigned median_target);

    /** \brief Second result from make_rgbd() without generating image.
     */
    float z_median(const std::vector<rgbpoint128>& cloud,
                   unsigned width, unsigned height, float xy_scale);

    /** \brief Create vertices and faces for output to ply from rgbd image.
     */
    std::pair<std::vector<point3f_rgbf>, std::vector<index_list> >
    render_rgbd(const plane& rgbd, float xy_scale);
}
