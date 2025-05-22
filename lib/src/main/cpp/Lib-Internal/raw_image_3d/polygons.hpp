#pragma once

#include "point3.hpp"

#include <stdext/span.hpp>
#include <stdext/forward_iterator.hpp>

#include <limits>


namespace raw_image {
    /** \brief RGB 8bit values along with a flag.
     *
     * The flag is not used internally.
     */
    struct rgbf {
        uint8_t r, g, b;
        uint8_t flag;
    };

    /// point in 3d along with rgbf (for vertices)
    struct point3f_rgbf : point3f, rgbf {};
    static_assert(sizeof(point3f_rgbf) == 4*sizeof(float));

    /// list of vertex indices to define triangle or quadrilateral faces
    struct index_list {
        static constexpr auto npos = std::numeric_limits<unsigned>::max();
        unsigned indices[4];

        index_list() : indices{npos,npos,npos,npos} {}
        index_list(unsigned i0, unsigned i1, unsigned i2, unsigned i3 = npos)
            : indices{i0,i1,i2,i3} {}

        constexpr bool is_triangle() const {
            return indices[3] == npos &&
                indices[2] != npos &&
                indices[1] != npos &&
                indices[0] != npos;
        }
        constexpr bool is_quad() const {
            return indices[3] != npos &&
                indices[2] != npos &&
                indices[1] != npos &&
                indices[0] != npos;
        }
    };
        
    /// list of indices along with rgbf (for polygon faces)
    struct indices_rgbf : index_list, rgbf {};
    static_assert(sizeof(indices_rgbf) == 5*sizeof(unsigned));
}

