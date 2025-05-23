#pragma once

#include "polygons.hpp"
#include <string_view>
#include <vector>

namespace raw_image {

    /// stl files don't have rgb and are triangles only
    void save_stl(stdx::span<const point3f> vertices,
                  stdx::forward_iterator<const index_list&> triangle_first,
                  stdx::forward_iterator<const index_list&> triangle_last,
                  std::string_view comment,
                  std::ostream& out);

    /// this overload ignores the rgbf data
    void save_stl(stdx::span<const point3f_rgbf> vertices,
                  stdx::forward_iterator<const index_list&> triangle_first,
                  stdx::forward_iterator<const index_list&> triangle_last,
                  std::string_view comment,
                  std::ostream& out);

    /// with rgb per vertex
    void save_ply(stdx::span<const point3f_rgbf> vertices,
                  stdx::forward_iterator<const index_list&> face_first,
                  stdx::forward_iterator<const index_list&> face_last,
                  std::string_view comment,
                  std::ostream& out);

    /// with rgb per face
    void save_ply(stdx::span<const point3f> vertices,
                  stdx::forward_iterator<const indices_rgbf&> face_first,
                  stdx::forward_iterator<const indices_rgbf&> face_last,
                  std::string_view comment,
                  std::ostream& out);


    struct point3f_rgbf_uv : point3f_rgbf {
        float u, v; // texture
    };

    enum ply_options : unsigned {
        ply_vertex_has_rgb = 1,
        ply_vertex_has_uv = 2,
        ply_face_has_rgb = 4
    };

    /** \brief Decode a ply file.
     *
     * \returns { vertices, faces, flags, comments }
     */
    std::tuple<std::vector<point3f_rgbf_uv>,
               std::vector<indices_rgbf>,
               ply_options,
               std::vector<std::string> >
    load_ply(const void* data, std::size_t size);
}
