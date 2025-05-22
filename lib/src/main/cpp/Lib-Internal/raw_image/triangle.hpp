#pragma once

#include "point2.hpp"
#include <algorithm>
#include <iterator>

namespace raw_image {

    /** \brief Point belonging to triangle along with per vertex weights.
     *
     * The 3 weights sum to denom > 0, so the actual weight per vertex is
     * the ratio weight / denom.
     */
    struct triangle_point : point2i {
        unsigned weights[3];
        unsigned denom;
    };


    /** \brief Iterate through the interior points of a triangle.
     *
     * This class uses barycentric coordinates to not only determine
     * which points belong to the triangle, but to also to compute
     * weights that may be applied to interpolate values between
     * the vertices.
     *
     * The points are produced in a top down left to right order.
     *
     * For the math, see https://codeplea.com/triangular-interpolation
     */
    class triangle_iterator {
    public:
        using iterator_category = std::forward_iterator_tag;
        using value_type = triangle_point;
        using difference_type = std::ptrdiff_t;
        using reference = value_type const&;
        using pointer = value_type const*;

        const value_type& operator*() const {
            return pos;
        }
        const value_type* operator->() const {
            return &pos;
        }

        bool operator==(const triangle_iterator& other) const {
            if (pos.y < y_bottom)
                return other.pos.y < other.y_bottom && pos == other.pos;
            return other.y_bottom <= other.pos.y;
        }
        bool operator!=(const triangle_iterator& other) const {
            return !(*this == other);
        }

        auto& operator++() {
            do {
                if (++pos.x >= x_right) {
                    pos.x = x_left;
                    if (++pos.y >= y_bottom)
                        break;
                }
            } while (!update_weights());
            return *this;
        }
        auto operator++(int) {
            auto orig = *this;  ++*this;  return orig;
        }

        triangle_iterator() : y_bottom(0) { pos.y = 0; }

        triangle_iterator(point2i v0, point2i v1, point2i v2)
            : verts{v0,v1,v2},
              x_left(std::min({v0.x,v1.x,v2.x})),
              x_right(std::max({v0.x,v1.x,v2.x})),
              y_bottom(std::max({v0.y,v1.y,v2.y})) {

            verts[1] -= verts[0];
            verts[2] -= verts[0];
            const auto d1 = int64_t(verts[1].x) * int64_t(verts[2].y);
            const auto d2 = int64_t(verts[2].x) * int64_t(verts[1].y);
            const auto neg = d1 < d2;
            pos.denom = neg ? d2 - d1 : d1 - d2;
            if (pos.denom == 0) {
                pos.y = y_bottom;
                return; // degenerate triangle
            }

            if (neg) {
                verts[1].x = -verts[1].x;
                verts[2].y = -verts[2].y;
            }
            else {
                verts[1].y = -verts[1].y;
                verts[2].x = -verts[2].x;
            }
            std::swap(verts[1].x, verts[1].y);
            std::swap(verts[2].x, verts[2].y);

            pos.x = x_left;
            pos.y = std::min({v0.y,v1.y,v2.y});

            // advance to first valid position
            while (!update_weights()) {
                if (++pos.x >= x_right) {
                    pos.x = x_left;
                    if (++pos.y >= y_bottom)
                        return; // no valid points found
                }
            }
        }


    private:
        triangle_point pos;
        point2i verts[3];
        int x_left, x_right; // horizontal range [left,right)
        int y_bottom;

        // returns true if this is a valid point
        bool update_weights() {
            const auto p = pos - verts[0];
            const auto w1 = dot<int64_t>(p,verts[2]);
            if (w1 < 0) return false;
            const auto w2 = dot<int64_t>(p,verts[1]);
            if (w2 < 0) return false;
            const auto sum = (pos.weights[1] = w1) + (pos.weights[2] = w2);
            if (sum > pos.denom)
                return false;
            pos.weights[0] = pos.denom - sum;
            return true;
        }
    };

    /// helper to create for loops over points of triangle
    inline auto triangle(point2i v0, point2i v1, point2i v2) {
        struct triangle_points {
            triangle_iterator first;
            auto& begin() const { return first; }
            auto end() const { return triangle_iterator{}; }
        };
        return triangle_points{triangle_iterator(v0,v1,v2)};
    }
}
