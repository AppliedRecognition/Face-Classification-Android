#pragma once

#include "core.hpp"
#include "color.hpp"
#include "point2.hpp"
#include <vector>

namespace raw_image {

    /** \brief Fill image with specified color.
     *
     * To fill a specific rectangular region of interest, do
     * <code>fill(crop(dest,x,y,w,h), color)</code>.
     */
    void fill(single_plane_arg dest, pixel_color color);

    /** \brief Flood fill image with specified color.
     *
     * This method replaces pixels having the color currently present
     * at the specified location with the specified replacement color.
     * The flood proceeds moving left, right, up and down.  It does not
     * move diagonally.
     *
     * If the specified pixel already holds the specified color, then
     * this method does nothing and returns zero.
     *
     * \throws std::invalid_argument if x or y is out of range
     *
     * \returns total number of pixels set
     */
    unsigned fill(single_plane_arg dest, unsigned x, unsigned y,
                  std::array<uint8_t,4> replacement_color);
    unsigned fill(single_plane_arg dest, unsigned x, unsigned y,
                  pixel_color replacement_color);

    /** \brief Draw line on image.
     *
     * The coordinates will be transformed to match the rotated and scaled
     * image by using the to_image_point() method.
     */
    void line(single_plane_arg dest,
              double x0, double y0, double x1, double y1,
              pixel_color color, unsigned width = 1);
    template <typename T0, typename T1>
    inline void line(single_plane_arg dest,
                     const point2<T0>& p0, const point2<T1>& p1,
                     pixel_color color, unsigned width = 1) {
        line(dest, double(p0.x), double(p0.y), double(p1.x), double(p1.y),
             color, width);
    }

    /** \brief Draw circle on image.
     *
     * If radius is negative, then circle is filled.
     *
     * The coordinates will be transformed to match the rotated and scaled
     * image by using the to_image_point() method.
     */
    void circle(single_plane_arg dest,
                double x, double y, pixel_color color, int radius = 1);
    template <typename T>
    inline void circle(single_plane_arg dest, const point2<T>& p,
                       pixel_color color, int radius = 1) {
        circle(dest, double(p.x), double(p.y), color, radius);
    }

    /** \brief Text rendering.
     *
     * Uses stb_easy_font to render text as a collection of rectangles.
     * Rendering text requires at least two steps:
     *   1. create an instance of this class with the text to render
     *   2. call render method to render text
     */
    class easy_font {
    public:
        struct rect_type { unsigned x,y,w,h; };
        std::vector<rect_type> rects;

        // construct empty
        easy_font() = default;

        /** \brief Compute rectangles to render text.
         *
         * The text may contain ascii codes from 32 to 126, inclusive.
         * Also, '\n' can be used to render multi-line text.
         * All other ascii codes are treated as extra space.
         *
         * The adjust_spacing values are relative to scale.
         * For example, if adjust_spacing = -3 and scale = 2, then
         * the actual adjustment is -3/2 = -1.5.  
         * To get the same result with scale = 4, set adjust_spacing = -6.
         */
        easy_font(const char* text, unsigned scale = 1,
                  int adjust_character_spacing = 0,
                  int adjust_line_spacing = 0);

        easy_font(const std::string& text, unsigned scale = 1,
                  int adjust_character_spacing = 0,
                  int adjust_line_spacing = 0)
            : easy_font(text.c_str(), scale,
                        adjust_character_spacing,
                        adjust_line_spacing) {
        }

        // minimum image width required to render complete text
        auto width() const {
            unsigned w = 0;
            for (auto& r : rects)
                w = std::max(w, r.x+r.w);
            return w;
        }

        // minimum image height required to render complete text
        auto height() const {
            unsigned h = 0;
            for (auto& r : rects)
                h = std::max(h, r.y+r.h);
            return h;
        }

        /** \brief Render text on image.
         *
         * Text is rendered with an effectively transparent background.
         * Use the crop() method to select the region where the text should
         * be rendered.
         * If the image is too small, then rectangles that don't fit will
         * not be rendered (ie. no error or exception).
         */
        void render(const plane& dest, pixel_color c = color_white) const;

        // helper with x,y location
        inline void render(const plane& dest, unsigned x, unsigned y,
                           pixel_color c = color_white) const {
            if (x < dest.width && y < dest.height)
                render(crop(dest,x,y,dest.width-x,dest.height-y),c);
        }
    };

}
