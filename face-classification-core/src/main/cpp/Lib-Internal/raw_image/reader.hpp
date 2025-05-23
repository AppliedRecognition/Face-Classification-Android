#pragma once

#include "core.hpp"

#include <array>
#include <cstdint>
#include <iterator>
#include <stdexcept>

namespace raw_image {

    /** \brief Abstract base class for line-by-line image reader.
     */
    class reader {
    public:
        virtual ~reader() = default;

        /** \brief Pixel layout.
         */
        inline pixel_layout layout() const { return m_layout; }

        /** \brief Number of bytes per pixel.
         */
        inline unsigned bytes_per_pixel() const { return m_bpp; }

        /** \brief Number of pixels per line.
         */
        inline unsigned width() const { return m_width; }

        /** \brief Size of line buffer.
         *
         * The number of pixels per line must be at least width, but
         * may be higher if a larger buffer has been allocated.
         */
        inline unsigned pixels_per_line() const { return m_pixels_per_line; }
        inline unsigned bytes_per_line() const { return m_bytes_per_line; }

        /** \brief Initial height of image.
         */
        inline unsigned height() const { return m_height; }

        /** \brief Number of lines remaining.
         *
         * This number includes the current line.
         */
        inline unsigned lines_remaining() const {
            return m_lines_remaining;
        }

        /** \brief Test if current line is valid.
         *
         * This test is the same as the test lines_remaining() > 0.
         */
        inline explicit operator bool() const {
            return m_lines_remaining > 0;
        }

        /** \brief Set minimum line buffer size.
         *
         * Use of this method may trigger the allocation and use
         * of a pixel buffer.
         *
         * This method should only be called before calling any other
         * non-const methods.
         */
        void set_pixels_per_line(unsigned ppl) {
            if (m_pixels_per_line < ppl) {
                m_pixels_per_line = ppl;
                m_bytes_per_line = m_bpp * ppl;
                if (m_buf)  // re-allocate larger buffer
                    m_buf = std::make_unique<uint8_t[]>(m_bytes_per_line);
            }
        }
        void set_bytes_per_line(unsigned bpl) {
            if (m_bytes_per_line < bpl)
                set_pixels_per_line((bpl+m_bpp-1)/m_bpp);
        }

        /** \brief Force the use of a buffer between source and destination.
         *
         * Providing dest_bytes the same as will be passed to copy_to()
         * allows for a more intellegent selection of input verses output
         * buffering.
         */
        void force_buffer(unsigned dest_bytes) {
            if (!m_buf &&
                (dest_bytes < bytes_per_line() || !buffered_internally()))
                m_buf = std::make_unique<uint8_t[]>(m_bytes_per_line);
        }
        inline void force_buffer() {
            force_buffer(m_bpp * m_width);
        }

        /** \brief Advance to next line.
         *
         * \pre lines_remaining() > 0
         * \returns false if no more lines available (lines_remaining() == 0)
         */
        bool next_line() {
            if (--m_lines_remaining > 0) {
                line_next();
                m_line_copied_to_buf = false;
                return true;
            }
            return false;
        }

        /** \brief Advance to next line.
         */
        inline reader& operator++() {
            next_line();
            return *this;
        }

        /** \brief Read-only access either directly to the source image or
         * a buffer to which the pixel data has been stored.
         *
         * Up to pixels_per_line may be safely accessed at this address.
         * The content of pixels beyond width is not specified.
         */
        const uint8_t* get_line();
        inline const uint8_t* operator*() {
            return get_line();
        }
        template <unsigned BPP>
        inline const std::array<uint8_t,BPP>* as_bpp() {
            static_assert(sizeof(std::array<uint8_t,BPP>) == BPP,
                          "std::array has incorrect size");
            return reinterpret_cast<const std::array<uint8_t,BPP>*>(get_line());
        }

        /** \brief Copy pixels to detination buffer.
         *
         * If possible, specify dest_bytes > bytes_per_pixel * width to enable
         * certain optimizations such as the use of SIMD vector operations.
         * In particular, pixel layout conversion benefits from passing
         * dest_bytes == bytes_per_pixel * (width rounded up to multiple of 4).
         *
         * If dest_bytes is not specified, then exactly
         * bytes_per_pixel * width bytes are copied.
         *
         * \returns dest
         */
        void* copy_to(void* dest, unsigned dest_bytes);
        inline void* copy_to(void* dest) {
            return copy_to(dest, m_bpp * m_width);
        }

        /** \brief Copy as many lines of image as possible to destination image.
         */
        void copy_to(const plane& dest, unsigned dest_bytes_per_line);
        inline void copy_to(const plane& dest) {
            return copy_to(
                dest, raw_image::bytes_per_pixel(dest.layout) * dest.width);
        }


        /** \brief Merge specified bytes into destination while leaving
         * other bytes unmodified.
         *
         * The dest_idx array specifies, for each source byte, which
         * destination byte that source byte should be copied to.
         * If a dest_idx value is >= dest_bpp, then that source byte is
         * not copied.
         * Any dest bytes that are not overwritten will retain their
         * initial value.
         *
         * \returns dest
         */
        void* map_to(void* dest, unsigned dest_bpp,
                     const std::array<unsigned,4>& dest_idx);

        /** \brief Map as many lines of image as possible to destination image.
         */
        void map_to(const plane& dest,
                    const std::array<unsigned,4>& dest_idx);


        /** \brief Copy to rotated destination image.
         */
        void rotate_to(const plane& dest, unsigned rotate);


        /** \brief Construct reader from multi-plane image with optional
         * rotate and pixel layout conversion.
         */
        static std::unique_ptr<reader> construct_with_opts(
            const multi_plane_arg& from,
            const stdx::options_tuple<rotate,pixel_layout>& opts);
        template <typename... Opts>
        static inline auto
        construct(const multi_plane_arg& from, Opts&&... opts) {
            return construct_with_opts(from, { std::forward<Opts>(opts)... } );
        }


    protected:
        const unsigned m_width;
        const unsigned m_height;
        const pixel_layout m_layout;
        const unsigned m_bpp;

        /** \brief Constructor.
         */
        reader(unsigned width, unsigned height,
               pixel_layout layout,
               unsigned min_pixels_per_line = 0)
            : m_width(width),
              m_height(height),
              m_layout(layout),
              m_bpp(raw_image::bytes_per_pixel(layout)),
              m_lines_remaining(height),
              m_pixels_per_line(std::max(width, min_pixels_per_line)),
              m_bytes_per_line(m_bpp * m_pixels_per_line) {
            if (m_bpp <= 0 || 4 < m_bpp)
                throw std::invalid_argument("reader: invalid pixel layout");
        }

        /** \brief Advance to next line.
         *
         * This method is called after the number of lines remaining has been
         * decremented.
         */
        virtual void line_next() = 0;

        /** \brief Copy up to pixels_per_line pixels to dest buffer.
         */
        virtual void line_copy(void* dest) = 0;

        /** \brief Direct access to line at source.
         *
         * This method must return nullptr if the line cannot be
         * accessed in this way.
         * In this case, a buffer must be allocated and line_copy() used
         * for all subsequent lines.
         *
         * If the return is not null, then at least pixels_per_line
         * must be accessable at that address.
         */
        virtual const uint8_t* line_direct() {
            return nullptr;
        }

        /** \brief Override and return true if buffering is internal.
         */
        virtual bool buffered_internally() const {
            return false;
        }

        /** \brief Test if address is that of the internal buffer.
         */
        inline bool is_buffer(const void* buf) const {
            return m_buf.get() == buf;
        }


    private:
        unsigned m_lines_remaining;
        unsigned m_pixels_per_line;
        unsigned m_bytes_per_line;
        bool m_line_copied_to_buf = false;
        std::unique_ptr<uint8_t[]> m_buf;

        reader(reader&&) = delete;
        reader(const reader&) = delete;
        reader& operator=(reader&&) = delete;
        reader& operator=(const reader&) = delete;
    };


    /** \brief Convert to another pixel layout.
     *
     * If dest_layout is pixel::none or is the same as src->layout, then
     * nothing is done and src is returned as is.
     *
     * If the conversion cannot be performed (is not implememnted), then
     * an error is logged and nullptr is returned.
     */
    std::unique_ptr<reader>
    convert(std::unique_ptr<reader> src,
            pixel_layout dest_layout);


    /** \brief Transform lines using arbitrary function.
     *
     * A "quad" is 4 pixels.
     * The function signature should be compatible with: <pre>
     *   void fn(uint8_t* dest_line, const uint8_t* src_line, unsigned nquads);
     * </pre>
     * The total number of pixels to transform is <code>4 * nquads</code>.
     *
     * The number of pixels per line available in both the source and
     * destination buffers is the width rounded up to a multiple of 4.
     * This allows for vector operations on up to 4 pixels at a time.
     */
    template <typename FN>
    std::unique_ptr<reader>
    transform_quads(std::unique_ptr<reader> src,
                    pixel_layout dest_layout,
                    FN&& func) {

        struct converter final : reader {
            const std::unique_ptr<reader> src;
            std::decay_t<FN> conv;
            const unsigned nquads;

            converter(std::unique_ptr<reader> src,
                      pixel_layout dest_layout,
                      unsigned nquads, FN&& func)
                : reader(src->width(), src->height(),
                                   dest_layout, 4*nquads),
                  src(move(src)),
                  conv(std::forward<FN>(func)),
                  nquads(nquads) {
                this->src->set_pixels_per_line(pixels_per_line());
            }

            void line_next() override {
                if (!src->next_line())
                    throw std::logic_error("unexpected end of image");
            }

            void line_copy(void* dest) override {
                conv(static_cast<uint8_t*>(dest), **src, nquads);
            }

            bool buffered_internally() const override {
                src->force_buffer();
                return true;
            }
        };

        const auto nquads = (src->width()+3) / 4;
        return std::make_unique<converter>(
            move(src), dest_layout, nquads, std::forward<FN>(func));
    }


    /** \brief Rotate image by integer number of gradians.
     *
     * This method respects the src rotate value in the sense that it is as
     * if the image is mirrored and rotated according to this value prior
     * to extraction of the rotated region.
     * The src scale value is ignored.
     *
     * Output is always upscaled by a factor of 2 in width and height, 
     * so ensure to choose output width and height accordingly.
     * For example, if width and height are 100x100, then only a
     * (roughly) 50x50 region of pixels from the source image is used
     * (upscaled to 100x100).
     *
     * The output image is the same layout as the input.
     * Each output pixel is a copy of an input pixel (or padding_value).
     * No interpolation is done.
     * In cases where a pixel is needed from outside the bounds of the
     * source image, either the nearest border pixel is used, or
     * if specified, the padding_value is used.
     */
    std::unique_ptr<reader>
    rotate_gradians(const plane& src,
                    int angle_gradians,
                    float center_x, float center_y,
                    unsigned width, unsigned height);
    std::unique_ptr<reader>
    rotate_gradians(const plane& src,
                    int angle_gradians,
                    float center_x, float center_y,
                    unsigned width, unsigned height,
                    uint32_t padding_value);

    /** \brief Rotate image by integer number of gradians
     * and expand pixel layout.
     *
     * This overload always produces a 4 bpp (bytes per pixel) output.
     *
     * If the input is already 4 bpp then this overload returns the
     * same result as the other overload (and base_value is ignored).
     *
     * If the input is RGB24 or BGR24, then the output will be
     * RGBA32 or BGRA32 (respectively).
     *
     * Otherwise the output will have a layout designated for 4 bpp
     * but will otherwise be an invalid layout (with value 0x4ff).
     *
     * Each destination pixel is first filled with base_value and then
     * the source bytes are copied.
     */
    std::unique_ptr<reader>
    rotate_gradians(const plane& src,
                    int angle_gradians,
                    float center_x, float center_y,
                    unsigned width, unsigned height,
                    uint32_t padding_value,
                    uint32_t base_value);


    /** \brief Scale image using nearest neighbour method.
     *
     * Pixels are dropped when downscaling and replicated when upscaling.
     * No interpolation, averaging or other pixel interpretation is done.
     */
    std::unique_ptr<reader>
    scale_nearest(std::unique_ptr<reader> src,
                  unsigned width, unsigned height);

    /** \brief Scale image using area method.
     *
     * This scaler gives good results when downscaling, but
     * may result in a blocky image when upscaling.
     */
    std::unique_ptr<reader>
    scale_area(std::unique_ptr<reader> src,
               unsigned width, unsigned height);

    /** \brief Scale image using interpolation.
     *
     * If not scaling up by at least a factor of 1.5, this method gives
     * the same result as scale_by_area().
     * When upscaling by at least 1.5 (width and/or height) this method
     * gives a more accurate result but may be slower.
     */
    std::unique_ptr<reader>
    scale_interpolate(std::unique_ptr<reader> src,
                      unsigned width, unsigned height);


    /** \brief Line iterator for reader.
     */
    struct line_iterator_base {
        std::unique_ptr<reader> m_reader;

        using difference_type = std::ptrdiff_t;
        using iterator_category = std::input_iterator_tag;

        inline bool operator==(const line_iterator_base& other) {
            return m_reader.get() == other.m_reader.get();
        }
        inline bool operator!=(const line_iterator_base& other) {
            return !(*this == other);
        }
        inline void operator++(int) {
            if (!m_reader->next_line()) m_reader = nullptr;
        }
    };
    template <unsigned BPP>
    struct line_iterator : line_iterator_base {
        line_iterator() = default;
        line_iterator(std::unique_ptr<reader> r)
            : line_iterator_base{move(r)} {
            if (m_reader) {
                if (!*m_reader)
                    m_reader = nullptr;
                else if (m_reader->bytes_per_pixel() != BPP)
                    throw std::invalid_argument("bytes per pixel mismatch");
            }
        }

        using value_type = stdx::span<const std::array<uint8_t, BPP> >;
        using pointer = const value_type*;
        using reference = const value_type&;

        inline value_type operator*() const {
            return { m_reader->as_bpp<BPP>(), m_reader->width() };
        }
        inline auto& operator++() {
            if (!m_reader->next_line()) m_reader = nullptr;
            return *this;
        }
    };
    template <>
    struct line_iterator<1> : line_iterator_base {
        line_iterator() = default;
        line_iterator(std::unique_ptr<reader> r)
            : line_iterator_base{move(r)} {
            if (m_reader) {
                if (!*m_reader)
                    m_reader = nullptr;
                else if (m_reader->bytes_per_pixel() != 1)
                    throw std::invalid_argument("bytes per pixel mismatch");
            }
        }

        using value_type = stdx::span<const uint8_t>;
        using pointer = const value_type*;
        using reference = const value_type&;

        inline value_type operator*() const {
            return { m_reader->get_line(), m_reader->width() };
        }
        inline auto& operator++() {
            if (!m_reader->next_line()) m_reader = nullptr;
            return *this;
        }
    };


    /** \brief Range-based for loop over the lines of an image.
     *
     * This method enables the following pattern: <pre>
     *   for (auto&& line : read_lines_bpp<3>(image, ...)) {
     *       // line is of type stdx::span<...>
     *       // line.size() == image width
     *       for (auto&& pixel : line) {
     *           // do something with the pixel
     *           auto channel0 = pixel[0];
     *           auto channel1 = pixel[1];
     *           auto channel2 = pixel[2];
     *       }
     *   }
     * </pre>
     * This method does not transform the image.
     * An exception is thrown if BPP does not match the image.
     *
     * If bytes per pixel > 1, then pixel has type std::array<uint8_t, BPP>.
     * If bytes per pixel == 1, then pixel is uint8_t.
     */
    template <unsigned BPP>
    auto read_lines_bpp(std::unique_ptr<reader> r) {
        struct range {
            using iterator = line_iterator<BPP>;
            std::unique_ptr<reader> m_reader;
            inline iterator begin() { return iterator{move(m_reader)}; }
            inline iterator end() const { return {}; }
        };
        return range{move(r)};
    }
    template <unsigned BPP, typename... Opts>
    inline auto
    read_lines_bpp(const multi_plane_arg& image, Opts&&... opts) {
        return read_lines_bpp<BPP>(
            reader::construct(image, std::forward<Opts>(opts)...));
    }

    /** \brief Range-based for loop over the lines of an image.
     *
     * This method enables the following pattern: <pre>
     *   for (auto&& line : read_lines_of<pixel::rgb24>(image, ...)) {
     *       // line is of type stdx::span<...>
     *       // line.size() == image width
     *       for (auto&& pixel : line) {
     *           // do something with the pixel
     *           auto red   = pixel[0];
     *           auto green = pixel[1];
     *           auto blue  = pixel[2];
     *       }
     *   }
     * </pre>
     * If the image is not in the specified format, it is transformed
     * line by line (on the fly) to that format.
     *
     * If bytes per pixel > 1, then pixel has type std::array<uint8_t, BPP>.
     * If bytes per pixel == 1, then pixel is uint8_t.
     */
    template <pixel_layout layout, typename... Opts>
    inline auto
    read_lines_of(const multi_plane_arg& image, Opts&&... opts) {
        static constexpr auto BPP = bytes_per_pixel(layout);
        return read_lines_bpp<BPP>(
            reader::construct(
                image, std::forward<Opts>(opts)..., layout));
    }
}
