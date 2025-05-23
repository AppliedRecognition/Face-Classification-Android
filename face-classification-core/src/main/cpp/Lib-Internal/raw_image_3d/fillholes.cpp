
#include "fillholes.hpp"

#include <raw_image/reader.hpp>
#include <raw_image/pixels.hpp>
#include <raw_image/point2.hpp>

#include <applog/core.hpp>

#include <algorithm>
#include <cassert>
#include <cstring>
#include <limits>
#include <vector>

using namespace raw_image;

namespace {
    /** \brief Interpolate and optionally extrapolate horizontal lines
     * of empty pixels.
     */
    template <typename CHTYPE>
    struct horz_interpolate {
        CHTYPE const* empty;  ///< value representing empty pixel to fill
        unsigned chpp;        ///< number of CHTYPE per pixel
        unsigned width;       ///< pixels per line
        bool extrapolate;

        using I = std::conditional_t<sizeof(CHTYPE) <= 2, int, int64_t>;

        inline bool is_empty(CHTYPE const* src) const {
            return memcmp(src, empty, chpp*sizeof(CHTYPE)) == 0;
        }

        auto interpolate(CHTYPE const* left, CHTYPE const* right,
                         unsigned n, CHTYPE* dest) const {
            const auto d = I(n+1);
            const auto r = d/2;
            for (I i = 1; i <= I(n); ++i) {
                const auto j = d - i;
                for (unsigned k = 0; k < chpp; ++k, ++dest)
                    *dest = CHTYPE((j*I(left[k]) + i*I(right[k]) + r) / d);
            }
            return dest;
        }
        
        void operator()(void* vdest, const void* vsrc, unsigned n) const {
            assert(width <= 4*n);
            if ((n = width) == 0)
                return;
            assert(n <= unsigned(std::numeric_limits<I>::max()));
            auto dest = static_cast<CHTYPE*>(vdest);
            auto src = static_cast<CHTYPE const*>(vsrc);

            if (is_empty(src)) {
                // line starts with an empty pixel
                auto px = src + chpp;
                for (unsigned j = 1; j < n; ++j, px += chpp) {
                    if (!is_empty(px)) {
                        n -= j;
                        if (!extrapolate)
                            dest = std::copy_n(src, j*chpp, dest);
                        else // fill start of line with first non-empty pixel
                            for ( ; j > 0; --j)
                                dest = std::copy_n(px, chpp, dest);
                        src = px;
                        px = nullptr;
                        break;
                    }
                }
                if (px) {
                    // entire line is empty
                    std::copy_n(src, n*chpp, dest);
                    return;
                }
            }

            while (1 <= n) {
                assert(!is_empty(src));

                if (n == 1) {
                    std::copy_n(src, chpp, dest);
                    break;
                }

                if (!is_empty(src + chpp)) {
                    dest = std::copy_n(src, chpp, dest);
                    src += chpp;
                    --n;
                    continue;
                }

                assert(is_empty(src + chpp));

                // find next non-empty pixel
                auto px = src + 2*chpp;
                for (unsigned j = 2; j < n; ++j, px += chpp) {
                    if (!is_empty(px)) {
                        // src[0] and px = src[j] are non-empty
                        // src[ 1 ... j-1 ] are empty
                        dest = std::copy_n(src, chpp, dest);
                        dest = interpolate(src, px, j-1, dest);
                        n -= j;
                        src += j*chpp;
                        px = nullptr;
                        break;
                    }
                }
                if (px) {
                    // remainder of line is empty
                    if (!extrapolate)
                        std::copy_n(src, n*chpp, dest);
                    else // fill with last non-empty pixel
                        do {
                            dest = std::copy_n(src, chpp, dest);
                        } while (--n > 0);
                    break;
                }
            }
        }
    };


    /** \brief Calculate mean of 2 images, pixel by pixel.
     *
     * Also check if either image has an empty pixel.  If either does then
     * use other image's pixel as is (no averaging).
     */
    template <typename CHTYPE>
    struct mean_pixel {
        CHTYPE const* empty;  ///< value representing empty pixel to fill
        unsigned chpp;        ///< number of CHTYPE per pixel
        std::unique_ptr<reader> rhs;  ///< other image to average

        using I = std::conditional_t<sizeof(CHTYPE) <= 2, int, int64_t>;

        inline bool is_empty(CHTYPE const* src) const {
            return memcmp(src, empty, chpp*sizeof(CHTYPE)) == 0;
        }

        void operator()(void* vdest, const void* vsrc, unsigned n) {
            assert(rhs);
            auto dest = static_cast<CHTYPE*>(vdest);
            auto src1 = static_cast<CHTYPE const*>(vsrc);
            auto src2 = reinterpret_cast<CHTYPE const*>(rhs->get_line());
            if (!rhs->next_line())
                rhs = nullptr;
            for (n *= 4; n > 0; --n, src1 += chpp, src2 += chpp) {
                if (is_empty(src1))
                    dest = std::copy_n(src2, chpp, dest);
                else if (is_empty(src2))
                    dest = std::copy_n(src1, chpp, dest);
                else // compute mean
                    for (unsigned k = 0; k < chpp; ++k, ++dest)
                        *dest = CHTYPE((I(src1[k]) + I(src2[k])) / 2);
            }
        }        
    };
}

template <typename T>
static void inplace_fill(const plane& img, T const* empty, bool extrapolate) {
    using HORZ = horz_interpolate<T>;
    using MEAN = mean_pixel<T>;
    const auto chpp = unsigned(bytes_per_pixel(img) / sizeof(T));
    if (chpp*sizeof(T) != bytes_per_pixel(img))
        throw std::logic_error("pixel is not an integral number of channels");
    
    // create temporary rotated by 90 degrees image with vertical columns
    // from the original filled
    const auto rot = create(img.height, img.width, img.layout);
    transform_quads(reader::construct(img, rotate(1)), img.layout,
                    HORZ{empty, chpp, img.height, extrapolate})->copy_to(*rot);

    // fill horizontal rows from original
    auto src = transform_quads(reader::construct(img), img.layout,
                               HORZ{empty, chpp, img.width, extrapolate});

    // average with temporary and write back to original image
    transform_quads(reader::construct(rot, rotate(3)), img.layout,
                    MEAN{empty,chpp,move(src)})->copy_to(img);
}

template<> 
void raw_image::in_place_fill_holes<uint8_t>(
    const plane& img, uint8_t const* empty, extrapolate_option ex) {
    inplace_fill(img, empty, ex);
}

template<> 
void raw_image::in_place_fill_holes<int8_t>(
    const plane& img, int8_t const* empty, extrapolate_option ex) {
    inplace_fill(img, empty, ex);
}

template<>
void raw_image::in_place_fill_holes<uint16_t>(
    const plane& img, uint16_t const* empty, extrapolate_option ex) {
    inplace_fill(img, empty, ex);
}

template<>
void raw_image::in_place_fill_holes<int16_t>(
    const plane& img, int16_t const* empty, extrapolate_option ex) {
    inplace_fill(img, empty, ex);
}


/****************  in_place_fill_bytes()  ****************/

static bool operator<(const point2i& a, const point2i& b) {
    return a.x <= b.x && (a.x < b.x || a.y < b.y);
}
static bool operator<=(const point2i& a, const point2i& b) {
    return a.x <= b.x && (a.x < b.x || a.y <= b.y);
}

namespace {
    struct hole_line {

        point2i first, last; // coords of non-hole pixel at each end

        bool is_vert() const { return first.x == last.x; }
        bool is_horz() const { return first.y == last.y; }

        // size of hole in pixels
        int size() const {
            return std::max(last.x - first.x, last.y - first.y) - 1;
        }

        // note: this method will match the end points too
        bool contains(const point2i& other) const {
            return first.x <= other.x && other.x <= last.x &&
                first.y <= other.y && other.y <= last.y;
        }

        template <typename T, std::size_t N>
        std::array<T,N>
        interpolate(const point2i& mid,
                    const std::array<T,N>& first_px,
                    const std::array<T,N>& last_px) const {
            const auto d0 = std::max(mid.x - first.x, mid.y - first.y);
            const auto d1 = std::max(last.x - mid.x, last.y - mid.y);
            const auto denom = d0 + d1;
            const auto half = denom / 2;
            std::array<T,N> px;
            for (unsigned i = 0; i < N; ++i)
                px[i] = T((first_px[i] * d1 + last_px[i] * d0 + half) / denom);
            return px;
        }

        bool operator==(const hole_line& other) const {
            return first == other.first && last == other.last;
        }

        bool operator<(const hole_line& other) const {
            if (size() < other.size())
                return true;
            if (size() == other.size())
                return first <= other.first &&
                    (first < other.first || last < other.last);
            return false;
        }

        struct iterator {
            using iterator_category = std::forward_iterator_tag;
            using value_type = point2i;
            using difference_type = std::ptrdiff_t;
            using pointer = point2i const*;
            using reference = point2i const&;

            point2i pt;
            bool going_down;

            iterator& operator++() {
                ++(going_down ? pt.y : pt.x);
                return *this;
            }

            bool operator==(const iterator& other) const {
                return pt == other.pt;
            }
            bool operator!=(const iterator& other) const {
                return pt != other.pt;
            }

            inline reference operator*() const { return pt; }
            inline pointer operator->() const { return &pt; }
        };

        iterator begin() const {
            return std::next(iterator{first,is_vert()});
        }
        iterator end() const {
            return iterator { last, is_vert() };
        }
    };
}

template <std::size_t BPP>
static void fill_bytes(const plane& img, std::array<uint8_t,BPP> hole) {

    raw_image::pixels_bpp<BPP> pixels(img);
    const auto is_hole = [&](const auto& px) {
        return memcmp(px.data(), hole.data(), BPP) == 0;
    };
    const auto is_hole_pt = [&](const auto& pt) {
        return is_hole(pixels(pt.x,pt.y));
    };

    // map out holes
    std::vector<hole_line> lines;
    raw_image::point2i pt = {0,0};
    for (const auto& line : pixels) {
        for (auto&& px : line) {
            if (is_hole(px)) {
                auto left = pt;
                if (0 <= --left.x && !is_hole_pt(left)) {
                    auto right = pt;
                    while (unsigned(++right.x) < pixels.width())
                        if (!is_hole_pt(right)) {
                            lines.push_back({left,right});
                            break;
                        }
                }
                auto top = pt;
                if (0 <= --top.y && !is_hole_pt(top)) {
                    auto bottom = pt;
                    while (unsigned(++bottom.y) < pixels.height())
                        if (!is_hole_pt(bottom)) {
                            lines.push_back({top,bottom});
                            break;
                        }
                }
            }
            ++pt.x;
        }
        pt.x = 0, ++pt.y;
    }

    // sort largest first
    sort(lines.begin(), lines.end(),
         [](const auto& a, const auto& b) { return b < a; });

    // fill holes from smallest (back) to largest
    while (!lines.empty()) {
        if (0) { // verify invariant
            FILE_LOG(logINFO) << lines.size() << '\t' << lines.back().size();
            auto prev_size = lines.front().size();
            for (const auto& line : lines) {
                assert(0 < line.size() && line.size() <= prev_size);
                prev_size = line.size();
                assert(line.is_vert() || line.is_horz());
                assert(!(line.is_vert() && line.is_horz()));
                assert(!is_hole_pt(line.first));
                assert(!is_hole_pt(line.last));
                for (auto pt : line)
                    assert(is_hole_pt(pt));
            }
        }

        auto line = lines.back();
        lines.pop_back();
        auto& px0 = pixels(line.first.x, line.first.y);
        auto& px1 = pixels(line.last.x, line.last.y);
        assert(!is_hole(px0) && !is_hole(px1));
        auto resort_idx = long(lines.size());
        for (auto& pt : line) {
            auto& dest = pixels(pt.x, pt.y);
            assert(is_hole(dest));
            dest = line.interpolate(pt, px0, px1);
            if (is_hole(dest)) {
                // this shouldn't happen but in case it does
                // tweak the first channel to avoid later assert
                if (dest[0] < 128) ++dest[0]; else --dest[0];
            }
            // can have at most one other line containing pt
            for (auto it = lines.begin(); it != lines.end(); ++it) {
                if (it->contains(pt)) {
                    // split line
                    auto l0 = hole_line{it->first, pt};
                    auto l1 = hole_line{pt, it->last};
                    const auto idx = long(it - lines.begin());
                    if (0 < l0.size()) {
                        *it = l0;
                        if (0 < l1.size())
                            lines.push_back(l1);
                        resort_idx = std::min(resort_idx, idx);
                    }
                    else if (0 < l1.size()) {
                        *it = l1;
                        resort_idx = std::min(resort_idx, idx);
                    }
                    else { // other line is erased
                        lines.erase(it);
                        if (idx < resort_idx)
                            resort_idx -= 1;
                    }
                    break;
                }
            }
        }
        if (resort_idx < long(lines.size())) // partial resort
            sort(lines.begin() + resort_idx, lines.end(),
                 [](const auto& a, const auto& b) { return b < a; });
    }
}

void raw_image::in_place_fill_bytes(
    const plane& img, std::array<uint8_t,4> hole) {

    const auto BPP = bytes_per_pixel(img);
    if (BPP <= 1)
        return in_place_fill_holes(img, hole.data());

    switch (BPP) {
    case 2:
        if (img.layout == pixel::a16_le)
            throw std::logic_error("for int16 pixels use fill_holes()");
        return fill_bytes(img, std::array<uint8_t,2>{hole[0],hole[1]});
    case 3:
        return fill_bytes(img, std::array<uint8_t,3>{hole[0],hole[1],hole[2]});
    case 4:
        return fill_bytes(img, hole);
    }
    throw std::runtime_error("unsupported bytes per pixel in fill_bytes()");
}
