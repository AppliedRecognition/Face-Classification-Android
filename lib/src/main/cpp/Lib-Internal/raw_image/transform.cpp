
#include "transform.hpp"
#include "reader.hpp"
#include "pixels.hpp"

#include <stdext/rounding.hpp>

#include <applog/core.hpp>

#include <algorithm>
#include <cassert>
#include <cstring>
#include <limits>
#include <vector>
#include <array>

using namespace raw_image;

void raw_image::in_place_flip(plane& img) {
    std::vector<unsigned char> buf;
    const auto n = img.width * bytes_per_pixel(img.layout);
    const auto bpl = long(img.bytes_per_line);
    for (auto l0 = img.data,
             l1 = img.data + img.height * bpl; (l1 -= bpl) > l0; l0 += bpl) {
        // swap line y with line (height-1-y)
        buf.assign(l0,l0+n);
        std::copy_n(l1,n,l0);
        std::copy_n(buf.data(),n,l1);
    }
    img.rotate ^= 6;
}

namespace {
    inline bool is_align_4(const plane& img) {
        return (img.bytes_per_line&3) == 0 &&
            (reinterpret_cast<std::size_t>(img.data)&3u) == 0;
    }

    template <unsigned N, typename ITER>
    inline void swap_N(ITER a, ITER b) {
        for (auto i = N; i > 0; --i, ++a, ++b)
            std::swap(*a, *b);
    }

    template <unsigned N, typename ITER>
    inline void reverse_N(ITER first, ITER last) {
        if (N == 1)
            std::reverse(first, last);
        else
            for ( ; first < last; first += N)
                swap_N<N>(first, last -= N);
    }

    template <unsigned N, typename T>
    void mirror(T* buf, unsigned rows, unsigned cols, unsigned els_per_row) {
        for ( ; rows > 0; --rows, buf += els_per_row)
            reverse_N<N>(buf, buf + N*cols);
    }
}

void raw_image::in_place_mirror(plane& img) {
    switch (bytes_per_pixel(img.layout)) {
    case 1:
        mirror<1>(img.data, img.height, img.width, img.bytes_per_line);
        break;
    case 2:
        mirror<2>(img.data, img.height, img.width, img.bytes_per_line);
        break;
    case 3:
        mirror<3>(img.data, img.height, img.width, img.bytes_per_line);
        break;
    case 4:
        if (!is_align_4(img))
            throw std::invalid_argument("32-bit pixel must be 32-bit aligned");
        mirror<1>(reinterpret_cast<uint32_t*>(img.data),
                  img.height, img.width, img.bytes_per_line/4);
        break;
    default:
        FILE_LOG(logERROR) << "invalid color space: " << int(img.layout);
        throw std::invalid_argument("invalid color space");
    }
    img.rotate ^= 4;
}

namespace {
    template <unsigned N, typename T>
    plane raw_from_buf(T* buf, unsigned rows, unsigned cols,
                                unsigned els_per_row) {
        plane raw;
        raw.height = rows;
        raw.width = cols;
        const auto bpl = els_per_row * sizeof(T);
        raw.bytes_per_line = unsigned(bpl);
        assert(raw.bytes_per_line == bpl);
        raw.data = reinterpret_cast<uint8_t*>(buf);
        switch (N*sizeof(T)) {
        case 1: raw.layout = pixel::gray8;  break;
        case 2: raw.layout = pixel::uv16_jpeg; break;
        case 3: raw.layout = pixel::rgb24;  break;
        case 4: raw.layout = pixel::argb32; break;
        default: assert(!"unknown color space");
        }
        return raw;
    }

    template <unsigned N, typename T>
    void transpose_square(T* buf, unsigned dim, unsigned els_per_row) {
        for ( ; dim > 1; buf += els_per_row + N) {
            auto upper = buf, lower = buf;
            for (auto i = --dim; i > 0; --i)
                swap_N<N>(upper += N, lower += els_per_row);
        }
    }

    // returns new elements_per_row
    template <unsigned N, typename T>
    unsigned swap_rows_cols(T* buf, unsigned rows, unsigned cols,
                            unsigned els_per_row) {

        const auto max_els = rows * els_per_row;

        auto new_els = rows * N;  // rows is new cols
        if (new_els <= els_per_row && cols <= rows) 
            return els_per_row;   // rows are already large enough
        
        if ((new_els&1) && cols*(new_els+1) <= max_els) ++new_els;
        if ((new_els&2) && cols*(new_els+2) <= max_els) new_els += 2;

        const auto dim = std::min(rows,cols);
        constexpr auto BPP = N * sizeof(T);

        if (new_els < els_per_row) {
            // contract rows
            auto dest = buf, src = buf;
            for (auto i = dim; i > 1; --i)
                std::memmove(dest += new_els, src += els_per_row, dim * BPP);
        }
        else {
            // expand rows
            assert(new_els > els_per_row);
            auto dest = buf + new_els*dim, src = buf + els_per_row*dim;
            for (auto i = dim; i > 1; --i)
                std::memmove(dest -= new_els, src -= els_per_row, dim * BPP);
        }

        return new_els;
    }
    
    // returns new elements_per_row
    template <unsigned N, typename T>
    unsigned transpose(T* buf, unsigned rows, unsigned cols,
                       unsigned els_per_row) {

        constexpr auto BPP = N * sizeof(T);
        
        if (rows <= 1)
            return N;  // result is single column with N els_per_row

        if (cols <= 1) {
            if (cols == 1 && N < els_per_row) {
                // compact elements to single row
                auto dest = buf, src = buf;
                for (auto i = rows; i > 1; --i) {
                    dest += N, src += els_per_row;
                    if (N == 1)
                        *dest = *src;
                    else
                        std::memmove(dest, src, BPP);
                }
            }
            return rows*els_per_row;
        }

        // top left square
        const auto dim = std::min(rows,cols);
        transpose_square<N>(buf, dim, els_per_row);

        if (rows < cols) {
            // wide: transpose strip on right
            auto raw = raw_from_buf<N>(buf, rows, cols, els_per_row);
            raw.data += dim * BPP;
            raw.width = cols - rows;
            auto strip = copy_transpose(raw);

            // reconfigure
            els_per_row = swap_rows_cols<N>(buf, rows, cols, els_per_row);
            std::swap(rows,cols);

            // copy to strip on bottom
            auto dest = raw_from_buf<N>(buf, rows, cols, els_per_row);
            dest.data += dim * std::size_t(dest.bytes_per_line);
            dest.height = strip->height;
            copy_pixels(*strip, dest);
        }

        else if (cols < rows) { // tall
            // tall: transpose strip on bottom
            auto raw = raw_from_buf<N>(buf, rows, cols, els_per_row);
            raw.data += dim * std::size_t(raw.bytes_per_line);
            raw.height = rows - cols;
            auto strip = copy_transpose(raw);

            // reconfigure
            els_per_row = swap_rows_cols<N>(buf, rows, cols, els_per_row);
            std::swap(rows,cols);

            // copy to strip on right
            auto dest = raw_from_buf<N>(buf, rows, cols, els_per_row);
            dest.data += dim * BPP;
            dest.width = strip->width;
            copy_pixels(*strip, dest);
        }

        return els_per_row;
    }

    template <unsigned N, typename Iter, typename OutIter>
    inline void copy_N(Iter from, OutIter to) {
        for (auto i = N; i > 0; --i, ++from, ++to)
            *to = *from;
    }

    // transpose by copying
    template <unsigned N, typename T>
    void transpose(T* dest, unsigned dest_rows, unsigned dest_per_row, 
                   const T* src, unsigned src_rows, unsigned src_per_row) {
        for ( ; dest_rows > 0; --dest_rows, dest += dest_per_row, src += N) {
            auto d = dest;
            auto s = src;
            for (auto i = src_rows; i > 0; --i, d += N, s += src_per_row)
                copy_N<N>(s,d);
        }
    }

    unsigned after_transpose(unsigned rot) {
        rot ^= 5 | ((rot<<1)^(rot>>1));
        return rot & 7;
        //static constexpr unsigned r[] = { 5, 6, 7, 4, 3, 0, 1, 2 };
        //return r[rot&7];
    }
}

void raw_image::in_place_transpose(plane& img) {
    switch (bytes_per_pixel(img.layout)) {
    case 1:
        img.bytes_per_line = 
            transpose<1>(img.data,img.height,img.width,img.bytes_per_line);
        break;
    case 2:
        img.bytes_per_line =
            transpose<2>(img.data,img.height,img.width,img.bytes_per_line);
        break;
    case 3:
        img.bytes_per_line = 
            transpose<3>(img.data,img.height,img.width,img.bytes_per_line);
        break;
    case 4:
        if (!is_align_4(img))
            throw std::invalid_argument("32-bit pixel must be 32-bit aligned");
        img.bytes_per_line = 
            transpose<1>(reinterpret_cast<uint32_t*>(img.data),
                         img.height,img.width,img.bytes_per_line/4) * 4;
        break;
    default:
        FILE_LOG(logERROR) << "invalid color space: " << int(img.layout);
        throw std::invalid_argument("invalid color space");
    }
    std::swap(img.width,img.height);
    img.rotate = after_transpose(img.rotate);
}
    
void raw_image::in_place_rotate(plane& img, unsigned rotate) {
    if (rotate&1) {
        in_place_transpose(img);
        rotate = after_transpose(rotate);
    }
    if (rotate&2) {
        in_place_flip(img);
        rotate ^= 6;
    }
    if (rotate&4)
        in_place_mirror(img);
}

std::unique_ptr<std::array<plane,2> >
raw_image::create_nv21(plane image) {

    // ensure width and height are multiples of 8
    while (image.width&7) --image.width;
    while (image.height&7) --image.height;
    if (image.width <= 0 || image.height <= 0)
        throw std::invalid_argument("empty image");

    using ARR = std::array<plane,2>;
    static_assert(sizeof(ARR) <= 2*plane_struct_padded_size,
                  "array is larger than excepted");
    const auto buf_size =
        2*plane_struct_padded_size + 3*image.width*image.height;
    const auto buf = operator new(buf_size);
    std::fill_n(static_cast<unsigned char*>(buf), 2*plane_struct_padded_size, 0);

    std::unique_ptr<ARR> r(static_cast<ARR*>(buf));
    r->front().rotate = image.rotate;
    r->front().scale = image.scale;
    r->front().width = image.width;
    r->front().height = image.height;
    r->front().data =
        static_cast<unsigned char*>(buf) + 2*plane_struct_padded_size;

    // convert source image to YUV
    r->front().bytes_per_line = 3*image.width;
    r->front().layout = pixel::yuv24_nv21;
    copy_pixels(image, r->front());

    // scale VU plane
    const auto vu = copy_resize(r->front(), image.width/2, image.height/2);

    // extract Y plane
    if (auto p = convert(r->front(), pixel::y8_nv21))
        throw std::runtime_error("unexpected image allocation");
    assert(r->front().layout == pixel::y8_nv21);

    {
        // pack Y plane
        auto dest = r->front().data, src = r->front().data;
        for (auto j = r->front().height; j > 0; --j,
                 dest += r->front().width, src += r->front().bytes_per_line)
            memmove(dest, src, r->front().width);
    }
    r->front().bytes_per_line = r->front().width;

    // create VU image
    r->back().rotate = r->front().rotate;
    r->back().scale = r->front().scale + 1;
    r->back().width = r->front().width / 2;
    r->back().height = r->front().height / 2;
    r->back().bytes_per_line = 2 * r->back().width;
    r->back().data = r->front().data + r->front().width * r->front().height;
    r->back().layout = pixel::vu16_nv21;
    copy_pixels(vu, r->back());

    return r;
}

plane_ptr
raw_image::copy_resize(const multi_plane_arg& image,
                       unsigned width, unsigned height,
                       pixel_layout destcs,
                       interpolation_type it) {

    const auto is_downscale =
        [=](const auto& plane) {
            return width <= plane.width && height <= plane.height;
        };

    std::unique_ptr<reader> r;

    if (image.size() == 1 &&
        bytes_per_pixel(image.front().layout) <= bytes_per_pixel(destcs) &&
        (bytes_per_pixel(image.front().layout) < bytes_per_pixel(destcs) ||
         is_downscale(image.front()))) {
        // scale first before changing pixel layout
        r = reader::construct(image);
    }
    else {
        // convert to destination pixel layout first
        r = reader::construct(image, destcs);
    }

    // scaling
    switch (it) {
    case inter::nearest:
        r = scale_nearest(move(r), width, height);
        break;
    case inter::area:
        r = scale_area(move(r), width, height);
        break;
    case inter::bilinear:
    default:
        r = scale_interpolate(move(r), width, height);
    }

    // change pixel layout if necessary
    if (r->layout() != destcs)
        r = convert(move(r), destcs);

    auto dest = create(r->width(), r->height(), r->layout());
    r->copy_to(*dest, dest->bytes_per_line);

    dest->rotate = image.front().rotate;
    dest->scale = image.front().scale;

    return dest;
}

static auto extract_reader(
    const plane& image,
    float cx, float cy, float w, float h, float angle,
    unsigned dest_width, unsigned dest_height) {

    if (image.scale) {
        const auto z = std::ldexp(1.0f,-image.scale);
        cx *= z;
        cy *= z;
        w *= z;
        h *= z;
    }

    auto reader = rotate_gradians(
        image, stdx::round_from(angle*10/9), cx, cy,
        stdx::round_from(2*w), stdx::round_from(2*h));

    // scale to final size
    // only use interpolate if scaling up in either dimension
    if (reader->layout() == pixel::a16_le ||
        reader->layout() == pixel::f32) // cannot average or interpolate
        reader = scale_nearest(move(reader), dest_width, dest_height);
    else if (reader->width() < dest_width || reader->height() < dest_height)
        reader = scale_interpolate(move(reader), dest_width, dest_height);
    else
        reader = scale_area(move(reader), dest_width, dest_height);

    return reader;
}

plane_ptr
raw_image::extract_region(const multi_plane_arg& multiplane,
                          float cx, float cy, float w, float h, float angle,
                          unsigned dest_width, unsigned dest_height,
                          pixel_layout dest_layout) {
    if (multiplane.size() <= 1) {
        if (multiplane.empty())
            throw std::invalid_argument("image has no planes");
        auto reader = extract_reader(
            multiplane.front(), cx, cy, w, h, angle, dest_width, dest_height);
        reader = convert(move(reader), dest_layout);
        auto dest = create(reader->width(), reader->height(), reader->layout());
        reader->copy_to(*dest);
        return dest;
    }

    // extract region from each plane separately and
    // then convert multiplane image to dest_layout
    std::vector<plane_ptr> dest_ptr;
    dest_ptr.reserve(multiplane.size());
    std::vector<plane> dest_raw;
    dest_raw.reserve(multiplane.size());

    auto& front = multiplane.front();
    for (auto& raw : multiplane) {
        auto copy = raw;
        copy.rotate = front.rotate;
        copy.scale = front.scale;
        if (copy.width != front.width || copy.height != front.height) {
            if (copy.width*2  == front.width &&
                copy.height*2 == front.height)
                ++copy.scale;
            else
                throw std::invalid_argument("image plane dimension mismatch");
        }
        auto reader = extract_reader(
            copy, cx, cy, w, h, angle, dest_width, dest_height);
        auto dest = create(reader->width(), reader->height(), reader->layout());
        reader->copy_to(*dest);
        dest_raw.emplace_back(*dest);
        dest_ptr.emplace_back(move(dest));
    }

    return copy({dest_raw.data(), dest_raw.size()}, dest_layout);
}

plane_ptr raw_image::matrix_inverse(const plane& mat) {
    if (empty(mat) || mat.width != mat.height || mat.layout != pixel::f32)
        throw std::invalid_argument("invert requires a square matrix");

    const auto dim = mat.width;
    const pixels<float> matG(mat);

    auto matLU_ptr = create(dim, dim, pixel::f32);
    pixels<float> matLU(matLU_ptr);

    // https://github.com/FerryYoungFan/NaiveMatrixLib

    // Step 1: row permutation (swap diagonal zeros)
    std::vector<unsigned> permuteLU; // Permute vector
    permuteLU.reserve(dim);
    for (unsigned i = 0; i < dim; ++i)
        permuteLU.push_back(i);

    const auto usePermute = true;
    if (usePermute) { // sort rows by pivot element
        for (unsigned j = 0; j < dim; ++j) {
            float maxv = 0;
            for (unsigned i = j; i < dim; ++i) {
                const float currentv = std::abs(matG[permuteLU[i]][j]);
                if (maxv < currentv) { // Swap rows
                    maxv = currentv;
                    std::swap(permuteLU[i], permuteLU[j]);
                }
            }
        }
        // make a permuted matrix with new row order
        auto it = permuteLU.begin();
        for (auto&& line : matLU) {
            auto src = matG[*it++];
            assert(line.size() == src.size());
            std::copy_n(src.data(), src.size(), line.data());
        }
    }
    else
        copy_pixels(mat, matLU_ptr); // Simply duplicate matrix

    // Step 2: LU decomposition (save both L & U in matLU)
    if (std::abs(matLU[0][0]) < std::numeric_limits<float>::min())
        throw std::runtime_error("matrix is singular");

    for (unsigned i = 1; i < dim; ++i)
        matLU[i][0] /= matLU[0][0]; // first column of L matrix

    for (unsigned i = 1; i < dim; ++i) {
        for (auto j = i; j < dim; ++j)
            for (unsigned k = 0; k < i; ++k)
                matLU[i][j] -= matLU[i][k] * matLU[k][j]; // U matrix

        if (std::abs(matLU[i][i]) < std::numeric_limits<float>::min())
            throw std::runtime_error("matrix is singular");

        for (auto k = i + 1; k < dim; ++k) {
            for (unsigned j = 0; j < i; ++j)
                matLU[k][i] -= matLU[k][j] * matLU[j][i]; // L matrix
            matLU[k][i] /= matLU[i][i];
        }
    }

    // Step 3: L & U inversion (save both L^-1 & U^-1 in matLU_inv)
    auto matLU_inv_ptr = create(dim, dim, pixel::f32);
    pixels<float> matLU_inv(matLU_inv_ptr);
    for (auto&& line : matLU_inv)
        for (auto& z : line)
            z = 0;

    // matL inverse & matU inverse
    for (unsigned i = 0; i < dim; ++i) {
        // L matrix inverse, omit diagonal ones
        matLU_inv[i][i] = 1;
        for (auto k = i + 1; k < dim; ++k) {
            for (auto j = i; j <= k - 1; ++j)
                matLU_inv[k][i] -= matLU[k][j] * matLU_inv[j][i];
        }
        // U matrix inverse
        matLU_inv[i][i] = 1.0f / matLU[i][i];
        for (auto k = i; k > 0; --k) {
            for (auto j = k; j <= i; ++j)
                matLU_inv[k - 1][i] -= matLU[k - 1][j] * matLU_inv[j][i];
            matLU_inv[k - 1][i] /= matLU[k - 1][k - 1];
        }
    }

    // Step 4: Calculate G^-1 = U^-1 * L^-1
    // lower part product
    for (unsigned i = 1; i < dim; ++i) {
        for (unsigned j = 0; j < i; ++j) {
            const auto jp = permuteLU[j]; // permute column back
            matLU[i][jp] = 0;
            for (auto k = i; k < dim; ++k)
                matLU[i][jp] += matLU_inv[i][k] * matLU_inv[k][j];
        }
    }

    // upper part product
    for (unsigned i = 0; i < dim; ++i) {
        for (auto j = i; j < dim; ++j) {
            const auto jp = permuteLU[j]; // permute column back
            matLU[i][jp] = matLU_inv[i][j];
            for (auto k = j + 1; k < dim; ++k)
                matLU[i][jp] += matLU_inv[i][k] * matLU_inv[k][j];
        }
    }

    return matLU_ptr;
}
