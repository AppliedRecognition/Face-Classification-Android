
#include "image_quality.hpp"

#include <cmath>
#include <type_traits>
#include <vector>

raw_image::bcsg_result raw_image::bcsg(reader& src) {
    if (bytes_per_pixel(src.layout()) != 1)
        throw std::invalid_argument(
            "can only compute bcs on single channel images");
    const auto width = src.width();
    const auto total_pixels = double(width) * double(src.lines_remaining());
    if (total_pixels <= 0)
        return {0,0,0,0,0}; // nothing to do

    const auto whalf = width / 2;
    const auto horz_pixels  = double(whalf) * double(src.lines_remaining());
    const auto vert_pixels  = double(width) * double(src.lines_remaining()/2);

    // sums
    uint64_t ps = 0, ps2 = 0;
    int64_t psh = 0, psv = 0;
    int64_t ls = 0, ls2 = 0;
    uint64_t laplace_px = 0;

    // previous 2 lines for laplacian
    std::vector<uint8_t> line0, line1;

    // keep track of vertical location for psv
    int vidx = int(src.lines_remaining()) - 1;

    do {
        const auto* line2 = src.get_line();
        uint64_t line_sum = 0;

        auto px = line2;
        for (auto end = line2 + whalf; px != end; ++px)
            ps2 += *px*unsigned(*px), line_sum += *px, psh += *px;
        if (width&1) {
            // if width is odd, don't include center column in psh
            ps2 += *px*unsigned(*px), line_sum += *px;
            ++px;
        }
        for (auto end = line2 + width; px != end; ++px)
            ps2 += *px*unsigned(*px), line_sum += *px, psh -= *px;
        ps += line_sum;

        if (vidx > 0) // top half
            psv += int64_t(line_sum);
        else if (vidx < 0) // bottom half
            psv -= int64_t(line_sum);
        // else (vidx == 0) don't include center row in psv (height is odd)
        vidx -= 2; // 2 at a time so we cross zero half way through image

        if (2 < width) {
            if (!line0.empty()) {
                for (unsigned i = 1; i < width-1; ++i) {
                    const auto z =
                        line0[i] + line2[i] +
                        line1[i-1] + line1[i+1]
                        - 4*line1[i];
                    static_assert(std::is_same_v<decltype(z), const int>);
                    ls += z;
                    ls2 += z*z;
                }
                laplace_px += width - 2;
            }
            line0.swap(line1);
            line1.assign(line2, line2 + width);
        }

    } while (src.next_line());

    raw_image::bcsg_result r;

    r.horz = float(double(psh) / horz_pixels);
    r.vert = float(double(psv) / vert_pixels);

    const auto mean = double(ps) / total_pixels;
    const auto var = double(ps2) / total_pixels - mean*mean;
    r.brightness = float(mean);
    r.contrast = 0 < var ? float(std::sqrt(var)) : 0.0f;

    if (laplace_px <= 0)
        r.sharpness = 0;
    else {
        const auto N = double(laplace_px);
        const auto lm = double(ls) / N;
        const auto lv = double(ls2) / N - lm*lm;
        const auto dev = 0 < lv ? float(std::sqrt(lv)) : 0.0f;
        r.sharpness = 100 * dev / std::max(r.contrast,1.0f);
    }

    return r;
}
