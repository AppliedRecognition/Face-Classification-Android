
#include "pointcloud.hpp"
#include "fillholes.hpp"

#include <raw_image/core.hpp>
#include <raw_image/pixels.hpp>

#include <algorithm>
#include <cassert>
#include <stdexcept>

using namespace raw_image;

void rgbpoint64::set_xyz(float x, float y, float z) {
    if (!(-512 < x && x < 512 &&
          -512 < y && y < 512 &&
          0 <= z && z < 8192))
        throw std::invalid_argument("X, Y or Z out of range");

    auto xi = stdx::round_to<int>(x*32)*2;
    if (xi < -32768) xi = -32768;
    else if (32766 < xi) xi = 32766;

    auto yi = stdx::round_to<int>(y*32)*2;
    if (yi < -32768) yi = -32768;
    else if (32766 < yi) yi = 32766;

    auto zi = stdx::round_to<unsigned>(z);
    if (1023 < zi) zi = 1023;

    m_x = int16_t(xi + int(zi&1));
    m_y = int16_t(yi + int((zi>>1)&1));
    m_z = uint8_t(zi>>2);
}

bool raw_image::reduce_cloud(std::vector<rgbpoint64>& vec, float thres) {
    if (!(0 < thres && thres <= 10))
        throw std::invalid_argument("threshold in mm out of range (0,10]");

    // sort by x, then y, then z
    std::sort(vec.begin(), vec.end(),
              [](const auto& a, const auto& b) {
                  return a.m_x <= b.m_x &&
                      (a.m_x < b.m_x ||
                       (a.m_y <= b.m_y &&
                        (a.m_y < b.m_y || a.m_z < b.m_z)));
              });

    // compare all to all
    const auto thres64 = stdx::round_to<int>(64*thres);
    std::vector<std::tuple<float,unsigned,unsigned> > pairs;
    for (unsigned i = 0; i < vec.size(); ++i) {
        const auto& pi = vec[i];
        for (unsigned j = i+1; j < vec.size(); ++j) {
            const auto& pj = vec[j];
            if (thres64 < pj.m_x - pi.m_x) break;
            const auto dsqr = length_squared(pi, pj);
            if (dsqr < thres*thres)
                pairs.emplace_back(std::sqrt(dsqr), i, j);
        }
    }

    if (pairs.empty())
        return false; // nothing to do

    // pair is { head, next } pointers for linked list
    std::vector<std::pair<unsigned,unsigned> > cluster_map;
    cluster_map.reserve(vec.size());
    for (unsigned i = 0; i < vec.size(); ++i)
        cluster_map.emplace_back(i,i);
    auto cluster_count = vec.size();
    assert(cluster_map.size() == cluster_count);

    const auto is_cluster = [&](auto i) {
        const auto& pr = cluster_map[i];
        return (pr.first != i || pr.second != i) ? 1 : 0;
    };
    const auto node_can_join_cluster = [&](auto i, auto j) {
        // can node i join cluster j
        const auto& pi = vec[i];
        for (auto k = cluster_map[j].first; ; ) {
            const auto dsqr = length_squared(pi, vec[k]);
            if (thres*thres <= dsqr)
                return false;
            if (k == cluster_map[k].second)
                return true;
            k = cluster_map[k].second;
        }
    };
    const auto clusters_can_merge = [&](auto i, auto j) {
        if (cluster_map[i].first == cluster_map[j].first)
            return false; // already same cluster
        for (auto k = cluster_map[i].first; ; ) {
            if (!node_can_join_cluster(k, j))
                return false;
            if (k == cluster_map[k].second)
                return true;
            k = cluster_map[k].second;
        }
    };

    // find kliques to coalesce into clusters
    sort(pairs.begin(), pairs.end());
    for (auto& t : pairs) {
        auto i = std::get<1>(t), j = std::get<2>(t);
        switch (is_cluster(j) * 2 + is_cluster(i)) {
        case 0: // neither is a cluster -- make them cluster of two
            assert(cluster_map[i].first == i);
            cluster_map[i].second = j;
            cluster_map[j].first = i;
            assert(cluster_map[j].second == j);
            --cluster_count;
            break;

        case 1: // i is cluster, j is not
            if (node_can_join_cluster(j,i)) {
                i = cluster_map[i].first;
                cluster_map[j] = cluster_map[i];
                cluster_map[i].second = j;
                --cluster_count;
            }
            break;

        case 2: // j is cluster, i is not
            if (node_can_join_cluster(i,j)) {
                j = cluster_map[j].first;
                cluster_map[i] = cluster_map[j];
                cluster_map[j].second = i;
                --cluster_count;
            }
            break;

        case 3: // both are clusters
            if (clusters_can_merge(i,j)) {
                // move all i elements to cluster j
                j = cluster_map[j].first;
                for (auto k = cluster_map[i].first; ; ) {
                    const auto next = cluster_map[k].second;
                    cluster_map[k] = cluster_map[j];
                    cluster_map[j].second = k;
                    if (k == next) break;
                    k = next;
                }
                --cluster_count;
            }
            break;

        default: assert(!"machine failure");
        }
    }
    assert(cluster_count <= vec.size());

    std::vector<rgbpoint64> new_vec;
    new_vec.reserve(cluster_count);
    for (unsigned i = 0; i < cluster_map.size(); ++i) {
        if (cluster_map[i].first != i)
            continue; // not the head of a cluster
        if (cluster_map[i].second == i) // single element
            new_vec.emplace_back(vec[i]);
        else { // multi-element
            unsigned n = 0;
            unsigned r = 0, g = 0, b = 0;
            float x = 0, y = 0, z = 0;
            for (auto k = i; ; ) { // i is head of cluster
                auto& pk = vec[k];
                r += pk.r;
                g += pk.g;
                b += pk.b;
                x += pk.x();
                y += pk.y();
                z += pk.z();
                ++n;
                if (k == cluster_map[k].second)
                    break;
                k = cluster_map[k].second;
            }
            auto& pt = new_vec.emplace_back();
            pt.set_rgb(r/n, g/n, b/n);
            const auto fn = float(n);
            pt.set_xyz(x/fn, y/fn, z/fn);
        }
    }
    assert(new_vec.size() == cluster_count);
    vec = move(new_vec);
    return true; // something was done
}

std::vector<rgbpoint128>
raw_image::to_rgbpoint128(const std::vector<rgbpoint64>& vec) {
    return std::vector<rgbpoint128>(vec.begin(), vec.end());
}

std::vector<rgbpoint128>
raw_image::to_rgbpoint128(const stdx::binary& bin) {
    if (bin.size() % 8 != 0)
        throw std::invalid_argument("rgbpoint64 binary has incorrect size");
    const auto ptr = static_cast<const rgbpoint64*>(bin.data());
    const auto len = bin.size() / sizeof(rgbpoint64);
    return std::vector<rgbpoint128>(ptr, ptr + len);
}

void raw_image::fill_rgbd(const std::vector<rgbpoint128>& cloud,
                          const plane& img,
                          float xy_scale, float z_nearest) {

    const auto w = int(img.width);
    const auto hw = float(w) / 2;
    const auto h = int(img.height);
    const auto hh = float(h) / 2;

    const auto total_px = std::size_t(w) * std::size_t(h);
    std::vector<rgbpoint128 const*> pt_map(total_px, nullptr);
    std::vector<unsigned> rank_map(total_px, 0);

    for (auto& pt : cloud) {
        // ignore points closer than z_nearest to camera
        if (pt.z < z_nearest) continue;

        const auto xf = hw + pt.x / xy_scale;
        const auto x0 = int(std::floor(xf));
        if (x0 < 0 || w <= x0 + 1) continue;

        const auto yf = hh + pt.y / xy_scale;
        const auto y0 = int(std::floor(yf));
        if (y0 < 0 || h <= y0 + 1) continue;

        const auto x1 = x0 + 1;
        const auto y1 = y0 + 1;

        const auto dist = [=](auto x, auto y) {
            auto dx = float(x) - xf;
            auto dy = float(y) - yf;
            return dx*dx + dy*dy;
        };

        std::pair<float,int> posn[4] = {
            std::pair(dist(x0,y0),x0+y0*w),
            std::pair(dist(x1,y0),x1+y0*w),
            std::pair(dist(x0,y1),x0+y1*w),
            std::pair(dist(x1,y1),x1+y1*w),
        };
        std::sort(std::begin(posn), std::end(posn));

        unsigned idx = 0;
        for (auto& t : posn) {
            auto& rank = rank_map[unsigned(t.second)];
            auto& ptr = pt_map[unsigned(t.second)];
            if (ptr == nullptr ||
                (idx <= rank &&
                 (idx < rank || ptr->z > pt.z))) {
                ptr = &pt;
                rank = idx;
            }
            ++idx;
        }
    }

    auto pt_iter = pt_map.begin();
    switch (img.layout) {
    case pixel::rgb24:
        for (auto&& line : pixels_bpp<3>(img))
            for (auto& px : line) {
                if (auto ptr = *pt_iter)
                    px = { ptr->r, ptr->g, ptr->b };
                ++pt_iter;
            }
        break;

    case pixel::rgba32:
        z_nearest += 255;
        for (auto&& line : pixels_bpp<4>(img))
            for (auto& px : line) {
                if (auto ptr = *pt_iter) {
                    px = {
                        ptr->r, ptr->g, ptr->b,
                        stdx::round_to<uint8_t>(z_nearest - ptr->z)
                    };
                }
                ++pt_iter;
            }
        break;

    default:
        throw std::invalid_argument("unsupported pixel layout");
    }
}

static constexpr auto hole32 = uint32_t(0xf5f5f5f5u);
static constexpr auto hole8 = std::array<uint8_t,4> { 0xf5, 0xf5, 0xf5, 0xf5 };

float raw_image::z_median(const std::vector<rgbpoint128>& cloud,
                          unsigned width, unsigned height, float xy_scale) {
    // compute median z for depth channel
    std::vector<float> z_vals;
    const auto radius = float(std::min(width,height))*xy_scale/2;
    for (auto& pt : cloud) {
        auto d = pt.x*pt.x + pt.y*pt.y;
        if (d < radius*radius)
            z_vals.emplace_back(pt.z);
    }
    if (z_vals.empty())
        throw std::runtime_error("face has no pixels");
    sort(z_vals.begin(), z_vals.end());
    return z_vals[z_vals.size()/2];
}

std::pair<plane_ptr, float>
raw_image::make_rgbd(const std::vector<rgbpoint128>& cloud,
                     unsigned width, unsigned height,
                     float xy_scale, unsigned median_target) {
    if (!(0 < median_target && median_target < 255))
        throw std::invalid_argument("median_target must be between 0 and 255");

    auto rgbd = create(width+2, height+2, raw_image::pixel::rgba32);
    auto roi = crop(rgbd,1,1,width,height);

    {   // fill border with zeros and interior with hole bytes
        assert(rgbd->bytes_per_line % 4 == 0);
        const auto u32pl = rgbd->bytes_per_line / 4;
        assert(width + 2 <= u32pl);
        auto ptr32 = reinterpret_cast<uint32_t*>(rgbd->data);
        ptr32 = std::fill_n(ptr32, u32pl, uint32_t(0));
        *ptr32++ = 0;
        for (auto n = height; 0 < n; --n) {
            ptr32 = std::fill_n(ptr32, width, hole32);
            ptr32 = std::fill_n(ptr32, u32pl - width, uint32_t(0));
        }
        ptr32 = std::fill_n(--ptr32, u32pl, uint32_t(0));
        assert(static_cast<void*>(ptr32) == rgbd->data + rgbd->height*rgbd->bytes_per_line);
    }

    const auto z_median = raw_image::z_median(cloud, width, height,  xy_scale);
    fill_rgbd(cloud, roi, xy_scale, z_median - float(255-median_target));

    in_place_fill_bytes(*rgbd, hole8);

    *rgbd = roi;

    return { move(rgbd), z_median };
}

std::pair<std::vector<point3f_rgbf>, std::vector<index_list> >
raw_image::render_rgbd(const plane& rgbd, float xy_scale) {
    if (empty(rgbd) || rgbd.layout != pixel::rgba32)
        throw std::invalid_argument("expected rgba32 image");
    if (!(0 < xy_scale && xy_scale <= 16))
        throw std::invalid_argument("invalid xy_scale");

    const auto x_ofs = xy_scale * float(rgbd.width) / 2;
    const auto y_ofs = xy_scale * float(rgbd.height) / 2;

    std::vector<raw_image::point3f_rgbf> vertices;
    vertices.reserve(rgbd.width*rgbd.height);
    auto pt = raw_image::point3f { 0, y_ofs, 0 };
    for (auto&& line : raw_image::pixels_bpp<4>(rgbd)) {
        pt.x = -x_ofs;
        for (auto& px : line) {
            pt.z = px[3];
            vertices.push_back({
                    pt,
                    raw_image::rgbf{px[0],px[1],px[2],0}
                });
            pt.x += xy_scale;
        }
        pt.y -= xy_scale;
    }

    std::vector<raw_image::index_list> faces;
    faces.reserve((rgbd.width-1)*(rgbd.height-1));
    for (unsigned j = 1; j < rgbd.height; ++j) {
        const auto j0 = (j-1) * rgbd.width;
        const auto j1 = j * rgbd.width;
        for (unsigned i = 1; i < rgbd.width; ++i)
            faces.emplace_back(
                j0 + i - 1,
                j0 + i,
                j1 + i,
                j1 + i - 1);
    }

    return std::pair(vertices,faces);
}
