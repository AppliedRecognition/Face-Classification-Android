#pragma once

#include "coordinates.hpp"
#include <vector>

namespace det {
    namespace retina {
        struct FaceObject {
            coordinate_type tl, br;
            coordinate_type landmark[5];
            float score;

            inline auto w() const { return br.x - tl.x + 1; }
            inline auto h() const { return br.y - tl.y + 1; }
            inline auto area() const { return w() * h(); }

            template <typename I>
            inline void mirror(I width) {
                const auto fw = float(width) - 1;
                const auto x = fw - br.x;
                br.x = fw - tl.x;
                tl.x = x;
                for (auto& p : landmark)
                    p.x = fw - p.x;
                std::swap(landmark[0], landmark[1]); // eyes
                std::swap(landmark[3], landmark[4]); // mouth corners
            }
        };

        inline float
        intersection_area(const FaceObject& a, const FaceObject& b) {
            const auto x0 = std::max(a.tl.x, b.tl.x);
            const auto y0 = std::max(a.tl.y, b.tl.y);
            const auto x1 = std::min(a.br.x, b.br.x);
            const auto y1 = std::min(a.br.y, b.br.y);
            const auto w = x1 - x0 + 1;
            const auto h = y1 - y0 + 1;
            return 0 < w && 0 < h ? w * h : 0;
        }

        inline void nms_sorted_bboxes(
            const std::vector<FaceObject>& faceobjects,
            std::vector<unsigned>& picked, float nms_threshold) {

            picked.clear();

            std::vector<float> areas;
            areas.reserve(faceobjects.size());
            for (const auto& face : faceobjects)
                areas.push_back(face.area());

            for (unsigned i = 0; i < faceobjects.size(); ++i) {
                const auto& a = faceobjects[i];
                bool keep = true;
                for (auto j : picked) {
                    const auto& b = faceobjects[j];
                    // intersection over union
                    const float inter_area = intersection_area(a, b);
                    const float union_area = areas[i] + areas[j] - inter_area;
                    // float IoU = inter_area / union_area
                    if (inter_area > nms_threshold * union_area) {
                        keep = false;
                        break;
                    }
                }
                if (keep)
                    picked.push_back(i);
            }
        }

        struct anchors {
            // each anchor is { xy, wh }
            // xy is top-left (x==y) and wh is width and height
            std::array<std::array<float,2>,2> coords;
            float stride;

            constexpr anchors(float stride, float scale0)
                : coords{}, stride(stride) {
                constexpr float center = 16 * 0.5f;
                const auto h0 = center * scale0;
                coords[0][0] = center - h0;
                coords[0][1] = 2*h0;
                const auto h1 = h0 / 2;
                coords[1][0] = center - h1;
                coords[1][1] = 2*h1;
            }

            // note: scores must start at positive scores
            // this is channel 2 for ncnn and channel 0 for dlib
            void proposals(unsigned w, unsigned h, std::size_t channel_size,
                           const float* score_blob,
                           const float* bbox_blob,
                           const float* landmark_blob,
                           float score_threshold,
                           std::vector<FaceObject>& dest) const {

                for (unsigned q = 0; q < coords.size(); ++q) {
                    const auto& anchor = coords[q];
                    const float wh = anchor[1];
                    const float xy_start = anchor[0] + wh * 0.5f;

                    const auto scores = score_blob + q*channel_size;
                    const auto bbox = bbox_blob + 4*q*channel_size;
                    const auto landmark = landmark_blob + 10*q*channel_size;

                    std::size_t idx = 0;
                    float cy = xy_start;
                    for (unsigned i = 0; i < h; ++i, cy += stride) {
                        float cx = xy_start;
                        for (unsigned j = 0; j < w; ++j, cx += stride, ++idx) {
                            const float score = scores[idx];
                            if (score < score_threshold)
                                continue;

                            // apply center size
                            auto bptr = bbox + idx;
                            const float dx = *bptr;
                            bptr += channel_size;
                            const float dy = *bptr;
                            bptr += channel_size;
                            const float dw = *bptr;
                            bptr += channel_size;
                            const float dh = *bptr;

                            const float pb_cx = cx + wh * dx;
                            const float pb_cy = cy + wh * dy;

                            const float pb_w = wh * std::exp(dw);
                            const float pb_h = wh * std::exp(dh);

                            const float x0 = pb_cx - pb_w * 0.5f;
                            const float y0 = pb_cy - pb_h * 0.5f;
                            const float x1 = pb_cx + pb_w * 0.5f;
                            const float y1 = pb_cy + pb_h * 0.5f;

                            auto& obj = dest.emplace_back();
                            obj.score = score;
                            obj.tl = { x0, y0 };
                            obj.br = { x1, y1 };
                            auto lptr = landmark + idx;
                            for (auto& pt : obj.landmark) {
                                pt.x = cx + (wh + 1) * *lptr;
                                lptr += channel_size;
                                pt.y = cy + (wh + 1) * *lptr;
                                lptr += channel_size;
                            }
                        }
                    }
                }
            }

            void proposals(unsigned w, unsigned h,
                           const float* score_blob,
                           const float* bbox_blob,
                           const float* landmark_blob,
                           float score_threshold,
                           std::vector<FaceObject>& dest) const {
                proposals(w,h,std::size_t(w)*h,
                          score_blob,bbox_blob,landmark_blob,
                          score_threshold,dest);
            }
        };
    }
}
