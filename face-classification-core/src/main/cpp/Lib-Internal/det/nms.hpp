#pragma once

#include "types.hpp"

#include <algorithm>

namespace det {
    namespace nms {

        /// face bounding box with confidence score
        struct bbox {
            float score;
            coordinate_type tl, br;

            inline auto w() const { return br.x - tl.x + 1; }
            inline auto h() const { return br.y - tl.y + 1; }
            inline auto area() const { return w() * h(); }

            template <typename I>
            inline void mirror(I width) {
                const auto fw = float(width) - 1;
                const auto x = fw - br.x;
                br.x = fw - tl.x;
                tl.x = x;
            }

            inline void scale(float z) {
                score *= z;
                tl *= z;
                br *= z;
            }

            inline void add(const bbox& other) {
                score += other.score;
                tl += other.tl;
                br += other.br;
            }

            friend float intersection_area(const bbox& a, const bbox& b) {
                const auto x0 = std::max(a.tl.x, b.tl.x);
                const auto y0 = std::max(a.tl.y, b.tl.y);
                const auto x1 = std::min(a.br.x, b.br.x);
                const auto y1 = std::min(a.br.y, b.br.y);
                const auto w = x1 - x0 + 1;
                const auto h = y1 - y0 + 1;
                return 0 < w && 0 < h ? w * h : 0;
            }
        };

        /// bounding box with additional landmarks
        template <std::size_t N>
        struct bbox_landmarks : bbox {
            coordinate_type landmarks[N];

            template <typename I>
            inline void mirror(I width) {
                bbox::mirror(width);
                const auto fw = float(width) - 1;
                for (auto& p : landmarks)
                    p.x = fw - p.x;
            }

            void scale(float z) {
                bbox::scale(z);
                for (auto& pt : landmarks)
                    pt *= z;
            }

            void add(const bbox_landmarks& other) {
                bbox::add(other);
                for (std::size_t i = 0; i < N; ++i)
                    landmarks[i] += other.landmarks[i];
            }
        };

        /// v7 retina detector specific landmarks
        struct retina_landmarks : bbox_landmarks<5> {
            template <typename I>
            inline void mirror(I width) {
                bbox_landmarks<5>::mirror(width);
                std::swap(landmarks[0], landmarks[1]); // eyes
                std::swap(landmarks[3], landmarks[4]); // mouth corners
            }
        };

        /// v8 blaze detector specific landmarks
        struct blaze_landmarks : bbox_landmarks<6> {
            template <typename I>
            inline void mirror(I width) {
                bbox_landmarks<6>::mirror(width);
                std::swap(landmarks[0], landmarks[1]); // eyes
                std::swap(landmarks[4], landmarks[5]); // tragion
            }
        };

        /** \brief Sort face candidates by decreasing score.
         *
         * This is a require prereq for the nms methods below.
         */
        template <typename T>
        inline void sort_decreasing_score(std::vector<T>& candidates) {
            sort(candidates.begin(), candidates.end(),
                 [](const auto& a, const auto& b) {
                     return a.score > b.score;
                 });
        }

        /** \brief Select face with maximum score.
         *
         * \pre input candidates sorted by decreasing score
         *
         * \returns vector of those selected
         */
        template <typename T>
        auto max_from_sorted(std::vector<T> candidates, float iou_threshold) {
            for (auto it = candidates.begin(); it != candidates.end(); ++it) {
                const auto it_area = it->area();
                for (auto jt = next(it); jt != candidates.end(); ) {
                    const float inter_area = intersection_area(*it, *jt);
                    const float union_area = it_area + jt->area() - inter_area;
                    // float IoU = inter_area / union_area
                    if (inter_area > iou_threshold * union_area)
                        jt = candidates.erase(jt);
                    else ++jt;
                }
            }
            return candidates;
        }

        /** \brief Group and blend face candidates weighted by score.
         *
         * \pre input candidates sorted by decreasing score
         */
        template <typename T>
        auto blend_from_sorted(std::vector<T> candidates, float iou_threshold) {
            for (auto it = candidates.begin(); it != candidates.end(); ++it) {
                auto last = next(it);
                const auto it_area = it->area();
                for (auto jt = next(it); jt != candidates.end(); ++jt) {
                    const float inter_area = intersection_area(*it, *jt);
                    const float union_area = it_area + jt->area() - inter_area;
                    // float IoU = inter_area / union_area
                    if (inter_area > iou_threshold * union_area)
                        rotate(last++, jt, next(jt));
                }
                if (last != next(it)) {
                    float total = 0;
                    for (auto jt = it; jt != last; ++jt)
                        total += std::exp(jt->score);
                    it->scale(std::exp(it->score) / total);
                    for (auto jt = next(it); jt != last; ++jt) {
                        jt->scale(std::exp(jt->score) / total);
                        it->add(*jt);
                    }
                    candidates.erase(next(it), last);
                }
            }
            return candidates;
        }
    }
}
