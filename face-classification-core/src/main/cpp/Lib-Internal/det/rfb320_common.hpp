#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace det {
    namespace rfb320 {
        /** \brief Face bounding box and score.
         */
        struct bbox {
            float x1 = 0;
            float y1 = 0;
            float x2 = 0;
            float y2 = 0;
            float score = 0;

            template <typename I>
            inline void mirror(I width) {
                const auto fw = float(width) - 1;
                const auto x = fw - x2;
                x2 = fw - x1;
                x1 = x;
            }
        };

        static constexpr auto hard_nms = 1;
        static constexpr auto blending_nms = 2;

        /** \brief NMS proposed in the paper blaze face.
         *
         * Aims to minimize the temporal jitter.
         */
        inline auto nms(std::vector<bbox>& input, float iou_threshold,
                        int type = blending_nms) {

            std::sort(input.begin(), input.end(),
                      [](const auto& a, const auto& b) {
                          return a.score > b.score;
                      });

            std::vector<int> merged(input.size(), 0);
            std::vector<bbox> buf;
            std::vector<bbox> output;
            output.reserve(input.size());

            for (unsigned i = 0; i < input.size(); i++) {
                if (merged[i])
                    continue;
                auto& ini = input[i];

                buf.clear();
                buf.push_back(ini);
                // merged[i] = 1;

                const auto h0 = ini.y2 - ini.y1 + 1;
                const auto w0 = ini.x2 - ini.x1 + 1;
                const auto area0 = h0 * w0;

                for (unsigned j = i + 1; j < input.size(); j++) {
                    if (merged[j])
                        continue;
                    auto& inj = input[j];

                    const auto inner_x0 = std::max(ini.x1, inj.x1);
                    const auto inner_y0 = std::max(ini.y1, inj.y1);
                    const auto inner_x1 = std::min(ini.x2, inj.x2);
                    const auto inner_y1 = std::min(ini.y2, inj.y2);

                    const auto inner_h = inner_y1 - inner_y0 + 1;
                    const auto inner_w = inner_x1 - inner_x0 + 1;

                    if (inner_h <= 0 || inner_w <= 0)
                        continue;

                    const auto inner_area = inner_h * inner_w;

                    const auto h1 = inj.y2 - inj.y1 + 1;
                    const auto w1 = inj.x2 - inj.x1 + 1;
                    const auto area1 = h1 * w1;

                    const auto score =
                        inner_area / (area0 + area1 - inner_area);
                    if (score > iou_threshold) {
                        buf.push_back(inj);
                        merged[j] = 1;
                    }
                }

                switch (type) {
                case hard_nms:
                    output.push_back(buf.front());
                    break;

                case blending_nms: {
                    float total = 0;
                    for (const auto& face : buf)
                        total += std::exp(face.score);
                    auto& rects = output.emplace_back();
                    for (const auto& face : buf) {
                        const float rate = std::exp(face.score) / total;
                        rects.x1 += face.x1 * rate;
                        rects.y1 += face.y1 * rate;
                        rects.x2 += face.x2 * rate;
                        rects.y2 += face.y2 * rate;
                        rects.score += face.score * rate;
                    }
                    break;
                }

                default:
                    throw std::invalid_argument("invalid nms type");
                }
            }
            return output;
        }

        /** \brief Priors are used to compute bounding box coordinates.
         */
        class priors {
            std::vector<std::array<float,4> > parr;

            template <typename T>
            static constexpr auto clip01(T x) {
                return x < 0 ? 0 : (x > 1 ? 1 : x);
            }

        public:
            const unsigned width;
            const unsigned height;

            /** \brief Construct for specified input image dimensions.
             */
            priors(unsigned width, unsigned height)
                : width(width), height(height) {
                if (width <= 0 || height <= 0)
                    throw std::invalid_argument(
                        "width and height must be positive");

                static constexpr auto K = 4;
                static const float strides[K] = { 8.0f, 16.0f, 32.0f, 64.0f };
                static const std::vector<float> min_boxes[K] = {
                    std::vector<float>{10.0f,  16.0f,  24.0f},
                    std::vector<float>{32.0f,  48.0f},
                    std::vector<float>{64.0f,  96.0f},
                    std::vector<float>{128.0f, 192.0f, 256.0f}
                };

                // generate prior anchors
                for (unsigned i = 0; i < K; ++i) {
                    const auto scale_w = float(width) / strides[i];
                    const auto scale_h = float(height) / strides[i];
                    for (float y = 0; y < scale_h; y += 1) {
                        const auto cy = (y + 0.5f) / scale_h;
                        for (float x = 0; x < scale_w; x += 1) {
                            const auto cx = (x + 0.5f) / scale_w;
                            for (auto k : min_boxes[i]) {
                                const auto w = k / float(width);
                                const auto h = k / float(height);
                                parr.push_back( {
                                        clip01(cx), 
                                        clip01(cy),
                                        clip01(w),
                                        clip01(h)
                                    } );
                            }
                        }
                    }
                }
                // for RFB-320 (320x240):
                //   3*40*30 + 2*20*15 + 2*10*8 + 3*5*4 == 4420
                //   assert(parr.size() == 4420);
            }

            static constexpr auto center_variance = 0.1f;
            static constexpr auto size_variance = 0.2f;

            /** \brief Extract boxes from separate scores and boxes matrices.
             *
             * This method is suitable for ncnn output (inputs are ncnn::Mat).
             *
             * Returned coordinates are in [0,1].
             */
            template <typename MAT>
            auto operator()(const MAT& boxes, const MAT& scores,
                            float score_threshold) const {

                if (scores.h != long(parr.size()))
                    throw std::invalid_argument("scores have wrong size");
                if (boxes.h != long(parr.size()))
                    throw std::invalid_argument("boxes have wrong size");
                if (scores.w != 2 || scores.c != 1 ||
                    boxes.w != 4 || boxes.c != 1)
                    throw std::invalid_argument("scores or boxes have wrong shape");

                std::vector<bbox> bboxes;
                bboxes.reserve(parr.size());
                for (unsigned i = 0; i < parr.size(); ++i) {
                    if (scores[i*2+1] > score_threshold) {
                        auto& rect = bboxes.emplace_back();
                        rect.score = clip01(scores[i*2+1]);
                        const auto& pr = parr[i];
                        const auto cx =
                            pr[0] + pr[2]*boxes[i*4]*center_variance;
                        const auto cy =
                            pr[1] + pr[3]*boxes[i*4+1]*center_variance;
                        const auto w =
                            float(pr[2]*std::exp(boxes[i*4+2]*size_variance));
                        const auto h =
                            float(pr[3]*std::exp(boxes[i*4+3]*size_variance));
                        rect.x1 = clip01(cx - w/2);
                        rect.y1 = clip01(cy - h/2);
                        rect.x2 = clip01(cx + w/2);
                        rect.y2 = clip01(cy + h/2);
                    }
                }
                return bboxes;
            }

            /** \brief Extract boxes from combined model output.
             *
             * This method is suitable for net::vector output
             * (input is std::vector<float>).
             *
             * Returned coordinates are in [0,1].
             */
            template <typename VEC>
            auto operator()(const VEC& combo, float score_threshold) const {
                if (combo.size() != 6 * parr.size())
                    throw std::invalid_argument("input has wrong size");
                const auto scores = combo.data() + parr.size();
                const auto b0 = scores + parr.size();
                const auto b1 = b0 + parr.size();
                const auto b2 = b1 + parr.size();
                const auto b3 = b2 + parr.size();

                std::vector<bbox> bboxes;
                bboxes.reserve(parr.size());
                for (unsigned i = 0; i < parr.size(); ++i) {
                    if (scores[i] > score_threshold) {
                        auto& rect = bboxes.emplace_back();
                        rect.score = clip01(scores[i]);
                        const auto& pr = parr[i];
                        const auto cx = pr[0] + pr[2]*b0[i]*center_variance;
                        const auto cy = pr[1] + pr[3]*b1[i]*center_variance;
                        const auto w =
                            float(pr[2] * std::exp(b2[i]*size_variance));
                        const auto h =
                            float(pr[3] * std::exp(b3[i]*size_variance));
                        rect.x1 = clip01(cx - w/2);
                        rect.y1 = clip01(cy - h/2);
                        rect.x2 = clip01(cx + w/2);
                        rect.y2 = clip01(cy + h/2);
                    }
                }
                return bboxes;
            }
        };
    }
}
