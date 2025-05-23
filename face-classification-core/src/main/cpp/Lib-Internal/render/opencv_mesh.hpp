#pragma once

#include <array>
#include <set>
#include <vector>
#include <opencv2/core/core.hpp>
#include <stdext/forward_iterator.hpp>

namespace cvx {

    // note: triangle indices must be unique and ordered (as in std::set)
    using triangle_type = std::array<unsigned,3>;
    using mesh_type = std::set<triangle_type>;
    
    mesh_type compute_mesh(stdx::forward_iterator<cv::Point> pts_first,
                           stdx::forward_iterator<cv::Point> pts_last);
    
    inline mesh_type compute_mesh(const std::vector<cv::Point>& pts) {
        return compute_mesh(pts.begin(), pts.end());
    }
    
    void warp_mesh(
        cv::Mat dest_img, const cv::Point* dest_pts,
        const cv::Mat& src_img, const cv::Point* src_pts,
        stdx::forward_iterator<const triangle_type&> mesh_first,
        stdx::forward_iterator<const triangle_type&> mesh_last,
        std::size_t num_pts);
    
    inline void warp_mesh(
        cv::Mat dest_img, const std::vector<cv::Point>& dest_pts,
        const cv::Mat& src_img, const std::vector<cv::Point>& src_pts,
        stdx::forward_iterator<const triangle_type&> mesh_first,
        stdx::forward_iterator<const triangle_type&> mesh_last) {
        const auto N = std::min(dest_pts.size(), src_pts.size());
        warp_mesh(dest_img, dest_pts.data(),
                  src_img, src_pts.data(),
                  mesh_first, mesh_last, N);
    }
    
    template <typename T>
    inline void warp_mesh(
        cv::Mat dest_img, const std::vector<cv::Point>& dest_pts,
        const cv::Mat& src_img, const std::vector<cv::Point>& src_pts,
        const T& mesh) {
        warp_mesh(dest_img, dest_pts,
                  src_img, src_pts,
                  mesh.begin(), mesh.end());
    }
}
