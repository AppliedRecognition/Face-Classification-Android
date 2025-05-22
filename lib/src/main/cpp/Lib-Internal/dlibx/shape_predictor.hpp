#pragma once

#include "pixel_intensity.hpp"

namespace dlibx {
    struct shape_predictor {
        dlib::matrix<float,0,1> initial_shape;
        struct regression_forest;
        std::vector<regression_forest> forests;
        std::vector<std::vector<unsigned> > anchor_idx;
        std::vector<std::vector<dlib::vector<float,2> > > deltas;

        shape_predictor();
        ~shape_predictor();
        shape_predictor(shape_predictor&&);
        shape_predictor& operator=(shape_predictor&&);

        shape_predictor(const shape_predictor&) = delete;
        shape_predictor& operator=(const shape_predictor&) = delete;


        // note: if min_max_contrast.second <= 0, then
        // contrast correction is disabled
        std::vector<dlib::point> detect(
            const pixel_intensity_base<float>& pi,
            const dlib::rectangle& rect,
            std::pair<float,float> min_max_contrast) const;
        std::vector<dlib::point> detect(
            const pixel_intensity_base<float>& pi,
            const std::vector<std::pair<unsigned,dlib::point> >& known,
            std::pair<float,float> min_max_contrast) const;
        
        template <typename image_type>
        inline dlib::full_object_detection operator()(
            const image_type& img, const dlib::rectangle& rect,
            std::pair<float,float> min_max_contrast = { -1, -1 }) const {
            const pixel_intensity_helper<float, image_type> pi(img);
            return { rect, detect(pi, rect, min_max_contrast) };
        }
        template <typename image_type>
        inline std::vector<dlib::point> operator()(
            const image_type& img, 
            const std::vector<std::pair<unsigned,dlib::point> >& known,
            std::pair<float,float> min_max_contrast = { -1, -1 }) const {
            const pixel_intensity_helper<float, image_type> pi(img);
            return detect(pi, known, min_max_contrast);
        }

        void deserialize(std::istream& in);
        friend inline void
        deserialize(shape_predictor& sp, std::istream& in) {
            sp.deserialize(in);
        }
    };
}
