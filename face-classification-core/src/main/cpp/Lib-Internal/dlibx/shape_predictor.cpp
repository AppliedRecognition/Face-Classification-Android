
#include "shape_predictor.hpp"
#include "huffman_decoder.hpp"


using namespace dlibx;


namespace {
    struct split_feature {
        unsigned idx1;  // index into feature_pixel_values
        unsigned idx2;
        float thresh;
        
        friend void deserialize(split_feature& item, std::istream& in) {
            dlib::deserialize(item.idx1, in);
            dlib::deserialize(item.idx2, in);
            dlib::deserialize(item.thresh, in);
        }
        friend void serialize(const split_feature& item, std::ostream& out) {
            dlib::serialize(item.idx1, out);
            dlib::serialize(item.idx2, out);
            dlib::serialize(item.thresh, out);
        }
    };

    struct regression_tree {
        using leaf_type = dlib::matrix<short,0,1>;
        
        std::vector<split_feature> splits;  // size 15
        std::vector<leaf_type> leaf_values; // vec 16 mat 136

        long leaf_size() const {
            return leaf_values.empty() ? 0 : leaf_values.front().size();
        }
        
        inline const leaf_type& operator()(
            const std::vector<float>& pixels, std::size_t& i) const {
            i = 0;
            while (i < splits.size()) {
                auto& s = splits[i];
                if (pixels[s.idx1] - pixels[s.idx2] > s.thresh)
                    i = 2*i + 1;  // left_child
                else
                    i = 2*i + 2;  // right_child
            }
            i -= splits.size();
            return leaf_values[i];
        }

        void deserialize(std::istream& in, huffman::decoder<short>& decoder) {
            dlib::deserialize(splits, in);
            std::size_t size;
            dlib::deserialize(size, in);
            leaf_values.resize(size);
            std::vector<short> buf;
            for (auto& leaf : leaf_values) {
                buf.clear();
                while (auto p = decoder(in))
                    buf.push_back(*p);
                leaf = dlib::mat(buf);
            }
        }
    };

    struct regression_tree_v1 {
        using leaf_type = dlib::matrix<float,0,1>;

        std::vector<split_feature> splits;
        std::vector<leaf_type> leaf_values;

        friend void deserialize(regression_tree_v1& t, std::istream& in) {
            dlib::deserialize(t.splits, in);
            dlib::deserialize(t.leaf_values, in);
        }
    };
}

struct shape_predictor::regression_forest {
    std::vector<regression_tree> forest;
    float coefficient;
    
    inline dlib::matrix<float,0,1>
    operator()(const std::vector<float>& pixels) const {
        const auto N = forest.empty() ? 0 : forest.front().leaf_size();
        assert(N > 0);
        dlib::matrix<int,0,1> correction = dlib::zeros_matrix<int>(N,1);
        std::size_t idx;  // ignored
        for (auto& tree : forest)
            correction += dlib::matrix_cast<int>(tree(pixels, idx));
        return coefficient * dlib::matrix_cast<float>(correction);
    }

    void deserialize_v1(std::istream& in) {
        std::vector<regression_tree_v1> v1;
        dlib::deserialize(v1, in);

        float max_value = 0;
        for (const auto& tree : v1)
            for (auto& leaf : tree.leaf_values)
                for (auto& x : leaf)
                    max_value = std::max(max_value, std::abs(x));
        coefficient = max_value / 32760;

        forest.reserve(v1.size());
        for (auto& src : v1) {
            forest.emplace_back();
            auto& dest = forest.back();
            dest.splits = move(src.splits);
            dest.leaf_values.reserve(src.leaf_values.size());
            for (const auto& leaf : src.leaf_values) {
                dest.leaf_values.emplace_back();
                dest.leaf_values.back() =
                    dlib::matrix_cast<short>(leaf / coefficient);
            }
        }
    }
    void deserialize(std::istream& in) {
        huffman::decoder<short> decoder;
        decoder.deserialize(in);

        dlib::deserialize(coefficient, in);

        std::size_t size;
        dlib::deserialize(size, in);
        forest.resize(size);
        for (auto& tree : forest)
            tree.deserialize(in, decoder);
    }
    friend void deserialize(regression_forest& f, std::istream& in) {
        f.deserialize(in);
    }
};

shape_predictor::shape_predictor() = default;
shape_predictor::~shape_predictor() = default;
shape_predictor::shape_predictor(shape_predictor&&) = default;
shape_predictor& shape_predictor::operator=(shape_predictor&&) = default;

template <typename MAT, typename I>
static inline auto location(const MAT& shape, I idx) {
    const auto i2 = 2*long(idx);
    return dlib::vector<float,2> { shape(i2), shape(i2+1) };
}

template <typename U>
static void apply_tform(dlib::matrix<float,0,1>& shape, U&& tform) {
    for (long i = 0; i < shape.nr(); i += 2) {
        const auto p = dlib::vector<float,2>(shape(i),shape(i+1));
        const auto r = tform(p);
        shape(i)   = float(r(0));
        shape(i+1) = float(r(1));
    }
}

// returns a transform that maps
//   (0,0) to rect.tl_corner() and
//   (1,1) to rect.br_corner()
static inline dlib::point_transform_affine
unnormalizing_tform(const dlib::rectangle& rect) {
    std::vector<dlib::vector<float,2> > from, to;
    to.push_back(rect.tl_corner()); from.push_back({0,0});
    to.push_back(rect.tr_corner()); from.push_back({1,0});
    to.push_back(rect.br_corner()); from.push_back({1,1});
    return dlib::find_affine_transform(from, to);
}

static dlib::point_transform_affine
find_tform_between_shapes(
    const dlib::matrix<float,0,1>& from_shape,
    const dlib::matrix<float,0,1>& to_shape) {

    DLIB_ASSERT(from_shape.size() == to_shape.size() &&
                (from_shape.size()%2) == 0 &&
                from_shape.size() > 2, "");

    std::vector<dlib::vector<float,2> > from_points, to_points;
    const auto num = from_shape.size()/2;
    from_points.reserve(std::size_t(num));  // num >= 0 by assert above
    to_points.reserve(std::size_t(num));
    for (long i = 0; i < num; ++i) {
        from_points.push_back(location(from_shape,i));
        to_points.push_back(location(to_shape,i));
    }
    return find_similarity_transform(from_points, to_points);
}

static void extract_feature_pixel_values (
    const pixel_intensity_base<float>& pi,
    const dlib::point_transform_affine& tform_to_img,
    const dlib::matrix<float,0,1>& current_shape,
    const dlib::matrix<float,2,2>& tform,
    const std::vector<unsigned>& reference_pixel_anchor_idx,
    const std::vector<dlib::vector<float,2> >& reference_pixel_deltas,
    std::vector<float>& feature_pixel_values) {

    feature_pixel_values.resize(reference_pixel_deltas.size());
    for (unsigned long i = 0; i < feature_pixel_values.size(); ++i) {
        const auto r = location(current_shape, reference_pixel_anchor_idx[i]);
        const dlib::point p = tform_to_img(r + tform*reference_pixel_deltas[i]);
        feature_pixel_values[i] = pi(p.y(),p.x());
    }
}

static void extract_feature_pixel_values (
    const pixel_intensity_base<float>& pi,
    const dlib::matrix<float,0,1>& current_shape,
    const dlib::matrix<float,2,2>& tform,
    const std::vector<unsigned>& ref_idx,
    const std::vector<dlib::vector<float,2> >& ref_delta,
    std::vector<float>& feature_pixel_values) {

    feature_pixel_values.resize(ref_delta.size());
    for (unsigned long i = 0; i < feature_pixel_values.size(); ++i) {
        const dlib::point p =
            location(current_shape, ref_idx[i]) + tform*ref_delta[i];
        feature_pixel_values[i] = pi(p.y(),p.x());
    }
}

static void
correct_standard_deviation(std::vector<float>& values,
                           const std::pair<float,float>& min_max) {
    assert(!values.empty());
    float sum = 0, s2 = 0;
    for (auto x : values)
        sum += x, s2 += x*x;
    const auto nv = float(values.size());
    const auto mean = sum/nv;
    auto stdev = std::sqrt(s2/nv - mean*mean);
    if (stdev < 5) stdev = 5;
    if (stdev < min_max.first)
        for (auto& x : values)
            x = (x-mean)*(min_max.first/stdev);
    else if (stdev > min_max.second)
        for (auto& x : values)
            x = (x-mean)*(min_max.second/stdev);
}

std::vector<dlib::point> shape_predictor::detect(
    const pixel_intensity_base<float>& pi, const dlib::rectangle& rect,
    std::pair<float,float> min_max_stdev) const {

    const auto tform_to_img = unnormalizing_tform(rect);
    auto current_shape = initial_shape;

    std::vector<float> feature_pixel_values;
    for (unsigned long iter = 0; iter < forests.size(); ++iter) {

        // tform is scale and rotation only (no translation)
        // from initial_shape to current_shape
        const auto tform_m =
            find_tform_between_shapes(initial_shape, current_shape).get_m();
        const auto tform = dlib::matrix_cast<float>(tform_m);

        extract_feature_pixel_values(
            pi, tform_to_img, current_shape, tform,
            anchor_idx[iter], deltas[iter], feature_pixel_values);
        if (min_max_stdev.second > 0)
            correct_standard_deviation(feature_pixel_values, min_max_stdev);

        // evaluate all the trees at this level of the cascade
        auto correction = forests[iter](feature_pixel_values);
        // todo: tform should be applied to correction before adding to shape
        current_shape += correction;
    }
    
    // convert the current_shape to pixel space
    const auto num = current_shape.size()/2;
    assert(num >= 0);
    std::vector<dlib::point> parts;
    parts.reserve(std::size_t(num));
    for (long i = 0; i < num; ++i)
        parts.push_back(tform_to_img(location(current_shape, i)));
    return parts;
}

std::vector<dlib::point> shape_predictor::detect(
    const pixel_intensity_base<float>& pi,
    const std::vector<std::pair<unsigned,dlib::point> >& known,
    std::pair<float,float> min_max_stdev) const {

    // transform initial_shape to best match known points
    auto current_shape = initial_shape;
    {
        std::vector<dlib::vector<float,2> > from, to;
        from.reserve(known.size());
        to.reserve(known.size());
        for (auto& p : known) {
            from.push_back(location(initial_shape, p.first));
            to.push_back(p.second);
        }
        const auto tform = find_similarity_transform(from, to);
        apply_tform(current_shape, tform);
    }

    std::vector<float> feature_pixel_values;
    for (unsigned long iter = 0; iter < forests.size(); ++iter) {

        // tform is scale and rotation only (no translation)
        // from initial_shape to current_shape
        const auto tform_full =
            find_tform_between_shapes(initial_shape, current_shape);
        const auto tform = dlib::matrix_cast<float>(tform_full.get_m());
        const auto tform_fn = [&](const auto& p) { return tform * p; };

        extract_feature_pixel_values(
            pi, current_shape, tform, anchor_idx[iter], deltas[iter],
            feature_pixel_values);
        if (min_max_stdev.second > 0)
            correct_standard_deviation(feature_pixel_values, min_max_stdev);

        // evaluate all the trees at this level of the cascade
        auto correction = forests[iter](feature_pixel_values);
        apply_tform(correction, tform_fn);
        current_shape += correction;
    }

    // round current_shape
    const auto num = current_shape.size()/2;
    assert(num >= 0);
    std::vector<dlib::point> parts;
    parts.reserve(std::size_t(num));
    for (long i = 0; i < current_shape.size(); i += 2)
        parts.emplace_back(
            lround(current_shape(i)), lround(current_shape(i+1)));
    return parts;
}

void shape_predictor::deserialize(std::istream& in) {
    int version = 0;
    dlib::deserialize(version, in);
    if (version != 1 && version != 87)
        throw dlib::serialization_error("Unexpected version found while deserializing dlib::shape_predictor.");

    dlib::deserialize(initial_shape, in);

    if (version == 1) {
        // forests
        std::size_t size;
        dlib::deserialize(size, in);
        forests.resize(size);
        for (auto& item : forests)
            item.deserialize_v1(in);
    }
    else // (version == 87)
        // why 87? because dlib uses 1 and could use 2, 3, ... in the future
        // we don't want to conflict
        dlib::deserialize(forests, in);

    
    dlib::deserialize(anchor_idx, in);
    dlib::deserialize(deltas, in);
}
