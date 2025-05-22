
#include <boost/test/unit_test.hpp>
#include <applog/core.hpp>

#include <dlibx/dnn_project.hpp>
#include <dlibx/dnn_fc_dynamic.hpp>
#include <dlibx/trainer.hpp>
#include <dlibx/dnn_traits.hpp>

#include <dlib/dnn/input.h>
#include <dlib/dnn/loss.h>

#include <raw_image_3d/point3.hpp>

#include <random>
#include <iostream>


BOOST_AUTO_TEST_SUITE(dlibx)

static constexpr auto num_cameras = 3;

using image_type = std::array<dlib::matrix<float,1,1>,3>;
using raw_image::point3f;
static_assert(sizeof(image_type) == sizeof(point3f));

static auto& operator<<(std::ostream& out, point3f p) {
    auto d = std::sqrt(length_squared(p));
    auto x = std::lround(9 * p.x / d);
    auto y = std::lround(9 * p.y / d);
    auto z = std::lround(9 * p.z / d);
    return out << std::showpos << x << y << z;
}

static auto cos(const point3f& a, const point3f& b) {
    const auto d = length_squared(a) * length_squared(b);
    return dot(a,b) / std::sqrt(d);
}

namespace {

    using net = dlibx::project<
        dlib::concat3<dlib::tag1,dlib::tag2,dlib::tag3,
        dlib::tag3<dlibx::fc<3,dlib::skip1<
        dlib::tag2<dlibx::fc<3,
        dlib::tag1<dlib::input<image_type> > > > > > > > >;

    //using net = dlibx::project<
    //    dlibx::fc<3*num_cameras,dlib::input<dlib::matrix<float,3,1> > > >;

    using loss = dlib::loss_mean_squared_multioutput<net>;
}

static std::mt19937 rgen(1);
static std::normal_distribution<float> norm_distr;

using mat3x3 = dlib::matrix<float,3,3>;

static_assert(sizeof(mat3x3) == 9*sizeof(float));
static_assert(sizeof(dlib::matrix<float,3,1>) == 3*sizeof(float));
static_assert(sizeof(dlib::matrix<float,1,3>) == 3*sizeof(float));

// 3x3 rotation matrix on plane ij
template <unsigned i, unsigned j>
static auto rot3deg(float degrees) {
    static_assert(i != j && i < 3 && j < 3);
    mat3x3 mat = dlib::identity_matrix<float>(3);
    mat(i,i) =   mat(j,j) = std::cos(degrees * (M_PIf / 180.0f));
    mat(i,j) = -(mat(j,i) = std::sin(degrees * (M_PIf / 180.0f)));
    return mat;
}

static void setup_camera(fc_dynamic_<3,HAS_BIAS>& fc, const mat3x3& rot,
                         const dlib::matrix<float,3,1>& ofs) {
    dlib::tensor& params = fc.get_layer_params();
    assert(params.size() == 12);
    auto* dest = params.host();
    *reinterpret_cast<mat3x3*>(dest) = rot;
    using COL = dlib::matrix<float,3,1>;
    *reinterpret_cast<COL*>(dest + 9) = trans(rot) * ofs;
}

static void verify_rotation(fc_dynamic_<3,HAS_BIAS>& fc) {
    dlib::tensor& params = fc.get_layer_params();
    auto& mat = *reinterpret_cast<mat3x3*>(params.host());
    for (unsigned i = 0; i < 3; ++i) {
        auto a = dot(colm(mat,i),colm(mat,i));
        assert(std::abs(1-a) < 1e-5);
        auto b = dot(rowm(mat,i),rowm(mat,i));
        assert(std::abs(1-b) < 1e-5);
        for (unsigned j = i + 1; j < 3; ++j) {
            auto a = dot(colm(mat,i),colm(mat,j));
            assert(std::abs(a) < 1e-5);
            auto b = dot(rowm(mat,i),rowm(mat,j));
            assert(std::abs(b) < 1e-5);
        }
    }
}

static void normalize_rotation(fc_dynamic_<3,HAS_BIAS>& fc) {
    dlib::tensor& params = fc.get_layer_params();
    assert(params.size() == 12);
    auto& mat = *reinterpret_cast<mat3x3*>(params.host());
    mat3x3 u, w, v;
    svd(mat, u, w, v);
    mat = u * trans(v);
}

static void normalize_distance(fc_dynamic_<3,HAS_BIAS>& fc, float target) {
    dlib::tensor& params = fc.get_layer_params();
    assert(params.size() == 12);
    auto* bias = params.host() + 9;
    target /=
        std::sqrt(bias[0]*bias[0] + bias[1]*bias[1] + bias[2]*bias[2]);
    bias[0] *= target;
    bias[1] *= target;
    bias[2] *= target;
}

static float camera_distance(const fc_dynamic_<3,HAS_BIAS>& fc) {
    dlib::tensor const& params = fc.get_layer_params();
    assert(params.size() == 12);
    auto* bias = reinterpret_cast<const point3f*>(params.host() + 9);
    return std::sqrt(length_squared(*bias));
}

static float camera_rotation_error(const fc_dynamic_<3,HAS_BIAS>& fc,
                                   const mat3x3& rot) {
    dlib::tensor const& params = fc.get_layer_params();
    assert(params.size() == 12);
    const mat3x3 diff = rot - *reinterpret_cast<const mat3x3*>(params.host());
    float err = 0;
    for (auto x : diff)
        err += x*x;
    return std::sqrt(err);
}

static void jitter_camera(fc_dynamic_<3,HAS_BIAS>& fc) {
    dlib::tensor& params = fc.get_layer_params();
    assert(params.size() == 12);
    auto p = params.host();
    for (auto n = 9; n > 0; --n, ++p)
        *p += 0.125f * norm_distr(rgen);
    for (auto n = 3; n > 0; --n, ++p)
        *p += 125 * norm_distr(rgen);
    normalize_rotation(fc);
}


BOOST_AUTO_TEST_CASE(camera_projection) {
    FILE_LOG(logINFO) << "--";

    std::vector<image_type> target_points;
    std::vector<image_type> learned_points;
    target_points.reserve(100);
    for (auto n = 100; n > 0; --n) {
        // 100mm std.dev.
        const auto x = 100*norm_distr(rgen);
        const auto y = 100*norm_distr(rgen);
        const auto z = 100*norm_distr(rgen) + 1000;
        using M = dlib::matrix<float,1,1>;
        target_points.push_back( { M { x }, M { y }, M { z } });
        learned_points.push_back( {
                M { x + 10*norm_distr(rgen) },
                M { y + 10*norm_distr(rgen) },
                M { z + 10*norm_distr(rgen) }
            });
    }

    // initialize net
    loss net;
    net(learned_points.front());

    FILE_LOG(logINFO) << "camera 2...";
    auto& fc2 = dlib::layer<dlib::tag2>(net).subnet().layer_details();
    const mat3x3 cam2_rot = rot3deg<0,2>(45);
    setup_camera(fc2, cam2_rot, { -1000, 0, 0 });
    verify_rotation(fc2);

    FILE_LOG(logINFO) << "camera 3...";
    auto& fc3 = dlib::layer<dlib::tag3>(net).subnet().layer_details();
    const mat3x3 cam3_rot = rot3deg<1,2>(45) * rot3deg<0,2>(-45);
    setup_camera(fc3, cam3_rot, { 1400, -1000, 0 });
    verify_rotation(fc3);

    FILE_LOG(logINFO) << "net setup complete";

    // project target_points
    const dlib::resizable_tensor target_image =
        net.subnet()(target_points.begin(), target_points.end());
    FILE_LOG(logINFO) << "target image: "
                      << target_image.num_samples() << 'x'
                      << target_image.k() << 'x'
                      << target_image.nr()*target_image.nc();
    assert(target_image.size() == 2*num_cameras*target_points.size());
    const auto labels = [&](unsigned j) {
        auto const* p = target_image.host() + 2*num_cameras*j;
        using M = dlib::matrix<float,2*num_cameras,1>;
        static_assert(sizeof(M) == 2*num_cameras*sizeof(float));
        return *reinterpret_cast<M const*>(p);
    };

    std::array<float,6> sum = {0};
    unsigned i = 0;
    for (auto p = target_image.host(),
             end = p + target_image.size(); p != end; p += 6) {
        for (unsigned i = 0; i < 6; ++i)
            sum[i] += p[i];
        if (0) {
            std::cout << i++ << '\t'
                      << std::lround(p[0]) << ' ' << std::lround(p[1]) << '\t'
                      << std::lround(p[2]) << ' ' << std::lround(p[3]) << '\t'
                      << std::lround(p[4]) << ' ' << std::lround(p[5])
                      << std::endl;
        }
    }
    for (unsigned i = 0; i < num_cameras; ++i)
        FILE_LOG(logINFO) << "camera " << (i+1) << ": "
                          << sum[2*i] / float(target_points.size()) << ','
                          << sum[2*i+1] / float(target_points.size());

    // jitter cameras so training has to re-learn where they were
    //jitter_camera(fc2);
    //jitter_camera(fc3);
    verify_rotation(fc2);
    verify_rotation(fc3);

    // minibatch
    static constexpr auto mb_size = 20;
    assert(3*mb_size <= learned_points.size());
    std::vector<image_type> mb_images(mb_size);
    std::vector<dlib::matrix<float,2*num_cameras,1> > mb_labels(mb_size);
    std::vector<unsigned> mb_indices;
    mb_indices.reserve(learned_points.size());
    for (unsigned i = 0; i < learned_points.size(); ++i)
        mb_indices.push_back(i);
    shuffle(mb_indices.begin(), mb_indices.end(), rgen);
    const auto mb_pivot = next(mb_indices.begin(), mb_size);
    const auto mb_half =
        next(mb_indices.begin(), long(learned_points.size()/2));

    // train cameras and learned_points
    dlibx::dnn_trainer<loss> trainer(net, dlibx::sgd(0.0005f, 0.9f));

    trainer.set_learning_rate(0.01);
    trainer.set_learning_rate_shrink_factor(0.1);
    trainer.set_iterations_without_progress_threshold(5000);

    fc2.set_learning_rate_multiplier(0.001);
    fc3.set_learning_rate_multiplier(0.001);

    // training
    for (auto step = 0; step < 100000; ++step) {
        normalize_distance(fc2, 1000);
        normalize_rotation(fc2);
        normalize_rotation(fc3);

        if (step%100 == 0)
            FILE_LOG(logINFO) << "cams:\t" << camera_distance(fc2)
                              << '\t' << camera_rotation_error(fc2, cam2_rot)
                              << '\t' << camera_distance(fc3)
                              << '\t' << camera_rotation_error(fc3, cam3_rot);

        // prepare minibatch including labels
        rotate(mb_indices.begin(), mb_pivot, mb_indices.end());
        shuffle(mb_indices.begin(), mb_half, rgen);
        for (unsigned i = 0; i < mb_size; ++i) {
            const auto j = mb_indices[i];
            mb_images[i] = learned_points[j];
            mb_labels[i] = labels(j);
        }

        // do training step
        trainer.train_one_step(
            mb_images.cbegin(), mb_images.cend(), mb_labels.cbegin());
        trainer.get_solvers(); // wait for completion
        const auto rate = float(trainer.get_learning_rate());

        // update data
        dlib::tensor const& data_grad = net.get_final_data_gradient();
        assert(data_grad.size() == mb_size * 3);
        auto const* dg = reinterpret_cast<const point3f*>(data_grad.host());
        float maxgrad = 0;
        for (unsigned i = 0; i < mb_size; ++i, ++dg) {
            const auto j = mb_indices[i];
            const auto tp =
                *reinterpret_cast<const point3f*>(&target_points[j]);
            auto& lp = *reinterpret_cast<point3f*>(&learned_points[j]);
            //FILE_LOG(logINFO) << '\t' << (tp-lp) << '\t' << -*dg << '\t' << std::noshowpos << cos(tp-lp,-*dg);
            const auto d = length_squared(*dg);
            maxgrad = std::max(maxgrad, d);
            if (d < 210*210)
                lp = lp - rate * *dg;
            else FILE_LOG(logWARNING) << "bad data gradient: " << d;
        }
        //FILE_LOG(logINFO) << std::sqrt(maxgrad);

        if (step % 100 == 0) {
            auto loss = trainer.get_average_loss();
            auto spin = trainer.get_steps_without_progress();
            FILE_LOG(logINFO) << step << '\t' << rate
                              << '\t' << loss << '\t' << spin;
            //trainer.clear_average_loss();
        }

        if (rate < 1e-8) break;

        fc2.set_learning_rate_multiplier(1e-5 / rate);
        fc3.set_learning_rate_multiplier(1e-5 / rate);
    }


    FILE_LOG(logINFO) << "camera projection: done";
}

BOOST_AUTO_TEST_SUITE_END()
