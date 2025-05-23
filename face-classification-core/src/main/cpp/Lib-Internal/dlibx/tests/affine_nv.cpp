#include <boost/test/unit_test.hpp>
#include <applog/base_directory.hpp>
#include <applog/core.hpp>

#include <dlibx/dnn_traits.hpp>
#include <dlibx/net_convert.hpp>
#include <dlibx/net_vector.hpp>
#include <dlibx/raw_image.hpp>

#include <random>

BOOST_AUTO_TEST_SUITE(dlibx)

template <long K, long NR, long NC, int SY, int SX>
using dlib_con_ = dlib::con_<K,NR,NC,SY,SX,
                            SY!=1 ? 0 : int(NR/2),
                            SX!=1 ? 0 : int(NC/2)>;

static std::mt19937 rgen(1);

template <typename PIXEL>
static auto randomize_image(dlib::matrix<PIXEL>& image) {
    std::uniform_int_distribution<unsigned> ui(0,255);
    const auto size = image.nr() * image.nc() * dlib::pixel_traits<PIXEL>::num;
    for (auto* it = reinterpret_cast<unsigned char*>(&image(0,0)),
             end = it + size; it != end; ++it)
        *it = static_cast<unsigned char>(ui(rgen));
}

template <typename DISTR>
static void set_random(dlib::tensor& t, DISTR&& distr) {
    for (auto d = t.host_write_only(), end = d + t.size(); d != end; ++d)
        *d = float(distr(rgen));
}

template <unsigned long K>
static auto randomize_bias(dlib::fc_<K,dlib::FC_HAS_BIAS>& fc) {
    auto& params = fc.get_layer_params();
    const auto num_outputs = fc.get_num_outputs();
    const auto num_inputs = params.size() / num_outputs - 1;
    assert(params.size() == num_outputs * (num_inputs + 1));
    std::uniform_real_distribution<float> real1(-1, 1);
    for (auto end = params.host() + params.size(),
             d = end - num_outputs; d != end; ++d) {
        BOOST_CHECK_EQUAL(*d,0);
        *d = real1(rgen);
    }
}

template <unsigned long K, bias_mode BM>
static auto randomize_bias(dlibx::fc_dynamic_<K,BM>& fc) {
    if (fc.get_bias_mode() == NO_BIAS) {
        FILE_LOG(logDETAIL) << "fc_dynamic has no bias";
        return; // nothing to do
    }
    auto& params = fc.get_layer_params();
    const auto num_outputs = fc.get_num_outputs();
    const auto num_inputs = params.size() / num_outputs - 1;
    assert(params.size() == num_outputs * (num_inputs + 1));
    std::uniform_real_distribution<float> real1(-1, 1);
    for (auto end = params.host() + params.size(),
             d = end - num_outputs; d != end; ++d) {
        BOOST_CHECK_EQUAL(*d,0);
        *d = real1(rgen);
    }
}

template <long K, long NR, long NC, int SY, int SX>
static auto randomize_bias(dlib_con_<K,NR,NC,SY,SX>& con) {
    auto& params = con.get_layer_params();
    const auto filter_size = params.size()/K - 1;
    assert(params.size() == K * (filter_size+1));
    assert(filter_size % (NR*NC) == 0);
    std::uniform_real_distribution<float> real1(-1, 1);
    for (auto end = params.host() + params.size(),
             d = end - K; d != end; ++d) {
        BOOST_CHECK_EQUAL(*d,0);
        *d = real1(rgen);
    }
}
template <long K, long NR, long NC, int SY, int SX>
static auto randomize_bias(dlibx::lm_con_<K,NR,NC,SY,SX>& con) {
    auto& params = con.get_layer_params();
    const auto filter_size = params.size()/K - 1;
    assert(params.size() == K * (filter_size+1));
    assert(filter_size % (NR*NC) == 0);
    std::uniform_real_distribution<float> real1(-1, 1);
    for (auto end = params.host() + params.size(),
             d = end - K; d != end; ++d) {
        BOOST_CHECK_EQUAL(*d,0);
        *d = real1(rgen);
    }
}
template <bias_mode MODE, long MULT, long NR, long NC, int SY, int SX>
static auto randomize_bias(dlibx::condw_<MODE,MULT,NR,NC,SY,SX>& con) {
    if (con.get_bias_mode() == NO_BIAS) return; // nothing to do
    auto& params = con.get_layer_params();
    const auto num_filters = params.size() / (NR*NC+1);
    assert(params.size() == num_filters * (NR*NC+1));
    std::uniform_real_distribution<float> real1(-1, 1);
    for (auto end = params.host() + params.size(),
             d = end - num_filters; d != end; ++d) {
        BOOST_CHECK_EQUAL(*d,0);
        *d = real1(rgen);
    }
}

static auto randomize_params(dlib::affine_& affine) {
    // have to serialize / deserialize to set affine params
    using dlib::serialize;
    using dlib::deserialize;

    std::stringstream in;
    serialize(affine, in);
    std::string version;
    dlib::deserialize(version, in);
    if (version != "affine_" && version != "affine_2")
        throw std::runtime_error("unknown version (affine)");
    dlib::resizable_tensor params;
    deserialize(params, in);

    std::uniform_real_distribution<float> real1(-1, 1);
    set_random(params, real1);

    assert((params.size()&1) == 0);
    auto alias = dlib::alias_tensor(1, long(params.size()/2));
    
    std::stringstream out;
    serialize("affine_", out);
    serialize(params, out);
    serialize(alias, out); // gamma
    serialize(alias, out); // beta
    serialize(int(dlib::CONV_MODE), out);

    deserialize(affine, out);
}

static void check_equal(const dlib::tensor& t0, const std::vector<float>& t1) {
    BOOST_REQUIRE_EQUAL(t0.size(), t1.size());
    float err = 0, m0 = 0, m1 = 0;
    for (auto d0 = t0.host(),
             d1 = t1.data(), end = d1 + t1.size(); d1 != end; ++d0, ++d1) {
        err = std::max(err, std::abs(*d0 - *d1));
        m0 = std::max(m0, std::abs(*d0));
        m1 = std::max(m1, std::abs(*d1));
    }
    FILE_LOG(logDETAIL) << "error: " << err << '\t' << m0 << '\t' << m1;
    BOOST_CHECK(err < 1e-5);
}

template <typename LAYER, typename PIXEL>
static auto do_tests(const dlib::matrix<PIXEL>& input) {

    using image = dlib::matrix<PIXEL>;
    using net_type = dlib::multiply<dlib::affine<dlib::add_layer<LAYER, dlib::input<image> > > >;

    FILE_LOG(logDETAIL) << "affine: setup net with affine";
    net_type net;

    BOOST_REQUIRE_EQUAL(net.layer_details().get_multiply_value(), 0.5f);
    auto& affine_layer = net.subnet().layer_details();
    affine_layer =
        { is_fc_layer<LAYER>::value ? dlib::FC_MODE : dlib::CONV_MODE };
    auto& main_layer = net.subnet().subnet().layer_details();

    dlib::resizable_tensor input_tensor;
    net.to_tensor(&input, &input+1, input_tensor);
    net.forward(input_tensor); // to trigger setup

    // randomize affine and layer bias (if it has bias)
    randomize_params(affine_layer);
    randomize_bias(main_layer);

    FILE_LOG(logDETAIL) << "affine: remove affine and convert to nv";
    auto lv = net::to_layers_vector(net);
    BOOST_REQUIRE_EQUAL(lv.size(), 4);
    remove_affine(lv);  // removes both affine and multiply layers
    BOOST_CHECK_EQUAL(lv.size(), 2);
    auto nv = net::vector(move(lv));

    FILE_LOG(logDETAIL) << "affine: forward tensor";
    const auto& out1 = net.forward(input_tensor);
    std::vector<float> out2;
    nv(to_raw_image(input),out2);

    FILE_LOG(logDETAIL) << "affine: compare results";
    BOOST_REQUIRE(out1.size() > 0);
    check_equal(out1, out2);
}

BOOST_AUTO_TEST_CASE(affine_removal_net_vector_test) {
    FILE_LOG(logINFO) << "--";

    dlib::matrix<unsigned char> sample_gray(11,13);
    randomize_image(sample_gray);
    dlib::matrix<dlib::rgb_pixel> sample_rgb(17,7);
    randomize_image(sample_rgb);

    FILE_LOG(logINFO) << "affine: con";
    do_tests<dlib_con_<17,1,1,1,1>>(sample_gray);
    do_tests<dlib_con_<17,3,3,1,1>>(sample_rgb);
    do_tests<dlib_con_<17,3,3,2,2>>(sample_gray);
    do_tests<dlib_con_<17,5,5,1,1>>(sample_rgb);
    do_tests<dlib_con_<17,7,7,2,2>>(sample_gray);
    
    FILE_LOG(logINFO) << "affine: lmcon";
    do_tests<dlibx::lm_con_<17,1,1,1,1>>(sample_rgb);
    do_tests<dlibx::lm_con_<17,3,3,1,1>>(sample_gray);
    do_tests<dlibx::lm_con_<17,3,3,2,2>>(sample_rgb);
    do_tests<dlibx::lm_con_<17,5,5,1,1>>(sample_gray);
    do_tests<dlibx::lm_con_<17,7,7,2,2>>(sample_rgb);

    FILE_LOG(logINFO) << "affine: condw (with bias)";
    do_tests<dlibx::condw_<HAS_BIAS,4,1,1,1,1>>(sample_rgb);
    do_tests<dlibx::condw_<HAS_BIAS,3,3,3,1,1>>(sample_gray);
    do_tests<dlibx::condw_<HAS_BIAS,2,3,3,2,2>>(sample_rgb);
    do_tests<dlibx::condw_<HAS_BIAS,1,5,5,1,1>>(sample_gray);
    do_tests<dlibx::condw_<HAS_BIAS,5,7,7,2,2>>(sample_rgb);

    FILE_LOG(logINFO) << "affine: condw (no bias)";
    do_tests<dlibx::condw_<NO_BIAS,5,1,1,1,1>>(sample_gray);
    do_tests<dlibx::condw_<NO_BIAS,4,3,3,1,1>>(sample_rgb);
    do_tests<dlibx::condw_<NO_BIAS,3,3,3,2,2>>(sample_gray);
    do_tests<dlibx::condw_<NO_BIAS,2,5,5,1,1>>(sample_rgb);
    do_tests<dlibx::condw_<NO_BIAS,1,7,7,2,2>>(sample_gray);

    FILE_LOG(logINFO) << "affine: fc";
    do_tests<dlib::fc_<1,HAS_BIAS>>(sample_rgb);
    do_tests<dlib::fc_<3,HAS_BIAS>>(sample_gray);
    do_tests<dlib::fc_<5,HAS_BIAS>>(sample_rgb);
    do_tests<dlib::fc_<7,HAS_BIAS>>(sample_gray);
    do_tests<dlib::fc_<12,HAS_BIAS>>(sample_rgb);

    FILE_LOG(logINFO) << "affine: fc_dynamic (with bias)";
    do_tests<dlibx::fc_dynamic_<1,HAS_BIAS>>(sample_gray);
    do_tests<dlibx::fc_dynamic_<3,HAS_BIAS>>(sample_rgb);
    do_tests<dlibx::fc_dynamic_<5,HAS_BIAS>>(sample_gray);
    do_tests<dlibx::fc_dynamic_<7,HAS_BIAS>>(sample_rgb);
    do_tests<dlibx::fc_dynamic_<12,HAS_BIAS>>(sample_gray);

    FILE_LOG(logINFO) << "affine: fc_dynamic_ (no bias)";
    do_tests<dlibx::fc_dynamic_<1,NO_BIAS>>(sample_rgb);
    do_tests<dlibx::fc_dynamic_<3,NO_BIAS>>(sample_gray);
    do_tests<dlibx::fc_dynamic_<5,NO_BIAS>>(sample_rgb);
    do_tests<dlibx::fc_dynamic_<7,NO_BIAS>>(sample_gray);
    do_tests<dlibx::fc_dynamic_<12,NO_BIAS>>(sample_rgb);

    FILE_LOG(logINFO) << "affine: done";
}

BOOST_AUTO_TEST_SUITE_END()
