#include <boost/test/unit_test.hpp>
#include <applog/base_directory.hpp>
#include <applog/core.hpp>

#include <dlibx/dnn_lmcon.hpp>
#include <dlibx/dnn_condw.hpp>
#include <dlibx/dnn_convert.hpp>
#include <dlib/dnn/input.h>

#include <random>

BOOST_AUTO_TEST_SUITE(dlibx)

template <long K, long NR, long NC, int SY, int SX>
using con_type = dlib::con_<K,NR,NC,SY,SX,
                            SY!=1 ? 0 : int(NR/2),
                            SX!=1 ? 0 : int(NC/2)>;

static std::mt19937 rgen(1);

template <std::size_t K>
static auto random_sample(long nr, long nc) {
    std::uniform_real_distribution<float> real1(-1, 1);
    std::array<dlib::matrix<float>,K> r;
    for (auto& m : r) {
        m.set_size(nr,nc);
        for (auto it = m.begin(), end = m.end(); it != end; ++it)
            *it = real1(rgen);
    }
    return r;
}

template <typename DISTR>
static void set_random(dlib::tensor& t, DISTR&& distr) {
    for (auto d = t.host_write_only(), end = d + t.size(); d != end; ++d)
        *d = float(distr(rgen));
}

template <long K, long NR, long NC, int SY, int SX>
static auto randomize_bias(con_type<K,NR,NC,SY,SX>& con) {
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
    if (MODE == NO_BIAS) return; // nothing to do
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

static void check_equal(const dlib::tensor& t0, const dlib::tensor& t1) {
    BOOST_REQUIRE_EQUAL(t0.num_samples(), t1.num_samples());
    BOOST_REQUIRE_EQUAL(t0.k(), t1.k());
    BOOST_REQUIRE_EQUAL(t0.nr(), t1.nr());
    BOOST_REQUIRE_EQUAL(t0.nc(), t1.nc());
    float err = 0;
    for (auto d0 = t0.host(), d1 = t1.host(),
             end = d1 + t1.size(); d1 != end; ++d0, ++d1)
        err = std::max(err, std::abs(*d0 - *d1));
    FILE_LOG(logDETAIL) << "error: " << err;
    BOOST_CHECK(err < 1e-5);
}

namespace {
    template <typename CON>
    struct add_bias {
        using type = CON;
    };
    template <long MULT, long NR, long NC, int SY, int SX>
    struct add_bias<dlibx::condw_<NO_BIAS,MULT,NR,NC,SY,SX> > {
        using type = dlibx::condw_<HAS_BIAS,MULT,NR,NC,SY,SX>;
    };
    template <typename CON>
    using add_bias_t = typename add_bias<CON>::type;
}

template <typename CON, std::size_t INPUT_CHANNELS>
static auto do_tests(
    const std::vector<std::array<dlib::matrix<float>,INPUT_CHANNELS> >& input) {

    using sample = std::array<dlib::matrix<float>,INPUT_CHANNELS>;
    using with_affine =
        dlib::affine<dlib::add_layer<CON, dlib::input<sample> > >;
    using no_affine = dlib::add_layer<add_bias_t<CON>, dlib::input<sample> >;

    FILE_LOG(logDETAIL) << "affine: setup net with affine";
    with_affine net_with_affine;
    net_with_affine.layer_details() = { dlib::CONV_MODE };
    dlib::resizable_tensor input_tensor;
    net_with_affine.to_tensor(input.begin(), input.end(), input_tensor);
    net_with_affine.forward(input_tensor); // to trigger setup

    // randomize affine and con bias (if it has bias)
    randomize_params(net_with_affine.layer_details());
    randomize_bias(net_with_affine.subnet().layer_details());

    FILE_LOG(logDETAIL) << "affine: convert to no affine version";
    no_affine net_no_affine;
    net_no_affine.to_tensor(input.begin(), input.end(), input_tensor);
    net_no_affine.forward(input_tensor); // to trigger setup

    convert_from_to(net_with_affine, net_no_affine,
                    copy_layer{}, remove_affine{});

    FILE_LOG(logDETAIL) << "affine: forward tensor";
    const auto& out1 = net_with_affine.forward(input_tensor);
    const auto& out2 = net_no_affine.forward(input_tensor);

    FILE_LOG(logDETAIL) << "affine: compare results";
    BOOST_REQUIRE(out1.size() > 0);
    check_equal(out1, out2);
}

BOOST_AUTO_TEST_CASE(affine_removal_test) {
    FILE_LOG(logINFO) << "--";

    static constexpr auto num_samples = 3;
    std::vector<std::array<dlib::matrix<float>,7> > input;
    input.reserve(num_samples);
    for (auto n = num_samples; n > 0; --n)
        input.emplace_back(random_sample<7>(11,13));

    FILE_LOG(logINFO) << "affine: con";
    do_tests<con_type<17,1,1,1,1>>(input);
    do_tests<con_type<17,3,3,1,1>>(input);
    do_tests<con_type<17,3,3,2,2>>(input);
    do_tests<con_type<17,5,5,1,1>>(input);
    do_tests<con_type<17,7,7,2,2>>(input);
    
    FILE_LOG(logINFO) << "affine: lmcon";
    do_tests<dlibx::lm_con_<17,1,1,1,1>>(input);
    do_tests<dlibx::lm_con_<17,3,3,1,1>>(input);
    do_tests<dlibx::lm_con_<17,3,3,2,2>>(input);
    do_tests<dlibx::lm_con_<17,5,5,1,1>>(input);
    do_tests<dlibx::lm_con_<17,7,7,2,2>>(input);

    FILE_LOG(logINFO) << "affine: condw (with bias)";
    do_tests<dlibx::condw_<HAS_BIAS,4,1,1,1,1>>(input);
    do_tests<dlibx::condw_<HAS_BIAS,3,3,3,1,1>>(input);
    do_tests<dlibx::condw_<HAS_BIAS,2,3,3,2,2>>(input);
    do_tests<dlibx::condw_<HAS_BIAS,1,5,5,1,1>>(input);
    do_tests<dlibx::condw_<HAS_BIAS,5,7,7,2,2>>(input);

    FILE_LOG(logINFO) << "affine: condw (no bias)";
    do_tests<dlibx::condw_<NO_BIAS,5,1,1,1,1>>(input);
    do_tests<dlibx::condw_<NO_BIAS,4,3,3,1,1>>(input);
    do_tests<dlibx::condw_<NO_BIAS,3,3,3,2,2>>(input);
    do_tests<dlibx::condw_<NO_BIAS,2,5,5,1,1>>(input);
    do_tests<dlibx::condw_<NO_BIAS,1,7,7,2,2>>(input);

    FILE_LOG(logINFO) << "affine: done";
}

BOOST_AUTO_TEST_SUITE_END()
