#include <boost/test/unit_test.hpp>
#include <applog/base_directory.hpp>
#include <applog/core.hpp>

#include <dlibx/dnn_condw.hpp>
#include <dlibx/dnn_relun.hpp>
#include <dlibx/dnn_traits.hpp>
#include <dlibx/tensor_tools.hpp>
#include <dlib/dnn/input.h>

#include <core/context.hpp>
#include <core/thread_data.hpp>
#include <core/job_queue.hpp>

#include <stdext/rounding.hpp>
#include <stdext/span.hpp>
#include <random>

BOOST_AUTO_TEST_SUITE(dlibx)

template <long K, long NR, long NC, int STRIDE>
using con_type = dlib::con_<K,NR,NC,STRIDE,STRIDE,
                            STRIDE!=1 ? 0 : int(NR/2),
                            STRIDE!=1 ? 0 : int(NC/2)>;

static std::mt19937 rgen(1);

template <typename DISTR>
static void set_random(dlib::tensor& t, DISTR&& distr) {
    for (auto d = t.host_write_only(), end = d + t.size(); d != end; ++d)
        *d = float(distr(rgen));
}

static auto compare(const dlib::tensor& t0, const dlib::tensor& t1) {
    BOOST_REQUIRE_EQUAL(t0.size(), t1.size());
    std::vector<float> err;
    err.reserve(t0.size());
    for (auto d0 = t0.host(), d1 = t1.host(),
             end1 = d1 + t1.size(); d1 != end1; ++d0, ++d1)
        err.emplace_back(std::abs(*d0 - *d1));
    std::sort(err.begin(), err.end());
    return err;
}

static void check_equal(const dlib::tensor& t0, const dlib::tensor& t1) {
    BOOST_REQUIRE_EQUAL(t0.num_samples(), t1.num_samples());
    BOOST_REQUIRE_EQUAL(t0.k(), t1.k());
    BOOST_REQUIRE_EQUAL(t0.nr(), t1.nr());
    BOOST_REQUIRE_EQUAL(t0.nc(), t1.nc());
    for (auto d0 = t0.host(), d1 = t1.host(),
             end = d1 + t1.size(); d1 != end; ++d0, ++d1)
        BOOST_CHECK_EQUAL(*d0, *d1);
}

namespace {
    struct subnet {
        dlib::resizable_tensor data;

        auto nr() const { return data.nr(); }
        auto nc() const { return data.nc(); }

        const dlib::tensor& get_output() const { return data; }

        dlib::resizable_tensor gradient;
        dlib::tensor& get_gradient_input() { return gradient; }

        void init_gradient() {
            gradient.copy_size(data);
            std::uniform_real_distribution<float> ud(-1, 1);
            set_random(data, ud);
        }
        
        void init_real_1(long n, long k, long nr, long nc) {
            data.set_size(n,k,nr,nc);
            std::uniform_real_distribution<float> ud(-1, 1);
            set_random(data, ud);
        }
        void init_int_3(long n, long k, long nr, long nc) {
            data.set_size(n,k,nr,nc);
            std::uniform_int_distribution<> ud(-3, 3);
            set_random(data, ud);
        }
    };

    template <long K>
    struct input : subnet {
        input(long nr, long nc) {
            init_int_3(1,K,nr,nc);
        }
    };
}

static auto to_subnet(const dlib::tensor& t) {
    struct subnet {
        const dlib::tensor& t;
        const dlib::tensor& get_output() const { return t; }
    };
    return subnet{t};
}

/** \brief Extract params for equivalent convolution for specified channel.
 *
 * \returns { filters_alias, biases_alias }
 */
template <bias_mode BM, long MULT, long NR, long NC, int STRIDE>
static auto extract_con_params(
    const condw_<BM,MULT,NR,NC,STRIDE,STRIDE>&,
    const dlib::tensor& params,
    long k) {

    static constexpr auto BIAS = BM == NO_BIAS ? 0 : MULT;
    const auto nchannels = params.size() / (MULT*NR*NC + BIAS);
    assert(params.size() == nchannels * (MULT*NR*NC + BIAS));
    assert(0 <= k && std::size_t(k) < nchannels);
    auto filt = dlib::alias_tensor(MULT,1,NR,NC);
    auto bias = dlib::alias_tensor(1,BIAS);
    const auto uk = std::size_t(k);
    return std::make_pair(filt(params,uk*filt.size()),
                          bias(params,nchannels*MULT*NR*NC + uk*BIAS));
}

/** \brief Extract equivalent convolution for specified channel.
 */
template <bias_mode BM, long MULT, long NR, long NC, int STRIDE>
static auto extract_con(
    const condw_<BM,MULT,NR,NC,STRIDE,STRIDE>& condw,
    long k) {

    const auto pr = extract_con_params(condw, condw.get_layer_params(), k);
    const dlib::tensor& filt = pr.first.get();
    assert(filt.size() == MULT*NR*NC);
    const dlib::tensor& bias = pr.second.get();

    subnet in;
    in.data.set_size(1,1,NR,NC);
    con_type<MULT,NR,NC,STRIDE> con;
    con.setup(in);

    dlib::tensor& dest_params = con.get_layer_params();
    assert(dest_params.size() == MULT*NR*NC + MULT);
    const auto p = std::copy_n(filt.host(), filt.size(), dest_params.host());
    // note: by default con.setup() will zero the bias
    std::copy_n(bias.host(), bias.size(), p);
    return con;
}

/// split filters by input channel
/// output size is (MULT,1,NR,NC)
template <long MULT, long NR, long NC, int SY, int SX>
static auto filters(condw_<NO_BIAS,MULT,NR,NC,SY,SX>& cdw) {
    return tensor_alias_span<dlib::tensor>(cdw.get_layer_params(), {MULT, 1, NR, NC});
}

template <long mult, long nr, long nc, long K>
static void do_test1(const input<K>& in) {
    static_assert(K >= 0);
    condw_<NO_BIAS,mult,nr,nc,1,1> cdw;
    cdw.setup(in);
    std::uniform_int_distribution<> dis(-3, 3);
    set_random(cdw.get_layer_params(), dis);

    dlib::resizable_tensor odw;
    cdw.forward(in, odw);
    BOOST_REQUIRE_EQUAL(odw.num_samples(), 1);
    BOOST_REQUIRE_EQUAL(odw.k(), mult*K);

    dlib::alias_tensor odw_alias(1,mult,odw.nr(),odw.nc());
    std::size_t odw_ofs = 0;

    unsigned k = 0;
    for (auto&& inch_alias : channels(in.data)) {
        const dlib::tensor& inch = inch_alias;
        auto ck = extract_con(cdw,k++);
        dlib::resizable_tensor of;
        ck.forward(to_subnet(inch), of);
        check_equal(of, odw_alias(odw,odw_ofs));
        odw_ofs += odw_alias.size();
    }
}

template <long MULT, long RC, int STRIDE = 1>
static void do_test2(const subnet& input) {
    static_assert(MULT > 0 && RC > 0 && STRIDE > 0);

    using condw_type = condw_<HAS_BIAS,MULT,RC,RC,STRIDE,STRIDE>;
    
    const auto num_channels = input.data.k();
    
    // setup random condw and run forward on image
    FILE_LOG(logDETAIL) << "condw: setup";
    condw_type condw;
    condw.setup(input);
    BOOST_REQUIRE_EQUAL(condw.get_layer_params().size(),
                        num_channels * (MULT*RC*RC+MULT));
    {
        // init biases
        tensor& params = condw.get_layer_params();
        auto biases = params.host() + num_channels*MULT*RC*RC;
        for (auto i = num_channels*MULT; i > 0; --i, ++biases)
            *biases = float(i) / float(num_channels*MULT) - 0.5f;
    }

    // serialize / deserialize
    condw_type other1, other2;
    {
        std::stringstream ss;
        serialize(condw, ss);
        deserialize(other1, ss);
        BOOST_CHECK(!other1.get_shared_qfilt());
    }
    {
        std::stringstream ss;
        set_parameter_format(ss, dlibx::quantize(12));
        serialize(condw, ss);
        deserialize(other2, ss);
        BOOST_CHECK(other2.get_shared_qfilt());
    }
    
    FILE_LOG(logDETAIL) << "condw: forward";
    dlib::resizable_tensor out_condw, out_other1, out_other2;
    condw.forward(input, out_condw);
    other1.forward(input, out_other1);
    other2.forward(input, out_other2);

    FILE_LOG(logDETAIL) << "con: forward";
    subnet channel;
    dlib::resizable_tensor out_con;
    out_con.copy_size(out_condw);

    // split image by channels and run each channel forward through full con
    for (unsigned k = 0; k < num_channels; ++k) {
        FILE_LOG(logDETAIL) << "con: channel " << k;

        // extract input channel and specific convolution
        channel.data = extract_channels<1>(input.data,k);
        auto con = extract_con(condw, k);
        
        // forward
        dlib::resizable_tensor out;
        con.forward(channel, out);

        // copy output MULT channels
        auto src = out.host();
        auto src_size = std::size_t(out.k()*out.nr()*out.nc());
        for (auto&& dest : sample_channels<MULT>(out_con, k)) {
            BOOST_REQUIRE_EQUAL(dest.size(), src_size);
            std::copy_n(src, src_size, dest.host());
            src += src_size;
        }
    }

    FILE_LOG(logDETAIL) << "compare";
    for (auto tp : { &out_con, &out_other1, &out_other2 }) {
        const auto err = compare(out_condw, *tp);
        FILE_LOG(logDETAIL) << "errors:\t" << err.front()
                            << '\t' << err[err.size()/2]
                            << '\t' << err[err.size()*95/100]
                            << '\t' << err.back();
        if (tp != &out_other2)
            BOOST_CHECK(err.back() < 1e-6);
        else
            BOOST_CHECK(err.back() < 0.01);
    }
    FILE_LOG(logDETAIL) << "done";
}

template <long MULT, long RC, int STRIDE = 1>
static void do_backward(subnet& sub) {
    static_assert(MULT > 0 && RC > 0 && STRIDE > 0);
    using condw_type = condw_<HAS_BIAS,MULT,RC,RC,STRIDE,STRIDE>;

    const auto num_channels = sub.data.k();
    FILE_LOG(logDETAIL) << "do_backward: " << MULT << ' ' << RC << ' ' << STRIDE << ' ' << num_channels;

    // initialize condw
    condw_type condw;
    condw.setup(sub);
    BOOST_REQUIRE_EQUAL(condw.get_layer_params().size(),
                        num_channels * (MULT*RC*RC+MULT));
    if (condw.get_bias_mode() == HAS_BIAS) {
        // init biases
        tensor& params = condw.get_layer_params();
        auto biases = params.host() + num_channels*MULT*RC*RC;
        for (auto i = num_channels*MULT; i > 0; --i, ++biases)
            *biases = float(i) / float(num_channels*MULT) - 0.5f;
    }

    // forward
    dlib::resizable_tensor out_condw;
    condw.forward(sub, out_condw);

    // initial values
    sub.init_gradient();
    auto con_sub_gradient = sub.gradient;

    dlib::resizable_tensor gradient_input;
    gradient_input.copy_size(out_condw);
    std::uniform_real_distribution<float> ud(-1, 1);
    set_random(gradient_input, ud);
    
    dlib::resizable_tensor params_grad;
    params_grad.copy_size(condw.get_layer_params());
    set_random(params_grad, ud);
    auto sub_params_grad = params_grad;
    
    // backward
    condw.backward(gradient_input, sub, params_grad);

    // compute using con_type
    for (auto k = 0; k < num_channels; ++k) {
        // subnet
        subnet con_sub;
        con_sub.data = extract_channels<1>(sub.data,k);
        con_sub.gradient = extract_channels<1>(con_sub_gradient,k);

        // con
        auto con = extract_con(condw, k);
        
        // forward
        dlib::resizable_tensor out_con;
        con.forward(con_sub, out_con);

        // backward
        const auto con_gradient_input =
            extract_channels<MULT>(gradient_input,k);
        dlib::resizable_tensor con_params_grad;
        {
            const auto pr = extract_con_params(condw, sub_params_grad, k);
            auto& filt = pr.first.get();
            auto& bias = pr.second.get();

            con_params_grad.copy_size(con.get_layer_params());
            assert(con_params_grad.size() == filt.size() + bias.size());
            std::copy_n(bias.host(), bias.size(),
                        std::copy_n(filt.host(), filt.size(),
                                    con_params_grad.host()));
        }

        con.backward(con_gradient_input, con_sub, con_params_grad);

        // check results
        {
            // con_sub.gradient
            const auto condw_gradient = extract_channels<1>(sub.gradient,k);
            const auto err = compare(condw_gradient, con_sub.gradient);
            BOOST_REQUIRE(!err.empty());
            FILE_LOG(err.back() < 1e-10 ? logDETAIL : logWARNING)
                << "data grad errors: " << k
                << '\t' << err.front()
                << '\t' << err[err.size()/2]
                << '\t' << err[err.size()*95/100]
                << '\t' << err.back();
            BOOST_CHECK_EQUAL(err[err.size()/2],0);
            BOOST_CHECK(err.back() < 1e-10);
        }
        {
            //   con_params_grad
            const auto pr = extract_con_params(condw, params_grad, k);
            auto& filt = pr.first.get();
            auto& bias = pr.second.get();

            dlib::resizable_tensor condw_params_grad;
            condw_params_grad.copy_size(con.get_layer_params());
            assert(condw_params_grad.size() == filt.size() + bias.size());
            std::copy_n(bias.host(), bias.size(),
                        std::copy_n(filt.host(), filt.size(),
                                    condw_params_grad.host()));
            const auto err = compare(condw_params_grad, con_params_grad);
            BOOST_REQUIRE(!err.empty());
            FILE_LOG(err.back() < 1e-10 ? logDETAIL : logWARNING)
                << "param grad errors: " << k
                << '\t' << err.front()
                << '\t' << err[err.size()/2]
                << '\t' << err[err.size()*95/100]
                << '\t' << err.back();
            BOOST_CHECK_EQUAL(err[err.size()/2],0);
            BOOST_CHECK(err.back() < 1e-10);
        }
    }
}

static void test_serialize() {
    // check serialized size (with add_layer)
    using input = dlib::input_rgb_image;
    using net_type = dlib::add_layer<condw_<HAS_BIAS,16,7,7,1,1>,input>;
    net_type net;
    setup(net,10,10);
    auto& params = net.layer_details().get_layer_params();
    const auto n = params.size();
    auto p = params.host();
    for (unsigned i = 0; i < n; )
        *p++ = float(++i);
    std::stringstream ss1;
    serialize(net, ss1);
    const auto size1 = ss1.str().size();
    std::stringstream ss2;
    set_parameter_format(ss2, dlibx::quantize(8));
    serialize(net, ss2);
    const auto size2 = ss2.str().size();
    FILE_LOG(logINFO) << "condw parameters: " << n
                      << "  q08: " << size2 << " bytes"
                      << "  float32: " << size1 << " bytes";
    BOOST_CHECK(8*size1 <= 33*n); // size1 must be only slightly more than 4*n
    BOOST_CHECK(8*size2 <= 9*n);  // size2 must be only slightly more than n
    if (0) {
        dlib::serialize("test_condw_ld.bin") << net.layer_details();
        dlib::serialize("test_condw_net.bin") << net;
    }
}

BOOST_AUTO_TEST_CASE(condw_tests) {
    FILE_LOG(logINFO) << "--";
    test_serialize();
    
    if (1) {
        const auto img0 = input<3>(7,11);
        do_test1<1,3,3>(img0);
        do_test1<2,3,5>(img0);
        do_test1<3,4,2>(img0);

        const auto img1 = input<4>(10,12);
        do_test1<1,3,3>(img1);
        do_test1<2,3,5>(img1);
        do_test1<3,4,2>(img1);

        const auto img2 = input<2>(20,13);
        do_test1<1,3,3>(img2);
        do_test1<2,3,5>(img2);
        do_test1<3,4,2>(img2);
    }

    if (1) {
        // multi-thread
        core::context_settings cs;
        cs.min_threads = 2;
        cs.max_threads = 8;
        const auto c = core::context::construct(cs);
        FILE_LOG(logDETAIL) << "context constructed";
        subnet image;
        image.init_real_1(11, 61, 13, 17);
        c->threads().run(
            [&](){
                do_test2<2,3>(image);
                do_test2<3,5,2>(image);
                return 0;
            });
        c->threads().run(
            [&](){
                do_backward<2,3>(image);
                do_backward<3,5,2>(image);
                return 0;
            });
        FILE_LOG(logDETAIL) << "context leave";
    }

    if (1) {
        // single thread
        subnet image;
        image.init_real_1(2, 5, 7, 11);
        do_test2<2,3>(image);
        do_test2<3,5,2>(image);
    }

    if (1) {
        FILE_LOG(logDETAIL) << "condw backward: start";
        subnet image;
        image.init_real_1(2, 5, 7, 11);
        do_backward<2,3>(image);
        do_backward<3,5,2>(image);
        FILE_LOG(logDETAIL) << "condw backward: done";
    }

    FILE_LOG(logINFO) << "condw: done";
}

BOOST_AUTO_TEST_SUITE_END()
