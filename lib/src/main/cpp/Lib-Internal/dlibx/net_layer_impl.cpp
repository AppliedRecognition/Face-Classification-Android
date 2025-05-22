
#include "net_layer_impl.hpp"

#include <applog/core.hpp>


using namespace dlibx;


/**************** class net::layer (deserialize) ****************/

template <typename DETAIL>
static std::unique_ptr<net::layer>
deserialize_input(std::istream& in) {
    auto p = std::make_unique<net::layer_input<DETAIL> >();
    deserialize(p->detail, in);
    return p;
}

template <typename DETAIL>
static std::unique_ptr<net::layer>
deserialize_generic(std::istream& in) {
    auto p = std::make_unique<net::layer_generic<DETAIL> >();
    deserialize(p->detail, in);
    return p;
}

template <long NR, long NC, int SY, int SX, int PY, int PX>
static std::unique_ptr<net::layer>
deserialize_con(std::istream& in) {
    return deserialize_generic<dlibx::lm_con_<1,NR,NC,SY,SX,PY,PX> >(in);
}

template <long NR, long NC, int DY, int DX, int SY, int SX, int PY, int PX>
static std::unique_ptr<net::layer>
deserialize_dcon(std::istream& in) {
    return deserialize_generic<dlibx::lm_con_<1,NR,NC,SY,SX,PY,PX,DY,DX> >(in);
}

template <std::size_t N>
static std::unique_ptr<net::layer>
deserialize_concat(std::istream&) {
    return std::make_unique<net::layer_concat<N> >();
}

template <typename LAYER>
static std::unique_ptr<net::layer>
deserialize_empty(std::istream&) {
    return std::make_unique<LAYER>();
}

using factory_ptr = std::unique_ptr<net::layer>(*)(std::istream&);
static auto make_table() {
    std::vector<std::pair<std::string_view, factory_ptr> > t;

    // input layers
    t.emplace_back("input_rgb_image",  &deserialize_input<dlib::input_rgb_image>);
    t.emplace_back("input_rgb_image_112", &deserialize_input<dlib::input_rgb_image>);
    t.emplace_back("input_rgb_image_150", &deserialize_input<dlib::input_rgb_image>);
    t.emplace_back("input_rgb_image_224", &deserialize_input<dlib::input_rgb_image>);
    t.emplace_back("input_matrix_rgb", &deserialize_input<dlib::input<dlib::matrix<dlib::rgb_pixel> > >);
    t.emplace_back("input_matrix_u8",  &deserialize_input<dlib::input<dlib::matrix<unsigned char> > >);
    t.emplace_back("input_matrix_float", &deserialize_input<dlib::input<dlib::matrix<float> > >);
    //t.emplace_back("input_array2d_rgb", &deserialize_input<dlib::input<dlib::array2d<dlib::rgb_pixel> > >);
    //t.emplace_back("input_array2d_u8", &deserialize_input<dlib::input<dlib::array2d<unsigned char> > >);
    t.emplace_back("input_image_u8",  &deserialize_input<dlibx::input_generic_image<dlib::matrix<unsigned char> > >);
    t.emplace_back("input_image_rgb", &deserialize_input<dlibx::input_generic_image<dlib::matrix<dlib::rgb_pixel> > >);
    t.emplace_back("input_image_rgba", &deserialize_input<dlibx::input_generic_image<dlib::matrix<dlib::rgb_alpha_pixel> > >);

    // padding
    t.emplace_back("padding_0_1", &deserialize_generic<dlibx::padding_<0,1> >);
    t.emplace_back("padding_1",   &deserialize_generic<dlibx::padding_<1> >);
    t.emplace_back("padding_2",   &deserialize_generic<dlibx::padding_<2> >);
    t.emplace_back("padding_3_4", &deserialize_generic<dlibx::padding_<3,4> >);
    t.emplace_back("padding_4",   &deserialize_generic<dlibx::padding_<4> >);
    
    // convolution (num_filters is runtime dynamic so always 1 here)
    t.emplace_back("con_1_1",         &deserialize_con<1,1,1,1,0,0>);
    t.emplace_back("con_1_1_2_2",     &deserialize_con<1,1,2,2,0,0>);
    t.emplace_back("con_2_2_1_1",     &deserialize_con<2,2,1,1,0,0>);
    t.emplace_back("con_2_2_2_2",     &deserialize_con<2,2,2,2,0,0>);
    t.emplace_back("con_1_3_1_1_0_1", &deserialize_con<1,3,1,1,0,1>);
    t.emplace_back("con_3_1_1_1_1_0", &deserialize_con<3,1,1,1,1,0>);
    t.emplace_back("con_3_3_1_1",     &deserialize_con<3,3,1,1,0,0>);
    t.emplace_back("con_3_3_1_1_1_1", &deserialize_con<3,3,1,1,1,1>);
    t.emplace_back("con_3_3_2_2",     &deserialize_con<3,3,2,2,0,0>);
    t.emplace_back("con_3_3_2_2_1_1", &deserialize_con<3,3,2,2,1,1>);
    t.emplace_back("con_5_5_1_1_2_2", &deserialize_con<5,5,1,1,2,2>);
    t.emplace_back("con_5_5_2_2",     &deserialize_con<5,5,2,2,0,0>);
    t.emplace_back("con_5_5_2_2_2_2", &deserialize_con<5,5,2,2,2,2>);
    t.emplace_back("con_1_7_1_1_0_3", &deserialize_con<1,7,1,1,0,3>);
    t.emplace_back("con_7_1_1_1_3_0", &deserialize_con<7,1,1,1,3,0>);
    t.emplace_back("con_7_7_2_2",     &deserialize_con<7,7,2,2,0,0>);
    t.emplace_back("con_7_7_2_2_3_3", &deserialize_con<7,7,2,2,3,3>);
    t.emplace_back("con_all", &deserialize_generic<dlib::con_<1,0,0,1,1> >);

    // dilated convolution (num_filters is runtime dynamic so always 1 here)
    t.emplace_back("con_3d2_3d2_1_1_2_2", &deserialize_dcon<3,3,2,2,1,1,2,2>);
    t.emplace_back("con_3d3_3d3_1_1_3_3", &deserialize_dcon<3,3,3,3,1,1,3,3>);
    t.emplace_back("con_3d5_3d5_1_1_5_5", &deserialize_dcon<3,3,5,5,1,1,5,5>);

    // depth-wise convolution (bias mode and multiplier are runtime dynamic)
    t.emplace_back("cdw_3_3_1_1_1_1", &deserialize_generic<dlibx::condw_<HAS_BIAS,1,3,3,1,1,1,1> >);
    t.emplace_back("cdw_7_7_1_1",     &deserialize_generic<dlibx::condw_<HAS_BIAS,1,7,7,1,1,0,0> >);
    t.emplace_back("cdw_3_3_2_2",     &deserialize_generic<dlibx::condw_<HAS_BIAS,1,3,3,2,2,0,0> >);
    t.emplace_back("cdw_3_3_2_2_1_1", &deserialize_generic<dlibx::condw_<HAS_BIAS,1,3,3,2,2,1,1> >);

    // average pool
    t.emplace_back("avg_pool_2_2_2_2", &deserialize_generic<dlib::avg_pool_<2,2,2,2,0,0> >);
    t.emplace_back("avg_pool_3_3_2_2", &deserialize_generic<dlib::avg_pool_<3,3,2,2,0,0> >);
    t.emplace_back("avg_pool_3_3_3_3", &deserialize_generic<dlib::avg_pool_<3,3,3,3,0,0> >);
    t.emplace_back("avg_pool_all", &deserialize_generic<dlib::avg_pool_<0,0,1,1> >);

    // max pool
    t.emplace_back("max_pool_2_2_2_2",     &deserialize_generic<dlib::max_pool_<2,2,2,2,0,0> >);
    t.emplace_back("max_pool_3_3_2_2",     &deserialize_generic<dlib::max_pool_<3,3,2,2,0,0> >);
    t.emplace_back("max_pool_3_3_2_2_1_1", &deserialize_generic<dlib::max_pool_<3,3,2,2,1,1> >);
    t.emplace_back("max_pool_all", &deserialize_generic<dlib::max_pool_<0,0,1,1> >);

    // upsample
    t.emplace_back("upsample_2", &deserialize_generic<dlib::upsample_<2,2> >);

    // fc (num_outputs is runtime dynamic)
    t.emplace_back("fc+bias", &deserialize_generic<dlibx::fc_dynamic_<1,HAS_BIAS> >);
    t.emplace_back("fcnb",    &deserialize_generic<dlibx::fc_dynamic_<1,HAS_BIAS> >); // HAS_BIAS is ok here

    // sum neighbours
    t.emplace_back("sum_neighbours_5", &deserialize_generic<dlibx::sum_neighbours_<5> >);

    // lambda
    t.emplace_back("lambda_sub1_mult2", &deserialize_generic<dlibx::lambda_<dlibx::fn_sub<1>,dlibx::fn_mult<2> > >);
    t.emplace_back("lambda_power2", &deserialize_generic<dlibx::lambda_<dlibx::fn_power2> >);
    t.emplace_back("lambda_sqrt", &deserialize_generic<dlibx::lambda_<dlibx::fn_sqrt> >);
    t.emplace_back("lambda_sqrt_mult9", &deserialize_generic<dlibx::lambda_<dlibx::fn_sqrt,dlibx::fn_mult<9> > >);
    t.emplace_back("lambda_scale", &deserialize_generic<dlibx::lambda_<dlibx::fn_scale<> > >);
    t.emplace_back("lambda_gauss", &deserialize_generic<dlibx::lambda_<dlibx::fn_gauss<> > >);
    t.emplace_back("lambda_lrn5default", &deserialize_generic<dlibx::lambda_<dlibx::fn_lrn<5> > >);

    // extract
    t.emplace_back("extract", &deserialize_generic<dlibx::extract_>);
    t.emplace_back("extract_1024", &deserialize_generic<dlib::extract_<0,1024,1,1> >);

    // classes without args
    t.emplace_back("sig",     &deserialize_generic<dlib::sig_>);
    t.emplace_back("softmax", &deserialize_generic<dlib::softmax_>);
    t.emplace_back("relu",    &deserialize_generic<dlib::relu_>);
    t.emplace_back("prelu",   &deserialize_generic<dlibx::prelu_>);
    t.emplace_back("dropout", &deserialize_generic<dlib::dropout_>);
    t.emplace_back("l2norm",  &deserialize_generic<dlib::l2normalize_>);
    t.emplace_back("affine",  &deserialize_generic<dlib::affine_>);
    t.emplace_back("multiply",&deserialize_generic<dlib::multiply_>);
    t.emplace_back("bncon",   &deserialize_generic<dlib::bn_<dlib::CONV_MODE> >);
    t.emplace_back("bnfc",    &deserialize_generic<dlib::bn_<dlib::FC_MODE> >);
    t.emplace_back("resize",  &deserialize_generic<dlibx::resize_>);
    t.emplace_back("transpose", &deserialize_generic<dlibx::transpose_>);

    // concat
    t.emplace_back("concat_2", &deserialize_concat<2>);
    t.emplace_back("concat_3", &deserialize_concat<3>);
    t.emplace_back("concat_4", &deserialize_concat<4>);
    
    // single tag id
    t.emplace_back("add_cropped", &deserialize_empty<net::layer_add_cropped>);
    t.emplace_back("add_prev",  &deserialize_empty<net::layer_add_prev>);
    t.emplace_back("mult_prev", &deserialize_empty<net::layer_mult_prev>);
    
    std::sort(t.begin(), t.end());
    return t;
}

std::unique_ptr<net::layer>
net::layer::deserialize(std::istream& in) {
    static const auto table = make_table();

    int version = 0;
    dlib::deserialize(version, in);
    if (version != 1)
        throw dlib::serialization_error("incorrect version number when deserializing net::layer");

    std::string name;
    dlib::deserialize(name, in);
    std::vector<std::string> inbound;
    dlib::deserialize(inbound, in);
    
    std::string code;
    dlib::deserialize(code, in);
    const auto key = decltype(table)::value_type(code,nullptr);
    const auto it = std::lower_bound(table.begin(), table.end(), key);
    if (it != table.end() && it->first == code) {
        std::unique_ptr<net::layer> p;
        try {
            p = it->second(in);
        }
        catch (const std::exception& e) {
            FILE_LOG(logERROR) << "net::layer: while deserializing '"
                               << code << "': " << e.what();
            throw;
        }
        p->name = move(name);
        p->inbound = move(inbound);
        return p;
    }
    
    FILE_LOG(logERROR) << "layer with code '" << code
                       << "' cannot be constructed";
    throw dlib::serialization_error("unknown layer type when deserializing net::layer");
}

