#pragma once

#include "float_constants.hpp"
#include "dnn_traits.hpp"
#include <dlib/dnn/core.h>
#include <json/types.hpp>

namespace dlibx {

    using dlib::tensor;

    template <typename... FUNCS>
    struct lambda_impl {
        // base case (FUNCS is empty)
        constexpr void init() const {}
        constexpr void calc() const {}
        static constexpr bool prep_required = false;
        constexpr void prep(float) const {}
        constexpr float operator()(float x) const { return x; }
        template <typename VISITOR>
        constexpr void visit_tail_first(VISITOR&&) const {}
        static constexpr auto name() { return ""; }
        friend void serialize(const lambda_impl&, std::ostream&) {}
        friend void deserialize(lambda_impl&, std::istream&) {}
    };

    // helper to call func.init() if it's available
    template <typename FUNC, typename = void>
    struct lambda_init_if_available { static constexpr void init(FUNC&) {} };
    template <typename FUNC>
    struct lambda_init_if_available<FUNC, std::enable_if_t<std::is_member_function_pointer<decltype(&FUNC::init)>::value> > {
        static inline void init(FUNC& obj) { obj.init(); }
    };

    // helper to call func.calc() if it's available
    template <typename FUNC, typename = void>
    struct lambda_calc_if_available { static constexpr void calc(FUNC&) {} };
    template <typename FUNC>
    struct lambda_calc_if_available<FUNC, std::enable_if_t<std::is_member_function_pointer<decltype(&FUNC::calc)>::value> > {
        static inline void calc(FUNC& obj) { obj.calc(); }
    };

    // helper to call func.prep() if it's available
    template <typename FUNC, typename = void>
    struct lambda_prep_if_available {
        static constexpr bool available = false;
        static constexpr void prep(FUNC&,float) {}
    };
    template <typename FUNC>
    struct lambda_prep_if_available<FUNC, std::enable_if_t<std::is_member_function_pointer<decltype(&FUNC::prep)>::value> > {
        static constexpr bool available = true;
        static inline void prep(FUNC& obj, float x) { obj.prep(x); }
    };

    template <typename FUNC0, typename... FUNCS>
    struct lambda_impl<FUNC0,FUNCS...> : lambda_impl<FUNCS...> {
        using base = lambda_impl<FUNCS...>;
        FUNC0 fn;
        lambda_impl() = default;
        template <typename... Args>
        lambda_impl(FUNC0 fn, Args&&... args)
            : base(std::forward<Args>(args)...), fn(fn) {}

        inline void init() {
            base::init();
            lambda_init_if_available<FUNC0>::init(fn);
        }
        inline void calc() {
            base::calc();
            lambda_calc_if_available<FUNC0>::calc(fn);
        }

        static constexpr bool prep_required =
            lambda_prep_if_available<FUNC0>::available || base::prep_required;
        inline void prep(float x) {
            base::prep(x);
            lambda_prep_if_available<FUNC0>::prep(fn,x);
        }

        inline float operator()(float x) { return fn(base::operator()(x)); }

        template <typename VISITOR>
        void visit_tail_first(VISITOR&& v) const {
            base::visit_tail_first(v);
            v(fn);
        }

        static auto name() {
            auto s = std::string("_");
            s.append(FUNC0::name());
            s.append(base::name());
            return s;
        }

        friend void serialize(const lambda_impl& item, std::ostream& out) {
            serialize(item.fn, out);
            serialize(static_cast<const base&>(item), out);
        }
        friend void deserialize(lambda_impl& item, std::istream& in) {
            deserialize(item.fn, in);
            deserialize(static_cast<base&>(item), in);
        }
    };
    

    /** \brief Lambda layer to apply arbitrary function per value.
     *
     * Each function class must implement the following:
     *   class func {
     *       float operator()(float) const;
     *       json::object args() const;
     *       static auto name();
     *       friend void serialize(const func&, std::ostream&);
     *       friend void deserialize(func&, std::istream&);
     *   };
     *
     * For a simple function without runtime parameters,
     * serialize() and deserialize() may do nothing,
     * args() returns an empty object, and
     * name() is constexpr.
     *
     * Optional member methods include:
     *       void init();
     *       void prep(float);
     *       void calc();
     * If present, init() is called before processing each sample,
     * then prep() is called with every value, then calc() is called,
     * and finally operator() is called to transform each value.
     * This enables the creation of lambdas that normalize or whiten
     * each sample in some way.  
     * Note that when combining multiple functions into a single lambda,
     * every function sees the same values during the prep() stage.
     * So this type of function should either be the right most function
     * or the only function of the lambda.
     *
     * If multiple functions are specified, then for the forward direction
     * they are applied from right to left.
     * For example, lambda<sqrt,mult9> does y = std::sqrt(9*x).
     */
    template <typename FUNC0, typename... FUNCS>
    class lambda_ {
    public:
        using impl_type = lambda_impl<FUNC0, FUNCS...>;
        impl_type impl;

        lambda_() = default;

        template <typename... Args>
        lambda_(FUNC0 fn, Args&&... args)
            : impl(std::move(fn), std::forward<Args>(args)...) {}

        template <typename SUBNET>
        void setup (const SUBNET&) {
        }

        void forward_inplace(const tensor& input, tensor& output) {
            auto src = input.host();
            auto dest = output.host_write_only();
            const auto sample_size =
                std::size_t(input.nc()*input.nr()*input.k());
            const auto sample_n8 = sample_size / 8;
            const auto sample_extra = sample_size - 8*sample_n8;
            for (auto n = input.num_samples(); n > 0; --n) {
                impl.init();
                if (impl_type::prep_required) {
                    auto s = src;
                    for (auto n8 = sample_n8; n8 > 0; --n8) {
                        // possible SIMD use on up to 8 values at a time
                        for (auto i = 8;  i > 0; --i, ++s)
                            impl.prep(*s);
                    }
                    for (auto extra = sample_extra; extra > 0; --extra, ++s)
                        impl.prep(*s);
                }
                impl.calc();
                for (auto n8 = sample_n8; n8 > 0; --n8) {
                    // possible SIMD use on up to 8 values at a time
                    for (auto i = 8;  i > 0; --i, ++dest, ++src)
                        *dest = impl(*src);
                }
                for (auto extra = sample_extra; extra > 0; --extra,
                         ++dest, ++src)
                    *dest = impl(*src);
            }
        } 

        void backward_inplace(
            const tensor&, const tensor& sub, tensor&, tensor&) {
            if (&sub != &input_layer(sub))
                throw std::runtime_error(
                    "lambda::backward() not implemented");
        }

        inline auto map_input_to_output(const dlib::dpoint& p) const {
            return p;
        }
        inline auto map_output_to_input(const dlib::dpoint& p) const {
            return p;
        }

        const tensor& get_layer_params() const { return params; }
        tensor& get_layer_params() { return params; }

        static inline auto name() {
            return std::string("lambda") + impl_type::name();
        }
        
        friend void serialize(const lambda_& item, std::ostream& out) {
            const auto version = name();
            dlib::serialize(version, out);
            serialize(item.impl, out);
        }

        friend void deserialize(lambda_& item, std::istream& in) {
            std::string version;
            dlib::deserialize(version, in);
            if (version != name())
                throw dlib::serialization_error("Unexpected version '"+version+"' found while deserializing dlibx::lambda_.");
            deserialize(item.impl, in);
        }

        friend std::ostream& operator<<(std::ostream& out, const lambda_&) {
            out << name();
            return out;
        }

        friend void to_xml(const lambda_&, std::ostream& out) {
            out << '<' << name() << "/>\n";
        }

    private:
        dlib::resizable_tensor params;
    };

    template <typename F1, typename SUBNET>
    using lambda = dlib::add_layer<lambda_<F1>, SUBNET>;
    template <typename F1, typename F2, typename SUBNET>
    using lambda2 = dlib::add_layer<lambda_<F1,F2>, SUBNET>;
    template <typename F1, typename F2, typename F3, typename SUBNET>
    using lambda3 = dlib::add_layer<lambda_<F1,F2,F3>, SUBNET>;


    // square
    struct fn_power2 {
        static constexpr auto name() { return "power2"; }
        inline auto operator()(float x) const { return x*x; }
        json::object args() const { return {}; }
        friend void serialize(const fn_power2&, std::ostream&) {}
        friend void deserialize(fn_power2&, std::istream&) {}
    };
    template <typename SUBNET>
    using lambda_power2 = lambda<fn_power2, SUBNET>;


    // square root
    struct fn_sqrt {
        static constexpr auto name() { return "sqrt"; }
        inline auto operator()(float x) const { return std::sqrt(x); }
        json::object args() const { return {}; }
        friend void serialize(const fn_sqrt&, std::ostream&) {}
        friend void deserialize(fn_sqrt&, std::istream&) {}
    };


    // add a constexpr integer
    template<long N>
    struct fn_add {
        static auto name() { return "add" + std::to_string(N); }
        inline auto operator()(float x) const { return x+N; }
        json::object args() const { return {}; }
        friend void serialize(const fn_add&, std::ostream&) {}
        friend void deserialize(fn_add&, std::istream&) {}
    };


    // subtract a constexpr integer
    template<long N>
    struct fn_sub {
        static auto name() { return "sub" + std::to_string(N); }
        inline auto operator()(float x) const { return x-N; }
        json::object args() const { return {}; }
        friend void serialize(const fn_sub&, std::ostream&) {}
        friend void deserialize(fn_sub&, std::istream&) {}
    };


    // multiply by constexpr integer
    template<long N>
    struct fn_mult {
        static auto name() { return "mult" + std::to_string(N); }
        inline auto operator()(float x) const { return N*x; }
        json::object args() const { return {}; }
        friend void serialize(const fn_mult&, std::ostream&) {}
        friend void deserialize(fn_mult&, std::istream&) {}
    };


    // multiply by runtime float
    // note: dlib has a multiply_ layer that does this same operation
    template <typename INIT = float_one>
    struct fn_scale {
        float coeff;
        fn_scale(float coeff = INIT{}) : coeff(coeff) {}
        inline auto operator()(float x) const { return coeff * x; }
        json::object args() const { return { {"scale",coeff} }; }
        static constexpr auto name() { return "scale"; }
        friend void serialize(const fn_scale& item, std::ostream& out) {
            dlib::serialize(item.coeff, out);
        }
        friend void deserialize(fn_scale& item, std::istream& in) {
            dlib::deserialize(item.coeff, in);
        }
    };
    template <typename INIT, typename SUBNET>
    using lambda_scale = dlibx::lambda<fn_scale<INIT>, SUBNET>;


    // per-sample normalize to gaussian
    template <typename MEAN = float_zero, typename STDDEV = float_one>
    struct fn_gauss {
        float mean, stddev;
        fn_gauss(float mean = MEAN{}, float stddev = STDDEV{})
            : mean(mean), stddev(stddev) {}

        float ofs, coeff;
        std::size_t count;
        inline void init() {
            ofs = coeff = 0;
            count = 0;
        }
        inline void prep(float x) {
            ofs += x, coeff += x*x, ++count;
        }
        inline void calc() {
            ofs /= float(count);
            coeff = std::sqrt(coeff/float(count) - ofs*ofs);
            coeff = coeff < 1e-3 ? 1000 : 1 / coeff;
            // y = mean + stddev * coeff * (x - ofs)
            coeff *= stddev;
            ofs = mean - coeff * ofs;
        }
        inline auto operator()(float x) const {
            return ofs + coeff * x;
        }

        json::object args() const {
            return { {"mean",mean}, {"stddev",stddev} };
        }
        static constexpr auto name() { return "gauss"; }
        friend void serialize(const fn_gauss& item, std::ostream& out) {
            dlib::serialize(item.mean, out);
            dlib::serialize(item.stddev, out);
        }
        friend void deserialize(fn_gauss& item, std::istream& in) {
            dlib::deserialize(item.mean, in);
            dlib::deserialize(item.stddev, in);
        }
    };
    template <typename MEAN, typename STDDEV, typename SUBNET>
    using lambda_gauss = dlibx::lambda<fn_gauss<MEAN,STDDEV>, SUBNET>;
    template <typename SUBNET>
    using lambda_stdnorm = dlibx::lambda<fn_gauss<>, SUBNET>;


    // Inter-Channel Local Response Normalization (SpatialCrossMapLRN)
    template <long N>
    struct fn_lrn {
        static auto name() {
            return "lrn" + std::to_string(N) + "default";
        }
        static constexpr auto alpha = 0.0001f;
        static constexpr auto beta = 0.75f;
        static constexpr auto k = 1;
        inline auto operator()(float x) const {
            return std::pow(k + (alpha/N)*x, -beta);
        }
        json::object args() const { return {}; }  // todo
        friend void serialize(const fn_lrn&, std::ostream&) {}  // todo
        friend void deserialize(fn_lrn&, std::istream&) {}  // todo
    };
    template <long N, typename SUBNET>
    using lambda_lrn = lambda<fn_lrn<N>, SUBNET>;
}
