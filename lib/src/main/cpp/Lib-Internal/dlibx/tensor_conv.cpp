
#include "tensor_conv.hpp"

#include <core/context.hpp>
#include <core/thread_data.hpp>
#include <core/job_queue.hpp>

#include <applog/core.hpp>

#include <numeric>
#include <optional>

using namespace dlibx;

namespace {
    struct gradient_for_bias {
        float const* const src;
        float* const dest;
        std::size_t channel_size, sample_size;
        unsigned num_channels, num_samples;

        std::atomic<unsigned> next;

        gradient_for_bias(const dlib::tensor& input, dlib::tensor& grad)
            : src(input.host()),
              dest(grad.host_write_only()),
              channel_size(std::size_t(input.nr() * input.nc())),
              sample_size(std::size_t(input.k()) * channel_size),
              num_channels(unsigned(input.k())),
              num_samples(unsigned(input.num_samples())),
              next{0} {
            DLIB_CASSERT(grad.k() > 0 &&
                         grad.size() == std::size_t(grad.k()) &&
                         input.k() == grad.k());
        }
    };

    struct gradient_for_bias_job {
        gradient_for_bias& s;

        auto operator()() {
            for (;;) {
                const auto k = s.next.fetch_add(1,std::memory_order_relaxed);
                if (k >= s.num_channels) break;
                auto sample = s.src + k * s.channel_size;
                auto& d = s.dest[k];
                d = 0;
                for (auto n = s.num_samples; n > 0; --n,
                         sample += s.sample_size)
                    d = std::accumulate(sample, sample + s.channel_size, d);
            }
            return 0;
        }
    };

    struct gradient_for_data_job {
        dlib::tt::tensor_conv& conv;

        const dlib::tensor& filters;
        const dlib::tensor& input;
        const dlib::alias_tensor input_sample;
        dlib::tensor& output;
        const dlib::alias_tensor output_sample;

        std::atomic<unsigned>& next;

        gradient_for_data_job(dlib::tt::tensor_conv& conv,
                              const dlib::tensor& filters,
                              const dlib::tensor& input,
                              dlib::tensor& output,
                              std::atomic<unsigned>& next)
            : conv(conv),
              filters(filters),
              input(input),
              input_sample(1,input.k(),input.nr(),input.nc()),
              output(output),
              output_sample(1,output.k(),output.nr(),output.nc()),
              next(next) {
            DLIB_CASSERT(input.num_samples() == output.num_samples());
        }

        inline auto operator()() {
            for (;;) {
                const auto n = next.fetch_add(1,std::memory_order_relaxed);
                if (n >= input.num_samples()) break;
                auto in  =  input_sample( input, n * input_sample.size());
                auto out = output_sample(output, n * output_sample.size());
                conv.get_gradient_for_data(true, in, filters, out);
            }
            return 0;
        }
    };

    struct gradient_for_filters_job {
        dlib::tt::tensor_conv& conv;

        const dlib::tensor& data;
        const dlib::alias_tensor data_sample;
        const dlib::tensor& input;
        const dlib::alias_tensor input_sample;
        const dlib::tensor& output;  // used for dimensions only

        std::atomic<unsigned>& next;

        gradient_for_filters_job(dlib::tt::tensor_conv& conv,
                                 const dlib::tensor& data,
                                 const dlib::tensor& input,
                                 const dlib::tensor& output,
                                 std::atomic<unsigned>& next)
            : conv(conv),
              data(data),
              data_sample(1,data.k(),data.nr(),data.nc()),
              input(input),
              input_sample(1,input.k(),input.nr(),input.nc()),
              output(output),
              next(next) {
            DLIB_CASSERT(data.num_samples() == input.num_samples());
        }

        inline void add_to(dlib::tensor& out) {
            for (;;) {
                const auto n = next.fetch_add(1,std::memory_order_relaxed);
                if (n >= input.num_samples()) break;
                auto in = input_sample(input, n * input_sample.size());
                auto d = data_sample(data, n * data_sample.size());
                conv.get_gradient_for_filters(true, in, d, out);
            }
        }

        inline auto operator()() {
            dlib::resizable_tensor out;
            out.copy_size(output);
            out = 0.0f;
            add_to(out);
            return out;
        }
    };
}

void tensor_conv::backward_conv(const dlib::tensor& filters,
                                const dlib::tensor& input,
                                dlib::tensor& output,
                                const dlib::tensor* data,
                                dlib::tensor* filters_grad,
                                dlib::tensor* bias_grad) {

    const auto context = core::job_context::this_context();
    const auto nthreads = context ? context->num_threads() : 0;
    if (nthreads > 0) {
        FILE_LOG(logTRACE) << "conv backward: " << (nthreads+1) << " threads";

        std::atomic<unsigned> fgnext{0};
        std::vector<core::job_function<gradient_for_filters_job> > fgjobs;
        if (data && filters_grad) {
            fgjobs.reserve(nthreads);
            for (auto n = nthreads; n > 0; --n)
                context->submit(
                    fgjobs.emplace_back(*this, *data, input,
                                        *filters_grad, fgnext));
        }

        std::atomic<unsigned> dgnext{0};
        std::vector<core::job_function<gradient_for_data_job> > dgjobs;
        dgjobs.reserve(nthreads);
        for (auto n = nthreads; n > 0; --n)
            context->submit(
                dgjobs.emplace_back(*this, filters, input, output, dgnext));

        std::optional<gradient_for_bias> bg;
        std::vector<core::job_function<gradient_for_bias_job> > biasjobs;
        if (bias_grad) {
            bg.emplace(input, *bias_grad);
            biasjobs.reserve(nthreads);
            for (auto n = nthreads; n > 0; --n)
                context->submit(biasjobs.emplace_back(*bg));
        }

        if (data && filters_grad) {
            *filters_grad = 0.0f;
            gradient_for_filters_job fg{
                *this, *data, input, *filters_grad, fgnext};
            fg.add_to(*filters_grad); // this thread
            context->wait_for_all(fgjobs.begin(), fgjobs.end());
            for (auto& job : fgjobs)
                *filters_grad += mat(*job);
        }

        // this thread
        gradient_for_data_job{*this, filters, input, output, dgnext}();
        if (bg) gradient_for_bias_job{*bg}();

        context->wait_for_all(dgjobs.begin(), dgjobs.end());
        for (auto& job : dgjobs) *job; // re-throw any exceptions
        context->wait_for_all(biasjobs.begin(), biasjobs.end());
        for (auto& job : biasjobs) *job; // re-throw any exceptions

        FILE_LOG(logTRACE) << "conv backward: done";
    }

    else { // single thread
        FILE_LOG(logTRACE) << "conv backward: single thread";
        get_gradient_for_data(true, input, filters, output);
        if (data && filters_grad) {
            get_gradient_for_filters(false, input, *data, *filters_grad);
            if (bias_grad) {
                gradient_for_bias bg(input, *bias_grad);
                gradient_for_bias_job{bg}();
            }
        }
    }
}

namespace {
    struct gradient_for_dw {
        dlib::tt::tensor_conv& conv;

        const long multiplier;
        const dlib::tensor& filters;

        const dlib::tensor& input;
        const std::size_t input_size;

        dlib::tensor& output;
        const std::size_t output_size;

        const dlib::tensor* data;
        dlib::tensor* filters_grad;

        std::atomic<unsigned> next;

        gradient_for_dw(dlib::tt::tensor_conv& conv,
                        const dlib::tensor& filters,
                        const dlib::tensor& input,
                        dlib::tensor& output,
                        const dlib::tensor* data,
                        dlib::tensor* filters_grad)
            : conv(conv),
              multiplier(long(input.k() / output.k())),
              filters(filters),
              input(input),
              input_size(input.size() / unsigned(input.num_samples())),
              output(output),
              output_size(output.size() / unsigned(output.num_samples())),
              data(data),
              filters_grad(filters_grad),
              next{0} {

            DLIB_CASSERT(output.num_samples() == input.num_samples());
            DLIB_CASSERT(input.k() == multiplier * output.k());
            DLIB_CASSERT(filters.num_samples() == input.k() &&
                         filters.k() == 1);

            if (!filters_grad)
                data = nullptr;
            else if (data) {
                DLIB_CASSERT(data->num_samples() == output.num_samples() &&
                             data->k() == output.k() &&
                             data->nr() == output.nr() &&
                             data->nc() == output.nc());
                *filters_grad = 0.0f;
            }
        }
    };

    struct gradient_for_dw_job {
        gradient_for_dw& s;

        // alias_tensor is not thread-safe so we need per-thread instances
        const dlib::alias_tensor channel_filters;
        const dlib::alias_tensor input_channels;
        const dlib::alias_tensor output_channel;

        gradient_for_dw_job(gradient_for_dw& s)
            : s(s),
              channel_filters(s.multiplier,1,s.filters.nr(),s.filters.nc()),
              input_channels(1,s.multiplier,s.input.nr(),s.input.nc()),
              output_channel(1,1,s.output.nr(),s.output.nc()) {
        }

        auto operator()() {
            for (;;) {
                const auto k =
                    s.next.fetch_add(1,std::memory_order_relaxed);
                if (k >= s.output.k()) break;
                const auto fofs = k * channel_filters.size();
                auto filt = channel_filters(s.filters, fofs);
                auto iofs = k * input_channels.size();
                auto oofs = k * output_channel.size();
                std::optional<dlib::alias_tensor_instance> fgout;
                if (s.data)
                    fgout.emplace(channel_filters(*s.filters_grad, fofs));
                for (auto n = s.output.num_samples(); n > 0; --n,
                         iofs += s.input_size, oofs += s.output_size) {
                    auto in  = input_channels(s.input, iofs);
                    auto out = output_channel(s.output, oofs);
                    s.conv.get_gradient_for_data(true, in, filt, out);
                    if (fgout)
                        s.conv.get_gradient_for_filters(
                            true, in, output_channel(*s.data,oofs), *fgout);
                }
            }
            return 0;
        }
    };
}

void tensor_conv::backward_dw(const dlib::tensor& filters,
                              const dlib::tensor& input,
                              dlib::tensor& output,
                              const dlib::tensor* data,
                              dlib::tensor* filters_grad,
                              dlib::tensor* bias_grad) {
    gradient_for_dw dw(*this, filters, input, output, data, filters_grad);

    const auto context = core::job_context::this_context();
    const auto nthreads = context ? context->num_threads() : 0;
    if (nthreads > 0) {
        FILE_LOG(logTRACE) << "condw backward: " << (nthreads+1) << " threads";

        std::vector<core::job_function<gradient_for_dw_job> > dwjobs;
        dwjobs.reserve(nthreads);
        for (auto n = nthreads; n > 0; --n)
            context->submit(dwjobs.emplace_back(dw));

        std::optional<gradient_for_bias> bg;
        std::vector<core::job_function<gradient_for_bias_job> > biasjobs;
        biasjobs.reserve(nthreads);
        if (bias_grad) {
            bg.emplace(input, *bias_grad);
            for (auto n = nthreads; n > 0; --n)
                context->submit(biasjobs.emplace_back(*bg));
        }

        gradient_for_dw_job{dw}(); // this thread
        if (bg) gradient_for_bias_job{*bg}(); // this thread

        FILE_LOG(logTRACE) << "condw backward: wait";
        context->wait_for_all(dwjobs.begin(), dwjobs.end());
        for (auto& job : dwjobs) *job; // re-throw any exceptions
        context->wait_for_all(biasjobs.begin(), biasjobs.end());
        for (auto& job : biasjobs) *job; // re-throw any exceptions

        FILE_LOG(logTRACE) << "condw backward: done";
    }

    else { // single thread
        FILE_LOG(logTRACE) << "condw backward: single thread";
        gradient_for_dw_job{dw}();
        if (bias_grad) {
            gradient_for_bias bg(input, *bias_grad);
            gradient_for_bias_job{bg}();
        }
    }
}
