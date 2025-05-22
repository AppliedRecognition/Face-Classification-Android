#pragma once

#include <dlib/dnn/core.h>
#include <core/context.hpp>
#include <core/job_queue.hpp>
#include <core/thread_data.hpp>

namespace dlibx {
    /** \brief Wrapper for loss layer to execute methods through job_queue.
     */
    template <typename NET>
    struct loss_wrapper : NET {
        core::job_queue* queue;
        std::function<void()> update_parameters_hook;

        template <typename... Args>
        loss_wrapper(core::job_queue& queue, Args&&... args)
            : NET(std::forward<Args>(args)...), queue(&queue) {}

        template <typename... Args>
        inline auto compute_loss(Args&&... args) {
            const core::job::external_job<core::thread_data> thread(*queue);
            return NET::compute_loss(std::forward<Args>(args)...);
        }

        template <typename... Args>
        inline auto compute_parameter_gradients(Args&&... args) {
            const core::job::external_job<core::thread_data> thread(*queue);
            return NET::compute_parameter_gradients(std::forward<Args>(args)...);
        }

        template <typename... Args>
        inline auto update_parameters(Args&&... args) {
            // hook here to average params_grad for multi-core training
            if (update_parameters_hook) update_parameters_hook();
            return NET::update_parameters(std::forward<Args>(args)...);
        }
    };
}
namespace dlib {
    template <typename T>
    struct is_loss_layer_type<dlibx::loss_wrapper<T> > : std::true_type {};
}
