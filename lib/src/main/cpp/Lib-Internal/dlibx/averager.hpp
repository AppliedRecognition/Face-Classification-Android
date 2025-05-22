#pragma once

#include <dlib/dnn/core.h>
#include <dlibx/tensor.hpp>

#include <atomic>
#include <mutex>
#include <condition_variable>

namespace dlibx {
    struct averager_layer_ref {
        virtual ~averager_layer_ref() = default;
        virtual dlib::tensor& get_layer_params() = 0;
        virtual dlib::tensor& get_parameter_gradient() = 0;
    };

    template <typename T, typename = void>
    struct averager_layer_ref_t;

    template <typename T>
    struct averager_layer_ref_t<
        T, std::enable_if_t<dlib::is_add_layer<T>::value> >
        : averager_layer_ref {
        static constexpr auto has_params = true;
        T& layer;
        averager_layer_ref_t(T& t) : layer(t) {}
        dlib::tensor& get_layer_params() override {
            return layer.layer_details().get_layer_params();
        }
        dlib::tensor& get_parameter_gradient() override {
            return layer.get_parameter_gradient();
        }
    };

    template <typename T>
    struct averager_layer_ref_t<
        T, std::enable_if_t<!dlib::is_add_layer<T>::value> >
        : averager_layer_ref {
        static constexpr auto has_params = false;
        averager_layer_ref_t(T&) { throw std::logic_error("invalid layer"); }
        dlib::tensor& get_layer_params() override {
            throw std::logic_error("invalid layer"); }
        dlib::tensor& get_parameter_gradient() override {
            throw std::logic_error("invalid layer"); }
    };

    /** \brief Compute average parameter gradient for group of trainers.
     *
     * This object is used to enable multi-core training with a group
     * of separate trainer objects each receiving distinct minibatches
     * and running in seperate threads.
     *
     * Before updating parameters the parameter gradients from each trainer
     * are averaged.  This results in an identical update being made to
     * each trainer.  Every so often the parameters from the first trainer
     * are copied to all the others to ensure they don't drift too far apart.
     */
    template <typename NET>
    class averager {
        averager(averager&&) = delete;
        averager(const averager&) = delete;
        averager& operator=(averager&&) = delete;
        averager& operator=(const averager&) = delete;

        std::mutex mux;
        std::condition_variable lobby;
        unsigned enter_count = 0, leave_count = 0;

        std::vector<NET*> nets;
        unsigned sync_count = 0; ///< for periodic resync of params

        using layer_ref = averager_layer_ref;

        // pointers to each computational layer with parameters in each net
        std::vector<std::vector<std::unique_ptr<layer_ref> > > layers;

        void find_layers() {
            assert(layers.empty());
            visit_layers(
                *nets.front(), [&](auto& t) {
                    using T = std::decay_t<decltype(t)>;
                    using R = averager_layer_ref_t<T>;
                    if (R::has_params) {
                        auto ptr = std::make_unique<R>(t);
                        if (0 < ptr->get_layer_params().size())
                            layers.emplace_back().emplace_back(move(ptr));
                    }
                });
            for (unsigned i = 1; i < nets.size(); ++i) {
                auto it = layers.begin();
                visit_layers(
                    *nets[i], [&](auto& t) {
                        using T = std::decay_t<decltype(t)>;
                        using R = averager_layer_ref_t<T>;
                        if (R::has_params) {
                            auto ptr = std::make_unique<R>(t);
                            if (0 < ptr->get_layer_params().size()) {
                                it->emplace_back(move(ptr));
                                ++it;
                            }
                        }
                    });
                assert(it == layers.end());
            }
        }

        template <dlib::tensor&(layer_ref::*M)()>
        static void copy_front_to_all(
            const std::vector<std::unique_ptr<layer_ref> >& vec) {
            dlib::tensor const& ref = ((*vec.front()).*M)();
            for (auto it = next(vec.begin()),
                     end = vec.end(); it != end; ++it) {
                dlib::tensor& dest = ((**it).*M)();
                memcpy(dest, ref);
            }
        }

        static void sum_to_front(
            const std::vector<std::unique_ptr<layer_ref> >& vec) {
            dlib::tensor& sum = vec.front()->get_parameter_gradient();
            for (auto it = next(vec.begin()),
                     end = vec.end(); it != end; ++it) {
                auto& src = (**it).get_parameter_gradient();
                assert(src.size() == sum.size());
                for (auto d = sum.host(), dend = d + sum.size(),
                         s = src.host(); d != dend; ++d, ++s)
                    *d += *s;
            }
        }

        static void scale_tensor(dlib::tensor& t, float scale) {
            for (auto el = t.host(), end = el + t.size(); el != end; ++el)
                *el *= scale;
        }

        mutable std::atomic<std::size_t> sync_next{0};

        void sync_layer_parameters() const {
            for (const auto size = layers.size(); ; ) {
                const auto i =
                    sync_next.fetch_add(1,std::memory_order_relaxed);
                if (i < size)
                    copy_front_to_all<&layer_ref::get_layer_params>(layers[i]);
                else break;
            }
        }

        mutable std::atomic<std::size_t> average_next{0};

        void average_parameter_gradients() const {
            for (const auto size = layers.size(); ; ) {
                const auto i =
                    average_next.fetch_add(1,std::memory_order_relaxed);
                if (i < size) {
                    auto& vec = layers[i];
                    sum_to_front(vec);
                    scale_tensor(vec.front()->get_parameter_gradient(),
                                 1.0 / nets.size());
                    copy_front_to_all<&layer_ref::get_parameter_gradient>(vec);
                }
                else break;
            }
        }

    public:
        /// every 100 steps copy parameters from first model to all others
        /// to avoid floating point drift between models
        /// also, in case of rollback to previous checkpoint
        static constexpr auto sync_steps = 100;

        /// constructor
        explicit averager(std::vector<NET*> nets) : nets(move(nets)) {
            assert(1 < this->nets.size());
        }
        
        /** \brief Trainers (threads) call this method before parameter update. 
         *
         * Operations are performed in lock step.
         *   1. wait for N = nets.size() threads to enter
         *   2. parallelized sync and average
         *   3. wait for N threads to complete
         *   4. reset for next training step
         */
        void operator()() {
            std::unique_lock lock(mux);
            if (++enter_count == nets.size()) {
                leave_count = 0;
                if (layers.empty())
                    find_layers();
                lobby.notify_all();
            }
            else
                while (enter_count != nets.size())
                    lobby.wait(lock);

            // all trainers (threads) are now here
            // the following operations are parallelized using all threads
            lock.unlock();
            if (sync_count >= sync_steps)
                sync_layer_parameters();
            average_parameter_gradients();
            lock.lock();

            // ensure all threads get here before we exit this method
            if (++leave_count == nets.size()) {
                enter_count = 0;
                average_next.store(0);
                if (sync_count >= sync_steps) {
                    sync_next.store(0);
                    sync_count = 0;
                }
                else ++sync_count;
                lobby.notify_all();
            }
            else
                while (leave_count != nets.size())
                    lobby.wait(lock);
        }
    };
}
