#pragma once

#include "types.hpp"
#include <atomic>
#include <functional>
#include <optional>
#include <stdexcept>
#include <map>
#include <mutex>

namespace rec {
    struct model_state;

    /** \brief Compile time static model parameters.
     */
    struct model_static {
        const version_type version;
        const variant default_compare_variant;
        const float cos_max_score;
        const float l2sqr_max_score;
        const float l2sqr_coeff;
        prototype_ptr(*const deserialize_prototype)(std::shared_ptr<const model_state>, const void*, std::size_t, const std::optional<uuid_type>&) = nullptr;
        prototype_ptr(*const random)(core::thread_data&,std::shared_ptr<const model_state>, const prototype*, float score, variant) = nullptr;
    };

    /** \brief Runtime per-context model state.
     */
    struct model_state : model_static {
        std::atomic<variant> compare_variant;
        std::atomic<int> serialize_format;
        model_state(const model_static& ms) noexcept
            : model_static(ms),
              compare_variant(ms.default_compare_variant),
              serialize_format(0) {
        }
    };

    /** \brief Per-context map of models.
     */
    class context_map {
    public:
        /** \brief Initialize map with static data.
         */
        context_map();


        /** \brief Get non-const model state without loading model.
         *
         * \throws if model version number not known
         * \returns non-null pointer
         */
        std::shared_ptr<model_state> get(version_type ver) const;


        /** \brief Load model data for known model.
         *
         * If model data has been loaded and is of the correct type,
         * then it is returned.
         *
         * If model data has been loaded but is not of the correct type,
         * then a logic_error is thrown.
         *
         * If model data has not been loaded, then the supplied loader
         * function will be called to attempt to load data.
         * If this function returns a nullptr (failure to load data), then
         * a runtime_error is thrown.
         *
         * The loader function is called with the provided arguments and
         * must return std::shared_ptr<T> for some class T.
         *
         * \throws if model version number not known
         * \throws if model loader failed
         * \throws if existing data has wrong type
         *
         * \returns pair {
         *     std::shared_ptr<const T>,
         *     std::shared_ptr<const model_state> }
         */
        template <typename FUNC, typename... Args>
        auto load(version_type ver, FUNC&& loader, Args&&... args) const {
            using T = typename
                decltype(loader(std::forward<Args>(args)...))::element_type;
            auto ptr = find_or_load(
                ver, &typemap<std::remove_cv_t<T> >,
                [&]() { return loader(std::forward<Args>(args)...); });
            auto& rec = *ptr;
            return std::pair(
                std::static_pointer_cast<const T>(rec.m_data),
                std::shared_ptr<const model_state>(move(ptr)));
        }

        /** \brief Store custom model in map.
         *
         * \throws if model version number already in use
         */
        inline void insert(const model_static& model) {
            insert(model, nullptr, {});
        }
        template <typename T>
        inline void insert(const model_static& model, std::shared_ptr<T> data) {
            insert(model, &typemap<std::remove_cv_t<T> >, move(data));
        }


        /** \brief Set of known versions (not including custom models).
         */
        static std::vector<version_type> known_versions();


    private:
        struct typeinfo {};

        template <typename T>
        static const typeinfo typemap;

        struct model_record : model_state {
            model_record(const model_static& ms) : model_state(ms) {}
            typeinfo const* m_type = nullptr;
            std::shared_ptr<const void> m_data;
        };

        std::map<version_type, std::shared_ptr<model_record> > map;
        mutable std::mutex mux;

        void insert(const model_static& model,
                    typeinfo const* type, std::shared_ptr<const void> data);

        std::shared_ptr<const model_record>
        find_or_load(version_type ver, typeinfo const* type,
                     std::function<std::shared_ptr<const void>()> loader) const;
    };

    template <typename T>
    const context_map::typeinfo context_map::typemap{};
}
