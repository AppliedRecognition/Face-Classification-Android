#pragma once

#include "types.hpp"
#include <json/types.hpp>
#include <stdext/forward_iterator.hpp>
#include <unordered_map>
#include <array>

namespace rec {
    namespace internal {

        json::object decode_object(json::value);

        struct hash_from_uuid {
            std::size_t operator()(const uuid_type& uuid) const {
                if (uuid.size() < sizeof(std::size_t))
                    throw std::invalid_argument("uuid is too small");
                return *reinterpret_cast<const std::size_t*>(uuid.data());
            }
        };

        // array is ids, value is encoded object
        using face_map_type =
            std::unordered_map<uuid_type, std::pair<json::array, json::value>,
                               hash_from_uuid>;

        using uuid_set_type = std::vector<uuid_type>;


        /** \brief Abstract base class for multiface objects.
         */
        class multiface {
        public:
            /** \brief Deserialize.
             */
            static std::unique_ptr<multiface>
            deserialize(const core::context_data& context_data,
                        const json::object& val,
                        face_map_type* face_map = nullptr);


        public:
            /** \brief Version of prototypes contained.
             */
            const version_type version;
            
            virtual ~multiface() = default;

            virtual void assign(
                stdx::forward_iterator<prototype_ptr> first,
                stdx::forward_iterator<prototype_ptr> last) = 0;
            
            virtual std::size_t size() const = 0;

            virtual uuid_set_type uuid_set() const = 0;

            // this method may return an empty vector if
            // the prototypes are not directly stored
            virtual std::vector<prototype_ptr> get_prototypes() const = 0;
            
            virtual json::object serialize(
                const face_map_type* face_map = nullptr) const = 0;

            virtual void compare_to_n(
                const prototype* const* p, std::size_t n,
                variant var, float* results) const = 0;

            virtual json::value diagnostic() const = 0;
            

        protected:
            multiface(version_type version) : version(version) {}

            multiface(multiface&&) = delete;
            multiface(const multiface&) = delete;
            multiface& operator=(multiface&&) = delete;
            multiface& operator=(const multiface&) = delete;
        };
    }
}
