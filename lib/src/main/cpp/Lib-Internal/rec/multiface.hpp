#pragma once

#include "types.hpp"
#include "serialize.hpp"
#include <stdext/arg.hpp>
#include <stdext/forward_iterator.hpp>

namespace rec {
    
    /** \brief Set of prototypes with the same version.
     */
    class multiface {
        float cluster_threshold;
        std::unique_ptr<internal::multiface> state;

        multiface(const multiface&) = delete;
        multiface& operator=(const multiface&) = delete;

    public:
        using size_type = std::size_t;
        
        ~multiface();
        multiface(multiface&&);
        multiface& operator=(multiface&&);

        /** \brief Construct empty multiface.
         */
        multiface(float cluster_threshold = 0);

        /** \brief Construct and assign prototypes.
         */
        multiface(stdx::forward_iterator<prototype_ptr> first,
                  stdx::forward_iterator<prototype_ptr> last,
                  float cluster_threshold = 0);

        /** \brief Deserialize.
         *
         * The data may be json::binary, base64 json::string, or
         * a decoded json::object.
         * The binary data may be either deflate compressed or not, and
         * either json or amf3 encoded.
         *
         * This method will also accept a flattened face or subject --
         * converting them to a multiface object.
         */
        multiface(stdx::arg<const core::context_data> context, json::value data);
        
        
        /** \brief Update multiface with new set of one or more prototypes.
         *
         * All prototypes must have the same version.
         *
         * More efficient than constructing a new multiface object.
         */
        void assign(stdx::forward_iterator<prototype_ptr> first,
                    stdx::forward_iterator<prototype_ptr> last);

        /** \brief Test if empty.
         */
        inline bool empty() const { return !state; }

        /** \brief Number of faces contained.
         */
        size_type size() const;

        /** \brief Version of contained prototypes.
         *
         * Returns 0 if empty.
         */
        version_type version() const;

        /** \brief Serialize to json object.
         */
        friend json::value to_json(const multiface& mf);

        /** \brief Serialize to binary.
         *
         * Don't use this method directly.
         * Use to_binary() instead (defined in types.hpp).
         *
         * Default is deflate compressed amf3.
         * Note that raw is the same as amf3.
         */
        friend stdx::binary to_binary_with_opts(
            const multiface& mf,
            const stdx::options_tuple<serialize_type,compression_type>& opts);


        /** \brief Access to internal structure for compare.
         */
        inline operator const internal::multiface&() const {
            return *state;
        }
        multiface_ptr release();


    private:
        template <typename Key, typename Data, typename Ops>
        friend class subject;
    };


    /** \brief Compare multiface to prototype.
     */
    compare_result compare(stdx::arg<const internal::multiface> mf,
                           stdx::arg<const prototype> proto,
                           variant var = variant::none);
    inline compare_result compare(stdx::arg<const prototype> proto,
                                  stdx::arg<const internal::multiface> mf,
                                  variant var = variant::none) {
        return compare(mf, proto, var);
    }
}
