#pragma once

#include "multiface.hpp"
#include "internal_multiface.hpp"

#include <json/types.hpp>

#include <type_traits>
#include <atomic>
#include <mutex>
#include <map>

namespace rec {
    
    /** \brief Map of integer id to prototype.
     *
     * Semantically like std::map<Key,Data> with just-in-time conversion
     * to multiface for comparisons.
     *
     * The Ops object must have methods like so: <pre>
         struct Ops {
             prototype_ptr prototype(const Data&) const;
             json::value encode(const Data&) const;
             Data decode(const core::context_data&, const json::value&) const;
         };
     * </pre>
     */
    template <typename Key, typename Data, typename Ops>
    class subject {
        static_assert(std::is_integral<Key>::value,
                      "subject id must be an integer");
        using map_type = std::map<Key,Data>;

    public:
        using key_type = Key;
        using mapped_type = Data;
        using size_type = typename map_type::size_type;

        subject(subject&&) = default;
        subject& operator=(subject&&) = default;

        /** \brief Construct empty subject.
         */
        subject() = default;

        /** \brief Construct empty subject.
         */
        template <typename... Args>
        subject(float cluster_threshold, Args&&... args)
            : m_mf(cluster_threshold),
              m_ops(std::forward<Args>(args)...) {}

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
        template <typename... Args>
        subject(const core::context_data& context,
                json::value data, Args&&... args)
            : m_ops(std::forward<Args>(args)...) {
            auto obj = internal::decode_object(move(data));
            m_mf.cluster_threshold = make_number(obj["cluster_threshold"]);
            if (get_integer_safe(obj["ver"]) != 2 ||
                !json::is_type<json::array>(obj["clusters"]) ||
                !get_array(obj["clusters"]).empty()) {
                internal::face_map_type face_map;
                m_mf.state =
                    internal::multiface::deserialize(context,obj,&face_map);
                for (auto& p : face_map) {
                    const auto d =
                        m_ops.decode(context,std::get<json::value>(p.second));
                    for (auto idv : std::get<json::array>(p.second))
                        m_map[make_number(idv)] = d;
                }
                if (m_map.empty())
                    throw std::invalid_argument("subject has no faces");
            }
        }
    
        
        /** \brief Test if empty.
         */
        inline bool empty() const { return m_map.empty(); }

        /** \brief Number of faces contained.
         */
        inline auto size() const { return m_map.size(); }
        
        /** \brief Const access to map.
         */
        inline const map_type& access() const { return m_map; }
        inline const map_type& operator*() const { return m_map; }
        inline const map_type* operator->() const { return &m_map; }

        /** \brief Modify map.
         *
         * This method marks the internal multiface as stale.
         * It will require an update before compare().
         */
        inline map_type& modify() {
            m_stale.store(true, std::memory_order_release);
            return m_map;
        }

        /** \brief Access to multiface for comparison.
         *
         * This method will update the internal multiface if necessary.
         * The update is done in a thread-safe manner with respect to this
         * method.  It is not safe to call modify() in another thread.
         */
        const rec::multiface& multiface() const {
            if (m_stale.load(std::memory_order_acquire)) {
                std::lock_guard<std::mutex> lock(m_mutex);
                if (m_stale.load(std::memory_order_acquire)) {
                    //FILE_LOG(logDETAIL) << "subject: update " << m_faces.size() << " faces";
                    const auto conv = [&](const auto& el) {
                        return m_ops.prototype(el.second);
                    };
                    m_mf.assign(
                        { m_map.begin(), conv },
                        { m_map.end(),   conv });
                    //FILE_LOG(logDETAIL) << "subject: update complete";
                    m_stale.store(false, std::memory_order_release);
                }
            }
            return m_mf;
        }
        inline operator const rec::multiface&() const {
            return multiface();
        }
        inline operator const rec::internal::multiface&() const {
            return multiface();
        }

        /** \brief Serialize to json object.
         */
        friend json::value to_json(const subject& sub) {
            return sub.serialize();
        }

        /** \brief Serialize to binary.
         *
         * Don't use this method directly.
         * Use to_binary() instead (defined in types.hpp).
         *
         * Default is deflate compressed amf3.
         * Note that raw is the same as amf3.
         */
        friend inline stdx::binary to_binary_with_opts(
            const subject& sub,
            const stdx::options_tuple<serialize_type,compression_type>& opts) {
            return to_binary_with_opts(to_json(sub),opts);
        }
 

    private:
        map_type m_map;
        mutable rec::multiface m_mf;
        mutable std::mutex m_mutex;
        mutable std::atomic<bool> m_stale{false};
        Ops m_ops;

        auto serialize() const {
            internal::face_map_type face_map;
            for (auto& el : m_map) {
                auto& rec = face_map[m_ops.prototype(el.second)->uuid];
                std::get<json::array>(rec).push_back(el.first);
                auto& val = std::get<json::value>(rec);
                if (val == json::null)
                    val = m_ops.encode(el.second);
            }
            auto& mf = multiface();
            if (!mf.state)
                throw std::runtime_error("subject is empty");
            return mf.state->serialize(&face_map);
        }
    };
}
