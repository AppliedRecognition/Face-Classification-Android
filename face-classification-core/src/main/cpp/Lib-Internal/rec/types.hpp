#pragma once

#include <raw_image/point2.hpp>
#include <stdext/binary.hpp>

namespace core {
    class context;
    struct context_data;
    class thread_data;
}
namespace det {
    struct face_coordinates;
}

namespace rec {
    using version_type = unsigned;
    static constexpr auto uuid_bytes = 16;
    using uuid_type = std::array<std::byte,uuid_bytes>;

    using raw_image::rotated_box;

    class prototype;
    using prototype_ptr = std::shared_ptr<const prototype>;

    class multiface;
    namespace internal {
        class multiface;
        struct multiface_deleter { void operator()(const multiface*) const; };
    }
    using multiface_ptr =
        std::unique_ptr<const internal::multiface, internal::multiface_deleter>;


    /** \brief Types of diagnostic face images.
     */
    enum class diagnostic {
        extracted,      ///< raw extracted face
        preprocessed,   ///< face after preprocessing
        reconstructed,  ///< face reconstructed from template
        features        ///< features of face weighted as used by recognition
    };


    /** \brief Comparison variants.
     */
    enum class variant : unsigned {
        // values < 16 are comparison classes (and are mutually exclusive)
        none = 0,  ///< use default
        cos = 1,   ///< cosine similarity a.k.a. normalized inner product
        l2sqr = 2, ///< square of L2 Euclidean distance

        // values >= 16 are modifier flags
        raw = 16,  ///< don't adjust score to standard normal distribution
        nomirror = 32,
        remove_subject_bias = 64
    };
    constexpr variant comparison_class(variant a) {
        return variant(unsigned(a) & 15);
    }
    constexpr bool operator&(variant a, variant b) {
        return unsigned(a) & unsigned(b);
    }
    constexpr variant operator|(variant a, variant b) {
        return variant(unsigned(a) | unsigned(b));
    }
    constexpr variant& operator|=(variant& a, variant b) {
        return a = a | b;
    }
    constexpr variant operator+(variant a, variant b) {
        return a | b;
    }
    constexpr variant& operator+=(variant& a, variant b) {
        return a = a + b;
    }


    /** \brief Comparison result.
     */
    enum class flags { none = 0, mirror = 1 };
    constexpr bool operator&(flags a, flags b) {
        return unsigned(a) & unsigned(b);
    }
    struct compare_result {
        float score;
        rec::flags flags;
        constexpr compare_result(
            float score = 0, rec::flags flags = rec::flags::none)
            : score(score), flags(flags) {}
        constexpr operator float() const { return score; }
    };
}
