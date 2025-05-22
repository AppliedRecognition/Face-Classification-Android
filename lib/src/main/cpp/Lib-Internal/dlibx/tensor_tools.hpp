#pragma once

#include "tensor.hpp"

namespace dlibx {
    /** \brief Iterator helper class for tensor_alias_span.
     */
    template <typename TENSOR>
    struct tensor_alias_iterator {
        TENSOR& tensor;
        dlib::alias_tensor alias;
        std::size_t ofs, incr;

        using iterator_category = std::forward_iterator_tag;
        using difference_type = std::ptrdiff_t;
        using value_type = decltype(alias(tensor,ofs));

        template <typename U>
        bool operator==(const tensor_alias_iterator<U>& other) const {
            return ofs == other.ofs;
        }
        template <typename U>
        bool operator!=(const tensor_alias_iterator<U>& other) const {
            return ofs != other.ofs;
        }
        auto& operator++() {
            ofs += incr;
            return *this;
        }
        value_type operator*() const {
            return alias(tensor,ofs);
        }
    };

    /** \brief Span of fixed size aliases of a tensor.
     *
     * Helper class for samples() and channels() below.
     */
    template <typename TENSOR>
    class tensor_alias_span {
        TENSOR& tensor;
        dlib::alias_tensor alias;
        std::size_t start, incr;

    public:
        tensor_alias_span(TENSOR& tensor, dlib::alias_tensor alias,
                          std::size_t start = 0, std::size_t incr = 1)
            : tensor(tensor),
              alias(alias),
              start(start*alias.size()),
              incr(incr*alias.size()) {
            if (alias.size() <= 0 || incr <= start ||
                tensor.size() % incr != 0)
                throw std::invalid_argument(
                    "tensor size must be a multiple of alias size");
        }

        inline bool empty() const { return tensor.size() <= 0; }
        inline auto size() const { return tensor.size() / alias.size(); }

        template <typename IDX>
        inline auto operator[](IDX i) {
            return alias(tensor, start + std::size_t(i)*incr);
        }
        template <typename IDX>
        inline auto operator[](IDX i) const {
            return alias(tensor, start + std::size_t(i)*incr);
        }

        using iterator = tensor_alias_iterator<TENSOR>;
        using const_iterator = tensor_alias_iterator<const TENSOR>;

        inline auto begin() {
            return iterator{tensor,alias,start,incr};
        }
        inline auto end() {
            return iterator{tensor,alias,start+tensor.size(),incr};
        }
        inline auto begin() const {
            return const_iterator{tensor,alias,start,incr};
        }
        inline auto end() const {
            return const_iterator{tensor,alias,start+tensor.size(),incr};
        }
        inline auto cbegin() const {
            return const_iterator{tensor,alias,start,incr};
        }
        inline auto cend() const {
            return const_iterator{tensor,alias,start+tensor.size(),incr};
        }
    };


    /** \brief Iterate through the samples of a tensor.
     */
    template <typename T>
    inline auto samples(T& tensor) {
        using TENSOR = std::conditional_t<
            std::is_const<T>::value, const dlib::tensor, dlib::tensor>;
        return tensor_alias_span<TENSOR>(
            tensor, {1, tensor.k(), tensor.nr(), tensor.nc()});
    }


    /** \brief Iterate through the channels of a tensor.
     *
     * If the input tensor has multiple samples then this class
     * iterates through all channels of all samples.
     * Use the samples() method above to separate the samples. 
     *
     * The second overload support iterating through channels
     * by groups of 1 or more.
     */
    template <typename T>
    inline auto channels(T& tensor) {
        using TENSOR = std::conditional_t<
            std::is_const<T>::value, const dlib::tensor, dlib::tensor>;
        return tensor_alias_span<TENSOR>(
            tensor, {1, 1, tensor.nr(), tensor.nc()});
    }
    template <long STRIDE, typename T>
    inline auto channels(T& tensor) {
        using TENSOR = std::conditional_t<
            std::is_const<T>::value, const dlib::tensor, dlib::tensor>;
        static_assert(STRIDE > 0);
        if (tensor.k() % STRIDE != 0)
            throw std::invalid_argument(
                "number of tensor channels must be a multiple of stride");
        return tensor_alias_span<TENSOR>(
            tensor, {1, STRIDE, tensor.nr(), tensor.nc()});
    }


    /** \brief View a specific channel or group of channels per sample.
     */
    template <long STRIDE, typename T>
    inline auto sample_channels(T& tensor, long group_index) {
        using TENSOR = std::conditional_t<
            std::is_const<T>::value, const dlib::tensor, dlib::tensor>;
        static_assert(STRIDE > 0);
        if (tensor.k() % STRIDE != 0)
            throw std::invalid_argument(
                "number of tensor channels must be a multiple of stride");
        const auto num_groups = tensor.k() / STRIDE;
        if (!(0 <= group_index && group_index < num_groups))
            throw std::invalid_argument("invalid group index");
        return tensor_alias_span<TENSOR>(
            tensor, {1, STRIDE, tensor.nr(), tensor.nc()},
            std::size_t(group_index), std::size_t(num_groups));
    }


    /** \brief Extract specific channel or group of channels per sample.
     */
    template <long STRIDE>
    inline auto extract_channels(const dlib::tensor& t, long group_index) {
        dlib::resizable_tensor r(t.num_samples(),STRIDE,t.nr(),t.nc());
        auto src = sample_channels<STRIDE>(t, group_index).begin();
        for (auto dest : samples(r)) {
            memcpy(dest, *src);
            ++src;
        }
        return r;
    }
}
