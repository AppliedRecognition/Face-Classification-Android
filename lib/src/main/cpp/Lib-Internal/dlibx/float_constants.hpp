#pragma once

#include <ratio>

namespace dlibx {
    template<std::intmax_t Num, std::intmax_t Denom = 1>
    struct float_ratio : std::ratio<Num,Denom> {
        constexpr operator float() const { return float(Num) / float(Denom); }
    };
    using float_zero = float_ratio<0>;
    using float_half = float_ratio<1,2>;
    using float_one = float_ratio<1>;
}
