
#include <dlib/matrix.h>
#include "linear_regression.hpp"
#include <applog/core.hpp>

using namespace dlibx;

template <typename T>
std::vector<T> linear_regression<T>::compute() const {
    if (ncols > Z.size())
        throw std::invalid_argument("insufficient data for linear regression");

    assert(X.size() == Z.size() * ncols);
    const auto xmat = dlib::mat(X.data(), long(Z.size()), long(ncols));
    const auto t = dlib::matrix<T>(trans(xmat)*xmat);
    assert(t.nr() == t.nc() && t.nr() == long(ncols));
    const auto i = dlib::matrix<T>(inv(t));
    assert(i.nr() == i.nc() && i.nr() == long(ncols));

    std::vector<T> r(ncols);
    const auto zcol = dlib::mat(Z.data(), long(Z.size()), 1);
    dlib::set_ptrm(r.data(),long(ncols),1) = i * trans(xmat) * zcol;
    return r;
}

template <typename T>
T linear_regression<T>::ssr(const std::vector<T>& coeff_) const {
    if (ncols != coeff_.size())
        throw std::invalid_argument("incorrect number of coefficients");
    const auto coeff = dlib::mat(coeff_.data(), long(ncols), 1);
    const auto xmat = dlib::mat(X.data(), long(Z.size()), long(ncols));
    const auto zcol = dlib::mat(Z.data(), long(Z.size()), 1);
    return length_squared(zcol - xmat * coeff);
}

// explicit template instantiation
template class dlibx::linear_regression<float>;
template class dlibx::linear_regression<double>;
