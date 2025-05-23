
#include "linear_regression.hpp"
#include "transform.hpp"
#include "adjust.hpp"

#include <applog/core.hpp>

#include <cassert>

using namespace raw_image;

static plane make_matrix(const std::vector<float>& vec, unsigned rows) {
    const auto cols = unsigned(vec.size() / rows);
    assert(vec.size() == rows*std::size_t(cols));
    plane mat;
    mat.height = rows;
    mat.width = cols;
    mat.bytes_per_line = 4*mat.width;
    mat.layout = pixel::f32;
    mat.data =
        const_cast<uint8_t*>(reinterpret_cast<const uint8_t*>(vec.data()));
    return mat;
}

template <typename T>
std::vector<T> linear_regression<T>::compute() const {
    if (ncols > Z.size())
        throw std::logic_error("insufficient data for linear regression");
    assert(X.size() == Z.size() * ncols);

    static_assert(std::is_same_v<T,float> && sizeof(T) == 4);

    const auto xmat = make_matrix(X, unsigned(Z.size()));
    const auto zcolt = make_matrix(Z, 1); // zcol transposed is a row

    const auto xt = copy_transpose(xmat);
    const auto square = create(xmat.width, xmat.width, pixel::f32);
    matrix_multiply(reader::construct(xt),*xt)->copy_to(*square);

    const auto inv = matrix_inverse(*square);

    std::vector<T> r(ncols);
    const auto out = make_matrix(r, unsigned(ncols));
    matrix_multiply(matrix_multiply(reader::construct(inv),xmat),zcolt)
        ->copy_to(out);
    return r;
}

template <typename T>
T linear_regression<T>::ssr(const std::vector<T>& coeff_) const {
    if (ncols != coeff_.size())
        throw std::invalid_argument("incorrect number of coefficients");

    const auto coeff = make_matrix(coeff_, 1); // a row
    const auto zcol = make_matrix(Z, unsigned(Z.size())); // a column
    const auto xmat = make_matrix(X, unsigned(Z.size()));

    auto r = blend(matrix_multiply(reader::construct(xmat), coeff), 1,
                   reader::construct(zcol), -1);
    assert(r->width() == 1);
    T sum = 0;
    do {
        auto z = *reinterpret_cast<const T*>(r->get_line());
        sum += z*z;
    } while (r->next_line());
    return sum;
}

// explicit template instantiation
template class raw_image::linear_regression<float>;
//template class raw_image::linear_regression<double>;
