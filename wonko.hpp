/*
 * Copyright (c) 2017  Pavel Kirienko  <pavel.kirienko@zubax.com>
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#pragma once

#include <cmath>
#include <type_traits>
#include <utility>
#include <limits>
#include <cassert>
#include <algorithm>
#include <cstdlib>

/**********************************************************************************************************************
 * MACRO DECLARATIONS
 * Wonko heavily relies on macros for performance reasons.
 * These macros are un-defined at the end of the file in order to not pollute the global scope.
 */

 /**
  * The macros WONKO_BEFORE_LOOP and WONKO_AFTER_LOOP are expanded before and after each loop over the elements of
  * the matrix, respectively. They can be used to inject compiler-specific and application-specific local optimization
  * flags, e.g. "unroll-all-loops".
  */
#if defined(WONKO_BEFORE_LOOP) != defined(WONKO_AFTER_LOOP)
# error "Both or neither of WONKO_BEFORE_LOOP and WONKO_AFTER_LOOP should be defined."
#endif

#ifndef  WONKO_BEFORE_LOOP
# define WONKO_BEFORE_LOOP     (void) 0
#endif

#ifndef  WONKO_AFTER_LOOP
# define WONKO_AFTER_LOOP      (void) 0
#endif


#define WONKO_ITER_ROWS_COLS(block)                     \
    WONKO_BEFORE_LOOP;                                  \
    for (Index r = 0; r < NRows; ++r) {                 \
        for (Index c = 0; c < NCols; ++c)               \
        {                                               \
            block;                                      \
        }                                               \
    }                                                   \
    WONKO_AFTER_LOOP;

#define WONKO_ITER_ELEMENTS(block)                      \
    WONKO_BEFORE_LOOP;                                  \
    for (Index i = 0; i < (NRows * NCols); ++i)         \
    {                                                   \
        block;                                          \
    }                                                   \
    WONKO_AFTER_LOOP;


namespace wonko
{
/**********************************************************************************************************************
 * GLOBAL DECLARATIONS
 */
static constexpr unsigned FloatComparisonRelativeEpsilonMultiplier = 10;

/**
 * We're intentionally using a signed type for index. Refer to Eigen docs (and STL) to learn why unsigned indexes is
 * really a bad idea.
 */
using Index = int;

enum class StringRepresentation
{
    SingleLine,
    MultiLine
};

/**********************************************************************************************************************
 * SCALAR AND OPERAND-AGNOSTIC API
 */
namespace impl_
{

template <typename T>
constexpr inline bool compareFloats(const T a,
                                    const T b,
                                    const T absolute_epsilon,
                                    const T relative_epsilon) noexcept
{
    if (std::isnan(a) || std::isnan(b))
    {
        return false;
    }

    if (std::isinf(a) || std::isinf(b))
    {
        if (std::isinf(a) && std::isinf(b))
        {
            return (a > 0) == (b > 0);
        }
        else
        {
            return false;
        }
    }

    // Close numbers near zero
    if (std::abs(a - b) <= absolute_epsilon)
    {
        return true;
    }

    // General case
    return std::abs(a - b) <= (std::max(std::abs(a), std::abs(b)) * relative_epsilon);
}

}

template <typename T, typename = std::enable_if_t<!std::is_floating_point<T>::value>>
inline bool areClose(const T& a, const T& b)   // NOT noexcept, because == can be overloaded and may throw
{
    return a == b;
}

template <typename T, typename = std::enable_if_t<std::is_floating_point<T>::value>>
constexpr inline bool areClose(const T a, const T b, const T epsilon = std::numeric_limits<T>::epsilon()) noexcept
{
    return impl_::compareFloats(a, b,
                                epsilon,
                                epsilon * FloatComparisonRelativeEpsilonMultiplier);
}

/**********************************************************************************************************************
 * MATRIX API
 */
template <typename Derived_>
class MatrixBase
{
public:
    using Derived = Derived_;
};


template <typename Scalar_, Index NRows, Index NCols = NRows>
class Matrix : public MatrixBase<Matrix<Scalar_, NRows, NCols>>
{
public:
    using Scalar = Scalar_;

    using Storage = Scalar[NRows * NCols];

private:
    using Self = Matrix<Scalar, NRows, NCols>;

    template <typename FS, Index FNR, Index FNC>
    friend class Matrix;

    template <typename FS, Index FNR, Index FNC>
    friend Matrix<FS, FNR, FNC> operator-(const FS&, const Matrix<FS, FNR, FNC>&);

    template <typename FS, Index FNR, Index FNC>
    friend Matrix<FS, FNR, FNC> operator/(const FS&, const Matrix<FS, FNR, FNC>&);

    static_assert(NRows > 0 && NCols > 0, "Invalid matrix size");

    static constexpr Index NElements = NRows * NCols;

    Storage data_;      ///< Zero-initialized by the default constructor


    template <Index N>
    void unfoldVariadicInitializer(const Scalar s)
    {
        data_[N] = Scalar(s);
        static_assert(N >= (NElements - 1), "Incomplete variadic initializer");
        static_assert(N <= (NElements - 1), "Excessive elements in the variadic initializer");
    }

    template <Index N, typename... Args>
    void unfoldVariadicInitializer(const Scalar s, Args&&... a)
    {
        data_[N] = Scalar(s);
        unfoldVariadicInitializer<N + 1>(std::forward<Args>(a)...);
    }

public:
    static constexpr bool isVector() { return (NRows == 1) || (NCols == 1); }

    static constexpr Index numRows()     { return NRows; }
    static constexpr Index numColumns()  { return NCols; }
    static constexpr Index numElements() { return NElements; }

    static constexpr Self ones()
    {
        Self m;
        std::fill(std::begin(m.data_), std::end(m.data_), Scalar(1));
        return m;
    }

    static constexpr Self identity()
    {
        Self m;
        WONKO_ITER_ROWS_COLS(m(r, c) = Scalar((r == c) ? 1 : 0))
        return m;
    }

    /**
     * Constructs a matrix initialized with random values in the range [-1, 1].
     */
    static constexpr Self random()
    {
        constexpr Scalar HalfRandMax = Scalar(RAND_MAX) / Scalar(2);
        Self m;
        WONKO_ITER_ELEMENTS(m.data_[i] = Scalar(std::rand()) / HalfRandMax - Scalar(1))
        return m;
    }

    /**
     * Matrix is a trivially copyable class, so we use default copy/move/assignment constructors/operators.
     */
    Matrix(const Self&)             = default;
    Matrix(Self&&)                  = default;
    Self& operator=(const Self&)    = default;
    Self& operator=(Self&&)         = default;

    /**
     * Default zero-initializing constructor.
     */
    constexpr Matrix() : data_{} { }

    /**
     * Constructor that initializes the matrix from a flat array of scalars.
     * It is made explicit in order to enforce additional type safety and to prevent it from competing with the
     * variadic constructor defined below.
     */
    explicit constexpr Matrix(const Scalar (&elements)[NElements])
    {
        std::copy(std::begin(elements), std::end(elements), std::begin(data_));
    }

    /**
     * Constructor for scalar matrix takes just one argument. It is made explicit in order to improve type safety.
     * The single-argument constructor can't be handled as a special case of variadic constructor, because
     * it would be matched instead of copy constructor, causing compilation errors on trivial use cases.
     */
    explicit constexpr Matrix(const Scalar& first) :
        data_{first}
    {
        static_assert(NElements == 1, "This constructor can only be used for scalar matrices");
    }

    /**
     * Variadic initialization constructor. We must explicitly define the type of the first element in order to
     * avoid shadowing the copy constructor.
     * This constructor is made non-explicit in order to facilitate easy return value initialization.
     */
    template <typename... Args, typename = std::enable_if_t<sizeof...(Args) == (NElements - 1)>>
    constexpr Matrix(const Scalar& first, Args&&... elements)
    {
        unfoldVariadicInitializer<0>(first, std::forward<Args>(elements)...);
    }


    Matrix<Scalar, NCols, NRows> transpose() const
    {
        Matrix<Scalar, NCols, NRows> m;
        WONKO_ITER_ROWS_COLS(m.data_[c * NRows + r] = data_[r * NCols + c])
        return m;
    }

    template <typename TargetScalar>
    Matrix<TargetScalar, NRows, NCols> cast() const
    {
        Matrix<TargetScalar, NRows, NCols> m;
        WONKO_ITER_ELEMENTS(m.data_[i] = TargetScalar(data_[i]))
        return m;
    }

    Storage& data()             { return data_; }
    const Storage& data() const { return data_; }

    template <typename T>
    void fill(const T& x)
    {
        WONKO_ITER_ELEMENTS(data_[i] = Scalar(x))
    }

    void setZero() { fill(Scalar(0)); }
    void setOnes() { fill(Scalar(1)); }

    Scalar min() const
    {
        Scalar s = data_[0];
        WONKO_ITER_ELEMENTS(s = std::min(s, data_[i]))
        return s;
    }

    Scalar max() const
    {
        Scalar s = data_[0];
        WONKO_ITER_ELEMENTS(s = std::max(s, data_[i]))
        return s;
    }

    Scalar absmax() const
    {
        Scalar s = Scalar(0);
        WONKO_ITER_ELEMENTS(s = std::max(s, std::abs(data_[i])))
        return s;
    }

    Self abs() const
    {
        Self m;
        WONKO_ITER_ELEMENTS(m.data_[i] = std::abs(data_[i]))
        return m;
    }

    Scalar sum() const
    {
        Scalar s = Scalar(0);
        WONKO_ITER_ELEMENTS(s += data_[i])
        return s;
    }

    Scalar mean() const
    {
        return sum() / Scalar(NElements);
    }

    bool hasNonFiniteElements() const
    {
        bool all_finite = true;
        WONKO_ITER_ELEMENTS(all_finite = all_finite && std::isfinite(data_[i]))
        return !all_finite;
    }

    Scalar norm() const
    {
        static_assert(isVector(), "Norm for matrices is not supported yet");
        Scalar s = Scalar(0);
        WONKO_ITER_ELEMENTS(s += data_[i] * data_[i])
        return std::sqrt(s);
    }

    Self normalized() const
    {
        return (*this) / norm();
    }

    Self power(unsigned p) const
    {
        Self m = identity();
        while (p --> 0)         // No point in unrolling this loop
        {
            m *=* this;
        }
        return m;
    }

    Scalar trace() const
    {
        Scalar s = Scalar(0);
        WONKO_BEFORE_LOOP;
        for (Index i = 0; i < std::min(NRows, NCols); ++i)
        {
            s += data_[i * NCols + i];
        }
        WONKO_AFTER_LOOP;
        return s;
    }

    Matrix<Scalar, std::min(NRows, NCols), 1> diagonal() const
    {
        Matrix<Scalar, std::min(NRows, NCols), 1> v;
        WONKO_BEFORE_LOOP;
        for (Index i = 0; i < std::min(NRows, NCols); ++i)
        {
            v.data_[i] = data_[i * NCols + i];
        }
        WONKO_AFTER_LOOP;
        return v;
    }

    Scalar determinant() const;

    bool inverse(Self& result,
                 const Scalar abs_min_determinant = std::numeric_limits<Scalar>::epsilon() * Scalar(10)) const;

    Self pseudoInverse() const;


    template <Index SNRows, Index SNCols>
    Matrix<Scalar, SNRows, SNCols> slice(const Index from_row, const Index from_column) const
    {
        static_assert(SNRows <= NRows, "Number of rows in the slice exceeds that of the source matrix");
        static_assert(SNCols <= NCols, "Number of columns in the slice exceeds that of the source matrix");

        assert(from_row >= 0);
        assert(from_column >= 0);

        const Index to_row    = SNRows + from_row;
        const Index to_column = SNCols + from_column;

        assert(to_row <= NRows);
        assert(to_column <= NCols);

        Matrix<Scalar, SNRows, SNCols> m;

        WONKO_BEFORE_LOOP;
        for (Index r = from_row; r < to_row; ++r)
        {
            for (Index c = from_column; c < to_column; ++c)
            {
                m.data_[(r - from_row) * SNCols + (c - from_column)] = data_[r * NCols + c];
            }
        }
        WONKO_AFTER_LOOP;

        return m;
    }

    template <Index SNRows, Index SNCols, Index FromRow, Index FromCol>
    Matrix<Scalar, SNRows, SNCols> slice() const
    {
        constexpr Index ToRow = SNRows + FromRow;
        constexpr Index ToCol = SNCols + FromCol;

        static_assert(SNRows <= NRows, "Number of rows in the slice exceeds that of the source matrix");
        static_assert(SNCols <= NCols, "Number of columns in the slice exceeds that of the source matrix");
        static_assert(FromRow >= 0,    "Invalid FromRow");
        static_assert(FromCol >= 0,    "Invalid FromCol");
        static_assert(ToRow <= NRows,  "Row range out of range");
        static_assert(ToCol <= NCols,  "Column range out of range");

        Matrix<Scalar, SNRows, SNCols> m;

        WONKO_BEFORE_LOOP;
        for (Index r = FromRow; r < ToRow; ++r)
        {
            for (Index c = FromCol; c < ToCol; ++c)
            {
                m.data_[(r - FromRow) * SNCols + (c - FromCol)] = data_[r * NCols + c];
            }
        }
        WONKO_AFTER_LOOP;

        return m;
    }


    void setRow(const Index index, const Matrix<Scalar, 1, NCols>& value)
    {
        assert((index >= 0) && (index < NRows));
        WONKO_BEFORE_LOOP;
        for (Index c = 0; c < NCols; ++c)
        {
            data_[index * NCols + c] = value.data_[c];
        }
        WONKO_AFTER_LOOP;
    }

    void setColumn(const Index index, const Matrix<Scalar, NRows, 1>& value)
    {
        assert((index >= 0) && (index < NCols));
        WONKO_BEFORE_LOOP;
        for (Index r = 0; r < NRows; ++r)
        {
            data_[r * NCols + index] = value.data_[r];
        }
        WONKO_AFTER_LOOP;
    }


    void swapRows(const Index a, const Index b)
    {
        const Matrix<Scalar, 1, NCols> tmp = slice<1, NCols>(a, 0);
        setRow(a, slice<1, NCols>(b, 0));
        setRow(b, tmp);
    }

    void swapColumns(const Index a, const Index b)
    {
        const Matrix<Scalar, NRows, 1> tmp = slice<NRows, 1>(0, a);
        setColumn(a, slice<NRows, 1>(0, b));
        setColumn(b, tmp);
    }


    Scalar& operator()(const Index row, const Index column)
    {
        assert((row >= 0) && (row < NRows) && (column >= 0) && (column < NCols));
        return data_[row * NCols + column];
    }

    const Scalar& operator()(const Index row, const Index column) const
    {
        assert((row >= 0) && (row < NRows) && (column >= 0) && (column < NCols));
        return data_[row * NCols + column];
    }

    Scalar& operator[](const Index i)
    {
        static_assert(isVector(), "operator[] can only be used with vectors");
        assert(i >= 0 && i < NElements);
        return data_[i];
    }

    const Scalar& operator[](const Index i) const
    {
        static_assert(isVector(), "operator[] can only be used with vectors");
        assert(i >= 0 && i < NElements);
        return data_[i];
    }

    template <Index Row, Index Column>
    Scalar& at()
    {
        static_assert((Row >= 0) && (Row < NRows) && (Column >= 0) && (Column < NCols), "Index out of range");
        return data_[Row * NCols + Column];
    }

    template <Index Row, Index Column>
    const Scalar& at() const
    {
        static_assert((Row >= 0) && (Row < NRows) && (Column >= 0) && (Column < NCols), "Index out of range");
        return data_[Row * NCols + Column];
    }


#define WONKO_SCALAR_OPERATOR(op)                               \
    Self& operator op##= (const Scalar& rhs)                    \
    {                                                           \
        WONKO_ITER_ELEMENTS(data_[i] op##= rhs)                 \
        return *this;                                           \
    }                                                           \
    Self operator op (const Scalar& rhs) const                  \
    {                                                           \
        Self m;                                                 \
        WONKO_ITER_ELEMENTS(m.data_[i] = data_[i] op rhs)       \
        return m;                                               \
    }

    WONKO_SCALAR_OPERATOR(+)
    WONKO_SCALAR_OPERATOR(-)
    WONKO_SCALAR_OPERATOR(*)
    WONKO_SCALAR_OPERATOR(/)
    WONKO_SCALAR_OPERATOR(%)

#undef WONKO_SCALAR_OPERATOR


#define WONKO_ELEMENTWISE_OPERATOR(op)                                  \
    Self& operator op##= (const Self& rhs)                              \
    {                                                                   \
        WONKO_ITER_ELEMENTS(data_[i] op##= rhs.data_[i])                \
        return *this;                                                   \
    }                                                                   \
    Self operator op (const Self& rhs) const                            \
    {                                                                   \
        Self m;                                                         \
        WONKO_ITER_ELEMENTS(m.data_[i] = data_[i] op rhs.data_[i])      \
        return m;                                                       \
    }

    WONKO_ELEMENTWISE_OPERATOR(+)
    WONKO_ELEMENTWISE_OPERATOR(-)

#undef WONKO_ELEMENTWISE_OPERATOR


    template <Index RhsNCols>
    Matrix<Scalar, NRows, RhsNCols> operator*(const Matrix<Scalar, NCols, RhsNCols>& rhs) const
    {
        Matrix<Scalar, NRows, RhsNCols> m;

        WONKO_BEFORE_LOOP;
        for (Index i = 0; i < NRows; ++i)
        {
            for (Index k = 0; k < RhsNCols; ++k)
            {
                for (Index j = 0; j < NCols; ++j)
                {
                    m.data_[i * RhsNCols + k] += data_[i * NCols + j] * rhs.data_[j * RhsNCols + k];
                }
            }
        }
        WONKO_AFTER_LOOP;

        return m;
    }

    Self& operator*=(const Matrix<Scalar, NCols, NRows>& rhs)
    {
        (*this) = (*this) * rhs;
        return *this;
    }


    Self operator-() const
    {
        Self m;
        WONKO_ITER_ELEMENTS(m.data_[i] = -data_[i])
        return m;
    }


    bool operator==(const Self& rhs) const
    {
        bool all_close = true;
        WONKO_ITER_ELEMENTS(all_close = all_close && areClose(data_[i], rhs.data_[i]))
        return all_close;
    }

    bool operator!=(const Self& rhs) const { return !operator==(rhs); }


    bool close(const Self& rhs, const Scalar& epsilon) const
    {
        bool all_close = true;
        WONKO_ITER_ELEMENTS(all_close = all_close && areClose(data_[i], rhs.data_[i], epsilon))
        return all_close;
    }


    template <typename Stream>
    Stream& toString(Stream& s, const StringRepresentation representation = StringRepresentation::MultiLine) const
    {
        const bool single_line = representation == StringRepresentation::SingleLine;

        for (Index r = 0; r < NRows; r++)
        {
            if (r > 0)
            {
                s << (single_line ? "; " : "\n");
            }

            for (Index c = 0; c < NCols; c++)
            {
                if (c > 0)
                {
                    s << (single_line ? ", " : "  ");
                }

                s << operator()(r, c);
            }
        }

        return s;
    }
};

/**********************************************************************************************************************
 * MATH ROUTINES
 */
namespace impl_
{

template<typename Scalar, Index NRows, Index NCols>
inline Scalar determinant(const Matrix<Scalar, NRows, NCols>& m)
{
    static_assert(NRows == NCols, "Cannot compute determinant of a non-square matrix");
    // This is not implemented yet
    assert(false);
    return Scalar(0);
}

template<typename Scalar>
inline Scalar determinant(const Matrix<Scalar, 1, 1>& m)
{
    return m.template at<0, 0>();
}

template<typename Scalar>
inline Scalar determinant(const Matrix<Scalar, 2, 2>& m)
{
    return m.template at<0, 0>() * m.template at<1, 1>() - m.template at<0, 1>() * m.template at<1, 0>();
}

template<typename Scalar>
inline Scalar determinant(const Matrix<Scalar, 3, 3>& m)
{
    auto& e = m.data();
    // Autogenerated with SymPy
    return e[0] * e[4] * e[8] - e[0] * e[5] * e[7] - e[1] * e[3] * e[8] + e[1] * e[5] * e[6] + e[2] * e[3] * e[7] -
           e[2] * e[4] * e[6];
}

template<typename Scalar>
inline Scalar determinant(const Matrix<Scalar, 4, 4>& m)
{
    // TODO: This is numerically unstable and should be removed in favor of the non-specialized variant
    auto& e = m.data();
    // Autogenerated with SymPy
    return -e[0]  * e[10] * e[13] * e[7] + e[0]  * e[10] * e[15] * e[5] + e[0]  * e[11] * e[13] * e[6] -
            e[0]  * e[11] * e[14] * e[5] + e[0]  * e[14] * e[7]  * e[9] - e[0]  * e[15] * e[6]  * e[9] +
            e[10] * e[12] * e[1]  * e[7] - e[10] * e[12] * e[3]  * e[5] + e[10] * e[13] * e[3]  * e[4] -
            e[10] * e[15] * e[1]  * e[4] - e[11] * e[12] * e[1]  * e[6] + e[11] * e[12] * e[2]  * e[5] -
            e[11] * e[13] * e[2]  * e[4] + e[11] * e[14] * e[1]  * e[4] - e[12] * e[2]  * e[7]  * e[9] +
            e[12] * e[3]  * e[6]  * e[9] + e[13] * e[2]  * e[7]  * e[8] - e[13] * e[3]  * e[6]  * e[8] -
            e[14] * e[1]  * e[7]  * e[8] - e[14] * e[3]  * e[4]  * e[9] + e[14] * e[3]  * e[5]  * e[8] +
            e[15] * e[1]  * e[6]  * e[8] + e[15] * e[2]  * e[4]  * e[9] - e[15] * e[2]  * e[5]  * e[8];
}


template<typename Scalar, Index NRows, Index NCols>
inline Matrix<Scalar, NRows, NCols> inverse(const Scalar det, const Matrix<Scalar, NRows, NCols>& m)
{
    static_assert(NRows == NCols, "Cannot compute inverse of a non-square matrix");
    // This is not implemented yet
    (void) det;
    (void) m;
    Matrix<Scalar, NRows, NCols> out;
    out.fill(std::numeric_limits<Scalar>::quiet_NaN());
    return out;
}

template<typename Scalar>
inline Matrix<Scalar, 1, 1> inverse(const Scalar, const Matrix<Scalar, 1, 1>& m)
{
    return Matrix<Scalar, 1, 1>(Scalar(1) / m.template at<0, 0>());
}

template<typename Scalar>
inline Matrix<Scalar, 2, 2> inverse(const Scalar det, const Matrix<Scalar, 2, 2>& m)
{
    return Matrix<Scalar, 2, 2>(m.template at<1, 1>(), -m.template at<0, 1>(),
                               -m.template at<1, 0>(),  m.template at<0, 0>()) / det;
}

template<typename Scalar>
inline Matrix<Scalar, 3, 3> inverse(const Scalar det, const Matrix<Scalar, 3, 3>& m)
{
    const Scalar a = m.template at<0, 0>();
    const Scalar b = m.template at<0, 1>();
    const Scalar c = m.template at<0, 2>();
    const Scalar d = m.template at<1, 0>();
    const Scalar e = m.template at<1, 1>();
    const Scalar f = m.template at<1, 2>();
    const Scalar g = m.template at<2, 0>();
    const Scalar h = m.template at<2, 1>();
    const Scalar i = m.template at<2, 2>();

    return Matrix<Scalar, 3, 3>(+(e * i - f * h), -(b * i - c * h), +(b * f - c * e),
                                -(d * i - f * g), +(a * i - c * g), -(a * f - c * d),
                                +(d * h - e * g), -(a * h - b * g), +(a * e - b * d)) / det;
}

template<typename Scalar>
inline Matrix<Scalar, 4, 4> inverse(const Scalar det, const Matrix<Scalar, 4, 4>& m)
{
    // TODO: This is numerically unstable and should be removed in favor of the non-specialized variant
    // Direct implementation of the Cayley-Hamilton method
    using Self = Matrix<Scalar, 4, 4>;

    const Self m2 = m * m;
    const Self m3 = m * m2;

    const Scalar tr  = m.trace();
    const Scalar tr2 = m2.trace();
    const Scalar tr3 = m3.trace();

    const Scalar x = ((tr * tr * tr) - Scalar(3) * tr * tr2 + Scalar(2) * tr3) / Scalar(6);

    Self A;                     // TODO: this should be a diagonal matrix
    A.template at<0, 0>() = x;
    A.template at<1, 1>() = x;
    A.template at<2, 2>() = x;
    A.template at<3, 3>() = x;

    const Self B = m * (Scalar(0.5) * (tr * tr - tr2));
    const Self C = m2 * tr;

    return (A - B + C - m3) / det;
}

}   // namespace impl_

/**********************************************************************************************************************
 * METHOD DEFINITIONS
 * This section is moved towards the end of the file in order to make the math functions defined above accessible.
 */
template <typename Scalar, Index NRows, Index NCols>
inline Scalar Matrix<Scalar, NRows, NCols>::determinant() const
{
    return impl_::determinant<Scalar>(*this);
}

template <typename Scalar, Index NRows, Index NCols>
inline bool Matrix<Scalar, NRows, NCols>::inverse(Self& result, const Scalar abs_min_determinant) const
{
    const Scalar det = determinant();

    if (std::isfinite(det) && (std::abs(det) > abs_min_determinant))
    {
        result = impl_::inverse<Scalar>(det, *this);
        return !result.hasNonFiniteElements();
    }

    return false;
}

/**********************************************************************************************************************
 * STATIC OPERATORS
 */
template <typename Stream, typename Scalar, Index NRows, Index NCols>
inline Stream& operator<<(Stream& s, const Matrix<Scalar, NRows, NCols>& m)
{
    return m.toString(s);
}


template <typename Scalar, Index NRows, Index NCols>
inline Matrix<Scalar, NRows, NCols> operator+(const Scalar& s, const Matrix<Scalar, NRows, NCols>& m)
{
    return m + s;
}

template <typename Scalar, Index NRows, Index NCols>
inline Matrix<Scalar, NRows, NCols> operator-(const Scalar& s, const Matrix<Scalar, NRows, NCols>& m)
{
    Matrix<Scalar, NRows, NCols> r;
    WONKO_ITER_ELEMENTS(r.data_[i] = s - m.data_[i])
    return r;
}

template <typename Scalar, Index NRows, Index NCols>
inline Matrix<Scalar, NRows, NCols> operator*(const Scalar& s, const Matrix<Scalar, NRows, NCols>& m)
{
    return m * s;
}

template <typename Scalar, Index NRows, Index NCols>
inline Matrix<Scalar, NRows, NCols> operator/(const Scalar& s, const Matrix<Scalar, NRows, NCols>& m)
{
    Matrix<Scalar, NRows, NCols> r;
    WONKO_ITER_ELEMENTS(r.data_[i] = s / m.data_[i])
    return r;
}

}

#undef WONKO_ITER_ROWS_COLS
#undef WONKO_ITER_ELEMENTS
