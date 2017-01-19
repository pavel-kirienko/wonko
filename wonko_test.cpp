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

#include "wonko.hpp"
#include <iostream>
#include <sstream>
#include <ctime>
#include <typeinfo>
#include <array>
#include <iomanip>


#define STRINGIZE_(x)   #x
#define STRINGIZE(x)    STRINGIZE_(x)

#define GLUE_(A, B)     A##B
#define GLUE(A, B)      GLUE_(A, B)

#define TEST                                                                                \
    static void GLUE(_test_body, __LINE__)();                                               \
    struct GLUE(_test_invoker, __LINE__)                                                    \
    {                                                                                       \
        GLUE(_test_invoker, __LINE__)()                                                     \
        {                                                                                   \
            std::cout << "\033[36m" "Running tests at " << __FILE__ << ":" << __LINE__      \
                      << "\033[39m" << std::endl;                                           \
            GLUE(_test_body, __LINE__)();                                                   \
        }                                                                                   \
    } GLUE(_test_invoker_instance, __LINE__);                                               \
    static void GLUE(_test_body, __LINE__)()

#define TEST_CONDITION(x, y, op)                                                                \
    do {                                                                                        \
        try                                                                                     \
        {                                                                                       \
            const auto evaluated_x = (x);                                                       \
            const auto evaluated_y = (y);                                                       \
            if (evaluated_x op evaluated_y)                                                     \
            {                                                                                   \
                TestEnvironment::registerSuccessfulTest();                                      \
            }                                                                                   \
            else                                                                                \
            {                                                                                   \
                TestEnvironment::registerFailedTest();                                          \
                std::cout << "\033[31m\nTEST FAILED AT " << __FILE__ << ":" << __LINE__         \
                          << ":\n\033[33m"                                                      \
                          << "LEFT SIDE\n"  << STRINGIZE(x) << "\nWHICH IS\n" << evaluated_x    \
                          << "\nOPERATOR " STRINGIZE(op) "\n"                                   \
                          << "RIGHT SIDE\n" << STRINGIZE(y) << "\nWHICH IS\n" << evaluated_y    \
                          << "\033[39m\n" << std::endl;                                         \
            }                                                                                   \
        }                                                                                       \
        catch (std::exception& ex)                                                              \
        {                                                                                       \
            std::cout << "\033[31m\nEXCEPTION AT " << __FILE__ << ":" << __LINE__               \
                          << ":\n\033[33m" << ex.what() << "\n" << std::endl;                   \
        }                                                                                       \
    } while (false)

#define EQUAL(x, y)         TEST_CONDITION(x, y, ==)
#define TRUE(x)             TEST_CONDITION(x, true, ==)
#define FALSE(x)            TEST_CONDITION(x, false, ==)


class TestEnvironment
{
    unsigned successful_ = 0;
    unsigned failed_ = 0;

    static TestEnvironment& instance()
    {
        static TestEnvironment obj;
        return obj;
    }

    TestEnvironment()
    {
        std::cout << std::fixed << std::setprecision(12) << std::scientific;
        std::srand(unsigned(std::time(nullptr)));
    }

public:
    static void registerSuccessfulTest() { instance().successful_++; }
    static void registerFailedTest()     { instance().failed_++; }

    static unsigned getNumFailedTests()  { return instance().failed_; }

    ~TestEnvironment()
    {
        std::printf("TEST REPORT: %u successful, %u failed\n", successful_, failed_);
    }
};


using namespace wonko;


TEST
{
    const Matrix<float, 3, 3> m(1.0F, 2.0F, 3.0F,
                                4.0F, 5.0F, 6.0F,
                                7.0F, 8.0F, 9.0F);
    {
        std::ostringstream os;
        EQUAL(m.toString(os, StringRepresentation::SingleLine).str(),
              "1, 2, 3; 4, 5, 6; 7, 8, 9");
    }
    {
        std::ostringstream os;
        EQUAL((m.slice<2, 2>(0, 0).toString(os).str()),
              "1  2\n"
              "4  5");
    }
    {
        std::ostringstream os;
        EQUAL((m.transpose().slice<3, 1>(0, 0).transpose().toString(os).str()),
              "1  2  3");
    }
}

TEST
{
    {
        EQUAL((Matrix<int, 1, 3>(42, 67, 23).trace()), 42);

        EQUAL((Matrix<int, 2, 4>(-234, 12, -74, 98,
                                  95, 41, 27, -87).trace()), -193);

        EQUAL((Matrix<int, 4, 2>(-234, 12,
                                 -74, 98,
                                  95, 41,
                                  27, -87).trace()), -136);

        EQUAL((Matrix<int, 1, 1>(123).trace()), 123);
    }
    {
        TRUE(areClose(Matrix<double, 1, 1>(123).power(3).at<0, 0>(), 1860867.0));

        EQUAL((Matrix<double, 5, 5>::random().power(0)), (Matrix<double, 5, 5>::identity()));

        const auto m10 = Matrix<double, 10, 10>::random();
        EQUAL(m10.power(1), m10);

        const auto m4 = Matrix<float, 4, 4>(  5.F, -4.F,  3.F,  2.F,
                                            -12.F,  7.F, -9.F, 11.F,
                                              6.F,  1.F,  2.F, 10.F,
                                              3.F,  5.F,  9.F, -4.F);

        EQUAL(m4.power(4), (Matrix<int, 4, 4>(19720,  -5919,  16278,   -108,
                                             -40437,  24622, -12429, -23184,
                                               6216,   6546,  14317,  -2745,
                                              -6012,   1935, -15021,  26791).cast<float>()));

        EQUAL(m4.power(5), (Matrix<int, 4, 4>(266972, -104575,  144015,  137543,
                                             -641775,  205753, -576423,  158414,
                                               30195,   21550,  -36337,  238588,
                                              -63033,  156527,  175626, -248113).cast<float>()));

        EQUAL(m4.diagonal(), (Matrix<int, 4, 1>(5, 7, 2, -4).cast<float>()));
    }
}

TEST    // Reference values were computed with SymPy
{
    static const auto inverse = [](auto mat)
    {
        auto inv = mat;
        if (mat.inverse(inv))
        {
            return inv;
        }
        std::ostringstream os;
        os << mat;
        throw std::runtime_error("Failed to invert the matrix:\n" + os.str());
    };

    {
        using M = Matrix<double, 2, 2>;
        M m(-21, 440,
            637, 115);

        EQUAL(inverse(m.transpose()),
              M(-0.000406798846813704, 0.00225331187322025,
                 0.001556447761722000, 7.4285006809459e-5));

        TRUE(areClose((m * 100500.0).determinant(), -2.85529017375000e+15));

        TRUE(areClose(inverse(m.transpose()).determinant(), -3.53738127664090e-6));

        TRUE(areClose(m.abs().min(),  21.0));
        TRUE(areClose(m.min(),       -21.0));
        TRUE(areClose(m.max(),       637.0));
        TRUE(areClose(m.absmax(),    637.0));
        TRUE(areClose((-m).max(),     21.0));
        TRUE(areClose((-m).absmax(), 637.0));
        TRUE(areClose(m.mean(),      292.75));
    }
    {
        using M = Matrix<float, 3, 3>;
        M m( -21.0F,  440.0F, -831.0F,
             637.0F,  115.0F,   35.0F,
            -683.0F,  967.0F,  -15.0F);

        EQUAL(inverse(m.transpose()),
              M( 6.10416928078817e-5F,  2.46260413773715e-5F,  -0.00119187294505767F,
                 0.00136769258388943F,  0.000973471705898605F,  0.000480873653831177F,
                -0.000190427085814636F, 0.000907151288123699F,  0.000485133015134218F));

        TRUE(areClose((m / 9000.0F).determinant(), -0.000799336727023320F));

        TRUE(areClose(inverse((-m).transpose()).determinant(), 1.71610044441613e-9F));
    }
    {
        using M = Matrix<double, 4, 4>;
        M m( -21.0, 440.0, -831.0,  23.0,
             637.0, 115.0,   35.0, -12.0,
            -683.0, 967.0,  -15.0, 224.0,
             523.0,   2.0,  -78.0, 631.0);

        EQUAL(inverse(m).transpose(),
              M( 5.02347911454767e-5,   6.06252886422982e-5,  -0.00117772595503502,   -0.000187411522724407,
                 0.00130889718999846,   0.0011693270943198,    0.000557840936104745,  -0.00101962011337823,
                -0.000178696461441125,  0.0008680749958999,    0.000469776807485104,   0.000203430571039218,
                 8.64966298675872e-5,  -0.000288131942295583, -0.000113230137365453,   0.00150001042115584));

        TRUE(areClose(inverse(m * -42).transpose().determinant(), -8.27255795673604e-19));

        EQUAL((m + 123.0).transpose(),
              M( 102.0, 760.0, -560.0, 646.0,
                 563.0, 238.0, 1090.0, 125.0,
                -708.0, 158.0,  108.0,  45.0,
                 146.0, 111.0,  347.0, 754.0));
    }
    {
        // This matrix was identified as problematic by the random inversion checks defined below
        Matrix<long double, 4, 4> m(-228.46100,  -90.49140, -276.73600,  236.49500,
                                     -13.31220,    1.90467,   38.36900,  -34.63270,
                                    -222.77200,  199.00600,   50.70680,  -63.82020,
                                      16.27670, -252.69100, -279.22400,  257.38300);

        std::cout << "long double: " << m.determinant() << std::endl;
        std::cout << "double:      " << m.cast<double>().determinant() << std::endl;
        std::cout << "float:       " << m.cast<float>().determinant() << std::endl;

        TRUE(areClose(float(m.determinant()), 722.047106399816F));
    }
    {
        // This matrix was identified as problematic by the random inversion checks defined below
        Matrix<long double, 4, 4> m(-4155.87500000,    84.96509552, -5386.86035156,  5403.66845703,
                                    -4276.23681641, -5092.47265625, -4621.94335938,  5282.27441406,
                                    -3884.02709961,  1158.95825195, -3934.80688477,  1043.06616211,
                                    -6007.82324219, -3909.71850586, -5175.27294922,  1630.80334473);

        std::cout << "long double: " << m.determinant() << std::endl;
        std::cout << "double:      " << m.cast<double>().determinant() << std::endl;
        std::cout << "float:       " << m.cast<float>().determinant() << std::endl;

        TRUE(areClose(float(m.determinant()), -59257103.1428230F));
    }
}

TEST    // Reference values were computed with SymPy
{
    EQUAL((Matrix<long, 3, 1>(1, 2, 3).transpose() * Matrix<long, 3, 1>(7, 8, 9))[0], 50);

    TRUE(areClose(Matrix<float, 3, 1>(1.0F, 2.2F, 3.4F).sum(), 6.6F));

    TRUE(areClose(Matrix<float, 3, 1>(1.0F, 2.0F, 3.0F).norm(), 3.74165738677394F));

    TRUE(areClose(Matrix<double, 3, 1>(1.0, 2.0, 3.0).normalized().norm(), 1.0));

    EQUAL((Matrix<float, 3, 1>(1.0F, 2.0F, 3.0F).normalized()),
          (Matrix<float, 3, 1>(0.267261241912424F, 0.534522483824849F, 0.801783725737273F)));

    {
        Matrix<float,  96, 100> a;
        Matrix<long,  143, 100> b;

        for (int r = 0; r < a.numRows(); r++)
        {
            for (int c = 0; c < a.numColumns(); c++)
            {
                a(r, c) = float(r - c);
            }
        }

        for (int r = 0; r < b.numRows(); r++)
        {
            for (int c = 0; c < b.numColumns(); c++)
            {
                b(r, c) = c - r;
            }
        }

        const Matrix<double, 96, 143> p = a.cast<double>() * b.transpose().cast<double>() * 0.00001;

        EQUAL((p.slice<3, 2>(23, 87)),
              (Matrix<double, 3, 2>(p(23, 87), p(23, 88),
                                    p(24, 87), p(24, 88),
                                    p(25, 87), p(25, 88))));

        EQUAL((p.slice<3, 2>(23, 87)),
              (p.slice<3, 2, 23, 87>()));

        EQUAL((p.slice<4, 3>(23, 87)),
              (Matrix<double, 4, 3>(0.1605, 0.1870, 0.2135,
                                    0.1230, 0.1485, 0.1740,
                                    0.0855, 0.1100, 0.1345,
                                    0.0480, 0.0715, 0.0950)));

        FALSE(a.hasNonFiniteElements());
        FALSE(b.hasNonFiniteElements());
        FALSE(p.hasNonFiniteElements());

        a(1, 3) = std::numeric_limits<float>::quiet_NaN();
        TRUE(a.hasNonFiniteElements());
        FALSE(b.hasNonFiniteElements());
        TRUE((a.cast<double>() * b.cast<double>().transpose()).hasNonFiniteElements());

        EQUAL((Matrix<double, 2, 1>(p.max(), p.min())).transpose(),
              (Matrix<double, 1, 2>(3.7455, -5.042)));

        // TODO: add this test later, when the generic determinant algorithm is implemented
        //TRUE(areClose(p.slice<20, 20>(10, 25).determinant(), 2.88946732480574e-285));
    }
    {
        using S = float;
        auto m = Matrix<S, 3, 3>::ones();

        Matrix<S, 3, 3> inv;
        FALSE(m.inverse(inv));
        EQUAL((Matrix<S, 3, 3>()), inv);

        m(1, 2) = 123.0F;
        m(2, 1) = 45.0F;
        TRUE(areClose(m.determinant(), S(-5368.0)));

        TRUE(m.inverse(inv));
        EQUAL(inv,
              (Matrix<S, 3, 3>( 1.03092399403875000F, -0.00819672131147541F, -0.0227272727272727F,
                               -0.02272727272727270F,                  0.0F,  0.0227272727272727F,
                               -0.00819672131147541F,  0.00819672131147541F,                 0.0F)));

        std::cout << m * inv << std::endl;
    }
}

TEST
{
    {
        Matrix<double, 3, 2> m;

        m.fill(123.0);
        EQUAL((Matrix<double, 3, 2>::ones() * 123.0), m);

        m.fill(0.0);
        EQUAL((Matrix<double, 3, 2>()), m);
    }
}


template <int N, typename Scalar, unsigned NumIterations = 1000000>
void runRandomizedInversionTest(const Scalar epsilon)
{
    std::cout << "Random inversion test: float" << (sizeof(Scalar) * 8)
              << ", epsilon " << std::numeric_limits<Scalar>::epsilon()
              << ", size " << N << "x" << N
              << ", " << NumIterations << " iters" << "..." << std::endl;

    const auto Eye = Matrix<Scalar, N, N>::identity();

    Scalar max_error = 0;

    Scalar min_element = std::numeric_limits<Scalar>::max();
    Scalar max_element = 0;
    Scalar abs_min_element = std::numeric_limits<Scalar>::max();

    std::array<Matrix<Scalar, N>, 3> worst_group;

    for (unsigned i = 0; i < NumIterations; i++)
    {
        const Scalar scale = Scalar(10000) * Scalar(std::rand()) / Scalar(RAND_MAX);

        // Singular matrices are rare
        const auto m = Matrix<Scalar, N>::random() * scale;
        Matrix<Scalar, N> inv;
        if (!m.inverse(inv))
        {
            std::cout << "NON-INVERTIBLE MATRIX:\n" << m << "\ndeterminant " << m.determinant()
                      << " (long double: " << (m.template cast<long double>().determinant()) << ")"
                      << "\n" << std::endl;
            continue;
        }

        // Checking the deviation from identity
        TRUE(Eye.close(m * inv, epsilon));

        // Statistics
        const Scalar new_error = (Eye - m * inv).abs().max();
        if (new_error > max_error)
        {
            max_error = new_error;
            worst_group = { m, inv, m * inv };
        }

        min_element = std::min(min_element, m.min());
        max_element = std::max(max_element, m.max());
        abs_min_element = std::min(abs_min_element, m.abs().min());
    }

    std::cout << "Done.\nWorst error: " << max_error
              << ", min/max/absmin: " << min_element << " / " << max_element << " / " << abs_min_element
              << "\nWorst set (m, m^-1, m*m^-1):\n---\n"
              << worst_group[0] << "\ndeterminant " << worst_group[0].determinant()
              << " (long double: " << (worst_group[0].template cast<long double>().determinant()) << ")"
              << "\n---\n"
              << worst_group[1] << "\n---\n"
              << worst_group[2] << "\n"
              << std::endl;
}

TEST
{
    // range of the random variables is sufficiently wide to render float numerically unstable at 2x2 and larger
    runRandomizedInversionTest<1, float>(0.1F);
    runRandomizedInversionTest<2, float>(1.0F);

    runRandomizedInversionTest<1, double>(1e-9);
    runRandomizedInversionTest<2, double>(1e-8);
    runRandomizedInversionTest<3, double>(1e-7);
    runRandomizedInversionTest<4, double>(1e-6);

    // we're not testing small matrices with long double because that wouldn't provide much useful information anyway
    runRandomizedInversionTest<4, long double>(1e-9);

    // TODO: test larger matrices when generic inversion is supported
}


TEST
{
    // Default initialization
    for (int i = 0; i < 10; i++)
    {
        Matrix<long, 10, 10> z;
        EQUAL(z.min(), 0);
        EQUAL(z.max(), 0);
        z = Matrix<long, 10, 10>::random();

        EQUAL((Matrix<long, 10, 10>{}).min(), 0);
        EQUAL((Matrix<long, 10, 10>{}).max(), 0);
    }
}


TEST
{
    Matrix<long, 3, 5> m{ 1,  2,  3,  4,  5,
                          6,  7,  8,  9, 10,
                         11, 12, 13, 14, 15};
    m.swapColumns(1, 4);
    EQUAL(m,
          (Matrix<long, 3, 5>( 1,  5,  3,  4,  2,
                               6, 10,  8,  9,  7,
                              11, 15, 13, 14, 12)));
    m.swapRows(1, 2);
    EQUAL(m,
          (Matrix<long, 3, 5>( 1,  5,  3,  4,  2,
                              11, 15, 13, 14, 12,
                               6, 10,  8,  9,  7)));
    m.swapColumns(2, 0);
    EQUAL(m,
          (Matrix<long, 3, 5>( 3,  5,  1,  4,  2,
                              13, 15, 11, 14, 12,
                               8, 10,  6,  9,  7)));
    m.swapRows(2, 0);
    EQUAL(m,
          (Matrix<long, 3, 5>( 8, 10,  6,  9,  7,
                              13, 15, 11, 14, 12,
                               3,  5,  1,  4,  2)));
    EQUAL(m.transpose(),
          (Matrix<long, 5, 3>{ 8, 13, 3,
                              10, 15, 5,
                               6, 11, 1,
                               9, 14, 4,
                               7, 12, 2}));
}


TEST
{
    using M = Matrix<long, 2, 2>;
    M m;

    m += 3;
    EQUAL(m, M(3, 3,
               3, 3));

    m -= M::identity() * 3;
    EQUAL(m, M(0, 3,
               3, 0));

    m *= 9;
    EQUAL(m, M( 0, 27,
                27, 0));

    m %= 10;
    EQUAL(m, M(0, 7,
               7, 0));

    m *= M::ones();
    EQUAL(m, M(7, 7,
               7, 7));

    m /= 3;
    EQUAL(m, M(2, 2,
               2, 2));

    EQUAL(-m, M(-2, -2,
                -2, -2));

    EQUAL(2L / m, M::ones());

    EQUAL(2L - m, M());

    EQUAL(10L + m, M::ones() * 12);

    EQUAL(10L * m, M::ones() * 20);
}


int main()
{
    return int(TestEnvironment::getNumFailedTests());
}
