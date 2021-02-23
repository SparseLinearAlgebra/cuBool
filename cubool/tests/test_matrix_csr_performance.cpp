/**********************************************************************************/
/*                                                                                */
/* MIT License                                                                    */
/*                                                                                */
/* Copyright (c) 2020 JetBrains-Research                                          */
/*                                                                                */
/* Permission is hereby granted, free of charge, to any person obtaining a copy   */
/* of this software and associated documentation files (the "Software"), to deal  */
/* in the Software without restriction, including without limitation the rights   */
/* to use, copy, modify, merge, publish, distribute, sublicense, and/or sell      */
/* copies of the Software, and to permit persons to whom the Software is          */
/* furnished to do so, subject to the following conditions:                       */
/*                                                                                */
/* The above copyright notice and this permission notice shall be included in all */
/* copies or substantial portions of the Software.                                */
/*                                                                                */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR     */
/* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,       */
/* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE    */
/* AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER         */
/* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,  */
/* OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE  */
/* SOFTWARE.                                                                      */
/*                                                                                */
/**********************************************************************************/

#include <gtest/gtest.h>
#include <test_common.hpp>
#include <cubool/cubool.h>

CuBoolMatrix createMatrixFromTestData(const testing::Matrix& source, CuBoolInstance instance) {
    CuBoolMatrix matrix = nullptr;

    CuBool_Matrix_New(instance, &matrix, source.mNrows, source.mNcols);
    CuBool_Matrix_Build(instance, matrix, source.mRowsIndex.data(), source.mColsIndex.data(), source.mNvals, CUBOOL_HINT_VALUES_SORTED);

    return matrix;
}

struct KronResult {
    double averageTimeMs;
    size_t averageNvalsA;
    size_t averageNvalsB;
    size_t averageNvalsR;
};

KronResult matrixKronSample(size_t N, CuBoolSize_t m, CuBoolSize_t n, CuBoolSize_t k, CuBoolSize_t t, double densityA, double densityB, CuBoolInstance instance) {
    testing::TimeQuery timeQuery;
    size_t nvalsA = 0.0;
    size_t nvalsB = 0.0;
    size_t nvalsR = 0.0;


    for (size_t i = 0; i < N; i++) {
        testing::Matrix ta = testing::Matrix::generateSparse(m, n, densityA);
        testing::Matrix tb = testing::Matrix::generateSparse(k, t, densityB);

        CuBoolMatrix a = createMatrixFromTestData(ta, instance);
        CuBoolMatrix b = createMatrixFromTestData(tb, instance);
        CuBoolMatrix r = nullptr;
        CuBoolSize_t nvals;

        CuBool_Matrix_New(instance, &r, 0, 0);
        {
            testing::TimeScope scope(timeQuery);

            CuBool_Kron(instance, r, a, b);
        }

        CuBool_Matrix_Nvals(instance, r, &nvals);

        nvalsA += ta.mNvals;
        nvalsB += tb.mNvals;
        nvalsR += nvals;

        CuBool_Matrix_Free(instance, a);
        CuBool_Matrix_Free(instance, b);
        CuBool_Matrix_Free(instance, r);
    }

    return {timeQuery.getAverageTimeMs(), nvalsA / N,nvalsB / N, nvalsR / N };
}

void matrixKronBench(size_t N, const std::vector<double> &da, const std::vector<double> &db, const std::vector<unsigned>& ms, const std::vector<unsigned>& ns, const std::vector<unsigned>& ks, const std::vector<unsigned>& ts, bool useUnifiedMemory = false) {
    assert(!da.empty());
    assert(!ms.empty());
    assert(!ns.empty());
    assert(da.size() == db.size());
    assert(ms.size() == ns.size());
    assert(ks.size() == ts.size());
    assert(ms.size() == ks.size());
    assert(ms.size() == da.size());

    CuBoolInstance instance;
    CuBoolInstanceDesc instanceDesc;

    testing::details::setupInstanceDescSilent(instanceDesc, useUnifiedMemory);
    CuBool_Instance_New(&instanceDesc, &instance);

    for (int i = 0; i < ms.size(); i++) {
        auto m = ms[i];
        auto n = ns[i];
        auto k = ks[i];
        auto t = ts[i];
        auto densityA = da[i];
        auto densityB = db[i];

        auto result = matrixKronSample(N, m, n, k, t, densityA, densityB, instance);

        std::cout
            << m << " x " << n << " (" << result.averageNvalsA << ") "
            << k << " x " << t << " (" << result.averageNvalsB << ") "
            << m * k << " x " << n * t << " (" << result.averageNvalsR << ") "
            << result.averageTimeMs << " ms\n";
    }

    CuBool_Instance_Free(instance);
}

struct MultAddResult {
    double averageTimeMs;
    size_t averageNvalsA;
    size_t averageNvalsB;
    size_t averageNvalsC;
    size_t averageNvalsR;
};

MultAddResult matrixMultAddSample(size_t N, CuBoolSize_t m, CuBoolSize_t n, CuBoolSize_t t, double densityA, double densityB, double densityC, CuBoolInstance instance) {
    testing::TimeQuery timeQuery;
    size_t nvalsA = 0.0;
    size_t nvalsB = 0.0;
    size_t nvalsC = 0.0;
    size_t nvalsR = 0.0;

    for (size_t i = 0; i < N; i++) {
        testing::Matrix ta = testing::Matrix::generateSparse(m, t, densityA);
        testing::Matrix tb = testing::Matrix::generateSparse(t, n, densityB);
        testing::Matrix tr = testing::Matrix::generateSparse(m, n, densityC);

        CuBoolMatrix a = createMatrixFromTestData(ta, instance);
        CuBoolMatrix b = createMatrixFromTestData(tb, instance);
        CuBoolMatrix r = createMatrixFromTestData(tr, instance);

        CuBoolSize_t nvals;

        {
            testing::TimeScope scope(timeQuery);

            CuBool_MxM(instance, r, a, b);
        }

        CuBool_Matrix_Nvals(instance, r, &nvals);

        nvalsA += ta.mNvals;
        nvalsB += tb.mNvals;
        nvalsC += tr.mNvals;
        nvalsR += nvals;

        CuBool_Matrix_Free(instance, a);
        CuBool_Matrix_Free(instance, b);
        CuBool_Matrix_Free(instance, r);
    }

    return {timeQuery.getAverageTimeMs(), nvalsA / N,nvalsB / N, nvalsC / N, nvalsR / N };
}

void matrixMultAddBench(size_t N, const std::vector<double> &da, const std::vector<double> &db, const std::vector<double> &dc, const std::vector<unsigned>& ms, const std::vector<unsigned>& ns, const std::vector<unsigned>& ts, bool useUnifiedMemory = false) {
    assert(!da.empty());
    assert(!ms.empty());
    assert(!ns.empty());
    assert(da.size() == db.size());
    assert(da.size() == dc.size());
    assert(ms.size() == ns.size());
    assert(ms.size() == ts.size());
    assert(ms.size() == da.size());

    CuBoolInstance instance;
    CuBoolInstanceDesc instanceDesc;

    testing::details::setupInstanceDescSilent(instanceDesc, useUnifiedMemory);
    CuBool_Instance_New(&instanceDesc, &instance);

    for (int i = 0; i < ms.size(); i++) {
        auto m = ms[i];
        auto n = ns[i];
        auto t = ts[i];
        auto densityA = da[i];
        auto densityB = db[i];
        auto densityC = dc[i];

        auto result = matrixMultAddSample(N, m, n, t, densityA, densityB, densityC, instance);

        std::cout
            << m << " x " << t << " (" << result.averageNvalsA << ") "
            << t << " x " << n << " (" << result.averageNvalsB << ") "
            << m << " x " << n << " (" << result.averageNvalsC << ") "
            << m << " x " << n << " (" << result.averageNvalsR << ") "
            << result.averageTimeMs << " ms\n";
    }

    CuBool_Instance_Free(instance);
}

static const size_t KRON_ITERATIONS = 2;
static const size_t MULT_ADD_ITERATIONS = 2;

TEST(MatrixBenchmark, KronDist) {
    std::vector<double> densityA = { 0.01   , 0.01    , 0.01     , 0.01       };
    std::vector<double> densityB = { 0.01   , 0.0001  , 0.000001 , 0.00000025 };
    std::vector<unsigned> ms     = { 100    , 100     , 100      , 100        };
    std::vector<unsigned> ns     = { 100    , 100     , 100      , 100        };
    std::vector<unsigned> ks     = { 10000  , 100000  , 1000000  , 2000000    };
    std::vector<unsigned> ts     = { 10000  , 100000  , 1000000  , 2000000    };

    size_t N = KRON_ITERATIONS;

    matrixKronBench(N, densityA, densityB, ms, ns, ks, ts);
}

TEST(MatrixBenchmark, KronSize) {
    std::vector<double> densityA = { 0.01      , 0.01      , 0.01      , 0.01      };
    std::vector<double> densityB = { 0.0000001 , 0.0000002 , 0.0000004 , 0.0000008 };
    std::vector<unsigned> ms     = { 100       , 100       , 100       , 100       };
    std::vector<unsigned> ns     = { 100       , 100       , 100       , 100       };
    std::vector<unsigned> ks     = { 2000000   , 2000000   , 2000000   , 2000000   };
    std::vector<unsigned> ts     = { 2000000   , 2000000   , 2000000   , 2000000   };

    size_t N = KRON_ITERATIONS;

    matrixKronBench(N, densityA, densityB, ms, ns, ks, ts);
}

TEST(MatrixBenchmark, KronDistManaged) {
    std::vector<double> densityA = { 0.01   , 0.01    , 0.01     , 0.01       , 0.01         };
    std::vector<double> densityB = { 0.01   , 0.0001  , 0.000001 , 0.00000025 , 0.0000000625 };
    std::vector<unsigned> ms     = { 100    , 100     , 100      , 100        , 100          };
    std::vector<unsigned> ns     = { 100    , 100     , 100      , 100        , 100          };
    std::vector<unsigned> ks     = { 10000  , 100000  , 1000000  , 2000000    , 4000000      };
    std::vector<unsigned> ts     = { 10000  , 100000  , 1000000  , 2000000    , 4000000      };

    size_t N = KRON_ITERATIONS;
    const auto useUnifiedMemory = true;

    matrixKronBench(N, densityA, densityB, ms, ns, ks, ts, useUnifiedMemory);
}

TEST(MatrixBenchmark, KronSizeManaged) {
    std::vector<double> densityA = { 0.01      , 0.01      , 0.01      , 0.01      };
    std::vector<double> densityB = { 0.0000001 , 0.0000002 , 0.0000004 , 0.0000008 };
    std::vector<unsigned> ms     = { 100       , 100       , 100       , 100       };
    std::vector<unsigned> ns     = { 100       , 100       , 100       , 100       };
    std::vector<unsigned> ks     = { 4000000   , 4000000   , 4000000   , 4000000   };
    std::vector<unsigned> ts     = { 4000000   , 4000000   , 4000000   , 4000000   };

    size_t N = KRON_ITERATIONS;
    const auto useUnifiedMemory = true;

    matrixKronBench(N, densityA, densityB, ms, ns, ks, ts, useUnifiedMemory);
}

int main(int argc, char *argv[]) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}