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

#ifndef CUBOOL_TEST_SHARED_HPP
#define CUBOOL_TEST_SHARED_HPP

#include <cubool/cubool.h>
#include <chrono>
#include <vector>
#include <iostream>
#include <algorithm>
#include <random>
#include <unordered_set>

namespace testing {
    namespace details {

        struct Pair {
            CuBoolIndex_t i;
            CuBoolIndex_t j;
        };

        struct PairHash {
        public:
            std::size_t operator()(const Pair &x) const {
                return std::hash<size_t>()(x.i) ^ std::hash<size_t>()(x.j);
            }
        };

        struct PairCmp {
        public:
            bool operator()(const Pair &a, const Pair& b) const {
                return a.i < b.i || (a.i == b.i && a.j < b.j);
            }
        };

        struct PairEq {
        public:
            bool operator()(const Pair &a, const Pair& b)  const {
                return a.i == b.i && a.j == b.j;
            }
        };

        bool operator ==(const Pair& a, const Pair& b) {
            PairEq pairEq;
            return pairEq(a, b);
        }

        static void testMsgFun(CuBoolStatus error, const char* message, void* _) {
            std::cout << "CuBool: " << message << std::endl;
        }

        static CuBoolCpuPtr_t testAllocateFun(size_t size, void* _) {
            CuBoolCpuPtr_t ptr = malloc(size);
            std::cout << "Cubool: Allocate: " << size << " " << ptr << std::endl;
            return ptr;
        }

        static void testDeallocateFun(CuBoolCpuPtr_t ptr, void* _) {
            std::cout << "Cubool: Deallocate: " << ptr << std::endl;
            free(ptr);
        }

        static CuBoolCpuPtr_t testAllocateFunSilent(size_t size, void* _) {
            return malloc(size);
        }

        static void testDeallocateFunSilent(CuBoolCpuPtr_t ptr, void* _) {
            free(ptr);
        }

        struct Condition1 {
            bool operator()(CuBoolIndex_t i, CuBoolIndex_t j) {
                return (((i - 1) & i) == 0 && ((j - 1) & j) == 0);
            }
        };

        struct Condition2 {
            bool operator()(CuBoolIndex_t i, CuBoolIndex_t j) {
                return !(i % 5) && !(j % 7);
            }
        };

        struct Condition3 {
        public:
            explicit Condition3(float density): mDensity(density) {
                mRandomEngine.seed(std::time(0));
            }
            bool operator()(CuBoolIndex_t i, CuBoolIndex_t j) {
                return std::uniform_real_distribution<float>(0.0f, 1.0f)(mRandomEngine) <= mDensity;
            }
        private:
            std::default_random_engine mRandomEngine;
            float mDensity = 1.0f;
        };

        template<typename Condition>
        static void generateTestData(size_t rows, size_t columns, std::vector<Pair> &values, Condition&& condition) {
            for (CuBoolIndex_t i = 0; i < rows; i++) {
                for (CuBoolIndex_t j = 0; j < columns; j++) {
                    // is i and j power of two or 0
                    if (condition(i, j)) {
                        values.push_back(Pair{i, j});
                    }
                }
            }
        }

        template<typename Condition>
        static void generateTestData(size_t nrows, size_t ncols, std::vector<CuBoolIndex_t> &rows, std::vector<CuBoolIndex_t> &cols, size_t& nvals, Condition&& condition) {
            nvals = 0;
            for (CuBoolIndex_t i = 0; i < nrows; i++) {
                for (CuBoolIndex_t j = 0; j < ncols; j++) {
                    // is i and j power of two or 0
                    if (condition(i, j)) {
                        rows.push_back(i);
                        cols.push_back(j);
                        nvals += 1;
                    }
                }
            }
        }

        static void setupInstanceDesc(CuBoolInstanceDesc& instanceDesc) {
            instanceDesc.memoryType = CuBoolGpuMemoryType::CUBOOL_GPU_MEMORY_TYPE_GENERIC;
            instanceDesc.errorCallback.userData = nullptr;
            instanceDesc.errorCallback.msgFun = testMsgFun;
            instanceDesc.allocationCallback.userData = nullptr;
            instanceDesc.allocationCallback.allocateFun = testAllocateFun;
            instanceDesc.allocationCallback.deallocateFun = testDeallocateFun;
        }

        static void setupInstanceDescSilent(CuBoolInstanceDesc& instanceDesc) {
            instanceDesc.memoryType = CuBoolGpuMemoryType::CUBOOL_GPU_MEMORY_TYPE_GENERIC;
            instanceDesc.errorCallback.userData = nullptr;
            instanceDesc.errorCallback.msgFun = testMsgFun;
            instanceDesc.allocationCallback.userData = nullptr;
            instanceDesc.allocationCallback.allocateFun = testAllocateFunSilent;
            instanceDesc.allocationCallback.deallocateFun = testDeallocateFunSilent;
        }

    }

    struct Matrix {
        std::vector<CuBoolIndex_t> mRowsIndex;
        std::vector<CuBoolIndex_t> mColsIndex;
        size_t mNvals = 0;
        size_t mNrows = 0;
        size_t mNcols = 0;

        bool areEqual(CuBoolMatrixDense matrix, CuBoolInstance instance) const {
            using namespace details;

            CuBoolIndex_t* extRows;
            CuBoolIndex_t* extCols;
            size_t extNvals;

            EXPECT_EQ(CuBool_MatrixDense_ExtractPairs(instance, matrix, &extRows, &extCols, &extNvals), CUBOOL_STATUS_SUCCESS);

            if (extNvals != mNvals)
                return false;

            std::vector<Pair> extracted(mNvals);
            std::vector<Pair> reference(mNvals);

            for (CuBoolIndex_t idx = 0; idx < mNvals; idx++) {
                extracted[idx] = Pair{extRows[idx], extCols[idx]};
                reference[idx] = Pair{mRowsIndex[idx], mColsIndex[idx]};
            }

            EXPECT_EQ(CuBool_Vals_Free(instance, extRows), CUBOOL_STATUS_SUCCESS);
            EXPECT_EQ(CuBool_Vals_Free(instance, extCols), CUBOOL_STATUS_SUCCESS);

            std::sort(extracted.begin(), extracted.end(), details::PairCmp());
            std::sort(reference.begin(), reference.end(), details::PairCmp());

            return extracted == reference;
        }

        bool areEqual(CuBoolMatrix matrix, CuBoolInstance instance) const {
            using namespace details;

            CuBoolIndex_t* extRows;
            CuBoolIndex_t* extCols;
            size_t extNvals;

            EXPECT_EQ(CuBool_Matrix_ExtractPairs(instance, matrix, &extRows, &extCols, &extNvals), CUBOOL_STATUS_SUCCESS);

            if (extNvals != mNvals)
                return false;

            std::vector<Pair> extracted(mNvals);
            std::vector<Pair> reference(mNvals);

            for (CuBoolIndex_t idx = 0; idx < mNvals; idx++) {
                extracted[idx] = Pair{extRows[idx], extCols[idx]};
                reference[idx] = Pair{mRowsIndex[idx], mColsIndex[idx]};
            }

            EXPECT_EQ(CuBool_Vals_Free(instance, extRows), CUBOOL_STATUS_SUCCESS);
            EXPECT_EQ(CuBool_Vals_Free(instance, extCols), CUBOOL_STATUS_SUCCESS);

            std::sort(extracted.begin(), extracted.end(), details::PairCmp());
            std::sort(reference.begin(), reference.end(), details::PairCmp());

            return extracted == reference;
        }

        template<typename Condition>
        static Matrix generate(size_t nrows, size_t ncols, Condition&& condition) {
            using namespace details;

            Matrix matrix;
            matrix.mNrows = nrows;
            matrix.mNcols = ncols;
            generateTestData(nrows, ncols, matrix.mRowsIndex, matrix.mColsIndex, matrix.mNvals, std::forward<Condition>(condition));
            return matrix;
        }
    };

    struct MatrixMultiplyAddFunctor {
        Matrix operator()(const Matrix& ma, const Matrix& mb, const Matrix& mc) {
            auto m = ma.mNrows;
            auto t = ma.mNcols;
            auto n = mb.mNcols;

            assert(ma.mNcols == mb.mNrows);
            assert(ma.mNrows == mc.mNrows);
            assert(mb.mNcols == mc.mNcols);

            std::vector<std::vector<uint8_t>> a(m, std::vector<uint8_t>(t, 0));
            std::vector<std::vector<uint8_t>> b(n, std::vector<uint8_t>(t, 0));
            std::vector<std::vector<uint8_t>> r(m, std::vector<uint8_t>(n, 0));

            for (size_t i = 0; i < ma.mNvals; i++) {
                a[ma.mRowsIndex[i]][ma.mColsIndex[i]] = 1;
            }

            for (size_t i = 0; i < mb.mNvals; i++) {
                b[mb.mColsIndex[i]][mb.mRowsIndex[i]] = 1;
            }

            for (size_t i = 0; i < m; i++) {
                for (size_t j = 0; j < n; j++) {
                    uint8_t v = 0;

                    for (size_t k = 0; k < t; k++) {
                        v |= a[i][k] & b[j][k] ? 1u: 0u;
                    }

                    r[i][j] = v;
                }
            }

            for (size_t i = 0; i < mc.mNvals; i++) {
                r[mc.mRowsIndex[i]][mc.mColsIndex[i]] |= 1u;
            }

            Matrix result;
            result.mNrows = m;
            result.mNcols = n;

            for (size_t i = 0; i < m; i++) {
                for (size_t j = 0; j < n; j++) {
                    if (r[i][j] != 0) {
                        result.mRowsIndex.push_back(i);
                        result.mColsIndex.push_back(j);
                        result.mNvals += 1;
                    }
                }
            }

            return std::move(result);
        }
    };

    struct MatrixAddFunctor {
        Matrix operator()(const Matrix& ma, const Matrix& mb) {
            auto m = ma.mNrows;
            auto n = ma.mNcols;

            assert(ma.mNrows == mb.mNrows);
            assert(ma.mNcols == mb.mNcols);

            std::vector<std::vector<uint8_t>> r(m, std::vector<uint8_t>(n, 0));

            for (size_t i = 0; i < ma.mNvals; i++) {
                r[ma.mRowsIndex[i]][ma.mColsIndex[i]] = 1;
            }

            for (size_t i = 0; i < mb.mNvals; i++) {
                r[mb.mRowsIndex[i]][mb.mColsIndex[i]] = 1;
            }

            Matrix result;
            result.mNrows = m;
            result.mNcols = n;

            for (size_t i = 0; i < m; i++) {
                for (size_t j = 0; j < n; j++) {
                    if (r[i][j] != 0) {
                        result.mRowsIndex.push_back(i);
                        result.mColsIndex.push_back(j);
                        result.mNvals += 1;
                    }
                }
            }

            return std::move(result);
        }
    };

    struct MatrixKronFunctor {
        Matrix operator()(const Matrix& ma, const Matrix& mb) {
            auto m = ma.mNrows;
            auto n = ma.mNcols;
            auto k = mb.mNrows;
            auto t = mb.mNcols;

            Matrix result;
            result.mNrows = m * k;
            result.mNcols = n * t;
            result.mNvals = ma.mNvals * mb.mNvals;
            result.mRowsIndex.reserve(result.mNvals);
            result.mColsIndex.reserve(result.mNvals);

            std::vector<details::Pair> vals;
            vals.reserve(result.mNvals);

            for (CuBoolIndex_t i = 0; i < ma.mNvals; i++) {
                auto blockI = ma.mRowsIndex[i];
                auto blockJ = ma.mColsIndex[i];

                for (CuBoolIndex_t j = 0; j < mb.mNvals; j++) {
                    auto valueI = mb.mRowsIndex[j];
                    auto valueJ = mb.mColsIndex[j];

                    CuBoolIndex_t idI = k * blockI + valueI;
                    CuBoolIndex_t idJ = t * blockJ + valueJ;

                    vals.push_back(details::Pair{idI, idJ});
                }
            }

            std::sort(vals.begin(), vals.end(), details::PairCmp{});

            for (auto& p: vals) {
                result.mRowsIndex.push_back(p.i);
                result.mColsIndex.push_back(p.j);
            }

            return std::move(result);
        }
    };

    namespace details {
        template <typename Stream>
        void printMatrix(Stream& stream, const CuBoolIndex_t* rowsIndex, const CuBoolIndex_t* colsIndex, CuBoolIndex_t nrows, CuBoolIndex_t ncols, CuBoolSize_t nvals) {
            CuBoolIndex_t currentRow = 0;
            CuBoolIndex_t currentCol = 0;
            CuBoolIndex_t currentId = 0;

            while (currentId < nvals) {
                auto i = rowsIndex[currentId];
                auto j = colsIndex[currentId];

                while (currentRow < i) {
                    while (currentCol < ncols) {
                        stream << "." << " ";
                        currentCol += 1;
                    }

                    stream << "\n";
                    currentRow += 1;
                    currentCol = 0;
                }

                while (currentCol < j) {
                    stream << "." << " ";
                    currentCol += 1;
                }

                stream << "1" << " ";
                currentId += 1;
                currentCol += 1;
            }

            while (currentRow < nrows) {
                while (currentCol < ncols) {
                    stream << "." << " ";
                    currentCol += 1;
                }

                stream << "\n";
                currentRow += 1;
                currentCol = 0;
            }
        }
    }

    struct Printing { CuBoolMatrix matrix; CuBoolInstance instance; };

    template <typename Stream>
    Stream& operator <<(Stream& stream, const Printing& printing) {
        assert(printing.matrix);
        assert(printing.instance);

        CuBoolMatrix matrix = printing.matrix;
        CuBoolInstance instance = printing.instance;

        CuBoolIndex_t* rowIndex;
        CuBoolIndex_t* colIndex;
        CuBoolIndex_t nrows;
        CuBoolIndex_t ncols;
        CuBoolSize_t nvals;

        // Query matrix data
        CuBool_Matrix_Nrows(instance, matrix, &nrows);
        CuBool_Matrix_Ncols(instance, matrix, &ncols);
        CuBool_Matrix_ExtractPairs(instance, matrix, &rowIndex, &colIndex, &nvals);

        details::printMatrix(stream, rowIndex, colIndex, nrows, ncols, nvals);

        // Release resources
        CuBool_Vals_Free(instance, rowIndex);
        CuBool_Vals_Free(instance, colIndex);

        return stream;
    }

    template <typename Stream>
    Stream& operator <<(Stream& stream, const Matrix& a) {
        details::printMatrix(stream, a.mRowsIndex.data(), a.mColsIndex.data(), a.mNrows, a.mNcols, a.mNvals);
        return stream;
    }

}

#endif //CUBOOL_TEST_SHARED_HPP