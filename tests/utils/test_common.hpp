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
#include <random>
#include <unordered_set>

struct CuBoolPairHash {
public:
    std::size_t operator()(const CuBoolPair &x) const {
        return std::hash<CuBoolSize_t>()(x.i) ^ std::hash<CuBoolSize_t>()(x.j);
    }
};

struct CuBoolPairEq {
public:
    bool operator()(const CuBoolPair &a, const CuBoolPair& b)  const {
        return a.i == b.i && a.j == b.j;
    }
};

static void testMsgFun(CuBoolStatus error, const char* message, void* _) {
    std::cout << "CuBool: " << message << std::endl;
}

static CuBoolCpuPtr_t testAllocateFun(CuBoolSize_t size, void* _) {
    CuBoolCpuPtr_t ptr = malloc(size);
    std::cout << "Cubool: Allocate: " << size << " " << ptr << std::endl;
    return ptr;
}

static void testDeallocateFun(CuBoolCpuPtr_t ptr, void* _) {
    std::cout << "Cubool: Deallocate: " << ptr << std::endl;
    free(ptr);
}

static CuBoolCpuPtr_t testAllocateFunSilent(CuBoolSize_t size, void* _) {
    return malloc(size);
}

static void testDeallocateFunSilent(CuBoolCpuPtr_t ptr, void* _) {
    free(ptr);
}

struct Condition1 {
    bool operator()(CuBoolSize_t i, CuBoolSize_t j) {
        return (((i - 1) & i) == 0 && ((j - 1) & j) == 0);
    }
};

struct Condition2 {
    bool operator()(CuBoolSize_t i, CuBoolSize_t j) {
        return !(i % 5) && !(j % 7);
    }
};

struct Condition3 {
public:
    explicit Condition3(float density): mDensity(density) {}
    bool operator()(CuBoolSize_t i, CuBoolSize_t j) {
        return std::uniform_real_distribution<float>(0.0f, 1.0f)(mRandomEngine) <= mDensity;
    }
private:
    std::default_random_engine mRandomEngine;
    float mDensity = 1.0f;
};

template<typename Condition>
static void generateTestData(CuBoolSize_t rows, CuBoolSize_t columns, std::vector<CuBoolPair> &values, Condition&& condition) {
    for (CuBoolSize_t i = 0; i < rows; i++) {
        for (CuBoolSize_t j = 0; j < columns; j++) {
            // is i and j power of two or 0
            if (condition(i, j)) {
                values.push_back(CuBoolPair{ i, j });
            }
        }
    }
}

template<typename Condition>
static void generateTestData(CuBoolSize_t nrows, CuBoolSize_t ncols, std::vector<CuBoolIndex_t> &rows, std::vector<CuBoolIndex_t> &cols, CuBoolSize_t& nvals, Condition&& condition) {
    nvals = 0;
    for (CuBoolSize_t i = 0; i < nrows; i++) {
        for (CuBoolSize_t j = 0; j < ncols; j++) {
            // is i and j power of two or 0
            if (condition(i, j)) {
                rows.push_back(i);
                cols.push_back(j);
                nvals += 1;
            }
        }
    }
}

static void packCmpSet(const std::vector<CuBoolPair>& values, std::unordered_set<CuBoolPair,CuBoolPairHash,CuBoolPairEq>& set) {
    for (auto p: values) {
        set.emplace(p);
    }
}

static void packCmpSet(std::vector<CuBoolIndex_t> &rows, std::vector<CuBoolIndex_t> &cols, std::unordered_set<CuBoolPair,CuBoolPairHash,CuBoolPairEq>& set) {
    assert(rows.size() == cols.size());
    for (size_t i = 0; i < rows.size(); i++) {
        set.emplace(CuBoolPair{rows[i], cols[i]});
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

static void evaluateMultiplyAdd(
        CuBoolSize_t m,
        CuBoolSize_t t,
        CuBoolSize_t n,
        const std::vector<CuBoolIndex_t> &arows,
        const std::vector<CuBoolIndex_t> &acols,
        const std::vector<CuBoolIndex_t> &brows,
        const std::vector<CuBoolIndex_t> &bcols,
        const std::vector<CuBoolIndex_t> &crows,
        const std::vector<CuBoolIndex_t> &ccols,
        std::unordered_set<CuBoolPair,CuBoolPairHash,CuBoolPairEq>& result) {

    assert(arows.size() == acols.size());
    assert(brows.size() == bcols.size());
    assert(crows.size() == ccols.size());

    std::vector<std::vector<uint8_t>> a(m, std::vector<uint8_t>(t, 0));
    std::vector<std::vector<uint8_t>> b(n, std::vector<uint8_t>(t, 0));
    std::vector<std::vector<uint8_t>> r(m, std::vector<uint8_t>(n, 0));

    for (size_t i = 0; i < arows.size(); i++) {
        a[arows[i]][acols[i]] = 1;
    }

    for (size_t i = 0; i < brows.size(); i++) {
        b[bcols[i]][brows[i]] = 1;
    }

    for (CuBoolSize_t i = 0; i < m; i++) {
        for (CuBoolSize_t j = 0; j < n; j++) {
            uint8_t v = 0;

            for (CuBoolSize_t k = 0; k < t; k++) {
                v |= a[i][k] & b[j][k] ? 1u: 0u;
            }

            r[i][j] = v;
        }
    }

    for (size_t i = 0; i < crows.size(); i++) {
        r[crows[i]][ccols[i]] |= 1u;
    }

    for (CuBoolIndex_t i = 0; i < m; i++) {
        for (CuBoolIndex_t j = 0; j < n; j++) {
            if (r[i][j] != 0) {
                result.emplace(CuBoolPair{i,j});
            }
        }
    }
}

#endif //CUBOOL_TEST_SHARED_HPP
