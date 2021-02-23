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

#ifndef CUBOOL_MATRIX_DENSE_HPP
#define CUBOOL_MATRIX_DENSE_HPP

#include <cubool/cubool.h>
#include <backend/matrix_base.hpp>
#include <cuda/details/host_allocator.hpp>
#include <cuda/details/device_allocator.cuh>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <cinttypes>
#include <vector>

namespace cubool {

    class MatrixDense final: public MatrixBase {
    public:
        // How we actually pack this matrix in memory
        // This info approached by kernels code
        using PackType_t = uint32_t;
        using Super = MatrixBase;
        template<typename T>
        using DeviceAlloc = details::DeviceAllocator<T>;
        template<typename T>
        using HostAlloc = details::HostAllocator<T>;
        static const size_t BYTE_SIZE_IN_BITS = 8; // 8 bits per byte?
        static const size_t PACK_TYPE_SIZE_BITS = sizeof(PackType_t) * BYTE_SIZE_IN_BITS;

        explicit MatrixDense(class Instance& instance);
        MatrixDense(const MatrixDense& other) = delete;
        MatrixDense(MatrixDense&& other) noexcept = delete;
        ~MatrixDense() override = default;

        void build(const index *rows, const index *cols, size_t nvals, bool isSorted) override;
        void extract(index* rows, index* cols, size_t& nvals) override;

        void clone(const MatrixBase& other) override;
        void transpose(const MatrixBase &other) override;

        void multiplySum(const MatrixBase &a, const MatrixBase &b, const MatrixBase &c) override;
        void multiply(const MatrixBase &aBase, const MatrixBase &bBase, bool accumulate) override;
        void kronecker(const MatrixBase& aBase, const MatrixBase& bBase) override;
        void eWiseAdd(const MatrixBase& aBase, const MatrixBase& bBase) override;

        index getNumRowsPacked() const { return mNumRowsPacked; }
        index getNumColsPadded() const { return mNumColsPadded; }
        thrust::device_vector<PackType_t, DeviceAlloc<PackType_t>>& getBuffer() { return mBuffer; }
        const thrust::device_vector<PackType_t, DeviceAlloc<PackType_t>>& getBuffer() const { return mBuffer; }

        static void getRowPackedIndex(index rowIndex, index &rowPackIdxMajor, index &rowPackIdxMinor);
        static index getNumRowsPackedFromRows(index rows);
        static index getNumColsPaddedFromCols(index cols);

    private:
        void extractVector(std::vector<Pair, details::HostAllocator<Pair>> &vals) const;

        thrust::device_vector<PackType_t, DeviceAlloc<PackType_t>> mBuffer;
        index mNumRowsPacked = 0;
        index mNumColsPadded = 0;
    };

}

#endif //CUBOOL_MATRIX_DENSE_HPP