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
#include <cubool/matrix_base.hpp>
#include <cubool/details/host_allocator.hpp>
#include <cubool/details/device_allocator.cuh>
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
        using Super::mNumRows;
        using Super::mNumCols;
        template<typename T>
        using DeviceAlloc = details::DeviceAllocator<T>;
        template<typename T>
        using HostAlloc = details::HostAllocator<T>;
        static const CuBoolSize_t BYTE_SIZE_IN_BITS = 8; // 8 bits per byte?
        static const CuBoolSize_t PACK_TYPE_SIZE_BITS = sizeof(PackType_t) * BYTE_SIZE_IN_BITS;

        explicit MatrixDense(class Instance& instance);
        MatrixDense(const MatrixDense& other) = delete;
        MatrixDense(MatrixDense&& other) noexcept = delete;
        ~MatrixDense() override = default;

        void resize(CuBoolSize_t nrows, CuBoolSize_t ncols) override;
        void build(const CuBoolIndex_t *rows, const CuBoolIndex_t *cols, CuBoolSize_t nvals) override;
        void extract(CuBoolIndex_t **rows, CuBoolIndex_t **cols, CuBoolSize_t *nvals) const override;
        void clone(const MatrixBase& other) override;

        void multiplySum(const MatrixBase &a, const MatrixBase &b, const MatrixBase &c) override;
        void multiplyAdd(const MatrixBase &a, const MatrixBase &b) override;
        void kron(const MatrixBase& a, const MatrixBase& b) override;

        CuBoolSize_t getNumRowsPacked() const { return mNumRowsPacked; }
        CuBoolSize_t getNumColsPadded() const { return mNumColsPadded; }
        thrust::device_vector<PackType_t, DeviceAlloc<PackType_t>>& getBuffer() { return mBuffer; }
        const thrust::device_vector<PackType_t, DeviceAlloc<PackType_t>>& getBuffer() const { return mBuffer; }

        static void getRowPackedIndex(CuBoolSize_t rowIndex, CuBoolSize_t &rowPackIdxMajor, CuBoolSize_t &rowPackIdxMinor);
        static CuBoolSize_t getNumRowsPackedFromRows(CuBoolSize_t rows);
        static CuBoolSize_t getNumColsPaddedFromCols(CuBoolSize_t cols);

    private:
        void extractVector(std::vector<CuBoolPair, details::HostAllocator<CuBoolPair>> &vals) const;

        thrust::device_vector<PackType_t, DeviceAlloc<PackType_t>> mBuffer;
        CuBoolSize_t mNumRowsPacked = 0;
        CuBoolSize_t mNumColsPadded = 0;
    };

}

#endif //CUBOOL_MATRIX_DENSE_HPP