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

#ifndef CUBOOL_MATRIX_CSR_CUH
#define CUBOOL_MATRIX_CSR_CUH

#include <cubool/matrix_base.hpp>
#include <cubool/details/host_allocator.hpp>
#include <cubool/details/device_allocator.cuh>
#include <matrix.h>

namespace cubool {

    class MatrixCsr: public MatrixBase {
    public:
        using Super = MatrixBase;
        using Super::mNumRows;
        using Super::mNumCols;
        using IndexType = CuBoolIndex_t;
        using DeviceAlloc = details::DeviceAllocator<IndexType>;
        using HostAlloc = details::HostAllocator<IndexType>;
        using MatrixImplType = nsparse::matrix<bool, IndexType, DeviceAlloc>;

        explicit MatrixCsr(Instance& instance);
        ~MatrixCsr() override = default;

        void resize(CuBoolSize_t nrows, CuBoolSize_t ncols) override;
        void build(const CuBoolIndex_t *rows, const CuBoolIndex_t *cols, CuBoolSize_t nvals) override;
        void extract(CuBoolIndex_t **rows, CuBoolIndex_t **cols, CuBoolSize_t *nvals) const override;
        void clone(const MatrixBase &other) override;

        void multiplySum(const MatrixBase &a, const MatrixBase &b, const MatrixBase &c) override;
        void multiplyAdd(const MatrixBase &a, const MatrixBase &b) override;

    private:
        // Uses nsparse csr matrix implementation as a backend
        MatrixImplType mMatrixImpl;
    };

}

#endif //CUBOOL_MATRIX_CSR_CUH