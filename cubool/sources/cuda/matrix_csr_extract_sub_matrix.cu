/**********************************************************************************/
/*                                                                                */
/* MIT License                                                                    */
/*                                                                                */
/* Copyright (c) 2020, 2021 JetBrains-Research                                    */
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

#include <cuda/matrix_csr.hpp>
#include <cuda/kernels/spsubmatrix.cuh>

namespace cubool {

    void MatrixCsr::extractSubMatrix(const MatrixBase &otherBase, index i, index j, index nrows, index ncols) {
        auto other = dynamic_cast<const MatrixCsr*>(&otherBase);

        CHECK_RAISE_ERROR(other != nullptr, InvalidArgument, "Provided matrix does not belong to matrix csr class");

        assert(nrows > 0);
        assert(ncols > 0);

        assert(other->getNrows() >= i + nrows);
        assert(other->getNcols() >= j + ncols);

        assert(this->getNrows() == nrows);
        assert(this->getNcols() == ncols);

        other->resizeStorageToDim();

        kernels::SpSubMatrix<index, details::DeviceAllocator<index>> spSubMatrix;
        auto result = spSubMatrix(other->mMatrixImpl, i, j, nrows, ncols);

        mMatrixImpl = std::move(result);
    }

}