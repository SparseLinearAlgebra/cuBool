/**********************************************************************************/
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
/**********************************************************************************/

#include <cuda/cuda_matrix.hpp>
#include <cuda/kernels/spmerge.cuh>

namespace cubool {

    void CudaMatrix::eWiseAdd(const MatrixBase &aBase, const MatrixBase &bBase, bool checkTime) {
        auto a = dynamic_cast<const CudaMatrix*>(&aBase);
        auto b = dynamic_cast<const CudaMatrix*>(&bBase);

        CHECK_RAISE_ERROR(a != nullptr, InvalidArgument, "Passed matrix does not belong to csr matrix class");
        CHECK_RAISE_ERROR(b != nullptr, InvalidArgument, "Passed matrix does not belong to csr matrix class");

        index M = this->getNrows();
        index N = this->getNcols();

        assert(a->getNrows() == M);
        assert(a->getNcols() == N);

        assert(b->getNrows() == M);
        assert(b->getNcols() == N);

        if (a->isMatrixEmpty()) {
            this->clone(bBase);
            return;
        }

        if (b->isMatrixEmpty()) {
            this->clone(aBase);
            return;
        }

        // Ensure csr proper csr format even if empty
        a->resizeStorageToDim();
        b->resizeStorageToDim();

        kernels::SpMergeFunctor<index, DeviceAlloc<index>> spMergeFunctor;
        auto result = spMergeFunctor(a->mMatrixImpl, b->mMatrixImpl);

        // Assign the actual impl result to this storage
        this->mMatrixImpl = std::move(result);
    }

}