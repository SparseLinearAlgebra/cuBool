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

#include <cuda/matrix_csr.hpp>
#include <nsparse/spgemm.h>

namespace cubool {

    void MatrixCsr::multiply(const MatrixBase &aBase, const MatrixBase &bBase, bool accumulate) {
        auto a = dynamic_cast<const MatrixCsr*>(&aBase);
        auto b = dynamic_cast<const MatrixCsr*>(&bBase);

        CHECK_RAISE_ERROR(a != nullptr, InvalidArgument, "Passed matrix do not belong to csr matrix class");
        CHECK_RAISE_ERROR(b != nullptr, InvalidArgument, "Passed matrix do not belong to csr matrix class");

        index M = a->getNrows();
        index N = b->getNcols();

        assert(this->getNrows() == M);
        assert(this->getNcols() == N);

        if (a->isMatrixEmpty() || b->isMatrixEmpty()) {
            // A or B has no values
            return;
        }

        if (this->isStorageEmpty()) {
            // If this was resized but actual data was not allocated
            this->resizeStorageToDim();
        }

        assert(accumulate);

        if (accumulate) {
            // Call backend r = c + a * b implementation, as C this is passed
            nsparse::spgemm_functor_t<bool, index, DeviceAlloc<index>> spgemmFunctor;
            auto result = spgemmFunctor(mMatrixImpl, a->mMatrixImpl, b->mMatrixImpl);

            // Assign result to this
            this->mMatrixImpl = std::move(result);
        }
    }

}