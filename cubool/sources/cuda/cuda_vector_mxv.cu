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

#include <cassert>
#include <core/error.hpp>
#include <cuda/cuda_matrix.hpp>
#include <cuda/cuda_vector.hpp>
#include <cuda/kernels/spgemv.cuh>

namespace cubool {

    void CudaVector::multiplyMxV(const struct MatrixBase& mBase, const VectorBase& vBase, bool checkTime) {
        const auto* m = dynamic_cast<const CudaMatrix*>(&mBase);
        const auto* v = dynamic_cast<const CudaVector*>(&vBase);

        CHECK_RAISE_ERROR(m != nullptr, InvalidArgument, "Provided matrix does not belong to cuda matrix class");
        CHECK_RAISE_ERROR(v != nullptr, InvalidArgument, "Provided vector does not belong to cuda vector class");

        assert(m->getNcols() == v->getNrows());
        assert(m->getNrows() == this->getNrows());

        m->resizeStorageToDim();

        kernels::SpGEMV<index, DeviceAlloc<index>> functor;
        auto                                       result = functor(m->mMatrixImpl, v->mVectorImpl);

        mVectorImpl = std::move(result);
    }

}// namespace cubool