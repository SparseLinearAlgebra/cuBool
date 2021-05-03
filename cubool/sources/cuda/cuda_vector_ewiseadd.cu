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

#include <cuda/cuda_vector.hpp>
#include <cuda/kernels/spewiseadd.cuh>
#include <core/error.hpp>
#include <cassert>

namespace cubool {

    void CudaVector::eWiseAdd(const VectorBase &aBase, const VectorBase &bBase, bool checkTime) {
        const auto* a = dynamic_cast<const CudaVector*>(&aBase);
        const auto* b = dynamic_cast<const CudaVector*>(&bBase);

        CHECK_RAISE_ERROR(a != nullptr, InvalidArgument, "Provided vector does not belong to cuda vector class");
        CHECK_RAISE_ERROR(b != nullptr, InvalidArgument, "Provided vector does not belong to cuda vector class");

        assert(a->getNrows() == b->getNrows());

        kernels::SpVectorEWiseAdd<index, DeviceAlloc<index>> functor;
        auto result = functor(a->mVectorImpl, b->mVectorImpl);

        mVectorImpl = std::move(result);
    }

}