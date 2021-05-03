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
#include <core/error.hpp>
#include <utils/data_utils.hpp>

namespace cubool {

    CudaVector::CudaVector(size_t nrows, CudaInstance &instance)
        : mVectorImpl(nrows), mInstance(instance) {

    }

    void CudaVector::setElement(index i) {
        RAISE_ERROR(NotImplemented, "This function is not supported for this vector class");
    }

    void CudaVector::build(const index *rows, size_t nvals, bool isSorted, bool noDuplicates) {
        if (nvals == 0) {
            // Empty vector, no values (but preserve dim)
            mVectorImpl = VectorImplType(getNrows());
            return;
        }

        // Validate data, sort, remove duplicates and etc.
        std::vector<index> data;
        DataUtils::buildVectorFromData(getNrows(), rows, nvals, data, isSorted, noDuplicates);

        // Transfer data to GPU
        thrust::device_vector<index, DeviceAlloc<index>> deviceData(data.size());
        thrust::copy(data.begin(), data.end(), deviceData.begin());

        // New vec instance
        mVectorImpl = VectorImplType(std::move(deviceData), getNrows(), data.size());
    }

    void CudaVector::extract(index *rows, size_t &nvals) {
        assert(nvals >= getNvals());

        nvals = getNvals();

        if (nvals > 0) {
            assert(rows);

            // Transfer data from GPU
            thrust::copy(mVectorImpl.m_rows_index.begin(), mVectorImpl.m_rows_index.end(), rows);
        }
    }

    void CudaVector::extractSubVector(const VectorBase &otherBase, index i, index nrows, bool checkTime) {
        RAISE_ERROR(NotImplemented, "This function is not implemented");

    }

    void CudaVector::clone(const VectorBase &otherBase) {
        auto other = dynamic_cast<const CudaVector*>(&otherBase);

        CHECK_RAISE_ERROR(other != nullptr, InvalidArgument, "Passed vector does not belong to vector class");
        CHECK_RAISE_ERROR(other != this, InvalidArgument, "Vectors must differ");

        assert(this->getNrows() == other->getNrows());
        this->mVectorImpl = other->mVectorImpl;
    }

    void CudaVector::reduce(index &result, bool checkTime) {
        result = getNvals();
    }

    void CudaVector::multiplyVxM(const VectorBase &vBase, const struct MatrixBase &mBase, bool checkTime) {
        RAISE_ERROR(NotImplemented, "This function is not implemented");

    }

    void CudaVector::multiplyMxV(const struct MatrixBase &mBase, const VectorBase &vBase, bool checkTime) {
        RAISE_ERROR(NotImplemented, "This function is not implemented");

    }

    index CudaVector::getNrows() const {
        return mVectorImpl.m_rows;
    }

    index CudaVector::getNvals() const {
        return mVectorImpl.m_vals;
    }

}