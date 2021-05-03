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

#include <sequential/sq_vector.hpp>
#include <sequential/sq_matrix.hpp>
#include <sequential/sq_reduce.hpp>
#include <sequential/sq_ewiseadd.hpp>
#include <sequential/sq_subvector.hpp>
#include <sequential/sq_spgemv.hpp>
#include <utils/data_utils.hpp>
#include <core/error.hpp>
#include <cassert>

namespace cubool {

    SqVector::SqVector(size_t nrows) {
        assert(nrows > 0);

        mData.nrows = nrows;
    }

    void SqVector::setElement(index i) {
        RAISE_ERROR(NotImplemented, "This function is not implemented");
    }

    void SqVector::build(const index *rows, size_t nvals, bool isSorted, bool noDuplicates) {
        // Utility used to reduce duplicates and sort values if needed
        DataUtils::buildVectorFromData(mData.nrows, rows, nvals, mData.indices, isSorted, noDuplicates);

        mData.nvals = mData.indices.size();
    }

    void SqVector::extract(index *rows, size_t &nvals) {
        assert(nvals >= getNvals());
        nvals = mData.nvals;

        if (nvals > 0) {
            std::copy(mData.indices.begin(), mData.indices.end(), rows);
        }
    }

    void SqVector::extractSubVector(const VectorBase &otherBase, index i, index nrows, bool checkTime) {
        auto other = dynamic_cast<const SqVector*>(&otherBase);

        CHECK_RAISE_ERROR(other != nullptr, InvalidArgument, "Provided vector does not belongs to sequential vector class");

        assert(this->getNrows() == nrows);

        sq_subvector(other->mData, i, nrows, this->mData);
    }

    void SqVector::clone(const VectorBase &otherBase) {
        auto other = dynamic_cast<const SqVector*>(&otherBase);

        CHECK_RAISE_ERROR(other != nullptr, InvalidArgument, "Provided vector does not belongs to sequential vector class");
        CHECK_RAISE_ERROR(other != this, InvalidArgument, "Vectors must differ");

        assert(other->getNrows() == this->getNrows());

        mData = other->mData;
    }

    void SqVector::reduce(index &result, bool checkTime) {
        result = getNvals();
    }

    void SqVector::reduceMatrix(const MatrixBase &otherBase, bool transpose, bool checkTime) {
        auto other = dynamic_cast<const SqMatrix*>(&otherBase);

        CHECK_RAISE_ERROR(other != nullptr, InvalidArgument, "Provided matrix does not belongs to sequential matrix class");

        if (transpose)
            assert(other->getNcols() == this->getNrows());
        else
            assert(other->getNrows() == this->getNrows());

        VecData out;
        out.nrows = this->getNrows();

        other->allocateStorage();

        if (transpose)
            sq_reduce_transposed(other->mData, out);
        else
            sq_reduce(other->mData, out);

        mData = std::move(out);
    }

    void SqVector::eWiseAdd(const VectorBase &aBase, const VectorBase &bBase, bool checkTime) {
        auto a = dynamic_cast<const SqVector*>(&aBase);
        auto b = dynamic_cast<const SqVector*>(&bBase);

        CHECK_RAISE_ERROR(a != nullptr, InvalidArgument, "Provided vector does not belongs to sequential vector class");
        CHECK_RAISE_ERROR(b != nullptr, InvalidArgument, "Provided vector does not belongs to sequential vector class");

        assert(a->getNrows() == this->getNrows());
        assert(a->getNrows() == b->getNrows());

        VecData out;
        out.nrows = this->getNrows();

        sq_ewiseadd(a->mData, b->mData, out);

        mData = std::move(out);
    }

    void SqVector::multiplyVxM(const VectorBase &vBase, const class MatrixBase &mBase, bool checkTime) {
        auto v = dynamic_cast<const SqVector*>(&vBase);
        auto m = dynamic_cast<const SqMatrix*>(&mBase);

        CHECK_RAISE_ERROR(v != nullptr, InvalidArgument, "Provided vector does not belongs to sequential vector class");
        CHECK_RAISE_ERROR(m != nullptr, InvalidArgument, "Provided matrix does not belongs to sequential matrix class");

        assert(v->getNrows() == m->getNrows());
        assert(this->getNrows() == m->getNcols());

        VecData out;
        out.nrows = this->getNrows();

        sq_spgemv_transposed(m->mData, v->mData, out);

        mData = std::move(out);
    }

    void SqVector::multiplyMxV(const class MatrixBase &mBase, const VectorBase &vBase, bool checkTime) {
        auto v = dynamic_cast<const SqVector*>(&vBase);
        auto m = dynamic_cast<const SqMatrix*>(&mBase);

        CHECK_RAISE_ERROR(v != nullptr, InvalidArgument, "Provided vector does not belongs to sequential vector class");
        CHECK_RAISE_ERROR(m != nullptr, InvalidArgument, "Provided matrix does not belongs to sequential matrix class");

        assert(v->getNrows() == m->getNcols());
        assert(this->getNrows() == m->getNrows());

        VecData out;
        out.nrows = this->getNrows();

        sq_spgemv(m->mData, v->mData, out);

        mData = std::move(out);
    }

    index SqVector::getNrows() const {
        return mData.nrows;
    }

    index SqVector::getNvals() const {
        return mData.nvals;
    }

}