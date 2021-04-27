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

#include <core/vector.hpp>
#include <core/error.hpp>
#include <core/library.hpp>
#include <io/logger.hpp>

namespace cubool {

    Vector::Vector(size_t nrows, BackendBase &backend) {
        mHnd = backend.createVector(nrows);
        mProvider = &backend;

        // By default marker is the address of the vector
        std::stringstream s;
        s << this;
        mMarker = s.str();
    }

    Vector::~Vector() {
        if (mHnd) {
            mProvider->releaseVector(mHnd);
            mHnd = nullptr;
            mProvider = nullptr;
        }
    }

    void Vector::setElement(index i) {
        RAISE_ERROR(NotImplemented, "This function is not implemented");
    }

    void Vector::build(const index *rows, size_t nvals, bool isSorted, bool noDuplicates) {
        CHECK_RAISE_ERROR(rows != nullptr || nvals == 0, InvalidArgument, "Null ptr rows array");

        this->releaseCache();

        LogStream stream(*Library::getLogger());
        stream << Logger::Level::Info
               << "Vector:build:" << this->getDebugMarker() << " "
               << "isSorted=" << isSorted << ", "
               << "noDuplicates=" << noDuplicates << LogStream::cmt;

        mHnd->build(rows, nvals, isSorted, noDuplicates);
    }

    void Vector::extract(index *rows, size_t &nvals) {
        CHECK_RAISE_ERROR(rows != nullptr || getNvals() == 0, InvalidArgument, "Null ptr rows array");
        CHECK_RAISE_ERROR(getNvals() <= nvals, InvalidArgument, "Passed arrays size must be more or equal to the nvals of the vector");

        this->commitCache();
        mHnd->extract(rows, nvals);
    }

    void Vector::extractSubVector(const VectorBase &otherBase, index i, index nrows, bool checkTime) {
        RAISE_ERROR(NotImplemented, "This function is not implemented");
    }

    void Vector::clone(const VectorBase &otherBase) {
        const auto* other = dynamic_cast<const Vector*>(&otherBase);

        CHECK_RAISE_ERROR(other != nullptr, InvalidArgument, "Passed vector does not belong to core vector class");

        if (this == other)
            return;

        auto M = other->getNrows();

        CHECK_RAISE_ERROR(M == this->getNrows(), InvalidArgument, "Cloned vector has incompatible size");

        other->commitCache();
        this->releaseCache(); // Values of this vector won't be used any more

        mHnd->clone(*other->mHnd);
    }

    void Vector::reduce(index &result, bool checkTime) {
        RAISE_ERROR(NotImplemented, "This function is not implemented");
    }

    void Vector::reduceMatrix(const MatrixBase &matrix, bool transpose, bool checkTime) {
        RAISE_ERROR(NotImplemented, "This function is not implemented");
    }

    index Vector::getNrows() const {
        return mHnd->getNrows();
    }

    index Vector::getNvals() const {
        this->commitCache();
        return mHnd->getNvals();
    }

    void Vector::releaseCache() const {
        mCachedI.clear();
    }

    void Vector::commitCache() const {
        // todo
    }

}