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

#include <core/error.hpp>
#include <core/library.hpp>
#include <core/matrix.hpp>
#include <core/vector.hpp>
#include <io/logger.hpp>
#include <utils/timer.hpp>

#define TIMER_ACTION(timer, action) \
    Timer timer;                    \
    timer.start();                  \
    action;                         \
    timer.end()

namespace cubool {

    Vector::Vector(size_t nrows, BackendBase& backend) {
        mHnd      = backend.createVector(nrows);
        mProvider = &backend;
    }

    Vector::~Vector() {
        if (mHnd) {
            mProvider->releaseVector(mHnd);
            mHnd      = nullptr;
            mProvider = nullptr;
        }
    }

    void Vector::setElement(index i) {
        CHECK_RAISE_ERROR(i < getNrows(), InvalidArgument, "Value out of vector bounds");

        // This values will be committed later
        mCachedI.push_back(i);
    }

    void Vector::build(const index* rows, size_t nvals, bool isSorted, bool noDuplicates) {
        CHECK_RAISE_ERROR(rows != nullptr || nvals == 0, InvalidArgument, "Null ptr rows array");

        this->releaseCache();

        LogStream stream(*Library::getLogger());
        stream << Logger::Level::Info
               << "Vector:build:" << this->getDebugMarker() << " "
               << "isSorted=" << isSorted << ", "
               << "noDuplicates=" << noDuplicates << LogStream::cmt;

        mHnd->build(rows, nvals, isSorted, noDuplicates);
    }

    void Vector::extract(index* rows, size_t& nvals) {
        CHECK_RAISE_ERROR(rows != nullptr || getNvals() == 0, InvalidArgument, "Null ptr rows array");
        CHECK_RAISE_ERROR(getNvals() <= nvals, InvalidArgument, "Passed arrays size must be more or equal to the nvals of the vector");

        this->commitCache();
        mHnd->extract(rows, nvals);
    }

    void Vector::extractSubVector(const VectorBase& otherBase, index i, index nrows, bool checkTime) {
        const auto* other = dynamic_cast<const Vector*>(&otherBase);

        CHECK_RAISE_ERROR(other != nullptr, InvalidArgument, "Passed vector does not belong to core vector class");

        auto bI = i + nrows;

        CHECK_RAISE_ERROR(nrows > 0, InvalidArgument, "Cannot extract sub-vector with zero dimension");
        CHECK_RAISE_ERROR(bI <= other->getNrows(), InvalidArgument, "Provided sub-vector range must be within matrix bounds");
        CHECK_RAISE_ERROR(nrows == this->getNrows(), InvalidArgument, "Result matrix has incompatible size for extracted sub-matrix range");

        other->commitCache();
        this->releaseCache();// Values of this vector won't be used any more

        if (checkTime) {
            TIMER_ACTION(timer, mHnd->extractSubVector(*other->mHnd, i, nrows, false));

            LogStream stream(*Library::getLogger());
            stream << Logger::Level::Info
                   << "Time: " << timer.getElapsedTimeMs() << " ms "
                   << "Vector::extractSubVector: "
                   << this->getDebugMarker() << " =subvector( "
                   << i << ", shape=(" << nrows << ") "
                   << other->getDebugMarker() << LogStream::cmt;

            return;
        }

        mHnd->extractSubVector(*other->mHnd, i, nrows, false);
    }

    void Vector::extractRow(const class MatrixBase& matrixBase, index i) {
        const auto* matrix = dynamic_cast<const Matrix*>(&matrixBase);

        CHECK_RAISE_ERROR(matrix != nullptr, InvalidArgument, "Passed matrix does not belong to core matrix class");
        CHECK_RAISE_ERROR(i < matrix->getNrows(), InvalidArgument, "Row index must be within matrix bounds");

        matrix->commitCache();
        this->releaseCache();

        mHnd->extractRow(*matrix->mHnd, i);
    }

    void Vector::extractCol(const class MatrixBase& matrixBase, index j) {
        const auto* matrix = dynamic_cast<const Matrix*>(&matrixBase);

        CHECK_RAISE_ERROR(matrix != nullptr, InvalidArgument, "Passed matrix does not belong to core matrix class");
        CHECK_RAISE_ERROR(j < matrix->getNcols(), InvalidArgument, "Column index must be within matrix bounds");

        matrix->commitCache();
        this->releaseCache();

        mHnd->extractCol(*matrix->mHnd, j);
    }

    void Vector::clone(const VectorBase& otherBase) {
        const auto* other = dynamic_cast<const Vector*>(&otherBase);

        CHECK_RAISE_ERROR(other != nullptr, InvalidArgument, "Passed vector does not belong to core vector class");

        if (this == other)
            return;

        auto M = other->getNrows();

        CHECK_RAISE_ERROR(M == this->getNrows(), InvalidArgument, "Cloned vector has incompatible size");

        other->commitCache();
        this->releaseCache();// Values of this vector won't be used any more

        mHnd->clone(*other->mHnd);
    }

    void Vector::reduce(index& result, bool checkTime) {
        this->commitCache();

        if (checkTime) {
            TIMER_ACTION(timer, mHnd->reduce(result, false));

            LogStream stream(*Library::getLogger());
            stream << Logger::Level::Info
                   << "Time: " << timer.getElapsedTimeMs() << " ms "
                   << "Vector::reduce: "
                   << "index" << " =reduce"
                   << this->getDebugMarker()
                   << LogStream::cmt;

            return;
        }

        mHnd->reduce(result, false);
    }

    void Vector::reduceMatrix(const MatrixBase& matrixBase, bool transpose, bool checkTime) {
        const auto* matrix = dynamic_cast<const Matrix*>(&matrixBase);

        CHECK_RAISE_ERROR(matrix != nullptr, InvalidArgument, "Passed matrix does not belong to core matrix class");

        if (transpose) {
            CHECK_RAISE_ERROR(matrix->getNcols() == this->getNrows(), InvalidArgument, "Passed matrix has incompatible size");
        } else {
            CHECK_RAISE_ERROR(matrix->getNrows() == this->getNrows(), InvalidArgument, "Passed matrix has incompatible size");
        }

        matrix->commitCache();
        this->releaseCache();

        if (checkTime) {
            TIMER_ACTION(timer, mHnd->reduceMatrix(*matrix->mHnd, transpose, false));

            LogStream stream(*Library::getLogger());
            stream << Logger::Level::Info
                   << "Time: " << timer.getElapsedTimeMs() << " ms "
                   << "Vector::reduceMatrix: "
                   << this->getDebugMarker() << " =reduce(trsp=" << transpose << ") "
                   << matrix->getDebugMarker() << LogStream::cmt;

            return;
        }

        mHnd->reduceMatrix(*matrix->mHnd, transpose, false);
    }

    void Vector::eWiseMult(const VectorBase& aBase, const VectorBase& bBase, bool checkTime) {
        const auto* a = dynamic_cast<const Vector*>(&aBase);
        const auto* b = dynamic_cast<const Vector*>(&bBase);

        CHECK_RAISE_ERROR(a != nullptr, InvalidArgument, "Passed vector does not belong to core vector class");
        CHECK_RAISE_ERROR(b != nullptr, InvalidArgument, "Passed vector does not belong to core vector class");

        index M = a->getNrows();

        CHECK_RAISE_ERROR(M == b->getNrows(), InvalidArgument, "Passed vectors have incompatible size");
        CHECK_RAISE_ERROR(M == this->getNrows(), InvalidArgument, "Vector has incompatible size for operation result");

        a->commitCache();
        b->commitCache();
        this->releaseCache();

        if (checkTime) {
            TIMER_ACTION(timer, mHnd->eWiseMult(*a->mHnd, *b->mHnd, false));

            LogStream stream(*Library::getLogger());
            stream << Logger::Level::Info
                   << "Time: " << timer.getElapsedTimeMs() << " ms "
                   << "Vector::eWiseMult: "
                   << this->getDebugMarker() << " = "
                   << a->getDebugMarker() << " & "
                   << b->getDebugMarker() << LogStream::cmt;

            return;
        }

        mHnd->eWiseMult(*a->mHnd, *b->mHnd, false);
    }

    void Vector::eWiseAdd(const VectorBase& aBase, const VectorBase& bBase, bool checkTime) {
        const auto* a = dynamic_cast<const Vector*>(&aBase);
        const auto* b = dynamic_cast<const Vector*>(&bBase);

        CHECK_RAISE_ERROR(a != nullptr, InvalidArgument, "Passed vector does not belong to core vector class");
        CHECK_RAISE_ERROR(b != nullptr, InvalidArgument, "Passed vector does not belong to core vector class");

        index M = a->getNrows();

        CHECK_RAISE_ERROR(M == b->getNrows(), InvalidArgument, "Passed vectors have incompatible size");
        CHECK_RAISE_ERROR(M == this->getNrows(), InvalidArgument, "Vector has incompatible size for operation result");

        a->commitCache();
        b->commitCache();
        this->releaseCache();

        if (checkTime) {
            TIMER_ACTION(timer, mHnd->eWiseAdd(*a->mHnd, *b->mHnd, false));

            LogStream stream(*Library::getLogger());
            stream << Logger::Level::Info
                   << "Time: " << timer.getElapsedTimeMs() << " ms "
                   << "Vector::eWiseAdd: "
                   << this->getDebugMarker() << " = "
                   << a->getDebugMarker() << " + "
                   << b->getDebugMarker() << LogStream::cmt;

            return;
        }

        mHnd->eWiseAdd(*a->mHnd, *b->mHnd, false);
    }

    void Vector::multiplyVxM(const VectorBase& vBase, const class MatrixBase& mBase, bool checkTime) {
        const auto* v = dynamic_cast<const Vector*>(&vBase);
        const auto* m = dynamic_cast<const Matrix*>(&mBase);

        CHECK_RAISE_ERROR(v != nullptr, InvalidArgument, "Passed vector does not belong to core vector class");
        CHECK_RAISE_ERROR(m != nullptr, InvalidArgument, "Passed matrix does not belong to core matrix class");

        CHECK_RAISE_ERROR(v->getNrows() == m->getNrows(), InvalidArgument, "Provided vector and matrix have incompatible size for operation");
        CHECK_RAISE_ERROR(this->getNrows() == m->getNcols(), InvalidArgument, "This vector has incompatible size for operation result");

        v->commitCache();
        m->commitCache();
        this->releaseCache();

        if (checkTime) {
            TIMER_ACTION(timer, mHnd->multiplyVxM(*v->mHnd, *m->mHnd, false));

            LogStream stream(*Library::getLogger());
            stream << Logger::Level::Info
                   << "Time: " << timer.getElapsedTimeMs() << " ms "
                   << "Vector::multiplyVxM: "
                   << this->getDebugMarker() << " = "
                   << v->getDebugMarker() << " x "
                   << m->getDebugMarker() << LogStream::cmt;

            return;
        }

        mHnd->multiplyVxM(*v->mHnd, *m->mHnd, false);
    }

    void Vector::multiplyMxV(const class MatrixBase& mBase, const VectorBase& vBase, bool checkTime) {
        const auto* v = dynamic_cast<const Vector*>(&vBase);
        const auto* m = dynamic_cast<const Matrix*>(&mBase);

        CHECK_RAISE_ERROR(v != nullptr, InvalidArgument, "Passed vector does not belong to core vector class");
        CHECK_RAISE_ERROR(m != nullptr, InvalidArgument, "Passed matrix does not belong to core matrix class");

        CHECK_RAISE_ERROR(v->getNrows() == m->getNcols(), InvalidArgument, "Provided vector and matrix have incompatible size for operation");
        CHECK_RAISE_ERROR(this->getNrows() == m->getNrows(), InvalidArgument, "This vector has incompatible size for operation result");

        v->commitCache();
        m->commitCache();
        this->releaseCache();

        if (checkTime) {
            TIMER_ACTION(timer, mHnd->multiplyMxV(*m->mHnd, *v->mHnd, false));

            LogStream stream(*Library::getLogger());
            stream << Logger::Level::Info
                   << "Time: " << timer.getElapsedTimeMs() << " ms "
                   << "Vector::multiplyMxV: "
                   << this->getDebugMarker() << " = "
                   << m->getDebugMarker() << " x "
                   << v->getDebugMarker() << LogStream::cmt;

            return;
        }

        mHnd->multiplyMxV(*m->mHnd, *v->mHnd, false);
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
        size_t cachedNvals = mCachedI.size();

        // Nothing to do if no value was cached on CPU side
        if (cachedNvals == 0)
            return;

        bool isSorted     = false;
        bool noDuplicates = false;

        if (mHnd->getNvals() > 0) {
            // We will have to join old and new values
            // Create tmp vector and merge values

            VectorBase* tmp = mProvider->createVector(getNrows());
            tmp->build(mCachedI.data(), cachedNvals, isSorted, noDuplicates);
            mHnd->eWiseAdd(*mHnd, *tmp, false);
            mProvider->releaseVector(tmp);
        } else {
            // Otherwise, new values are used to build vector content
            mHnd->build(mCachedI.data(), cachedNvals, isSorted, noDuplicates);
        }

        // Clear arrays
        releaseCache();
    }

}// namespace cubool