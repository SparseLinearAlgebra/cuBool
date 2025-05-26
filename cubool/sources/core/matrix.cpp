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
#include <core/library.hpp>
#include <core/matrix.hpp>
#include <io/logger.hpp>
#include <utils/timer.hpp>

#define TIMER_ACTION(timer, action) \
    Timer timer;                    \
    timer.start();                  \
    action;                         \
    timer.end()

namespace cubool {

    Matrix::Matrix(size_t nrows, size_t ncols, BackendBase& backend) {
        mHnd      = backend.createMatrix(nrows, ncols);
        mProvider = &backend;
    }

    Matrix::~Matrix() {
        if (mHnd) {
            mProvider->releaseMatrix(mHnd);
            mHnd      = nullptr;
            mProvider = nullptr;
        }
    }

    void Matrix::setElement(index i, index j) {
        CHECK_RAISE_ERROR(i < getNrows(), InvalidArgument, "Value out of matrix bounds");
        CHECK_RAISE_ERROR(j < getNcols(), InvalidArgument, "Value out of matrix bounds");

        // This values will be committed later
        mCachedI.push_back(i);
        mCachedJ.push_back(j);
    }

    void Matrix::build(const index* rows, const index* cols, size_t nvals, bool isSorted, bool noDuplicates) {
        CHECK_RAISE_ERROR(rows != nullptr || nvals == 0, InvalidArgument, "Null ptr rows array");
        CHECK_RAISE_ERROR(cols != nullptr || nvals == 0, InvalidArgument, "Null ptr cols array");

        this->releaseCache();

        LogStream stream(*Library::getLogger());
        stream << Logger::Level::Info
               << "Matrix:build:" << this->getDebugMarker() << " "
               << "isSorted=" << isSorted << ", "
               << "noDuplicates=" << noDuplicates << LogStream::cmt;

        mHnd->build(rows, cols, nvals, isSorted, noDuplicates);
    }

    void Matrix::extract(index* rows, index* cols, size_t& nvals) {
        CHECK_RAISE_ERROR(rows != nullptr || getNvals() == 0, InvalidArgument, "Null ptr rows array");
        CHECK_RAISE_ERROR(cols != nullptr || getNvals() == 0, InvalidArgument, "Null ptr cols array");
        CHECK_RAISE_ERROR(getNvals() <= nvals, InvalidArgument, "Passed arrays size must be more or equal to the nvals of the matrix");

        this->commitCache();
        mHnd->extract(rows, cols, nvals);
    }

    void Matrix::extractSubMatrix(const MatrixBase& otherBase, index i, index j, index nrows, index ncols, bool checkTime) {
        const auto* other = dynamic_cast<const Matrix*>(&otherBase);

        CHECK_RAISE_ERROR(other != nullptr, InvalidArgument, "Passed matrix does not belong to core matrix class");

        auto bI = i + nrows;
        auto bJ = j + ncols;

        CHECK_RAISE_ERROR(nrows > 0, InvalidArgument, "Cannot extract sub-matrix with zero dimension");
        CHECK_RAISE_ERROR(ncols > 0, InvalidArgument, "Cannot extract sub-matrix with zero dimension");

        CHECK_RAISE_ERROR(bI <= other->getNrows(), InvalidArgument, "Provided sub-matrix range must be within matrix bounds");
        CHECK_RAISE_ERROR(bJ <= other->getNcols(), InvalidArgument, "Provided sub-matrix range must be within matrix bounds");

        CHECK_RAISE_ERROR(nrows == this->getNrows(), InvalidArgument, "Result matrix has incompatible size for extracted sub-matrix range");
        CHECK_RAISE_ERROR(ncols == this->getNcols(), InvalidArgument, "Result matrix has incompatible size for extracted sub-matrix range");

        other->commitCache();
        this->releaseCache();// Values of this matrix won't be used any more

        if (checkTime) {
            TIMER_ACTION(timer, mHnd->extractSubMatrix(*other->mHnd, i, j, nrows, ncols, false));

            LogStream stream(*Library::getLogger());
            stream << Logger::Level::Info
                   << "Time: " << timer.getElapsedTimeMs() << " ms "
                   << "Matrix::extractSubMatrix: "
                   << this->getDebugMarker() << " =submatrix( "
                   << i << "," << j << ", shape=(" << nrows << "," << ncols << ") "
                   << other->getDebugMarker() << LogStream::cmt;

            return;
        }

        mHnd->extractSubMatrix(*other->mHnd, i, j, nrows, ncols, false);
    }

    void Matrix::clone(const MatrixBase& otherBase) {
        const auto* other = dynamic_cast<const Matrix*>(&otherBase);

        CHECK_RAISE_ERROR(other != nullptr, InvalidArgument, "Passed matrix does not belong to core matrix class");

        if (this == other)
            return;

        auto M = other->getNrows();
        auto N = other->getNcols();

        CHECK_RAISE_ERROR(M == this->getNrows(), InvalidArgument, "Cloned matrix has incompatible size");
        CHECK_RAISE_ERROR(N == this->getNcols(), InvalidArgument, "Cloned matrix has incompatible size");

        other->commitCache();
        this->releaseCache();// Values of this matrix won't be used any more

        mHnd->clone(*other->mHnd);
    }

    void Matrix::transpose(const MatrixBase& otherBase, bool checkTime) {
        const auto* other = dynamic_cast<const Matrix*>(&otherBase);

        CHECK_RAISE_ERROR(other != nullptr, InvalidArgument, "Passed matrix does not belong to core matrix class");

        auto M = other->getNrows();
        auto N = other->getNcols();

        CHECK_RAISE_ERROR(M == this->getNcols(), InvalidArgument, "Transposed matrix has incompatible size");
        CHECK_RAISE_ERROR(N == this->getNrows(), InvalidArgument, "Transposed matrix has incompatible size");

        this->commitCache();
        this->releaseCache();// Values of this matrix won't be used any more

        if (checkTime) {
            TIMER_ACTION(timer, mHnd->transpose(*other->mHnd, false));

            LogStream stream(*Library::getLogger());
            stream << Logger::Level::Info
                   << "Time: " << timer.getElapsedTimeMs() << " ms "
                   << "Matrix::transpose: "
                   << this->getDebugMarker() << " =transposed "
                   << other->getDebugMarker() << LogStream::cmt;

            return;
        }

        mHnd->transpose(*other->mHnd, false);
    }

    void Matrix::reduce(const MatrixBase& otherBase, bool checkTime) {
        const auto* other = dynamic_cast<const Matrix*>(&otherBase);

        CHECK_RAISE_ERROR(other != nullptr, InvalidArgument, "Passed matrix does not belong to core matrix class");

        auto M = other->getNrows();

        CHECK_RAISE_ERROR(M == this->getNrows(), InvalidArgument, "Matrix has incompatible size");
        CHECK_RAISE_ERROR(1 == this->getNcols(), InvalidArgument, "Matrix has incompatible size");

        other->commitCache();
        this->releaseCache();// Values of this matrix won't be used any more

        if (checkTime) {
            TIMER_ACTION(timer, mHnd->reduce(*other->mHnd, false));

            LogStream stream(*Library::getLogger());
            stream << Logger::Level::Info
                   << "Time: " << timer.getElapsedTimeMs() << " ms "
                   << "Matrix::reduce: "
                   << this->getDebugMarker() << " =reduce "
                   << other->getDebugMarker() << LogStream::cmt;

            return;
        }

        mHnd->reduce(*other->mHnd, false);
    }

    void Matrix::multiply(const MatrixBase& aBase, const MatrixBase& bBase, bool accumulate, bool checkTime) {
        const auto* a = dynamic_cast<const Matrix*>(&aBase);
        const auto* b = dynamic_cast<const Matrix*>(&bBase);

        CHECK_RAISE_ERROR(a != nullptr, InvalidArgument, "Passed matrix does not belong to core matrix class");
        CHECK_RAISE_ERROR(b != nullptr, InvalidArgument, "Passed matrix does not belong to core matrix class");

        auto M = a->getNrows();
        auto T = a->getNcols();
        auto N = b->getNcols();

        CHECK_RAISE_ERROR(M == this->getNrows(), InvalidArgument, "Matrix has incompatible size for operation result");
        CHECK_RAISE_ERROR(N == this->getNcols(), InvalidArgument, "Matrix has incompatible size for operation result");
        CHECK_RAISE_ERROR(T == b->getNrows(), InvalidArgument, "Cannot multiply passed matrices");

        a->commitCache();
        b->commitCache();

        if (accumulate)
            this->commitCache();
        else
            this->releaseCache();

        if (checkTime) {
            TIMER_ACTION(timer, mHnd->multiply(*a->mHnd, *b->mHnd, accumulate, false));

            LogStream stream(*Library::getLogger());
            stream << Logger::Level::Info
                   << "Time: " << timer.getElapsedTimeMs() << " ms "
                   << "Matrix::multiply: "
                   << this->getDebugMarker() << (accumulate ? " += " : " = ")
                   << a->getDebugMarker() << " x "
                   << b->getDebugMarker() << LogStream::cmt;

            return;
        }

        mHnd->multiply(*a->mHnd, *b->mHnd, accumulate, false);
    }

    void Matrix::kronecker(const MatrixBase& aBase, const MatrixBase& bBase, bool checkTime) {
        const auto* a = dynamic_cast<const Matrix*>(&aBase);
        const auto* b = dynamic_cast<const Matrix*>(&bBase);

        CHECK_RAISE_ERROR(a != nullptr, InvalidArgument, "Passed matrix does not belong to core matrix class");
        CHECK_RAISE_ERROR(b != nullptr, InvalidArgument, "Passed matrix does not belong to core matrix class");

        index M = a->getNrows();
        index N = a->getNcols();
        index K = b->getNrows();
        index T = b->getNcols();

        CHECK_RAISE_ERROR(M * K == this->getNrows(), InvalidArgument, "Matrix has incompatible size for operation result");
        CHECK_RAISE_ERROR(N * T == this->getNcols(), InvalidArgument, "Matrix has incompatible size for operation result");

        a->commitCache();
        b->commitCache();
        this->releaseCache();

        if (checkTime) {
            TIMER_ACTION(timer, mHnd->kronecker(*a->mHnd, *b->mHnd, false));

            LogStream stream(*Library::getLogger());
            stream << Logger::Level::Info
                   << "Time: " << timer.getElapsedTimeMs() << " ms "
                   << "Matrix::kronecker: "
                   << this->getDebugMarker() << " = "
                   << a->getDebugMarker() << " (x) "
                   << b->getDebugMarker() << LogStream::cmt;

            return;
        }

        mHnd->kronecker(*a->mHnd, *b->mHnd, false);
    }

    void Matrix::eWiseAdd(const MatrixBase& aBase, const MatrixBase& bBase, bool checkTime) {
        const auto* a = dynamic_cast<const Matrix*>(&aBase);
        const auto* b = dynamic_cast<const Matrix*>(&bBase);

        CHECK_RAISE_ERROR(a != nullptr, InvalidArgument, "Passed matrix does not belong to core matrix class");
        CHECK_RAISE_ERROR(b != nullptr, InvalidArgument, "Passed matrix does not belong to core matrix class");

        index M = a->getNrows();
        index N = a->getNcols();

        CHECK_RAISE_ERROR(M == b->getNrows(), InvalidArgument, "Passed matrices have incompatible size");
        CHECK_RAISE_ERROR(N == b->getNcols(), InvalidArgument, "Passed matrices have incompatible size");

        CHECK_RAISE_ERROR(M == this->getNrows(), InvalidArgument, "Matrix has incompatible size for operation result");
        CHECK_RAISE_ERROR(N == this->getNcols(), InvalidArgument, "Matrix has incompatible size for operation result");

        a->commitCache();
        b->commitCache();
        this->releaseCache();

        if (checkTime) {
            TIMER_ACTION(timer, mHnd->eWiseAdd(*a->mHnd, *b->mHnd, false));

            LogStream stream(*Library::getLogger());
            stream << Logger::Level::Info
                   << "Time: " << timer.getElapsedTimeMs() << " ms "
                   << "Matrix::eWiseAdd: "
                   << this->getDebugMarker() << " = "
                   << a->getDebugMarker() << " + "
                   << b->getDebugMarker() << LogStream::cmt;

            return;
        }

        mHnd->eWiseAdd(*a->mHnd, *b->mHnd, false);
    }

    void Matrix::eWiseMult(const MatrixBase& aBase, const MatrixBase& bBase, bool checkTime) {
        const auto* a = dynamic_cast<const Matrix*>(&aBase);
        const auto* b = dynamic_cast<const Matrix*>(&bBase);

        CHECK_RAISE_ERROR(a != nullptr, InvalidArgument, "Passed matrix does not belong to core matrix class");
        CHECK_RAISE_ERROR(b != nullptr, InvalidArgument, "Passed matrix does not belong to core matrix class");

        index M = a->getNrows();
        index N = a->getNcols();

        CHECK_RAISE_ERROR(M == b->getNrows(), InvalidArgument, "Passed matrices have incompatible size");
        CHECK_RAISE_ERROR(N == b->getNcols(), InvalidArgument, "Passed matrices have incompatible size");

        CHECK_RAISE_ERROR(M == this->getNrows(), InvalidArgument, "Matrix has incompatible size for operation result");
        CHECK_RAISE_ERROR(N == this->getNcols(), InvalidArgument, "Matrix has incompatible size for operation result");

        a->commitCache();
        b->commitCache();
        this->releaseCache();

        if (checkTime) {
            TIMER_ACTION(timer, mHnd->eWiseMult(*a->mHnd, *b->mHnd, false));

            LogStream stream(*Library::getLogger());
            stream << Logger::Level::Info
                   << "Time: " << timer.getElapsedTimeMs() << " ms "
                   << "Matrix::eWiseMult: "
                   << this->getDebugMarker() << " = "
                   << a->getDebugMarker() << " + "
                   << b->getDebugMarker() << LogStream::cmt;

            return;
        }

        mHnd->eWiseMult(*a->mHnd, *b->mHnd, false);
    }

    void Matrix::eWiseMultInverted(const MatrixBase& matrix, const MatrixBase& mask, bool checkTime) {

        const auto* mat = dynamic_cast<const Matrix*>(&matrix);
        const auto* msk = dynamic_cast<const Matrix*>(&mask);

        CHECK_RAISE_ERROR(mat != nullptr, InvalidArgument, "Passed matrix does not belong to core matrix class");
        CHECK_RAISE_ERROR(msk != nullptr, InvalidArgument, "Passed matrix does not belong to core matrix class");

        index M = mat->getNrows();
        index N = msk->getNcols();

        CHECK_RAISE_ERROR(M == msk->getNrows(), InvalidArgument, "Passed matrices have incompatible size");
        CHECK_RAISE_ERROR(N == msk->getNcols(), InvalidArgument, "Passed matrices have incompatible size");

        CHECK_RAISE_ERROR(M == this->getNrows(), InvalidArgument, "Matrix has incompatible size for operation result");
        CHECK_RAISE_ERROR(N == this->getNcols(), InvalidArgument, "Matrix has incompatible size for operation result");

        mat->commitCache();
        msk->commitCache();
        this->releaseCache();

        if (checkTime) {
            TIMER_ACTION(timer, mHnd->eWiseMultInverted(*mat->mHnd, *msk->mHnd, false));

            LogStream stream(*Library::getLogger());
            stream << Logger::Level::Info
                   << "Time: " << timer.getElapsedTimeMs() << " ms "
                   << "Matrix::eWiseMultInverted: "
                   << this->getDebugMarker() << " = "
                   << mat->getDebugMarker() << " + "
                   << msk->getDebugMarker() << LogStream::cmt;

            return;
        }

        mHnd->eWiseMultInverted(*mat->mHnd, *msk->mHnd, false);
    }


    index Matrix::getNrows() const {
        return mHnd->getNrows();
    }

    index Matrix::getNcols() const {
        return mHnd->getNcols();
    }

    index Matrix::getNvals() const {
        this->commitCache();
        return mHnd->getNvals();
    }

    void Matrix::releaseCache() const {
        mCachedI.clear();
        mCachedJ.clear();
    }

    void Matrix::commitCache() const {
        assert(mCachedI.size() == mCachedJ.size());

        size_t cachedNvals = mCachedI.size();

        // Nothing to do if no value was cached on CPU side
        if (cachedNvals == 0)
            return;

        bool isSorted     = false;
        bool noDuplicates = false;

        if (mHnd->getNvals() > 0) {
            // We will have to join old and new values
            // Create tmp matrix and merge values

            MatrixBase* tmp = mProvider->createMatrix(getNrows(), getNcols());
            tmp->build(mCachedI.data(), mCachedJ.data(), cachedNvals, isSorted, noDuplicates);
            mHnd->eWiseAdd(*mHnd, *tmp, false);
            mProvider->releaseMatrix(tmp);
        } else {
            // Otherwise, new values are used to build matrix content
            mHnd->build(mCachedI.data(), mCachedJ.data(), cachedNvals, isSorted, noDuplicates);
        }

        // Clear arrays
        releaseCache();
    }
}// namespace cubool
