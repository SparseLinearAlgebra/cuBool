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

#include <core/matrix.hpp>

namespace cubool {

    Matrix::Matrix(size_t nrows, size_t ncols, BackendBase &backend) {
        mHnd = backend.createMatrix(nrows, ncols);
        mProvider = &backend;
    }

    Matrix::~Matrix() {
        if (mHnd) {
            mProvider->releaseMatrix(mHnd);
            mHnd = nullptr;
            mProvider = nullptr;
        }
    }

    void Matrix::build(const index *rows, const index *cols, size_t nvals, bool isSorted) {
        // todo: validation of the state and arguments

        mHnd->build(rows, cols, nvals, isSorted);
    }

    void Matrix::extract(index *rows, index *cols, size_t &nvals) {
        // todo: validation of the state and arguments

        mHnd->extract(rows, cols, nvals);
    }

    void Matrix::clone(const MatrixBase &otherBase) {
        const auto* other = dynamic_cast<const Matrix*>(&otherBase);

        // todo: validation of the state and arguments

        mHnd->clone(*other->mHnd);
    }

    void Matrix::transpose(const MatrixBase &otherBase) {
        const auto* other = dynamic_cast<const Matrix*>(&otherBase);

        // todo: validation of the state and arguments

        mHnd->transpose(*other->mHnd);
    }

    void Matrix::multiplySum(const MatrixBase &aBase, const MatrixBase &bBase, const MatrixBase &cBase) {
        const auto* a = dynamic_cast<const Matrix*>(&aBase);
        const auto* b = dynamic_cast<const Matrix*>(&bBase);
        const auto* c = dynamic_cast<const Matrix*>(&cBase);

        // todo: validation of the state and arguments

        mHnd->multiplySum(*a->mHnd, *b->mHnd, *c->mHnd);
    }

    void Matrix::multiply(const MatrixBase &aBase, const MatrixBase &bBase) {
        const auto* a = dynamic_cast<const Matrix*>(&aBase);
        const auto* b = dynamic_cast<const Matrix*>(&bBase);

        // todo: validation of the state and arguments

        mHnd->multiply(*a->mHnd, *b->mHnd);
    }

    void Matrix::kronecker(const MatrixBase &aBase, const MatrixBase &bBase) {
        const auto* a = dynamic_cast<const Matrix*>(&aBase);
        const auto* b = dynamic_cast<const Matrix*>(&bBase);

        // todo: validation of the state and arguments

        mHnd->kronecker(*a->mHnd, *b->mHnd);
    }

    void Matrix::eWiseAdd(const MatrixBase &aBase, const MatrixBase &bBase) {
        const auto* a = dynamic_cast<const Matrix*>(&aBase);
        const auto* b = dynamic_cast<const Matrix*>(&bBase);

        // todo: validation of the state and arguments

        mHnd->eWiseAdd(*a->mHnd, *b->mHnd);
    }

    index Matrix::getNrows() const {
        return mHnd->getNrows();
    }

    index Matrix::getNcols() const {
        return mHnd->getNcols();
    }

    index Matrix::getNvals() const {
        return mHnd->getNvals();
    }
}