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

    void Matrix::build(const index *rows, const index *cols, size_t nvals, bool isSorted) {

    }

    void Matrix::extract(index *rows, index *cols, size_t &nvals) {

    }

    void Matrix::clone(const MatrixBase &other) {

    }

    void Matrix::transpose(const MatrixBase &other) {

    }

    void Matrix::multiplySum(const MatrixBase &aBase, const MatrixBase &bBase, const MatrixBase &cBase) {

    }

    void Matrix::multiply(const MatrixBase &aBase, const MatrixBase &bBase) {

    }

    void Matrix::kronecker(const MatrixBase &aBase, const MatrixBase &bBase) {

    }

    void Matrix::eWiseAdd(const MatrixBase &aBase, const MatrixBase &bBase) {

    }

    index Matrix::getNrows() const {
        return 0;
    }

    index Matrix::getNcols() const {
        return 0;
    }

    index Matrix::getNvals() const {
        return 0;
    }
}