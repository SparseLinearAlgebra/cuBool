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

#ifndef CUBOOL_MATRIX_HPP
#define CUBOOL_MATRIX_HPP

#include <backend/matrix_base.hpp>

namespace cubool {

    class Matrix final: public MatrixBase {
    public:
        ~Matrix() override = default;

        void build(const index *rows, const index *cols, size_t nvals, bool isSorted) override;
        void extract(index *rows, index *cols, size_t &nvals) override;

        void clone(const MatrixBase &other) override;
        void transpose(const MatrixBase &other) override;

        void multiplySum(const MatrixBase &aBase, const MatrixBase &bBase, const MatrixBase &cBase) override;
        void multiply(const MatrixBase &aBase, const MatrixBase &bBase) override;
        void kronecker(const MatrixBase &aBase, const MatrixBase &bBase) override;
        void eWiseAdd(const MatrixBase &aBase, const MatrixBase &bBase) override;

        index getNrows() const override;
        index getNcols() const override;
        index getNvals() const override;
    };

}

#endif //CUBOOL_MATRIX_HPP