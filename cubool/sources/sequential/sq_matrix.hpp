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

#ifndef CUBOOL_SQ_MATRIX_HPP
#define CUBOOL_SQ_MATRIX_HPP

#include <backend/matrix_base.hpp>
#include <sequential/sq_data.hpp>

namespace cubool {

    /**
     * Csr matrix for Cpu side operations in sequential backend.
     */
    class SqMatrix final: public MatrixBase {
    public:
        SqMatrix(size_t nrows, size_t ncols);
        ~SqMatrix() override = default;

        void setElement(index i, index j) override;
        void build(const index *rows, const index *cols, size_t nvals, bool isSorted, bool noDuplicates) override;
        void extract(index *rows, index *cols, size_t &nvals) override;
        void extractSubMatrix(const MatrixBase &otherBase, index i, index j, index nrows, index ncols, bool checkTime) override;

        void clone(const MatrixBase &otherBase) override;
        void transpose(const MatrixBase &otherBase, bool checkTime) override;
        void reduce(const MatrixBase &otherBase, bool checkTime) override;

        void multiply(const MatrixBase &aBase, const MatrixBase &bBase, bool accumulate, bool checkTime) override;
        void kronecker(const MatrixBase &aBase, const MatrixBase &bBase, bool checkTime) override;
        void eWiseAdd(const MatrixBase &aBase, const MatrixBase &bBase, bool checkTime) override;
        void eWiseMult(const MatrixBase &a, const MatrixBase &b, bool checkTime) override;

        index getNrows() const override;
        index getNcols() const override;
        index getNvals() const override;

    private:
        friend class SqVector;
        void allocateStorage() const;

        mutable CsrData mData;
    };

}

#endif //CUBOOL_SQ_MATRIX_HPP
