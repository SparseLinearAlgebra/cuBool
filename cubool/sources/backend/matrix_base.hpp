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

#ifndef CUBOOL_MATRIX_BASE_HPP
#define CUBOOL_MATRIX_BASE_HPP

#include <core/config.hpp>

namespace cubool {

    /**
     * Base class for boolean matrix representation.
     */
    class MatrixBase {
    public:
        virtual ~MatrixBase() = default;

        virtual void setElement(index i, index j)                                                                              = 0;
        virtual void build(const index* rows, const index* cols, size_t nvals, bool isSorted, bool noDuplicates)               = 0;
        virtual void extract(index* rows, index* cols, size_t& nvals)                                                          = 0;
        virtual void extractSubMatrix(const MatrixBase& otherBase, index i, index j, index nrows, index ncols, bool checkTime) = 0;

        virtual void clone(const MatrixBase& otherBase)                     = 0;
        virtual void transpose(const MatrixBase& otherBase, bool checkTime) = 0;
        virtual void reduce(const MatrixBase& otherBase, bool checkTime)    = 0;

        virtual void multiply(const MatrixBase& aBase, const MatrixBase& bBase, bool accumulate, bool checkTime) = 0;
        virtual void kronecker(const MatrixBase& aBase, const MatrixBase& bBase, bool checkTime)                 = 0;
        virtual void eWiseAdd(const MatrixBase& aBase, const MatrixBase& bBase, bool checkTime)                  = 0;
        virtual void eWiseMult(const MatrixBase& aBase, const MatrixBase& bBase, bool checkTime)                 = 0;
        virtual void eWiseMultInverted(const MatrixBase& matrix, const MatrixBase& mask, bool checkTime)         = 0;

        virtual index getNrows() const = 0;
        virtual index getNcols() const = 0;
        virtual index getNvals() const = 0;

        bool isZeroDim() const { return (size_t) getNrows() * (size_t) getNcols() == 0; }
    };

}// namespace cubool

#endif//CUBOOL_MATRIX_BASE_HPP
