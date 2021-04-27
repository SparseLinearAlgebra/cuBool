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

#ifndef CUBOOL_SQ_VECTOR_HPP
#define CUBOOL_SQ_VECTOR_HPP

#include <backend/vector_base.hpp>
#include <sequential/sq_data.hpp>

namespace cubool {

    class SqVector final: public VectorBase {
    public:
        ~SqVector() override = default;

        void setElement(index i) override;
        void build(const index *rows, size_t nvals, bool isSorted, bool noDuplicates) override;
        void extract(index *rows, size_t &nvals) override;
        void extractSubVector(const VectorBase &otherBase, index i, index nrows, bool checkTime) override;

        void clone(const VectorBase &otherBase) override;
        void reduce(index &result, bool checkTime) override;
        void reduceMatrix(const class MatrixBase &matrix, bool transpose, bool checkTime) override;

        index getNrows() const override;
        index getNvals() const override;

    private:

        mutable VecData mData;
    };

}

#endif //CUBOOL_SQ_VECTOR_HPP
