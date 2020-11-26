/**********************************************************************************/
/*                                                                                */
/* MIT License                                                                    */
/*                                                                                */
/* Copyright (c) 2020 JetBrains-Research                                          */
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

#ifndef CUBOOL_MATRIX_BASE_HPP
#define CUBOOL_MATRIX_BASE_HPP

#include <cubool/cubool.h>
#include <cubool/build.hpp>
#include <string>

namespace cubool {

    /** Base class for boolean matrix representation */
    class MatrixBase {
    public:
        explicit MatrixBase(class Instance& instance) : mInstanceRef(instance) {}
        virtual ~MatrixBase() = default;

        virtual void resize(CuBoolSize_t nrows, CuBoolSize_t ncols) = 0;
        virtual void build(const CuBoolIndex_t* rows, const CuBoolIndex_t* cols, CuBoolSize_t nvals) = 0;
        virtual void extract(CuBoolIndex_t** rows, CuBoolIndex_t** cols, CuBoolSize_t* nvals) const = 0;
        virtual void clone(const MatrixBase& other) = 0;

        virtual void multiplySum(const MatrixBase& a, const MatrixBase& b, const MatrixBase& c) = 0;
        virtual void multiplyAdd(const MatrixBase& a, const MatrixBase& b) = 0;
        // virtual void kron(const MatrixBase& a, const MatrixBase& b) = 0;

        void setDebugMarker(std::string string) { mDebugMarker = std::move(string); }

        const std::string& getDebugMarker() const { return mDebugMarker; }
        CuBoolSize_t getNumRows() const { return mNumRows; }
        CuBoolSize_t getNumCols() const { return mNumCols; }
        Instance& getInstance() const { return mInstanceRef; }
        bool isZeroDim() const { return mNumRows * mNumCols == 0; }

    protected:
        std::string mDebugMarker;
        CuBoolSize_t mNumRows = 0;
        CuBoolSize_t mNumCols = 0;
        class Instance& mInstanceRef;
    };

}

#endif //CUBOOL_MATRIX_BASE_HPP