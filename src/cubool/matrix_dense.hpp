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

#ifndef CUBOOL_MATRIX_DENSE_HPP
#define CUBOOL_MATRIX_DENSE_HPP

#include <cubool/cubool.h>
#include <cubool/utils/gpu_buffer.hpp>

#include <cinttypes>
#include <vector>

namespace cubool {

    class MatrixDense {
    public:
        // How we actually pack this matrix in memory
        // This info approached by kernels code
        using PackType_t = uint32_t;
        static const CuBoolSize_t BYTE_SIZE_IN_BITS = 8; // 8 bits per byte?
        static const CuBoolSize_t PACK_TYPE_SIZE_BITS = sizeof(PackType_t) * BYTE_SIZE_IN_BITS;

        explicit MatrixDense(class Instance& instance);
        MatrixDense(const MatrixDense& other) = delete;
        MatrixDense(MatrixDense&& other) noexcept = delete;
        ~MatrixDense() = default;

        CuBoolStatus resize(CuBoolSize_t rows, CuBoolSize_t columns);

        CuBoolStatus writeValues(const std::vector<CuBoolPair> &values);
        CuBoolStatus readValues(std::vector<CuBoolPair> &values) const;

        CuBoolStatus writeValues(CuBoolSize_t count, const CuBoolPair* values);
        CuBoolStatus readValues(CuBoolSize_t* count, CuBoolPair** values) const;

        CuBoolSize_t getRowsCount() const { return mRows; }
        CuBoolSize_t getColumnsCount() const { return mColumns; }
        CuBoolSize_t getRowsPackedCount() const { return mRowsPacked; }
        CuBoolSize_t getColumnsPaddedCount() const { return mColumnsPadded; }
        CuBoolSize_t getStride() const { return mStride; }
        const GpuBuffer& getBuffer() const { return mBuffer; }

        bool isZeroDim() const { return mRows * mColumns == 0; }


        static void getRowPackedIndex(CuBoolSize_t rowIndex, CuBoolSize_t &rowPackIdxMajor, CuBoolSize_t &rowPackIdxMinor);
        static CuBoolSize_t getRowsPackedFromRows(CuBoolSize_t rows);
        static CuBoolSize_t getColumnsPaddedFromColumns(CuBoolSize_t columns);

    private:
        GpuBuffer mBuffer;
        CuBoolSize_t mRows = 0;
        CuBoolSize_t mColumns = 0;
        CuBoolSize_t mRowsPacked = 0;
        CuBoolSize_t mColumnsPadded = 0;
        CuBoolSize_t mStride = 0;
        class Instance* mInstancePtr = nullptr;
    };

}

#endif //CUBOOL_MATRIX_DENSE_HPP