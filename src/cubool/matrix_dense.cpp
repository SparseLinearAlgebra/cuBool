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

#include <cubool/matrix_dense.hpp>
#include <cubool/instance.hpp>
#include <cubool/utils/cpu_buffer.hpp>

namespace cubool {

    MatrixDense::MatrixDense(class Instance &instance) : mBuffer(instance) {
        mInstancePtr = &instance;
    }

    CuBoolStatus MatrixDense::resize(CuBoolSize_t rows, CuBoolSize_t columns) {
        if (rows == mRows && columns == mColumns) {
            // nothing to do
            return CUBOOL_STATUS_SUCCESS;
        }

        mRows = rows;
        mColumns = columns;
        mRowsPacked = getRowsPackedFromRows(rows);
        mColumnsPadded = getColumnsPaddedFromColumns(columns);

        mStride = mColumnsPadded * sizeof(PackType_t);

        CuBoolSize_t bufferSize = mStride * mRowsPacked;
        return mBuffer.resizeNoContentKeep(bufferSize);
    }

    CuBoolStatus MatrixDense::writeValues(const std::vector<CuBoolPair> &values) {
        return writeValues(values.size(), values.data());
    }

    CuBoolStatus MatrixDense::readValues(std::vector<CuBoolPair> &values) const {
        if (mBuffer.isEmpty()) {
            // nothing to read, buffer is empty (matrix 0x0)
            return CUBOOL_STATUS_SUCCESS;
        }

        CuBoolSize_t bufferSize = mBuffer.getSize();
        CpuBuffer bufferTmp(*mInstancePtr);

        CuBoolStatus error = bufferTmp.resizeNoContentKeep(bufferSize);

        if (error != CUBOOL_STATUS_SUCCESS) {
            return error;
        }

        // Read values to host into tmp buffer.
        // May take a while
        error = mBuffer.transferToCpu(bufferTmp.getMemory(), bufferSize, 0);

        if (error != CUBOOL_STATUS_SUCCESS) {
            return error;
        }

        auto readBuffer = (const PackType_t*) bufferTmp.getMemory();

        for (CuBoolSize_t i = 0; i < mRowsPacked; i++) {
            for (CuBoolSize_t j = 0; j < mColumns; j++) {
                PackType_t pack = readBuffer[i * mColumnsPadded + j];

                for (CuBoolSize_t k = 0; k < PACK_TYPE_SIZE_BITS; k++) {
                    CuBoolSize_t rowIdx = i * PACK_TYPE_SIZE_BITS + k;
                    CuBoolSize_t columnIdx = j;

                    if (rowIdx < mRows && ((pack & 0x1u) != 0x0u)) {
                        values.push_back(CuBoolPair{ rowIdx, columnIdx });
                    }

                    pack = pack >> 1u;
                }
            }
        }

        return CUBOOL_STATUS_SUCCESS;
    }

    CuBoolStatus MatrixDense::writeValues(CuBoolSize_t count, const CuBoolPair *values) {
        if (mBuffer.isEmpty()) {
            mInstancePtr->sendMessage(CUBOOL_STATUS_INVALID_STATE, "An attempt to write values to empty matrix");
            return CUBOOL_STATUS_INVALID_STATE;
        }

        // Allocate buffer to form matrix image on the cpu side
        CuBoolSize_t bufferSize = mBuffer.getSize();
        CpuBuffer bufferTmp(*mInstancePtr);

        CuBoolStatus error = bufferTmp.resizeNoContentKeep(bufferSize);

        if (error != CUBOOL_STATUS_SUCCESS) {
            return error;
        }

        memset(bufferTmp.getMemory(), 0x0, bufferSize);

        auto writeBuffer = (PackType_t*) bufferTmp.getMemory();

        for (CuBoolSize_t idx = 0; idx < count; idx++) {
            CuBoolSize_t rowIdx = values[idx].i;
            CuBoolSize_t columnIdx = values[idx].j;

            if (rowIdx >= mRows || columnIdx >= mColumns) {
                mInstancePtr->sendMessage(CUBOOL_STATUS_INVALID_ARGUMENT, "Out of matrix bounds value");
                continue;
            }

            CuBoolSize_t i;
            CuBoolSize_t k;
            CuBoolSize_t j = columnIdx;

            getRowPackedIndex(rowIdx, i, k);

            writeBuffer[i * mColumnsPadded + j] |= (1u << k);
        }

        return mBuffer.transferFromCpu(bufferTmp.getMemory(), bufferSize, 0);
    }

    CuBoolStatus MatrixDense::readValues(CuBoolSize_t *count, CuBoolPair **values) const {
        std::vector<CuBoolPair> valuesVector;

        CuBoolStatus status = readValues(valuesVector);

        if (status != CUBOOL_STATUS_SUCCESS) {
            return status;
        }

        *count = valuesVector.size();
        *values = nullptr;

        if (*count > 0) {
            CuBoolCpuPtr_t mem;
            status = mInstancePtr->allocate(&mem, valuesVector.size()  * sizeof(CuBoolPair));

            if (status != CUBOOL_STATUS_SUCCESS) {
                return status;
            }

            *values = (CuBoolPair*) mem;
            memcpy(mem, valuesVector.data(), valuesVector.size() * sizeof(CuBoolPair));
        }

        return CUBOOL_STATUS_SUCCESS;
    }

    void MatrixDense::getRowPackedIndex(CuBoolSize_t rowIndex, CuBoolSize_t &rowPackIdxMajor, CuBoolSize_t &rowPackIdxMinor) {
        rowPackIdxMajor = rowIndex / PACK_TYPE_SIZE_BITS;
        rowPackIdxMinor = rowIndex % PACK_TYPE_SIZE_BITS;
    }

    CuBoolSize_t MatrixDense::getRowsPackedFromRows(CuBoolSize_t rows) {
        return rows / PACK_TYPE_SIZE_BITS + (rows % PACK_TYPE_SIZE_BITS? 1: 0);
    }

    CuBoolSize_t MatrixDense::getColumnsPaddedFromColumns(CuBoolSize_t columns) {
        return columns + (columns % PACK_TYPE_SIZE_BITS? PACK_TYPE_SIZE_BITS - columns % PACK_TYPE_SIZE_BITS: 0);
    }

}