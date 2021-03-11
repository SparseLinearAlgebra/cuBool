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

#ifndef CUBOOL_MATRIX_CSR_HPP
#define CUBOOL_MATRIX_CSR_HPP

#include <backend/matrix_base.hpp>
#include <cuda/details/host_allocator.hpp>
#include <cuda/details/device_allocator.cuh>
#include <nsparse/matrix.h>

namespace cubool {

    class MatrixCsr: public MatrixBase {
    public:
        template<typename T>
        using DeviceAlloc = details::DeviceAllocator<T>;
        template<typename T>
        using HostAlloc = details::HostAllocator<T>;
        using MatrixImplType = nsparse::matrix<bool, index, DeviceAlloc<index>>;

        explicit MatrixCsr(size_t nrows, size_t ncols, Instance& instance);
        ~MatrixCsr() override = default;

        void setElement(index i, index j) override;
        void build(const index *rows, const index *cols, size_t nvals, bool isSorted, bool noDuplicates) override;
        void extract(index* rows, index* cols, size_t &nvals) override;
        void extractSubMatrix(const MatrixBase &otherBase, index i, index j, index nrows, index ncols, bool checkTime) override;

        void clone(const MatrixBase &other) override;
        void transpose(const MatrixBase &other, bool checkTime) override;
        void reduce(const MatrixBase &other, bool checkTime) override;

        void multiply(const MatrixBase &a, const MatrixBase &b, bool accumulate, bool checkTime) override;
        void kronecker(const MatrixBase &a, const MatrixBase &b, bool checkTime) override;
        void eWiseAdd(const MatrixBase &a, const MatrixBase &b, bool checkTime) override;

        index getNrows() const override;
        index getNcols() const override;
        index getNvals() const override;

    private:
        void resizeStorageToDim() const;
        void clearAndResizeStorageToDim() const;
        bool isStorageEmpty() const;
        bool isMatrixEmpty() const;
        void transferToDevice(const std::vector<index> &rowOffsets, const std::vector<index> &colIndices);
        void transferFromDevice(std::vector<index> &rowOffsets, std::vector<index> &colIndices) const;

        // Uses nsparse csr matrix implementation as a backend
        mutable MatrixImplType mMatrixImpl;

        size_t mNrows = 0;
        size_t mNcols = 0;
        Instance& mInstance;
    };
};

#endif //CUBOOL_MATRIX_CSR_HPP