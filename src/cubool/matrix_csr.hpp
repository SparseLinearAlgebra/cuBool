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

#ifndef CUBOOL_MATRIX_CSR_HPP
#define CUBOOL_MATRIX_CSR_HPP

#include <cubool/matrix_base.hpp>
#include <cubool/details/host_allocator.hpp>
#include <cubool/details/device_allocator.cuh>
#include <nsparse/matrix.h>

namespace cubool {

    class MatrixCsr: public MatrixBase {
    public:
        using Super = MatrixBase;
        using Super::mNumRows;
        using Super::mNumCols;
        template<typename T>
        using DeviceAlloc = details::DeviceAllocator<T>;
        template<typename T>
        using HostAlloc = details::HostAllocator<T>;
        using MatrixImplType = nsparse::matrix<bool, index, DeviceAlloc<index>>;

        explicit MatrixCsr(Instance& instance);
        ~MatrixCsr() override = default;

        void resize(index nrows, index ncols) override;
        void build(const index *rows, const index *cols, size nvals) override;
        void extract(index* rows, index* cols, size_t &nvals) override;
        void extractExt(index* &rows, index* &cols, size_t &nvals) const override;
        void clone(const MatrixBase &other) override;
        void transpose(const MatrixBase &other);

        void multiplySum(const MatrixBase &a, const MatrixBase &b, const MatrixBase &c) override;
        void multiplyAdd(const MatrixBase &a, const MatrixBase &b) override;
        void kron(const MatrixBase& a, const MatrixBase& b) override;
        void add(const MatrixBase& a) override;

        size_t getNumVals() const { return mMatrixImpl.m_vals; }

    private:
        void resizeStorageToDim();
        bool isStorageEmpty() const;
        bool isMatrixEmpty() const;

        // Uses nsparse csr matrix implementation as a backend
        MatrixImplType mMatrixImpl;
    };

}

#endif //CUBOOL_MATRIX_CSR_HPP