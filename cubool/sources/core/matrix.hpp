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

#ifndef CUBOOL_MATRIX_HPP
#define CUBOOL_MATRIX_HPP

#include <core/config.hpp>
#include <backend/matrix_base.hpp>
#include <backend/backend_base.hpp>
#include <vector>

namespace cubool {

    /**
     * Proxy matrix for the actual backend matrix implementation.
     * Behaves as validation/auxiliary layer.
     */
    class Matrix final: public MatrixBase {
    public:
        Matrix(size_t nrows, size_t ncols, BackendBase& backend);
        ~Matrix() override;

        void setElement(index i, index j) override;
        void build(const index *rows, const index *cols, size_t nvals, bool isSorted, bool noDuplicates) override;
        void extract(index *rows, index *cols, size_t &nvals) override;
        void extractSubMatrix(const MatrixBase& otherBase, index i, index j, index nrows, index ncols) override;

        void clone(const MatrixBase &otherBase) override;
        void transpose(const MatrixBase &otherBase) override;
        void reduce(const MatrixBase& otherBase) override;

        void multiply(const MatrixBase &aBase, const MatrixBase &bBase, bool accumulate) override;
        void kronecker(const MatrixBase &aBase, const MatrixBase &bBase) override;
        void eWiseAdd(const MatrixBase &aBase, const MatrixBase &bBase) override;

        index getNrows() const override;
        index getNcols() const override;
        index getNvals() const override;

        void setDebugMarker(const char* marker);
        const char* getDebugMarker() const;
        index getDebugMarkerSizeWithNullT() const;

    private:

        void releaseCache() const;
        void commitCache() const;

        // Cached values by the set functions
        mutable std::vector<index> mCachedI;
        mutable std::vector<index> mCachedJ;

        // Marker for debugging
        std::string mMarker;

        // Implementation handle references
        MatrixBase* mHnd = nullptr;
        BackendBase* mProvider = nullptr;
    };

}

#endif //CUBOOL_MATRIX_HPP