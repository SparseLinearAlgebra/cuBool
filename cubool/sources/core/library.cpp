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

#include <core/library.hpp>
#include <core/error.hpp>
#include <core/matrix.hpp>
#include <backend/backend_base.hpp>
#include <backend/matrix_base.hpp>

#ifdef CUBOOL_WITH_CUDA
#include <cuda/cuda_backend.hpp>
#endif

#ifdef CUBOOL_WITH_CPU
#endif

namespace cubool {

    BackendBase* Library::mBackend = nullptr;

    void Library::initialize(hints initHints) {
        CHECK_RAISE_CRITICAL_ERROR(mBackend == nullptr, InvalidState, "Library already initialized");

        bool preferCpu = initHints & CUBOOL_HINT_CPU_BACKEND;

        // If user do not force the cpu backend usage
        if (!preferCpu) {
#ifdef CUBOOL_WITH_CUDA
            mBackend = new CudaBackend();
            mBackend->initialize(initHints);

            // Failed to setup cuda, release backend and go to try cpu
            if (!mBackend->isInitialized()) {
                delete mBackend;
                mBackend = nullptr;
            }
#endif
        }

#ifdef CUBOOL_WITH_CPU
#endif

        CHECK_RAISE_ERROR(mBackend != nullptr, BackendError, "Failed to select backend");
    }

    void Library::finalize() {
        if (mBackend) {
            // Remember to finalize backend
            mBackend->finalize();
            delete mBackend;
            mBackend = nullptr;
        }
    }

    void Library::validate() {
        CHECK_RAISE_CRITICAL_ERROR(mBackend != nullptr, InvalidState, "Library is not initialized");
    }

    MatrixBase *Library::createMatrix(size_t nrows, size_t ncols) {
        return new Matrix(nrows, ncols, *mBackend);
    }

    void Library::releaseMatrix(MatrixBase *matrixBase) {
        delete matrixBase;
    }
}