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

#ifndef CUBOOL_BACKEND_BASE_HPP
#define CUBOOL_BACKEND_BASE_HPP

#include <backend/matrix_base.hpp>
#include <core/config.hpp>

namespace cubool {

    class BackendBase {
    public:
        virtual ~BackendBase() = default;
        virtual void initialize(hints initHints) = 0;
        virtual void finalize() = 0;
        virtual bool isInitialized() const = 0;
        virtual MatrixBase* createMatrix(size_t nrows, size_t ncols) = 0;
        virtual void releaseMatrix(MatrixBase* matrixBase) = 0;
        virtual void queryCapabilities(cuBool_DeviceCaps& caps) = 0;
    };

}

#endif //CUBOOL_BACKEND_BASE_HPP
