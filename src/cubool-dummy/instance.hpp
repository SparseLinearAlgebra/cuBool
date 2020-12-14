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

#ifndef CUBOOL_INSTANCE_HPP
#define CUBOOL_INSTANCE_HPP

#include <cubool-dummy/matrix.hpp>
#include <unordered_set>
#include <iostream>
#include <exception>

namespace cubool_dummy {

    class Instance {
    public:
        Instance() = default;

        ~Instance() {
            if (!mMatrices.empty()) {
                std::cerr << "Some matrix instances was not properly released: " << mMatrices.size() << std::endl;
            }

            for (auto matrix: mMatrices) {
                delete matrix;
            }
        }

        Matrix* createMatrix(CuBoolIndex_t nrow, CuBoolIndex_t ncols) {
            auto matrix = new Matrix(*this);
            matrix->resize(nrow, ncols);
            mMatrices.emplace(matrix);
            return matrix;
        }

        void validateMatrix(Matrix* matrix) const {
            if(mMatrices.find(matrix) == mMatrices.end()) {
                throw std::runtime_error("No such matrix");
            }
        }

        void destroyMatrix(Matrix* matrix) {
            if (matrix == nullptr) {
                throw std::runtime_error("An attempt to release null matrix");
            }

            auto pos = mMatrices.find(matrix);
            mMatrices.erase(pos);

            delete matrix;
        }

    private:
        std::unordered_set<Matrix*> mMatrices;
    };

}

#endif //CUBOOL_INSTANCE_HPP
