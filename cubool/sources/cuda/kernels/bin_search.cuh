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

#ifndef CUBOOL_BIN_SEARCH_CUH
#define CUBOOL_BIN_SEARCH_CUH

#include <thrust/device_ptr.h>

namespace cubool {
    namespace kernels {

        template <typename IndexType>
        __device__ IndexType findNearestRowIdx(IndexType value, IndexType range, thrust::device_ptr<const IndexType> rowIndex) {
            IndexType left = 0;
            IndexType right = range - 1;

            while (left <= right) {
                IndexType i = (left + right) / 2;

                if (value < rowIndex[i]) {
                    // value is not in =>row i range
                    right = i - 1;
                } else if (value < rowIndex[i + 1]) {
                    // value is in =>row i range and value actually in row i
                    return i;
                } else {
                    // value is in =>row i+1 range
                    left = i + 1;
                }
            }

            // never goes here since kron row index always has value index
            return range;
        }

    }
}

#endif //CUBOOL_BIN_SEARCH_CUH
