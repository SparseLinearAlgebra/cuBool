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

#include <cuBool_Common.hpp>

cuBool_Status cuBool_Vector_EWiseAdd(
        cuBool_Vector result,
        cuBool_Vector left,
        cuBool_Vector right,
        cuBool_Hints hints
) {
    CUBOOL_BEGIN_BODY
        CUBOOL_VALIDATE_LIBRARY
        CUBOOL_ARG_NOT_NULL(result)
        CUBOOL_ARG_NOT_NULL(left)
        CUBOOL_ARG_NOT_NULL(right)
        auto resultM = (cubool::Vector *) result;
        auto leftM = (cubool::Vector *) left;
        auto rightM = (cubool::Vector *) right;
        resultM->eWiseAdd(*leftM, *rightM, hints & CUBOOL_HINT_TIME_CHECK);
    CUBOOL_END_BODY
}