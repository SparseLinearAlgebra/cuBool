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

const char* cuBool_GetAbout() {
    static const char about[] =
        "CuBool is a linear boolean algebra library primitives and operations for \n"
        "work with dense and sparse matrices written on the NVIDIA CUDA platform. The primary \n"
        "goal of the library is implementation, testing and profiling algorithms for\n"
        "solving *formal-language-constrained problems*, such as *context-free* \n"
        "and *recursive* path queries with various semantics for graph databases."
        ""
        "Contributors:"
        "- Egor Orachyov (Github: https://github.com/EgorOrachyov)\n"
        "- Pavel Alimov (Github : https://github.com/Krekep)\n"
        "- Semyon Grigorev (Github: https://github.com/gsvgit)";

    return about;
}