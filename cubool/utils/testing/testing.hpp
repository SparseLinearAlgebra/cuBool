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

#ifndef CUBOOL_TESTING_TESTING_HPP
#define CUBOOL_TESTING_TESTING_HPP

#include <gtest/gtest.h>
#include <testing/pair.hpp>
#include <testing/timer.hpp>
#include <testing/matrix.hpp>
#include <testing/matrix_printing.hpp>
#include <testing/matrix_generator.hpp>
#include <testing/matrix_ewiseadd.hpp>
#include <testing/matrix_mxm.hpp>
#include <testing/matrix_kronecker.hpp>
#include <testing/vector.hpp>

// Put in the end of the unit test file
#define CUBOOL_GTEST_MAIN                                                       \
    int main(int argc, char *argv[]) {                                          \
        ::testing::InitGoogleTest(&argc, argv);                                 \
        return RUN_ALL_TESTS();                                                 \
    }

#endif //CUBOOL_TESTING_TESTING_HPP
