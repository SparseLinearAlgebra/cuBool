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

#include <testing/testing.hpp>
#include <cstring>

TEST(cuBool_Vector, Duplicate) {
    cuBool_Vector vector = nullptr, duplicated = nullptr;
    cuBool_Index m = 900;
    float density = 0.31;

    testing::Vector tvector = testing::Vector::generateSparse(m, density);

    ASSERT_EQ(cuBool_Initialize(CUBOOL_HINT_CPU_BACKEND), CUBOOL_STATUS_SUCCESS);

    ASSERT_EQ(cuBool_Vector_New(&vector, m), CUBOOL_STATUS_SUCCESS);
    ASSERT_EQ(cuBool_Vector_Build(vector, tvector.index.data(), tvector.nvals, CUBOOL_HINT_VALUES_SORTED), CUBOOL_STATUS_SUCCESS);

    ASSERT_EQ(cuBool_Vector_Duplicate(vector, &duplicated), CUBOOL_STATUS_SUCCESS);

    ASSERT_EQ(tvector.areEqual(duplicated), true);

    ASSERT_EQ(cuBool_Vector_Free(vector), CUBOOL_STATUS_SUCCESS);
    ASSERT_EQ(cuBool_Vector_Free(duplicated), CUBOOL_STATUS_SUCCESS);

    ASSERT_EQ(cuBool_Finalize(), CUBOOL_STATUS_SUCCESS);
}

TEST(cuBool_Vector, PropertyQuery) {
    cuBool_Vector vector = nullptr;
    cuBool_Index m = 900;
    float density = 0.21;

    testing::Vector tvector = std::move(testing::Vector::generateSparse(m, density));

    ASSERT_EQ(cuBool_Initialize(CUBOOL_HINT_CPU_BACKEND), CUBOOL_STATUS_SUCCESS);

    ASSERT_EQ(cuBool_Vector_New(&vector, m), CUBOOL_STATUS_SUCCESS);
    ASSERT_EQ(cuBool_Vector_Build(vector, tvector.index.data(), tvector.nvals, CUBOOL_HINT_VALUES_SORTED), CUBOOL_STATUS_SUCCESS);

    cuBool_Index nrows;
    cuBool_Index nvals;

    ASSERT_EQ(cuBool_Vector_Nvals(vector, &nvals), CUBOOL_STATUS_SUCCESS);
    ASSERT_EQ(cuBool_Vector_Nrows(vector, &nrows), CUBOOL_STATUS_SUCCESS);

    ASSERT_EQ(nvals, tvector.nvals);
    ASSERT_EQ(nrows, tvector.nrows);

    ASSERT_EQ(cuBool_Vector_Free(vector), CUBOOL_STATUS_SUCCESS);

    ASSERT_EQ(cuBool_Finalize(), CUBOOL_STATUS_SUCCESS);
}

TEST(cuBool_Vector, ExtractValues) {
    cuBool_Vector vector = nullptr;
    cuBool_Index m = 900;
    float density = 0.21;

    testing::Vector tvector = std::move(testing::Vector::generateSparse(m, density));

    ASSERT_EQ(cuBool_Initialize(CUBOOL_HINT_CPU_BACKEND), CUBOOL_STATUS_SUCCESS);

    ASSERT_EQ(cuBool_Vector_New(&vector, m), CUBOOL_STATUS_SUCCESS);
    ASSERT_EQ(cuBool_Vector_Build(vector, tvector.index.data(), tvector.nvals, CUBOOL_HINT_VALUES_SORTED), CUBOOL_STATUS_SUCCESS);

    cuBool_Index nvals = tvector.nvals;
    std::vector<cuBool_Index> rows(tvector.nvals);

    ASSERT_EQ(cuBool_Vector_ExtractValues(vector, rows.data(), &nvals), CUBOOL_STATUS_SUCCESS);

    ASSERT_EQ(nvals, tvector.nvals);

    ASSERT_EQ(cuBool_Vector_Free(vector), CUBOOL_STATUS_SUCCESS);
    ASSERT_EQ(cuBool_Finalize(), CUBOOL_STATUS_SUCCESS);
}

TEST(cuBool_Vector, Marker) {
    cuBool_Vector vector = nullptr;
    cuBool_Index m;

    m = 100;

    const cuBool_Index BUFFER_SIZE = 100;
    const char* marker = "Test Vector Marker";
    cuBool_Index size = 0;
    char buffer[BUFFER_SIZE];

    ASSERT_EQ(cuBool_Initialize(CUBOOL_HINT_CPU_BACKEND), CUBOOL_STATUS_SUCCESS);
    ASSERT_EQ(cuBool_Vector_New(&vector, m), CUBOOL_STATUS_SUCCESS);
    ASSERT_EQ(cuBool_Vector_SetMarker(vector, marker), CUBOOL_STATUS_SUCCESS);
    ASSERT_EQ(cuBool_Vector_Marker(vector, nullptr, &size), CUBOOL_STATUS_SUCCESS);
    ASSERT_LE(size, BUFFER_SIZE);
    ASSERT_EQ(cuBool_Vector_Marker(vector, buffer, &size), CUBOOL_STATUS_SUCCESS);
    ASSERT_EQ(cuBool_Vector_Free(vector), CUBOOL_STATUS_SUCCESS);
    ASSERT_EQ(cuBool_Finalize(), CUBOOL_STATUS_SUCCESS);

    ASSERT_EQ(std::strlen(buffer), size - 1);
    ASSERT_LE(std::strcmp(marker, buffer), 0);

    std::cout << "Source marker: " << marker << std::endl;
    std::cout << "Returned marker: " << buffer << std::endl;
}

TEST(cuBool_Vector, MarkerShort) {
    cuBool_Vector vector = nullptr;
    cuBool_Index m;

    m = 100;

    const cuBool_Index BUFFER_SIZE = 10;
    const char* marker = "Test Vector Marker";
    cuBool_Index size = BUFFER_SIZE;
    char buffer[BUFFER_SIZE];

    ASSERT_EQ(cuBool_Initialize(CUBOOL_HINT_CPU_BACKEND), CUBOOL_STATUS_SUCCESS);
    ASSERT_EQ(cuBool_Vector_New(&vector, m), CUBOOL_STATUS_SUCCESS);
    ASSERT_EQ(cuBool_Vector_SetMarker(vector, marker), CUBOOL_STATUS_SUCCESS);
    ASSERT_EQ(cuBool_Vector_Marker(vector, buffer, &size), CUBOOL_STATUS_SUCCESS);
    ASSERT_EQ(cuBool_Vector_Free(vector), CUBOOL_STATUS_SUCCESS);
    ASSERT_EQ(cuBool_Finalize(), CUBOOL_STATUS_SUCCESS);

    ASSERT_GE(std::strcmp(marker, buffer), 0);

    std::cout << "Source marker: " << marker << std::endl;
    std::cout << "Returned marker: " << buffer << std::endl;
}

CUBOOL_GTEST_MAIN
