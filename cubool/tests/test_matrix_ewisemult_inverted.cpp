#include "cubool/cubool.h"
#include <algorithm>
#include <testing/testing.hpp>

using DataMatrix = std::vector<std::vector<int>>;

void testApplyNotMask(const DataMatrix& matrix_data, const DataMatrix& mask_data) {
    cuBool_Index nrows, ncols;
    nrows                       = matrix_data.size();
    ncols                       = matrix_data[0].size();
    testing::Matrix test_matrix = testing::Matrix::generatet(nrows, ncols,
                                                             [&matrix_data](cuBool_Index i, cuBool_Index j) { return matrix_data[i][j]; });
    nrows                       = mask_data.size();
    ncols                       = mask_data[0].size();
    testing::Matrix test_mask   = testing::Matrix::generatet(nrows, ncols,
                                                             [&mask_data](cuBool_Index i, cuBool_Index j) { return mask_data[i][j]; });

    cuBool_Matrix matrix, mask, result;
    ASSERT_EQ(cuBool_Matrix_New(&matrix, test_matrix.nrows, test_matrix.ncols), CUBOOL_STATUS_SUCCESS);
    ASSERT_EQ(cuBool_Matrix_New(&mask, test_mask.nrows, test_mask.ncols), CUBOOL_STATUS_SUCCESS);
    ASSERT_EQ(cuBool_Matrix_New(&result, test_matrix.nrows, test_matrix.ncols), CUBOOL_STATUS_SUCCESS);

    ASSERT_EQ(cuBool_Matrix_Build(matrix, test_matrix.rowsIndex.data(), test_matrix.colsIndex.data(), test_matrix.nvals,
                                  CUBOOL_HINT_VALUES_SORTED & CUBOOL_HINT_NO_DUPLICATES),
              CUBOOL_STATUS_SUCCESS);
    ASSERT_EQ(cuBool_Matrix_Build(mask, test_mask.rowsIndex.data(), test_mask.colsIndex.data(), test_mask.nvals,
                                  CUBOOL_HINT_VALUES_SORTED & CUBOOL_HINT_NO_DUPLICATES),
              CUBOOL_STATUS_SUCCESS);

    cuBool_Matrix_EWiseMulInverted(result, matrix, mask, CUBOOL_HINT_NO);

    // validate value of algorithm
    cuBool_Index nvals;
    cuBool_Matrix_Nvals(result, &nvals);
    std::vector<cuBool_Index> rows(nvals), cols(nvals);
    cuBool_Matrix_ExtractPairs(result, rows.data(), cols.data(), &nvals);

    cuBool_Matrix_Free(matrix);
    cuBool_Matrix_Free(mask);
    cuBool_Matrix_Free(result);

    auto mask_data_inverted = mask_data;
    for (auto& row : mask_data_inverted) {
        for (int& value : row) {
            value = !value;
        }
    }

    std::vector result_data(matrix_data.size(), std::vector(matrix_data[0].size(), 0));
    for (int i = 0; i < nvals; i++) {
        result_data[rows[i]][cols[i]] = 1;
    }

    for (int i = 0; i < matrix_data.size(); i++) {
        for (int j = 0; j < matrix_data[0].size(); j++) {
            ASSERT_EQ(matrix_data[i][j] * mask_data_inverted[i][j], result_data[i][j]);
        }
    }
}

TEST(cuBool_Matrix, ApplyMatrix) {
    ASSERT_EQ(cuBool_Initialize(CUBOOL_HINT_NO), CUBOOL_STATUS_SUCCESS);

    DataMatrix matrix{
            {1, 0, 0},
            {0, 0, 0},
            {0, 1, 0},
    };

    DataMatrix mask{
            {0, 1, 1},
            {1, 0, 1},
            {0, 1, 1},
    };
    // iverted is
    // 1 0 0
    // 0 1 0
    // 1 0 0
    // matrix & ~mask must have (0, 0)

    testApplyNotMask(matrix, mask);

    ASSERT_EQ(cuBool_Finalize(), CUBOOL_STATUS_SUCCESS);
}

TEST(cuBool_Matrix, ApplyMatrixRandom) {
    ASSERT_EQ(cuBool_Initialize(CUBOOL_HINT_NO), CUBOOL_STATUS_SUCCESS);

    for (int i = 0; i < 102; i++) {
        int n = rand() % 10 + 1;
        int m = rand() % 10 + 1;

        DataMatrix matrix(n, std::vector(m, 0));
        DataMatrix mask(n, std::vector(m, 0));

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                matrix[i][j] = rand() & 1;
                mask[i][j]   = rand() & 1;
            }
        }

        testApplyNotMask(matrix, mask);
    }

    ASSERT_EQ(cuBool_Finalize(), CUBOOL_STATUS_SUCCESS);
}


CUBOOL_GTEST_MAIN
