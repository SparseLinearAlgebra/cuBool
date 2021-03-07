import unittest
import test_utils
import utils.io_file


class TestMatrixExtractMatrix(unittest.TestCase):
    def setUp(self) -> None:
        self.input_matrices = [["matrix_1.mtx"], ["matrix_2.mtx"]]
        self.result_matrices = ["extract_res_1.mtx", "extract_res_2.mtx"]

    def test_extract_matrix(self):
        """
        Unit test for extract submatrix from left-upper corner of matrix
        """
        for i in range(len(self.input_matrices)):
            matrices = list()
            for matrix in self.input_matrices[i]:
                matrices.append(utils.io_file.build_matrix_by_name(matrix))
            expected_matrix = utils.io_file.build_matrix_by_name(self.result_matrices[i])

            actual_matrix = matrices[0].extract_matrix(0, 0, expected_matrix.shape)

            self.assertTrue(test_utils.compare_matrix(expected_matrix, actual_matrix))


if __name__ == "__main__":
    unittest.main()
