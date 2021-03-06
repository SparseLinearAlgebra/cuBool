import unittest
import test_utils
import utils.io_file


class TestMatrixKronecker(unittest.TestCase):
    def setUp(self) -> None:
        self.input_matrices = [["matrix_1.mtx", "matrix_2.mtx"], ["matrix_3.mtx", "matrix_4.mtx"]]
        self.result_matrices = ["kronecker_res_12.mtx", "kronecker_res_34.mtx"]

    def test_kronecker(self):
        """
        Unit test for kronecker product of two matrices
        """
        for i in range(len(self.input_matrices)):
            matrices = list()
            for matrix in self.input_matrices[i]:
                matrices.append(utils.io_file.build_matrix_by_name(matrix))
            expected_matrix = utils.io_file.build_matrix_by_name(self.result_matrices[i])
            first_matrix = matrices[0]
            second_matrix = matrices[1]
            actual_matrix = first_matrix.kronecker(second_matrix)

            self.assertTrue(test_utils.compare_matrix(expected_matrix, actual_matrix))


if __name__ == "__main__":
    unittest.main()
