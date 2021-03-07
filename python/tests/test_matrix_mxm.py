import unittest
import test_utils
import utils.io_file


class TestMatrixMxm(unittest.TestCase):
    def setUp(self) -> None:
        self.input_matrices = [["matrix_1.mtx", "matrix_2.mtx"], ["matrix_3.mtx", "matrix_4.mtx"]]
        self.result_matrices = ["mxm_res_12.mtx", "mxm_res_34.mtx"]

    def test_mxm(self):
        """
        Unit test for multiplication of two matrices
        """
        for i in range(len(self.input_matrices)):
            matrices = list()
            for matrix in self.input_matrices[i]:
                matrices.append(utils.io_file.build_matrix_by_name(matrix))
            expected_matrix = utils.io_file.build_matrix_by_name(self.result_matrices[i])
            first_matrix = matrices[0]
            second_matrix = matrices[1]
            actual_matrix = first_matrix.mxm(second_matrix)

            self.assertTrue(test_utils.compare_matrix(expected_matrix, actual_matrix))


if __name__ == "__main__":
    unittest.main()
