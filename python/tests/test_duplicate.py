import unittest
import test_utils
import utils.io_file


class TestMatrixDuplicate(unittest.TestCase):
    def setUp(self) -> None:
        self.input_matrices = [["matrix_1.mtx"], ["matrix_2.mtx"]]
        self.result_matrices = ["duplicate_res_1.mtx", "duplicate_res_2.mtx"]

    def test_duplicate(self):
        """
        Unit test for duplicate of matrix
        """
        for i in range(len(self.input_matrices)):
            matrices = list()
            for matrix in self.input_matrices[i]:
                matrices.append(utils.io_file.build_matrix_by_name(matrix))
            expected_matrix = utils.io_file.build_matrix_by_name(self.result_matrices[i])
            actual_matrix = matrices[0].dup()

            self.assertTrue(test_utils.compare_matrix(expected_matrix, actual_matrix))


if __name__ == "__main__":
    unittest.main()
