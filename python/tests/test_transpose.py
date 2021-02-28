import unittest
import test_utils


class TestMatrixTranspose(unittest.TestCase):
    def test_transpose(self):
        """
        Unit test for transpose of matrix
        """
        first_matrix = test_utils.build_matrix_from_file("matrices/transpose.txt")

        actual_matrix = first_matrix.transpose()
        expected_matrix = test_utils.build_matrix_from_file("matrices/transpose_result.txt")

        self.assertTrue(test_utils.compare_matrix(expected_matrix, actual_matrix))


if __name__ == "__main__":
    unittest.main()
