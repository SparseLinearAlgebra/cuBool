import unittest
import test_utils


class TestMatrixKronecker(unittest.TestCase):
    def test_kronecker(self):
        """
        Unit test for kronecker product of two matrices
        """
        first_matrix = test_utils.build_matrix_from_file("/matrices/kronecker_4.txt")
        second_matrix = test_utils.build_matrix_from_file("/matrices/kronecker_5.txt")

        actual_matrix = first_matrix.kronecker(second_matrix)
        expected_matrix = test_utils.build_matrix_from_file("/matrices/kronecker_result.txt")

        self.assertTrue(test_utils.compare_matrix(expected_matrix, actual_matrix))


if __name__ == "__main__":
    unittest.main()
