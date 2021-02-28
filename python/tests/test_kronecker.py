import unittest
import test_utils


class TestMatrixKronecker(unittest.TestCase):
    def test_kronecker(self):
        """
        Unit test for kronecker product of two matrices
        """
        first_matrix = test_utils.build_matrix_from_file("../data/kronecker_1.mtx")
        second_matrix = test_utils.build_matrix_from_file("../data/kronecker_2.mtx")

        actual_matrix = first_matrix.kronecker(second_matrix)
        expected_matrix = test_utils.build_matrix_from_file("../data/kronecker_result.mtx")

        self.assertTrue(test_utils.compare_matrix(expected_matrix, actual_matrix))


if __name__ == "__main__":
    unittest.main()
