import unittest
import test_utils


class TestMatrixReduce(unittest.TestCase):
    def test_reduce(self):
        """
        Unit test for reduce of matrix
        """
        first_matrix = test_utils.build_matrix_from_file("matrices/reduce.txt")

        actual_matrix = first_matrix.reduce()
        expected_matrix = test_utils.build_matrix_from_file("matrices/reduce_result.txt")

        self.assertTrue(test_utils.compare_matrix(expected_matrix, actual_matrix))


if __name__ == "__main__":
    unittest.main()
