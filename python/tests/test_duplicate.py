import unittest
import test_utils


class TestMatrixDuplicate(unittest.TestCase):
    def test_duplicate(self):
        """
        Unit test for duplicate of matrix
        """
        first_matrix = test_utils.build_matrix_from_file("/matrices/duplicate.txt")

        actual_matrix = first_matrix.dup()

        self.assertTrue(test_utils.compare_matrix(first_matrix, actual_matrix))


if __name__ == "__main__":
    unittest.main()
