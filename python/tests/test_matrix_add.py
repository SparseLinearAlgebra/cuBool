import unittest
import test_utils


class TestMatrixAdd(unittest.TestCase):
    def test_add(self):
        """
        Unit test for addition of two matrices
        """
        first_matrix = test_utils.build_matrix_from_file("/matrices/add_1.txt")
        second_matrix = test_utils.build_matrix_from_file("/matrices/add_2.txt")

        actual_matrix = first_matrix.ewiseadd(second_matrix)
        expected_matrix = test_utils.build_matrix_from_file("/matrices/add_result.txt")

        self.assertTrue(test_utils.compare_matrix(expected_matrix, actual_matrix))


if __name__ == "__main__":
    unittest.main()
