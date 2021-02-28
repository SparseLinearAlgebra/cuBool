import unittest
import test_utils


class TestMatrixToLists(unittest.TestCase):
    def setUp(self) -> None:
        self.matrix = test_utils.build_matrix_from_file("matrices/to_lists.txt")
        self.result_lists = list()
        with open("matrices/to_lists_result.txt", 'r') as _file:
            line = list(map(int, _file.readline().split()))
            self.result_lists.append(line)

    def test_to_lists(self):
        """
        Unit test to extract a matrix as two lists
        """

        actual_rows, actual_cols = self.matrix.to_lists()

        self.assertListEqual(self.result_lists[0], actual_rows)
        self.assertListEqual(self.result_lists[1], actual_cols)


if __name__ == "__main__":
    unittest.main()
