import unittest
import utils.io_file


class TestMatrixToLists(unittest.TestCase):
    def setUp(self) -> None:
        self.input_matrices = [["matrix_1.mtx"], ["matrix_2.mtx"]]
        self.result_matrices = ["to_lists_res_1.mtx", "to_lists_res_2.mtx"]

        self.matrices = list()
        self.result_lists = list()
        for i in range(len(self.input_matrices)):
            for matrix in self.input_matrices[i]:
                self.matrices.append(utils.io_file.build_matrix_by_name(matrix))

        for i in range(len(self.result_matrices)):
            self.result_lists.append(utils.io_file.build_matrix_by_name(self.result_matrices[i]))

    def test_to_lists(self):
        """
        Unit test to extract a matrix as two lists
        """
        for i in range(len(self.matrices)):

            actual_rows, actual_cols = self.matrices[i].to_lists()
            expected_rows, expected_cols = self.result_lists[i].to_lists()

            # element-wise compare
            if len(actual_cols) != len(expected_cols) and len(expected_rows) != len(actual_rows):
                self.assertTrue(False, msg="Length of lists doesn't match")
            for j in range(len(actual_cols)):
                self.assertEqual(expected_rows[j], actual_rows[j])
                self.assertEqual(expected_cols[j], actual_cols[j])


if __name__ == "__main__":
    unittest.main()
