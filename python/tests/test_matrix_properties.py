import unittest
import utils.io_file


class TestMatrixMethods(unittest.TestCase):
    def setUp(self) -> None:
        self.input_matrices = [["matrix_1.mtx"], ["matrix_2.mtx"]]
        self.result_matrices = ["property_res_1.mtx", "property_res_2.mtx"]

        self.matrices = list()
        self.result_property = list()
        for i in range(len(self.input_matrices)):
            for matrix in self.input_matrices[i]:
                self.matrices.append(utils.io_file.build_matrix_by_name(matrix))

        for i in range(len(self.result_matrices)):
            self.result_property.append(utils.io_file.build_matrix_by_name(self.result_matrices[i]))

    def tearDown(self) -> None:
        """
        Final actions
        Performed AFTER tests
        """

        pass

    def test_nrows(self):
        for i in range(len(self.matrices)):
            self.assertEqual(self.result_property[i].nrows, self.matrices[i].nrows)

    def test_ncols(self):
        for i in range(len(self.matrices)):
            self.assertEqual(self.result_property[i].ncols, self.matrices[i].ncols)

    def test_nvals(self):
        for i in range(len(self.matrices)):
            self.assertEqual(self.result_property[i].nvals, self.matrices[i].nvals)

    def test_shape(self):
        for i in range(len(self.matrices)):
            self.assertEqual(self.result_property[i].shape, self.matrices[i].shape)


if __name__ == "__main__":
    unittest.main()
