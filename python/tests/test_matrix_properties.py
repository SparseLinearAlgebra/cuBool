import unittest
import test_utils


class TestMatrixMethods(unittest.TestCase):

    def setUp(self) -> None:
        self.matrix = test_utils.build_matrix_from_file("matrices/property.txt")
        self.result_property = list()
        with open("/matrices/property_result.txt", 'r') as _file:
            line = list(map(int, _file.readline().split()))
            self.result_property.append(line)

    def tearDown(self) -> None:
        """
        Final actions
        Performed AFTER tests
        """

        pass

    def test_nrows(self):
        self.assertEqual(self.result_property[0][0], self.matrix.nrows)

    def test_ncols(self):
        self.assertEqual(self.result_property[1][0], self.matrix.ncols)

    def test_nvals(self):
        self.assertEqual(self.result_property[2][0], self.matrix.nvals)

    def test_shape(self):
        self.assertEqual(self.result_property[0], self.matrix.shape)


if __name__ == "__main__":
    unittest.main()
