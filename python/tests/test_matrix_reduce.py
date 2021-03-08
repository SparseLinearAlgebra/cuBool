import unittest
from tests.config import cfg
from pycubool import io


class TestMatrixReduce(unittest.TestCase):

    def setUp(self) -> None:
        matrices, self.total = cfg.get_test_cases("reduce", 2)
        self.input_matrices_0, self.result_matrices = matrices[0], matrices[1]

    def test_reduce(self):
        """
        Unit test for reduce of matrix
        """
        for i in range(self.total):
            first = io.import_matrix_from_mtx(self.input_matrices_0[i])
            expected = io.import_matrix_from_mtx(self.result_matrices[i])
            actual = first.reduce()

            self.assertTrue(expected.equals(actual))


if __name__ == "__main__":
    unittest.main()
