import unittest
from tests.config import cfg
from pycubool import io


class TestMatrixDuplicate(unittest.TestCase):

    def setUp(self) -> None:
        matrices, self.total = cfg.get_test_cases("matrix_dup", 2)
        self.input_matrices_0, self.result_matrices = matrices[0], matrices[1]

    def test_duplicate(self):
        """
        Unit test for duplicate of matrix
        """
        for i in range(self.total):
            first = io.import_matrix_from_mtx(self.input_matrices_0[i])
            expected = io.import_matrix_from_mtx(self.result_matrices[i])
            actual = first.dup()

            self.assertTrue(expected.equals(actual))


if __name__ == "__main__":
    unittest.main()

