import unittest
from tests.config import cfg
from pycubool import io


class TestMatrixReduceVector(unittest.TestCase):

    def setUp(self) -> None:
        matrices, self.total = cfg.get_test_cases("reduce_vector", 2)
        self.input_matrices_0, self.result_vectors = matrices[0], matrices[1]

    def test_reduce_vector(self):
        """
        Unit test for reduce to vector of matrix
        """
        for i in range(self.total):
            first = io.import_matrix_from_mtx(self.input_matrices_0[i])
            expected = io.import_matrix_from_mtx(self.result_vectors[i]).extract_row(0)
            actual = first.reduce_vector()

            self.assertTrue(expected.equals(actual))


if __name__ == "__main__":
    unittest.main()
