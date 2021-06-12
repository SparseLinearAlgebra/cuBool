import unittest
from tests.config import cfg
from pycubool import io


class TestVectorEWiseMult(unittest.TestCase):

    def setUp(self) -> None:
        matrices, self.total = cfg.get_test_cases("vector_ewisemult", 3)
        self.input_vectors_0, self.input_vectors_1, self.result_vectors = matrices[0], matrices[1], matrices[2]

    def test_mult(self):
        """
        Unit test for element-wise multiplication of two vectors
        """
        for i in range(self.total):
            first = io.import_matrix_from_mtx(self.input_vectors_0[i]).extract_row(0)
            second = io.import_matrix_from_mtx(self.input_vectors_1[i]).extract_row(0)
            expected = io.import_matrix_from_mtx(self.result_vectors[i]).extract_row(0)
            actual = first.ewisemult(second)

            self.assertTrue(expected.equals(actual))


if __name__ == "__main__":
    unittest.main()
