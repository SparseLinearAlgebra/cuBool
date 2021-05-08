import unittest
from tests.config import cfg
from pycubool import io


class TestVectorDuplicate(unittest.TestCase):

    def setUp(self) -> None:
        matrices, self.total = cfg.get_test_cases("vector_dup", 2)
        self.input_vectors_0, self.result_vectors = matrices[0], matrices[1]

    def test_duplicate(self):
        """
        Unit test for duplicate of vector
        """
        for i in range(self.total):
            first = io.import_matrix_from_mtx(self.input_vectors_0[i]).extract_row(0)
            expected = io.import_matrix_from_mtx(self.result_vectors[i]).extract_row(0)
            actual = first.dup()

            self.assertTrue(expected.equals(actual))


if __name__ == "__main__":
    unittest.main()

