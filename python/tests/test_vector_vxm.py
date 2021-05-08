import unittest
from tests.config import cfg
from pycubool import io


class TestVectorVxm(unittest.TestCase):

    def setUp(self) -> None:
        matrices, self.total = cfg.get_test_cases("vxm", 3)
        self.input_vectors_0, self.input_matrices_1, self.result_vectors = matrices[0], matrices[1], matrices[2]

    def test_vxm(self):
        """
        Unit test for vector-matrix multiplication
        """
        for i in range(self.total):
            first = io.import_matrix_from_mtx(self.input_vectors_0[i]).extract_row(0)
            second = io.import_matrix_from_mtx(self.input_matrices_1[i])
            expected = io.import_matrix_from_mtx(self.result_vectors[i]).extract_row(0)
            actual = first.vxm(second)

            self.assertTrue(expected.equals(actual))


if __name__ == "__main__":
    unittest.main()
