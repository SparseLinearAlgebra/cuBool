import unittest
import pycubool


class TestMatrixMethods(unittest.TestCase):

    def setUp(self) -> None:
        """
        Preparatory actions
        Performed BEFORE tests
        """

        pass

    def tearDown(self) -> None:
        """
        Final actions
        Performed AFTER tests
        """

        pass

    def test_fill(self):
        """
        Unit test for creation and filling a matrix
        """
        a = pycubool.Matrix.empty([4, 4])
        a_rows = [0, 1, 2, 3, 3, 3, 3]
        a_cols = [0, 1, 2, 0, 1, 2, 3]
        a.build(a_rows, a_cols, nvals=7)

        dim = a.shape()
        rows, cols = a.to_lists()
        self.assertEqual(b.nvals, 7)
        self.assertListEqual([4, 4], dim)
        self.assertListEqual(rows, a_rows)
        self.assertListEqual(cols, a_cols)

    def test_resize(self):
        """
        Unit test for resizing a matrix
        """
        a = pycubool.Matrix.empty([4, 4])
        a_rows = []
        a_cols = []
        a.build(a_rows, a_cols, nvals=0)

        a.resize(5, 5)
        dim = a.shape()
        self.assertListEqual([5, 5], dim)

    def test_duplicate(self):
        """
        Unit test for duplication a matrix
        """

        a = pycubool.Matrix.empty([4, 4])
        a_rows = [0, 1, 2, 3, 3, 3, 3]
        a_cols = [0, 1, 2, 0, 1, 2, 3]
        a.build(a_rows, a_cols, nvals=7)

        b = a.duplicate()

        dim = b.shape()
        b_rows, b_cols = b.to_lists()
        self.assertEqual(b.nvals, 7)
        self.assertListEqual([4, 4], dim)
        self.assertListEqual(b_rows, a_rows)
        self.assertListEqual(b_cols, a_cols)

    def test_transpose(self):
        """
        Unit test for transposing a matrix
        """

        a = pycubool.Matrix.empty([2, 3])
        a_rows = [1, 0]
        a_cols = [0, 1]
        a.build(a_rows, a_cols, nvals=2)

        b = a.transpose()

        dim = b.shape()
        b_rows, b_cols = b.to_lists()
        self.assertEqual(b.nvals, 2)
        self.assertListEqual([3, 2], dim)
        self.assertListEqual(b_cols, a_rows)
        self.assertListEqual(b_rows, a_cols)


if __name__ == "__main__":
    unittest.main()
