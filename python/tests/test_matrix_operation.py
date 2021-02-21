import unittest
import pycubool


class TestMatrixOperations(unittest.TestCase):

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

    def test_add(self):
        """
        Unit test for addition of two matrices
        """

        a = pycubool.Matrix.empty([4, 4])
        a_rows = [0, 1, 2, 3, 3, 3, 3]
        a_cols = [0, 1, 2, 0, 1, 2, 3]
        a.build(a_rows, a_cols, nvals=7)

        b = pycubool.Matrix.empty([4, 4])
        b_rows = []
        b_cols = []
        b.build(b_rows, b_cols, nvals=0)

        pycubool.add(b, a)
        rows, cols = b.to_lists()

        self.assertEqual(b.nvals, 7)
        self.assertListEqual(rows, a_rows)
        self.assertListEqual(cols, a_cols)

    def test_mxm(self):
        """
        Unit test for multiplication of two matrices
        """

        a = pycubool.Matrix.empty([4, 4])
        a_rows = [0, 1, 2, 3, 3, 3, 3]
        a_cols = [0, 1, 2, 0, 1, 2, 3]
        a.build(a_rows, a_cols, nvals=7)

        b = pycubool.Matrix.empty([4, 4])
        b_rows = [0, 1]
        b_cols = [0, 1]
        b.build(b_rows, b_cols, nvals=2)

        pycubool.mxm(b, b, a)
        rows, cols = b.to_lists()

        self.assertEqual(b.nvals, 2)
        self.assertListEqual(rows, b_rows)
        self.assertListEqual(cols, b_cols)


if __name__ == "__main__":
    unittest.main()
