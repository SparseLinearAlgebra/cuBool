import pycubool


def _read_matrix(path: str):
    """
    Read matrix from file
    """

    file = open(path, "r")
    n, m = map(int, file.readline().split())
    rows = list(map(int, file.readline().split()))
    cols = list(map(int, file.readline().split()))
    return n, m, [rows, cols]


def build_matrix_from_file(path: str) -> pycubool.Matrix:
    n, m, matrix = _read_matrix(path)
    nvals = len(matrix[0])
    result = pycubool.Matrix.empty([n, m])
    result.build(matrix[0], matrix[1], nvals)

    return result


def compare_matrix(a: pycubool.Matrix, b: pycubool.Matrix) -> bool:
    a_rows, a_cols = a.to_lists()
    b_rows, b_cols = b.to_lists()

    if len(a_rows) != len(b_rows):
        return False
    n = a_rows
    for i in range(n):
        if a_rows[i] != b_rows[i]:
            return False

    if len(a_cols) != len(b_cols):
        return False
    m = a_cols
    for i in range(m):
        if a_cols[i] != b_cols[i]:
            return False

    return True
