import pycubool


def compare_matrix(a: pycubool.Matrix, b: pycubool.Matrix) -> bool:
    a_rows, a_cols = a.to_lists()
    b_rows, b_cols = b.to_lists()

    if len(a_rows) != len(b_rows):
        return False
    n = len(a_rows)
    for i in range(n):
        if a_rows[i] != b_rows[i]:
            return False

    if len(a_cols) != len(b_cols):
        return False
    m = len(a_cols)
    for i in range(m):
        if a_cols[i] != b_cols[i]:
            return False

    return True
