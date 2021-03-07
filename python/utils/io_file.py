import os

import pycubool

__all__ = [
    "build_matrix_by_name",
    "write_matrix_to_file",
    "path_to_matrices"
]

path_to_matrices = os.environ["TestMatrix_Path"]


def read_matrix_mtx(path: str):
    """
    Read matrix from file
    """
    with open(path, 'r') as file:
        n, m, nvals = map(int, file.readline().split())
        rows = list()
        cols = list()
        for k in range(nvals):
            i, j = map(int, file.readline().split())
            rows.append(i)
            cols.append(j)
    return n, m, [rows, cols]


def build_matrix_by_name(matrix_name: str) -> pycubool.Matrix:
    n, m, matrix = read_matrix_mtx(path_to_matrices + matrix_name)
    nvals = len(matrix[0])
    result = pycubool.Matrix.empty([n, m])
    result.build(matrix[0], matrix[1], nvals)

    return result


def write_matrix_to_file(size: list, rows: list, cols: list, name: str):
    if len(rows) != len(cols):
        raise Exception("Size of rows and cols arrays must match the nval values")

    nvals = len(rows)
    path = path_to_matrices + name
    with open(path, "w") as file:
        file.write(f"{size[0]} {size[1]} {nvals}\n")
        for i in range(nvals):
            file.write(f"{rows[i]} {cols[i]}\n")
