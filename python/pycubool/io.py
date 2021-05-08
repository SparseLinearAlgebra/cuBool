"""
IO operations for exporting/importing mtx data.
Provides features to import/export data or pycubool matrix in mtx format.
"""

from . import Matrix


__all__ = [
    "read_mtx_file",
    "write_mtx_file",
    "import_matrix_from_mtx",
    "export_matrix_to_mtx"
]


def read_mtx_file(path: str):
    """
    Reads mtx file data.

    :param path: Path and name of the file with data
    :return: shape of the matrix, rows data, columns data, number of values
    """

    with open(str(path), 'r') as file:
        line = file.readline()
        while line[0].startswith("#"):
            line = file.readline()

        m, n, nvals = map(int, line.split())
        rows = list()
        cols = list()
        for k in range(nvals):
            line = file.readline()
            if line.startswith("#"):
                continue

            i, j = map(int, line.split())
            rows.append(i)
            cols.append(j)

    return (m, n), rows, cols, nvals


def write_mtx_file(path: str, shape, rows, cols, nvals):
    """
    Save data to mtx file.

    :param path: Path and file name of the file to save data
    :param shape: Matrix shape as tuple
    :param rows: Rows data
    :param cols: Columns data
    :param nvals: Number of values
    :return: None
    """

    assert len(rows) == nvals
    assert len(cols) == nvals

    with open(str(path), "w") as file:
        file.write("# pycubool sparse boolean matrix\n")
        file.write(f"{shape[0]} {shape[1]} {nvals}\n")
        for i in range(nvals):
            file.write(f"{rows[i]} {cols[i]}\n")


def import_matrix_from_mtx(path: str):
    """
    Read matrix from file in the mtx format.

    :param path: Path and name of the file with matrix
    :return: Matrix created from data
    """

    shape, rows, cols, nvals = read_mtx_file(path)
    return Matrix.from_lists(shape=shape, rows=rows, cols=cols, is_sorted=False, no_duplicates=False)


def export_matrix_to_mtx(path: str, matrix: Matrix):
    """
    Save matrix to mtx file.

    :param path: Path and file name of the file to save data
    :param matrix: Matrix to export
    :return: None
    """

    rows, cols = matrix.to_lists()
    nvals = matrix.nvals
    shape = matrix.shape
    write_mtx_file(path=path, shape=shape, rows=rows, cols=cols, nvals=nvals)
