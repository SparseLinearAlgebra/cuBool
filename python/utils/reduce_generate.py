import io_file


def reduce(first_name: str, res_name: str):
    first_matrix = io_file.build_matrix_by_name(first_name)
    result_matrix = first_matrix.reduce()

    size = result_matrix.shape
    rows, cols = result_matrix.to_lists()
    io_file.write_matrix_to_file(size, rows, cols, res_name)
