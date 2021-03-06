import io_file


def duplicate(first_name: str, res_name: str):
    first_matrix = io_file.build_matrix_by_name(first_name)
    result_matrix = first_matrix.dup()

    size = result_matrix.shape
    rows, cols = result_matrix.to_lists()
    io_file.write_matrix_to_file(size, rows, cols, res_name)
