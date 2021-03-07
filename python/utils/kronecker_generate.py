import io_file


def kronecker(first_name: str, second_name: str, res_name: str):
    first_matrix = io_file.build_matrix_by_name(first_name)
    second_matrix = io_file.build_matrix_by_name(second_name)
    result_matrix = first_matrix.kronecker(second_matrix)

    size = result_matrix.shape
    rows, cols = result_matrix.to_lists()
    io_file.write_matrix_to_file(size, rows, cols, res_name)
