import random

from utils import extract_generate, reduce_generate, to_lists_generate, transpose_generate, \
    duplicate_generate, mxm_generate, add_generate, kronecker_generate, io_file, properties_generate


def gen_matrix_data(size: list, density: int):
    m = size[0]
    n = size[1]
    if density > 100:
        density = 100
    elif density < 0:
        density = 0
    nvals = m * n * density // 100

    rows = list()
    cols = list()

    for _ in range(nvals):
        i = random.randrange(0, m)
        j = random.randrange(0, n)
        rows.append(i)
        cols.append(j)

    return rows, cols


def gen_matrices(sizes: list, names: list):
    amount = len(sizes)
    for i in range(amount):
        m = sizes[i][0]
        n = sizes[i][1]
        density = random.randint(0, 100)
        rows, cols = gen_matrix_data([m, n], density)
        io_file.write_matrix_to_file([m, n], rows, cols, names[i % len(names)])


max_size = 50
names = ["matrix_1.mtx", "matrix_2.mtx",
         "matrix_3.mtx", "matrix_4.mtx",
         "matrix_5.mtx", "matrix_6.mtx"]
sizes = [[50, 50], [50, 50],
         [25, 30], [30, 50],
         [10, 50], [50, 10]]
gen_matrices(sizes, names)

properties_generate.properties(names[0], "property_res_1.mtx")
properties_generate.properties(names[1], "property_res_2.mtx")

reduce_generate.reduce(names[0], "reduce_res_1.mtx")
reduce_generate.reduce(names[1], "reduce_res_2.mtx")

to_lists_generate.to_lists(names[0], "to_lists_res_1.mtx")
to_lists_generate.to_lists(names[1], "to_lists_res_2.mtx")

transpose_generate.transpose(names[0], "transpose_res_1.mtx")
transpose_generate.transpose(names[1], "transpose_res_2.mtx")

extract_generate.extract(names[0], "extract_res_1.mtx")
extract_generate.extract(names[1], "extract_res_2.mtx")

duplicate_generate.duplicate(names[0], "duplicate_res_1.mtx")
duplicate_generate.duplicate(names[1], "duplicate_res_2.mtx")

mxm_generate.mxm(names[0], names[1], "mxm_res_12.mtx")
mxm_generate.mxm(names[2], names[3], "mxm_res_34.mtx")

add_generate.add(names[0], names[1], "add_res_12.mtx")
add_generate.add(names[2], names[3], "add_res_34.mtx")

kronecker_generate.kronecker(names[0], names[1], "kronecker_res_12.mtx")
kronecker_generate.kronecker(names[2], names[3], "kronecker_res_34.mtx")
