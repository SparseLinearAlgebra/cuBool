import random
import pycubool
from examples import transitive_closure as ts


def lists_to_pairs(rows, cols):
    pairs = set()
    for i in range(len(rows)):
        pairs.add((rows[i], cols[i]))

    return pairs


def gen_matrix_data(size, seed):
    m = size[0]
    n = size[1]

    values = set()
    rows = list()
    cols = list()

    for _ in range(seed):
        i = random.randrange(0, m)
        j = random.randrange(0, n)
        values.add((i, j))

    for (i, j) in values:
        rows.append(i)
        cols.append(j)

    return rows, cols, len(values)


def gen_matrix(size, seed):
    rows, cols, nvals = gen_matrix_data(size, seed)
    mat = pycubool.Matrix.empty(size)
    mat.build(rows, cols, no_duplicates=True)
    return mat, lists_to_pairs(rows, cols)


dim = (10, 10)
to_gen = 50

# pycubool.setup_logger(pycubool.get_default_log_name())
a, a_set = gen_matrix(dim, to_gen)
b, b_set = gen_matrix(dim, to_gen)

print("Matrix a din:", a.shape, "values count:", a.nvals)
print("Matrix b dim:", b.shape, "values count:", b.nvals)

r = a.ewiseadd(b, time_check=True)
c = a.mxm(b, accumulate=True)
print(a, b, c, sep="\n")

print("Matrix r values count:", r.nvals)

rows, cols = r.to_lists()
res_set = lists_to_pairs(rows, cols)

print(b_set.union(a_set) == res_set)

t = ts.transitive_closure(a)

print(a.nvals, a.shape)
print(t.nvals, t.shape)

rows = [0, 1, 2, 3, 3, 3, 3]
cols = [0, 1, 2, 0, 1, 2, 3]

matrix = pycubool.Matrix.from_lists((4, 4), rows, cols, is_sorted=True)
transposed = matrix.transpose()
submatrix = matrix[0:3, 1:]
rows, cols = transposed.to_lists()

print([(rows[i], cols[i]) for i in range(transposed.nvals)])

print(matrix)
print(transposed)
print(submatrix)

print(list(iter(matrix)))

matrix = pycubool.Matrix.from_lists((4, 4), [0, 1, 2, 3], [0, 1, 2, 0], is_sorted=True)
print(matrix.extract_matrix(0, 1, shape=(3, 3)))

matrix = pycubool.Matrix.empty(shape=(4, 4))
matrix[0, 0] = True
matrix[1, 1] = True
matrix[2, 3] = True
matrix[3, 1] = True
print(matrix)

matrix = pycubool.Matrix.from_lists((4, 4), [0, 1, 2, 2], [0, 1, 0, 2])
print(matrix.reduce())

a = pycubool.Matrix.empty(shape=(4, 4))
a[0, 0] = True
a[0, 3] = True
print(a)
a = a.ewiseadd(a.transpose())
print(a)
a[3, 3] = True
print(a)

a = pycubool.Matrix.empty(shape=(4, 4))
a[0, 0] = True
a[1, 3] = True
a[1, 0] = True
a[2, 2] = True
vals = a.to_list()
print(vals)
print(a.equals(a))

a = pycubool.Matrix.empty(shape=(4, 4))
print(a.marker)
a.set_marker("meow")
print(a.marker)

matrix = pycubool.Matrix.from_lists((5, 4), [0, 1, 2, 4], [0, 1, 1, 3])
vector = pycubool.Vector.from_list(4, [0, 1, 2])
print(matrix.mxv(vector))

matrix = pycubool.Matrix.from_lists((5, 4), [0, 1, 2, 4], [0, 1, 1, 3])
print(matrix.reduce_vector(), matrix.reduce_vector(transpose=True), sep="")

matrix = pycubool.Matrix.from_lists((5, 4), [0, 1, 2, 4], [0, 1, 1, 3])
print(matrix.extract_row(1))

matrix = pycubool.Matrix.from_lists((5, 4), [0, 1, 2, 4], [0, 1, 1, 3])
print(matrix.extract_col(1))

matrix = pycubool.Matrix.generate(shape=(5, 4), density=0.5)
print(matrix, matrix.extract_row(1), matrix.extract_col(2), sep="\n")

matrix = pycubool.Matrix.generate(shape=(5, 4), density=0.5)
vector = pycubool.Vector.generate(nrows=4, density=0.6)
print(matrix, vector, matrix.mxv(vector), sep="\n")

matrix = pycubool.Matrix.generate(shape=(5, 4), density=0.5)
vector = pycubool.Vector.generate(nrows=5, density=0.6)
print(matrix, vector, vector.vxm(matrix), sep="\n")

matrix = pycubool.Matrix.generate(shape=(10, 6), density=0.2)
print(matrix, matrix.reduce_vector(transpose=False), matrix.reduce_vector(transpose=True), sep="\n")

vector = pycubool.Vector.from_list(4, [0, 1, 3], is_sorted=True, no_duplicates=True)
print(vector)

vector = pycubool.Vector.generate(nrows=4, density=0.5)
print(vector)

vector = pycubool.Vector.empty(4)
vector.build([0, 1, 3], is_sorted=True, no_duplicates=True)
print(vector)

a = pycubool.Vector.from_list(4, [0, 1, 3], is_sorted=True, no_duplicates=True)
b = a.dup()
b[2] = True
print(a, b, sep="")

a = pycubool.Vector.empty(nrows=4)
print(a.marker)
a.set_marker("meow")
print(a.marker)

a = pycubool.Vector.empty(nrows=4)
a[0] = True
a[2] = True
a[3] = True
rows = a.to_list()
print(list(rows))

vector = pycubool.Vector.from_list(4, [0, 1, 3], is_sorted=True, no_duplicates=True)
print(vector)

vector = pycubool.Vector.from_list(5, [0, 1, 3])
print(vector.extract_vector(1, nrows=3))

matrix = pycubool.Matrix.from_lists((5, 4), [0, 1, 2, 4], [0, 1, 1, 3])
vector = pycubool.Vector.from_list(5, [2, 3, 4])
print(vector.vxm(matrix))

a = pycubool.Vector.from_list(4, [0, 1, 3])
b = pycubool.Vector.from_list(4, [1, 2, 3])
print(a.ewiseadd(b))

vector = pycubool.Vector.from_list(5, [2, 3, 4])
print(vector.reduce())

vector = pycubool.Vector.from_list(5, [1, 3, 4])
print(list(iter(vector)))

vector = pycubool.Vector.from_list(5, [1, 3, 4])
print(vector[0:3], vector[2:], sep="")

vector = pycubool.Vector.empty(4)
vector[0] = True
vector[2] = True
vector[3] = True
print(vector)

a = pycubool.Vector.from_list(4, [0, 1, 3])
b = pycubool.Vector.from_list(4, [1, 2, 3])
print(a.ewisemult(b))
