import random
from python import pycubool
from python.tests import test_transitive_closure


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
    mat.build(rows, cols, nvals)
    return mat, lists_to_pairs(rows, cols)


dim = (100, 100)
to_gen = 500

a, a_set = gen_matrix(dim, to_gen)
b, b_set = gen_matrix(dim, to_gen)

print("Matrix a din:", a.shape, "values count:", a.nvals)
print("Matrix b dim:", b.shape, "values count:", b.nvals)

r = a.ewiseadd(b)

print("Matrix r values count:", r.nvals)

rows, cols = r.to_lists()
res_set = lists_to_pairs(rows, cols)

print(b_set.union(a_set) == res_set)

t = test_transitive_closure.transitive_closure(a)

print(a.nvals, a.shape)
print(t.nvals, t.shape)

rows = [0, 1, 2, 3, 3, 3, 3]
cols = [0, 1, 2, 0, 1, 2, 3]

matrix = pycubool.Matrix.from_lists((4, 4), rows, cols, is_sorted=True)
transposed = matrix.transpose()
submatrix = matrix[1:, 1:]
rows, cols = transposed.to_lists()

print([(rows[i], cols[i]) for i in range(transposed.nvals)])

print(matrix)
print(transposed)
print(submatrix)

