from python import pycubool
from python.tests import test_example
import random


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
r, r_set = gen_matrix(dim, to_gen)

print("Matrix a din:", a.shape, "values count:", a.nvals)
print("Matrix r dim:", r.shape, "values count:", r.nvals)

pycubool.add(r, a)

print("Matrix r values count:", r.nvals)

rows, cols = r.to_lists()
res_set = lists_to_pairs(rows, cols)

print(r_set.union(a_set) == res_set)

t = test_example.transitive_closure(a)

print(a.nvals, a.shape)
print(t.nvals, t.shape)

rows = [0, 1, 2, 3, 3, 3, 3]
cols = [0, 1, 2, 0, 1, 2, 3]

matrix = pycubool.Matrix.empty((4, 4))
matrix.build(rows, cols, nvals=7)

transposed = matrix.transpose()
rows, cols = transposed.to_lists()

print([(rows[i], cols[i]) for i in range(transposed.nvals)])
