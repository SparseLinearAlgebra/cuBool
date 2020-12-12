import CuBoolWrapper
from CuBoolWrapper.add import Add
from CuBoolWrapper.matrix import Matrix

print("Enter num of rows and cols (a, b)")
a, b = map(int, input().split())
matr = Matrix.sparse(a, b)

print("Enter num of values, rows and cols ")
n = int(input())
rows = list(map(int, input().split()))
cols = list(map(int, input().split()))
matr.build(rows, cols, n)

matr = Add(matr, matr)