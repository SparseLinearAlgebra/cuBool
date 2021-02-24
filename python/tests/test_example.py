from python import pycubool


def transitive_closure(a: pycubool.Matrix):
    """
    Evaluates transitive closure for the provided
    adjacency matrix of the graph

    :param a: Adjacency matrix of the graph
    :return: The transitive closure adjacency matrix
    """

    t = a.dup()                       # Duplicate matrix where to store result
    total = 0                         # Current number of values

    while total != t.nvals:
        total = t.nvals
        t.mxm(t, t, accumulate=True)  # t += t * t

    return t
