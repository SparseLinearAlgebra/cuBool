from python import pycubool


def transitive_closure(a: pycubool.Matrix):
    """
    Evaluates transitive closure for the provided
    adjacency matrix of the graph

    :param a: Adjacency matrix of the graph
    :return: The transitive closure adjacency matrix
    """

    t = a.duplicate()                 # Duplicate matrix where to store result

    total = t.nvals                   # Current number of values
    changing = True                   # track changing

    while changing:
        pycubool.mxm(t, t, t)         # t += t * t
        changing = t.nvals != total
        total = t.nvals

    return t
