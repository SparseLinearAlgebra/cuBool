from io import StringIO
from . import Matrix


__all__ = [
    "matrix_data_to_gviz",
    "matrices_data_to_gviz",
    "matrices_to_gviz"
]


DEFAULT_VERTEX_COLOR = "black"
DEFAULT_EDGE_COLOR = "black"
DEFAULT_LABEL = ""
DEFAULT_BASE_VERTEX = 0


def matrices_to_gviz(matrices: dict, **kwargs) -> str:
    """
    Export the labeled square matrices dictionary to the graph viz graph description script.
    All matrices must have the save shape.

    >>> name = "Test"                           # Displayed graph name
    >>> shape = (4, 4)                          # Adjacency matrices shape
    >>> colors = {"a": "red", "b": "green"}     # Colors per label
    >>>
    >>> a = Matrix.empty(shape=shape)        # Edges labeled as 'a'
    >>> a[0, 1] = True
    >>> a[1, 2] = True
    >>> a[2, 0] = True
    >>>
    >>> b = Matrix.empty(shape=shape)        # Edges labeled as 'b'
    >>> b[2, 3] = True
    >>> b[3, 2] = True
    >>>
    >>> print(matrices_to_gviz(matrices={"a": a, "b": b}, graph_name=name, edge_colors=colors))
    '
        digraph G {
            graph [label=Test];
            node [color=black];
            0 -> 1 [label=a,color=red];
            1 -> 2 [label=a,color=red];
            2 -> 0 [label=a,color=red];
            2 -> 3 [label=b,color=green];
            3 -> 2 [label=b,color=green];
        }
    '

    :param matrices: The dictionary with matrix label mapping to the pycubool Matrix instance
    :param kwargs: Optional arguments to tweak gviz graph appearance
        :param graph_name: Name of the graph
        :param base_vertex: Index of the base vertex (by default is 0)
        :param vertex_color: Color of the vertices (use \" - \" to pass html style hex colors)
        :param edges_colors: Dictionary with mapping matrix label to its edges colors (use \" - \" to pass html style hex colors)

    :return: Text script gviz graph representation
    """

    assert len(matrices) > 0

    base = next(iter(matrices.values()))
    shape = base.shape

    for m in matrices.values():
        assert m.shape == shape

    matrices_data = dict()

    for key, value in matrices.items():
        assert isinstance(value, Matrix)
        matrices_data[key] = value.to_lists()

    return matrices_data_to_gviz(shape=shape, matrices=matrices_data, **kwargs)


def matrix_data_to_gviz(shape, rows, cols, **kwargs) -> str:
    check_len_rows_cols(rows, cols)
    check_shape(shape)

    _result = StringIO()
    args = unboxing_kwargs([], **kwargs)
    _result.write("digraph G {\n")
    _result.write(args["graph_name"])
    _result.write(args["vertex_color"])
    n = shape[0]

    used_vertex = [False] * n
    _result.write(build_edges(rows, cols, args["base_vertex"], args["label"], args["edge_color"], used_vertex))

    for i, v in enumerate(used_vertex):
        if not v:
            _result.write(f'{i + args["base_vertex"]};\n')

    _result.write("}\n")
    result = _result.getvalue()
    _result.close()

    return result


def matrices_data_to_gviz(shape, matrices: dict, **kwargs):
    check_shape(shape)
    for name in matrices.keys():
        check_len_rows_cols(matrices[name][0], matrices[name][1])

    _result = StringIO()
    args = unboxing_kwargs(matrices.keys(), **kwargs)
    _result.write("digraph G {\n")
    _result.write(args["graph_name"])
    _result.write(args["vertex_color"])
    n = shape[0]

    used_vertex = [False] * n
    for name in matrices.keys():
        rows = matrices[name][0]
        cols = matrices[name][1]
        label = name
        edge_color = args["edge_colors"].get(name)
        _result.write(build_edges(rows, cols, args["base_vertex"], label, edge_color, used_vertex))

    for i, v in enumerate(used_vertex):
        if not v:
            _result.write(f'{i + args["base_vertex"]};\n')

    _result.write("}\n")
    result = _result.getvalue()
    _result.close()

    return result


def build_edges(rows, cols, base_vertex, label, edge_color, used_vertex):
    _result = StringIO()

    for i, j in zip(rows, cols):
        _result.write(f'{i + base_vertex} -> {j + base_vertex} [label={label},color={edge_color}];\n')
        used_vertex[i] = True
        used_vertex[j] = True

    result = _result.getvalue()
    _result.close()

    return result


def unboxing_kwargs(labels_set, **kwargs) -> dict:
    _graph_name = kwargs.get("graph_name")
    _vertex_color = kwargs.get("vertex_color")
    _edge_color = kwargs.get("edge_color")
    _base_vertex = kwargs.get("base_vertex")
    _label = kwargs.get("label")
    edge_colors = kwargs.get("edge_colors")

    graph_name = "" if _graph_name is None else f'graph [label={_graph_name}];\n'
    vertex_color = f'node [color={DEFAULT_VERTEX_COLOR if _vertex_color is None else _vertex_color}];\n'
    edge_color = DEFAULT_EDGE_COLOR if _edge_color is None else _edge_color
    base_vertex = DEFAULT_BASE_VERTEX if _base_vertex is None else _base_vertex
    label = DEFAULT_LABEL if _label is None else _label

    presenter_edge_colors = edge_colors.keys()
    for l in labels_set:
        if l not in presenter_edge_colors:
            edge_colors[l] = DEFAULT_EDGE_COLOR

    result = dict()
    result["graph_name"] = graph_name
    result["vertex_color"] = vertex_color
    result["edge_color"] = edge_color
    result["label"] = label
    result["base_vertex"] = base_vertex
    result["edge_colors"] = edge_colors

    return result


def check_shape(shape: list):
    if shape[0] != shape[1]:
        raise Exception("Matrix must be square")


def check_len_rows_cols(rows, cols):
    if len(rows) != len(cols):
        raise Exception("Rows and cols arrays must have equal size")
