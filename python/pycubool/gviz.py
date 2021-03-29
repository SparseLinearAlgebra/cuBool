from io import StringIO

__all__ = [
    "matrix_data_to_gviz",
    "matrices_data_to_gviz"
]


def matrix_data_to_gviz(shape, rows, cols, **kwargs) -> str:
    check_len_rows_cols(rows, cols)
    check_shape(shape)

    _result = StringIO()
    args = unboxing_kwargs(kwargs)
    _result.write("digraph {\n")
    _result.write(args["graph_name"])
    _result.write(args["vertex_color"])
    n = shape[0]

    used_vertex = [False] * n
    temp = build_edges(rows, cols, args["base_vertex"], args["label"], args["edge_color"], used_vertex)
    _result.write(temp[0])
    used_vertex = temp[1]
    for i in range(n):
        if not used_vertex[i]:
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
    args = unboxing_kwargs(kwargs)
    _result.write("digraph {\n")
    _result.write(args["graph_name"])
    _result.write(args["vertex_color"])
    n = shape[0]

    used_vertex = [False] * n
    for name in matrices.keys():
        rows = matrices[name][0]
        cols = matrices[name][0]
        label = name
        edge_color = args["edge_colors"].get(name)
        temp = build_edges(rows, cols, args["base_vertex"], label, edge_color, used_vertex)
        _result.write(temp[0])
        used_vertex = temp[1]
    for i in range(n):
        if not used_vertex[i]:
            _result.write(f'{i + args["base_vertex"]};\n')
    _result.write("}\n")
    result = _result.getvalue()
    _result.close()
    return result


def build_edges(rows, cols, base_vertex, label, edge_color, used_vertex):
    _result = StringIO()
    for i in range(len(rows)):
        _result.write(f'{rows[i] + base_vertex} -> {cols[i] + base_vertex} [label={label},color={edge_color}];\n')
        used_vertex[rows[i]] = True
        used_vertex[cols[i]] = True
    result = _result.getvalue()
    _result.close()
    return result, used_vertex


def unboxing_kwargs(kwargs: dict) -> dict:
    graph_name = kwargs.get("graph_name")
    vertex_color = kwargs.get("vertex_color")
    edge_color = kwargs.get("edge_color")
    _base_vertex = kwargs.get("base_vertex")
    _label = kwargs.get("label")
    label = "" if _label is None else _label
    base_vertex = 0 if _base_vertex is None else _base_vertex
    edge_colors = kwargs.get("edge_colors")

    if not (graph_name is None):
        graph_name = f'graph [label={graph_name}];\n'
    else:
        graph_name = ""
    if not (vertex_color is None):
        vertex_color = f'node [color={vertex_color}];\n'
    else:
        vertex_color = ""

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
