def matrix_data_to_gviz(shape, rows, cols, **kwargs):
    if len(rows) != len(cols):
        raise Exception("Rows and cols arrays must have equal size")
    result = "digraph {\n"
    graph_name = kwargs.get("graph_name")
    vertex_color = kwargs.get("vertex_color")
    edge_color = kwargs.get("edge_color")
    _base_vertex = kwargs.get("base_vertex")
    _label = kwargs.get("label")
    label = "" if _label is None else _label
    base_vertex = 0 if _base_vertex is None else _base_vertex

    n = max(shape[0], shape[1])

    if not (graph_name is None):
        result += f'graph [label={graph_name}];\n'
    if not (vertex_color is None):
        result += f'node [color={vertex_color}];\n'
    if not (vertex_color is None):
        result += f'edge [color={edge_color}];\n'

    used_vertex = [False] * n
    for i in range(len(rows)):
        result += f"{rows[i] + base_vertex} -> {cols[i] + base_vertex} [label={label}];\n"
        used_vertex[rows[i]] = True
        used_vertex[cols[i]] = True
    for i in range(n):
        if not used_vertex[i]:
            result += f"{i + base_vertex};\n"
    result += "}\n"
    return result
