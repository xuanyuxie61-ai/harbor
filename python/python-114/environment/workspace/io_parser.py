
import numpy as np
from typing import Tuple, Optional, List, Dict
import re


def parse_tec_file(content: str) -> Dict:
    lines = [line.strip() for line in content.splitlines()]
    lines = [line for line in lines if line]

    idx = 0


    if lines[idx].upper().startswith("TITLE"):
        idx += 1


    if not lines[idx].upper().startswith("VARIABLES"):
        raise ValueError("VARIABLES line not found")

    var_line = lines[idx]
    idx += 1


    var_part = var_line.split('=', 1)[1] if '=' in var_line else var_line

    raw_vars = re.findall(r'"([^"]*)"', var_part)
    if not raw_vars:
        raw_vars = var_part.replace(',', ' ').split()

    variable_names = [v.strip() for v in raw_vars if v.strip()]
    n_vars = len(variable_names)


    coord_vars = {'X', 'Y', 'Z'}
    dim_num = sum(1 for v in variable_names if v.upper() in coord_vars)
    n_data_per_node = n_vars - dim_num


    if idx >= len(lines) or not lines[idx].upper().startswith("ZONE"):
        raise ValueError("ZONE line not found")

    zone_line = lines[idx]
    idx += 1


    n_match = re.search(r'N\s*=\s*(\d+)', zone_line, re.IGNORECASE)
    e_match = re.search(r'E\s*=\s*(\d+)', zone_line, re.IGNORECASE)
    type_match = re.search(r'ZONETYPE\s*=\s*(\w+)', zone_line, re.IGNORECASE)

    if not n_match or not e_match:
        raise ValueError("ZONE line must contain N= and E=")

    node_num = int(n_match.group(1))
    element_num = int(e_match.group(1))

    zone_type = type_match.group(1).upper() if type_match else "FETETRAHEDRON"
    element_order_map = {
        "FETRIANGLE": 3,
        "FEQUADRILATERAL": 4,
        "FETETRAHEDRON": 4,
        "FEBRICK": 8,
    }
    element_order = element_order_map.get(zone_type, 4)


    node_coord = np.zeros((dim_num, node_num), dtype=np.float64)
    node_data = np.zeros((n_data_per_node, node_num), dtype=np.float64)

    for node in range(node_num):
        if idx >= len(lines):
            raise ValueError("unexpected end of file while reading nodes")
        tokens = lines[idx].split()
        idx += 1
        vals = list(map(float, tokens))
        if len(vals) < dim_num + n_data_per_node:
            raise ValueError(f"insufficient values on node line {node+1}")
        node_coord[:, node] = vals[:dim_num]
        if n_data_per_node > 0:
            node_data[:, node] = vals[dim_num:dim_num + n_data_per_node]


    element_node = np.zeros((element_order, element_num), dtype=np.int64)
    for elem in range(element_num):
        if idx >= len(lines):
            raise ValueError("unexpected end of file while reading elements")
        tokens = lines[idx].split()
        idx += 1
        vals = list(map(int, tokens))
        if len(vals) < element_order:
            raise ValueError(f"insufficient values on element line {elem+1}")
        element_node[:, elem] = vals[:element_order]

    return {
        "dim_num": dim_num,
        "node_num": node_num,
        "element_num": element_num,
        "element_order": element_order,
        "node_coord": node_coord,
        "element_node": element_node,
        "node_data": node_data,
        "variable_names": variable_names,
    }


def build_tec_file(
    node_coord: np.ndarray,
    element_node: np.ndarray,
    node_data: Optional[np.ndarray] = None,
    variable_names: Optional[List[str]] = None,
    title: str = "DNA_Repair_Focus",
) -> str:
    dim_num, node_num = node_coord.shape
    element_order, element_num = element_node.shape

    if variable_names is None:
        coord_names = ['X', 'Y', 'Z'][:dim_num]
        data_names = [f"Var{i+1}" for i in range(node_data.shape[0] if node_data is not None else 0)]
        variable_names = coord_names + data_names

    lines = []
    lines.append(f'TITLE = "{title}"')
    var_str = ', '.join([f'"{v}"' for v in variable_names])
    lines.append(f'VARIABLES = {var_str}')

    zone_type = "FETETRAHEDRON" if element_order == 4 else "FETRIANGLE"
    lines.append(f'ZONE N={node_num}, E={element_num}, ZONETYPE={zone_type}')

    n_data = node_data.shape[0] if node_data is not None else 0
    for i in range(node_num):
        parts = [f"{node_coord[d, i]:.8e}" for d in range(dim_num)]
        if n_data > 0:
            parts += [f"{node_data[d, i]:.8e}" for d in range(n_data)]
        lines.append('  '.join(parts))

    for e in range(element_num):
        parts = [str(int(element_node[d, e] + 1)) for d in range(element_order)]
        lines.append('  '.join(parts))

    return "\n".join(lines)


def grid_double_resolution(
    values: np.ndarray,
    mode: str = "2d",
) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)

    if mode == "2d":
        m, n = values.shape
        doubled = np.zeros((2 * m, 2 * n), dtype=np.float64)
        for i in range(m):
            for j in range(n):
                doubled[2 * i:2 * i + 2, 2 * j:2 * j + 2] = values[i, j]

        doubled /= 4.0
    elif mode == "3d":
        m, n, p = values.shape
        doubled = np.zeros((2 * m, 2 * n, 2 * p), dtype=np.float64)
        for i in range(m):
            for j in range(n):
                for k in range(p):
                    doubled[2 * i:2 * i + 2, 2 * j:2 * j + 2, 2 * k:2 * k + 2] = values[i, j, k]
        doubled /= 8.0
    else:
        raise ValueError("mode must be '2d' or '3d'")

    return doubled


def adaptive_mesh_refinement_2d(
    field: np.ndarray,
    gradient_threshold: float = 0.05,
    max_level: int = 2,
) -> List[np.ndarray]:
    field = np.asarray(field, dtype=np.float64)
    levels = [field.copy()]

    current = field
    for level in range(max_level):

        grad_x = np.zeros_like(current)
        grad_y = np.zeros_like(current)
        grad_x[1:-1, :] = (current[2:, :] - current[:-2, :]) / 2.0
        grad_y[:, 1:-1] = (current[:, 2:] - current[:, :-2]) / 2.0
        grad_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)


        if np.max(grad_mag) < gradient_threshold:
            break

        current = grid_double_resolution(current, mode="2d")
        levels.append(current)

    return levels
