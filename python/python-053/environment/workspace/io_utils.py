
import numpy as np
from typing import Tuple, List, Dict, Optional
import hashlib


def compute_checksum(data: np.ndarray) -> int:
    byte_data = data.tobytes()
    hash_val = int(hashlib.md5(byte_data).hexdigest(), 16)
    return hash_val % (2 ** 32)


def write_mesh_data(filename: str,
                    title: str,
                    variable_names: List[str],
                    node_coords: np.ndarray,
                    node_data: np.ndarray,
                    element_nodes: Optional[np.ndarray] = None,
                    element_type: str = "FETRIANGLE") -> None:
    dim_num = node_coords.shape[0]
    n_nodes = node_coords.shape[1]
    n_data_vars = node_data.shape[0]

    if node_data.shape[1] != n_nodes:
        raise ValueError("node_data and node_coords must have same number of nodes")

    with open(filename, 'w') as f:

        f.write(f'TITLE = "{title}"\n')


        f.write('VARIABLES = ')
        vars_all = list(variable_names)
        f.write(', '.join([f'"{v}"' for v in vars_all]))
        f.write('\n')


        n_elements = 0 if element_nodes is None else element_nodes.shape[1]
        element_order = 0 if element_nodes is None else element_nodes.shape[0]
        f.write(f'ZONE N={n_nodes}, E={n_elements}, DATAPACKING=POINT, ZONETYPE={element_type}\n')


        for i in range(n_nodes):
            line_parts = []
            for d in range(dim_num):
                line_parts.append(f"{node_coords[d, i]:.8e}")
            for v in range(n_data_vars):
                line_parts.append(f"{node_data[v, i]:.8e}")
            f.write(' '.join(line_parts) + '\n')


        if element_nodes is not None:
            for e in range(n_elements):
                line = ' '.join([str(int(element_nodes[o, e]) + 1)
                                 for o in range(element_order)])
                f.write(line + '\n')


def read_mesh_data(filename: str) -> Dict:
    with open(filename, 'r') as f:
        lines = [line.strip() for line in f.readlines()]


    title = ""
    variable_names = []
    zone_info = {}
    data_start_idx = 0

    for idx, line in enumerate(lines):
        if line.startswith('TITLE'):
            title = line.split('=', 1)[1].strip().strip('"')
        elif line.startswith('VARIABLES'):
            var_part = line.split('=', 1)[1].strip()

            import re
            variable_names = re.findall(r'"([^"]*)"', var_part)
        elif line.startswith('ZONE'):

            parts = line.replace('ZONE', '').strip().split(',')
            for part in parts:
                if '=' in part:
                    k, v = part.strip().split('=', 1)
                    zone_info[k.strip()] = v.strip()
            data_start_idx = idx + 1
            break

    dim_num = 2
    if 'ZONETYPE' in zone_info:
        if zone_info['ZONETYPE'] == 'FETRIANGLE':
            element_order = 3
        elif zone_info['ZONETYPE'] == 'FEQUADRILATERAL':
            element_order = 4
        elif zone_info['ZONETYPE'] == 'FETETRAHEDRON':
            element_order = 4
            dim_num = 3
        elif zone_info['ZONETYPE'] == 'FEBRICK':
            element_order = 8
            dim_num = 3
        else:
            element_order = 3
    else:
        element_order = 3

    n_nodes = int(zone_info.get('N', 0))
    n_elements = int(zone_info.get('E', 0))


    n_total_vars = len(variable_names)
    n_data_vars = n_total_vars - dim_num


    node_coords = np.zeros((dim_num, n_nodes), dtype=float)
    node_data = np.zeros((n_data_vars, n_nodes), dtype=float)

    for i in range(n_nodes):
        line_idx = data_start_idx + i
        if line_idx >= len(lines):
            break
        values = lines[line_idx].split()
        if len(values) < n_total_vars:
            continue
        for d in range(dim_num):
            node_coords[d, i] = float(values[d])
        for v in range(n_data_vars):
            node_data[v, i] = float(values[dim_num + v])


    element_nodes = None
    if n_elements > 0:
        element_nodes = np.zeros((element_order, n_elements), dtype=int)
        elem_start = data_start_idx + n_nodes
        for e in range(n_elements):
            line_idx = elem_start + e
            if line_idx >= len(lines):
                break
            vals = lines[line_idx].split()
            if len(vals) < element_order:
                continue
            for o in range(element_order):
                element_nodes[o, e] = int(vals[o]) - 1

    return {
        "title": title,
        "variable_names": variable_names,
        "node_coords": node_coords,
        "node_data": node_data,
        "element_nodes": element_nodes,
        "n_nodes": n_nodes,
        "n_elements": n_elements,
        "element_order": element_order,
    }


def write_sst_timeseries(filename: str,
                         times: np.ndarray,
                         nino34: np.ndarray,
                         metadata: Optional[Dict] = None) -> None:
    with open(filename, 'w') as f:
        if metadata:
            for k, v in metadata.items():
                f.write(f"# {k}={v}\n")
        f.write("time_months nino34_index\n")
        for t, n in zip(times, nino34):
            f.write(f"{t:.4f} {n:.6f}\n")


def read_sst_timeseries(filename: str) -> Tuple[np.ndarray, np.ndarray, Dict]:
    metadata = {}
    times = []
    nino = []

    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('#'):
                kv = line[1:].strip().split('=', 1)
                if len(kv) == 2:
                    metadata[kv[0].strip()] = kv[1].strip()
            elif line.startswith('time'):
                continue
            else:
                parts = line.split()
                if len(parts) >= 2:
                    times.append(float(parts[0]))
                    nino.append(float(parts[1]))

    return np.array(times), np.array(nino), metadata
