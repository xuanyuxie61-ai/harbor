
import numpy as np
from io import StringIO


def export_tecplot_2d(filename, x, y, fields, field_names):
    nx, ny = len(x), len(y)
    var_str = ", ".join(f'"{name}"' for name in (["X", "Y"] + field_names))
    lines = []
    lines.append(f'TITLE = "Molecular Field Data"')
    lines.append(f'VARIABLES = {var_str}')
    lines.append(f'ZONE T="Simulation", I={nx}, J={ny}, F=POINT')

    for j in range(ny):
        for i in range(nx):
            vals = [x[i], y[j]]
            for f in fields:
                vals.append(f[i, j])
            lines.append("  ".join(f"{v:.8e}" for v in vals))

    with open(filename, "w") as fh:
        fh.write("\n".join(lines))


def export_xml_state(filename, state_dict):
    lines = []
    lines.append('<?xml version="1.0"?>')
    lines.append('<SimulationState>')
    for key, val in state_dict.items():
        if isinstance(val, (int, float, np.integer, np.floating)):
            lines.append(f'  <Scalar name="{key}" value="{float(val):.12e}"/>')
        elif isinstance(val, (list, tuple, np.ndarray)):
            arr = np.asarray(val).flatten()
            arr_str = " ".join(f"{v:.8e}" for v in arr)
            lines.append(f'  <Vector name="{key}" size="{len(arr)}" data="{arr_str}"/>')
        elif isinstance(val, str):
            lines.append(f'  <String name="{key}" value="{val}"/>')
    lines.append('</SimulationState>')
    with open(filename, "w") as fh:
        fh.write("\n".join(lines))


def read_tecplot_zone(data_string):
    lines = data_string.strip().splitlines()
    var_names = []
    nx, ny = 1, 1
    data_lines = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.upper().startswith("VARIABLES"):

            import re
            var_names = re.findall(r'"([^"]+)"', line)
        elif line.upper().startswith("ZONE"):
            dims = {}
            tokens = line[4:].split(",")
            for tok in tokens:
                if "=" in tok:
                    k, v = tok.split("=", 1)
                    dims[k.strip().upper()] = v.strip().strip('"')
            nx = int(dims.get("I", 1))
            ny = int(dims.get("J", 1))
        elif line.upper().startswith("TITLE"):
            continue
        else:
            data_lines.append(line)

    if not data_lines:
        return np.array([]), nx, ny, var_names

    arr = np.loadtxt(StringIO("\n".join(data_lines)))
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr, nx, ny, var_names


def serialize_sparse_matrix_coo(rows, cols, data, filename):
    with open(filename, "w") as f:
        f.write("%%SparseMatrix coordinate real general\n")
        f.write(f"{max(rows)+1} {max(cols)+1} {len(data)}\n")
        for r, c, d in zip(rows, cols, data):
            f.write(f"{r+1} {c+1} {d:.16e}\n")
