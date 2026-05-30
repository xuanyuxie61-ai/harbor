# -*- coding: utf-8 -*-

import numpy as np
import json


def r8vec2_write(filename, x, y, fmt="%.8e"):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = min(x.size, y.size)
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(f"# Paired vectors, n = {n}\n")
        for i in range(n):
            f.write(f"{fmt}  {fmt}\n" % (x[i], y[i]))


def i4vec_transpose_print(vec, title="", elems_per_line=10):
    vec = np.asarray(vec, dtype=int)
    n = vec.size
    if title:
        print(title)
    for ilo in range(0, n, elems_per_line):
        ihi = min(ilo + elems_per_line, n)
        line = " ".join(f"{v:6d}" for v in vec[ilo:ihi])
        print(line)


def index_set_to_string(index_set, name="I"):
    indices = sorted(set(int(i) for i in index_set))
    if not indices:
        return f"{name} = ∅"
    return f"{name} = {{{', '.join(str(i) for i in indices)}}}"


def parse_solution_vector(text_data, var_prefix="x", clean_tol=1e-6):
    values = {}
    prefix_len = len(var_prefix)
    for line in text_data.strip().split('\n'):
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        parts = line.split('=')
        if len(parts) != 2:
            continue
        name = parts[0].strip()
        val_str = parts[1].strip()
        if not name.startswith(var_prefix):
            continue
        try:
            idx = int(name[prefix_len:])
            val = float(val_str)
            if abs(val - round(val)) < clean_tol:
                val = float(round(val))
            values[idx] = val
        except ValueError:
            continue
    return values


def write_simulation_results(filename, data_dict, metadata=None):
    output = {}
    if metadata:
        output['metadata'] = metadata
    output['data'] = {}
    for key, val in data_dict.items():
        if isinstance(val, np.ndarray):
            output['data'][key] = {
                'shape': list(val.shape),
                'values': val.tolist()
            }
        elif isinstance(val, (list, tuple)):
            output['data'][key] = list(val)
        else:
            output['data'][key] = float(val) if isinstance(val, (int, float, np.number)) else str(val)

    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)


def read_simulation_results(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        content = json.load(f)
    metadata = content.get('metadata', {})
    data = content.get('data', {})

    for key, val in data.items():
        if isinstance(val, dict) and 'shape' in val and 'values' in val:
            data[key] = np.array(val['values'], dtype=float).reshape(val['shape'])
    return metadata, data


def format_moment_vector(moments, names=None):
    moments = np.asarray(moments, dtype=float)
    if names is None:
        names = [f"μ_{i}" for i in range(len(moments))]
    lines = []
    for name, val in zip(names, moments):
        lines.append(f"  {name:8s} = {val:14.6e}")
    return "\n".join(lines)
