
import os
import re
import numpy as np


def file_char_count(filepath):
    if not os.path.exists(filepath):
        return -1
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        return len(content)
    except Exception:
        return -1


def string_to_float_vector(s, expected_n):
    parts = s.strip().split()
    if len(parts) < expected_n:
        raise ValueError(
            f"Expected {expected_n} numbers, found {len(parts)} in string."
        )
    vec = np.zeros(expected_n, dtype=float)
    for i in range(expected_n):
        try:
            vec[i] = float(parts[i])
        except ValueError as exc:
            raise ValueError(f"Cannot convert token '{parts[i]}' to float.") from exc
    return vec


def parse_complex_matrix(lines, n_rows, n_cols):
    if len(lines) < n_rows:
        raise ValueError(
            f"Expected {n_rows} rows, got {len(lines)}."
        )
    M = np.zeros((n_rows, n_cols), dtype=complex)
    for i in range(n_rows):
        tokens = lines[i].strip().split()
        if len(tokens) < n_cols:
            raise ValueError(
                f"Row {i}: expected {n_cols} columns, got {len(tokens)}."
            )
        for j in range(n_cols):
            token = tokens[j]

            try:
                M[i, j] = complex(token)
            except ValueError as exc:
                raise ValueError(
                    f"Cannot parse complex token '{token}' at ({i},{j})."
                ) from exc
    return M


def read_parameter_config(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Config file not found: {filepath}")
    params = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        for line_num, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line or line.startswith('#'):
                continue
            if '=' in line:
                key, val = line.split('=', 1)
            else:
                parts = line.split(None, 1)
                if len(parts) != 2:
                    continue
                key, val = parts
            key = key.strip()
            val = val.strip()

            try:
                params[key] = int(val)
            except ValueError:
                try:
                    params[key] = float(val)
                except ValueError:
                    params[key] = val
    return params
