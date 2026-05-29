"""
config_parser.py
---------------
Configuration and string parsing utilities.
Adapted from seed project 431_filum (file/string I/O utilities).

Role in synthesis:
  Parse Hamiltonian parameter files and convert string representations
  of complex matrices into NumPy arrays. Handles robust boundary checks
  for malformed inputs.
"""

import os
import re
import numpy as np


def file_char_count(filepath):
    """
    Count the number of characters in a text file.
    Returns -1 if the file cannot be opened.
    """
    if not os.path.exists(filepath):
        return -1
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        return len(content)
    except Exception:
        return -1


def string_to_float_vector(s, expected_n):
    """
    Parse a string into a vector of floats.
    Adapted from s_to_r8vec.

    Parameters
    ----------
    s : str
        Input string containing whitespace-separated numbers.
    expected_n : int
        Expected number of values.

    Returns
    -------
    vec : ndarray, shape (expected_n,)
        Parsed float vector.

    Raises
    ------
    ValueError
        If the number of parsed values does not match expected_n.
    """
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
    """
    Parse a complex matrix from a list of strings.
    Each line should contain either real numbers or 'a+bj' complex literals.

    Parameters
    ----------
    lines : list of str
        Lines representing the matrix row-wise.
    n_rows : int
        Expected row count.
    n_cols : int
        Expected column count.

    Returns
    -------
    M : ndarray, shape (n_rows, n_cols), dtype=complex
    """
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
            # Accept Python complex literal like '1+2j' or '1-2j'
            try:
                M[i, j] = complex(token)
            except ValueError as exc:
                raise ValueError(
                    f"Cannot parse complex token '{token}' at ({i},{j})."
                ) from exc
    return M


def read_parameter_config(filepath):
    """
    Read a structured parameter configuration file for the non-Hermitian
    simulation. The file format is:

        # Comments start with #
        N_SITES 4
        GAMMA 0.5
        Kappa 1.0
        ...

    Parameters
    ----------
    filepath : str

    Returns
    -------
    params : dict
        Dictionary of parameter name -> float or int.
    """
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
            # Try int, then float, then keep string
            try:
                params[key] = int(val)
            except ValueError:
                try:
                    params[key] = float(val)
                except ValueError:
                    params[key] = val
    return params
