
import os
import numpy as np


def file_column_count(filename):
    if not os.path.exists(filename):
        return -1

    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            return len(parts)

    return -1


def file_row_count(filename):
    if not os.path.exists(filename):
        return 0

    row_num = 0
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            row_num += 1

    return row_num


def read_matrix_file(filename, dtype=np.float64):
    n_cols = file_column_count(filename)
    n_rows = file_row_count(filename)

    if n_cols <= 0 or n_rows <= 0:
        return np.array([])

    data = np.zeros((n_rows, n_cols), dtype=dtype)

    with open(filename, 'r') as f:
        i = 0
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) == n_cols:
                data[i, :] = [dtype(x) for x in parts]
                i += 1
                if i >= n_rows:
                    break

    return data


def write_matrix_file(filename, data, fmt='%.6e', header=None):
    data = np.asarray(data)

    with open(filename, 'w') as f:
        if header is not None:
            f.write(f"# {header}\n")
        if data.ndim == 1:
            for val in data:
                f.write(fmt % val + "\n")
        else:
            for row in data:
                f.write(" ".join(fmt % x for x in row) + "\n")


def increment_indices(indices, increment=1):
    arr = np.asarray(indices, dtype=np.int64)
    return arr + increment


def convert_index_base(indices, from_base, to_base):
    if from_base not in (0, 1) or to_base not in (0, 1):
        raise ValueError("base must be 0 or 1")
    diff = to_base - from_base
    return np.asarray(indices, dtype=np.int64) + diff


def save_oscillation_results(results, filename_prefix):
    for key, value in results.items():
        if isinstance(value, np.ndarray):
            fname = f"{filename_prefix}_{key}.txt"
            write_matrix_file(fname, value, header=key)
        elif isinstance(value, (int, float, complex)):
            fname = f"{filename_prefix}_{key}.txt"
            with open(fname, 'w') as f:
                f.write(f"# {key}\n")
                if isinstance(value, complex):
                    f.write(f"{value.real:.10e} {value.imag:.10e}\n")
                else:
                    f.write(f"{value:.10e}\n")


def load_density_profile(filename):
    data = read_matrix_file(filename)
    if len(data) == 0:
        return np.array([]), np.array([])
    if data.shape[1] < 2:
        raise ValueError("Density profile file must have at least 2 columns")
    return data[:, 0], data[:, 1]
