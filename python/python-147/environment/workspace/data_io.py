"""
data_io.py
==========
Scientific data input/output utilities adapted from seed project 1197_tec_io.

Provides formatted text output for simulation results, neural network
checkpoints, and convergence metrics.  All functions are designed for
batch scientific computing pipelines without graphical visualization.

Key capabilities:
  - Matrix transpose printing with controlled precision
  - Structured checkpoint serialization for PINN weights
  - TEC-like variable parsing for multi-column scientific data
  - Scientific string formatting with fixed-width alignment
"""

import numpy as np


def r8mat_transpose_print(m, n, a, title="", max_cols=6):
    """
    Print an MxN real matrix in transpose format (column-major display).

    Adapted from seed project 1197_tec_io (r8mat_transpose_print).
    """
    if title:
        print(title)
    for j in range(n):
        row_vals = []
        for i in range(min(m, max_cols)):
            row_vals.append(f"{a[i, j]:12.6g}")
        print(f"  Col {j:3d}: " + " ".join(row_vals))
        if m > max_cols:
            print(f"  ... ({m - max_cols} more rows)")


def checkpoint_save(network, filepath, metadata=None):
    """
    Save PINN network weights and metadata to a structured text file.

    Format:
      # Metadata lines
      LAYER k
      W: rows x cols
      [W matrix entries in row-major order]
      b: length
      [b vector entries]
    """
    with open(filepath, 'w') as f:
        f.write("# PINN Checkpoint\n")
        if metadata:
            for key, val in metadata.items():
                f.write(f"# {key}: {val}\n")
        f.write(f"N_LAYERS: {network.n_layers}\n")
        f.write(f"INPUT_DIM: {network.input_dim}\n")
        f.write(f"OUTPUT_DIM: {network.output_dim}\n")
        f.write(f"ACTIVATION: {network.activation_name}\n")
        for k in range(network.n_layers):
            W = network.weights[k]
            b = network.biases[k]
            f.write(f"LAYER {k}\n")
            f.write(f"W_SHAPE: {W.shape[0]} {W.shape[1]}\n")
            f.write("W:\n")
            np.savetxt(f, W, fmt="%.16e")
            f.write(f"B_LEN: {len(b)}\n")
            f.write("B:\n")
            np.savetxt(f, b.reshape(1, -1), fmt="%.16e")
        f.write("# END\n")


def checkpoint_load(network, filepath):
    """
    Load PINN weights from a structured checkpoint file.
    """
    with open(filepath, 'r') as f:
        lines = f.readlines()

    layer_idx = -1
    mode = None
    W_rows = 0
    W_cols = 0
    W_data = []
    b_data = []

    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("N_LAYERS"):
            continue
        if line.startswith("INPUT_DIM"):
            continue
        if line.startswith("OUTPUT_DIM"):
            continue
        if line.startswith("ACTIVATION"):
            continue
        if line.startswith("LAYER"):
            if layer_idx >= 0:
                # Save previous layer
                W_arr = np.array(W_data).reshape(W_rows, W_cols)
                b_arr = np.array(b_data)
                network.weights[layer_idx] = W_arr
                network.biases[layer_idx] = b_arr
            layer_idx += 1
            W_data = []
            b_data = []
            mode = None
            continue
        if line.startswith("W_SHAPE"):
            parts = line.split(":")[1].strip().split()
            W_rows, W_cols = int(parts[0]), int(parts[1])
            continue
        if line == "W:":
            mode = 'W'
            continue
        if line.startswith("B_LEN"):
            mode = None
            continue
        if line == "B:":
            mode = 'B'
            continue
        if mode == 'W':
            vals = [float(v) for v in line.split()]
            W_data.extend(vals)
        elif mode == 'B':
            vals = [float(v) for v in line.split()]
            b_data.extend(vals)

    if layer_idx >= 0 and W_data and b_data:
        W_arr = np.array(W_data).reshape(W_rows, W_cols)
        b_arr = np.array(b_data)
        network.weights[layer_idx] = W_arr
        network.biases[layer_idx] = b_arr


def parse_variable_line(line, expected_vars=None):
    """
    Parse a space-separated variable line into a dictionary.

    Example:
        "t=0.5 x=3.14 residual=0.001"
    returns {'t': 0.5, 'x': 3.14, 'residual': 0.001}
    """
    result = {}
    tokens = line.strip().split()
    for tok in tokens:
        if '=' in tok:
            key, val = tok.split('=', 1)
            try:
                result[key] = float(val)
            except ValueError:
                result[key] = val
    return result


def write_metrics_log(filepath, metrics_dict, append=True):
    """
    Write training metrics to a log file.

    Parameters
    ----------
    filepath : str
    metrics_dict : dict
        Keys are metric names, values are scalars.
    append : bool
        If True, append; otherwise overwrite.
    """
    mode = 'a' if append else 'w'
    with open(filepath, mode) as f:
        line = ", ".join([f"{k}={v:.8e}" for k, v in metrics_dict.items()])
        f.write(line + "\n")


def print_scientific_summary(title, data_dict):
    """
    Print a formatted scientific summary block.
    """
    width = 60
    print("=" * width)
    print(f"  {title}")
    print("=" * width)
    for key, val in data_dict.items():
        if isinstance(val, float):
            print(f"  {key:30s}: {val:16.8e}")
        elif isinstance(val, int):
            print(f"  {key:30s}: {val:16d}")
        else:
            print(f"  {key:30s}: {str(val):>16s}")
    print("=" * width)
