"""
io_utils.py -- Scientific I/O Utilities
=========================================
Provides data I/O functions for 2D point data, face/line connectivity,
and structured text files used throughout the UQ pipeline.

Implements the XY/XYF/XYL family of file formats for geometry data exchange,
plus specialized UQ data serialization.

Seed references:
  - 1420_xy_io: xy_read/write, xyf_read/write, xyl_read/write,
    header/data parsing, comment handling
  - 824_octopus: environment detection (is_octave -> is_python_module)
"""
import numpy as np
import os
from typing import Tuple, Optional


# ---------------------------------------------------------------------------
# String utilities
# ---------------------------------------------------------------------------
def s_len_trim(s: str) -> int:
    """Return the length of string s with trailing whitespace removed."""
    return len(s.rstrip())


def s_word_count(s: str) -> int:
    """Count the number of whitespace-separated words in s."""
    return len(s.split())


# ---------------------------------------------------------------------------
# XY point data I/O (2D coordinates)
# ---------------------------------------------------------------------------
def xy_header_read(filename: str) -> int:
    """
    Read the header of an XY point file to determine the number of points.
    Skips comment lines (starting with '#') and blank lines.
    """
    count = 0
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            count += 1
    return count


def xy_data_read(filename: str) -> np.ndarray:
    """
    Read 2D point coordinates from an XY file.

    Format: each non-comment, non-blank line contains "x y" coordinates.

    Returns ndarray of shape (n_points, 2).
    """
    points = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) >= 2:
                points.append([float(parts[0]), float(parts[1])])
    return np.array(points) if points else np.zeros((0, 2))


def xy_data_write(filename: str, points: np.ndarray, header: str = ""):
    """
    Write 2D point coordinates to an XY file.

    Parameters
    ----------
    filename : str
        Output file path.
    points : ndarray, shape (n, 2)
        Point coordinates.
    header : str
        Optional header comment.
    """
    with open(filename, 'w') as f:
        if header:
            f.write(f"# {header}\n")
        f.write(f"# {len(points)} points\n")
        for i in range(len(points)):
            f.write(f"  {points[i, 0]:.15e}  {points[i, 1]:.15e}\n")


def xy_read(filename: str) -> Tuple[int, np.ndarray]:
    """High-level XY file reader: returns (n_points, points)."""
    points = xy_data_read(filename)
    return len(points), points


def xy_write(filename: str, points: np.ndarray, header: str = ""):
    """High-level XY file writer."""
    xy_data_write(filename, points, header)


# ---------------------------------------------------------------------------
# XYF face data I/O (polygon connectivity)
# ---------------------------------------------------------------------------
def xyf_data_read(filename: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Read face (polygon) data from an XYF file.

    Format:
        # header
        n_points
        x1 y1
        x2 y2
        ...
        ---
        n1 n2 n3 ... -1    (face vertex indices, terminated by -1)
        ...

    Returns
    -------
    points : ndarray, shape (n_points, 2)
    face_pointer : ndarray, shape (n_faces + 1,)
        CSR-style pointer into face_data.
    face_data : ndarray
        Concatenated face vertex indices.
    """
    points = []
    faces = []
    current_face = []
    reading_points = True

    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if line.startswith('---'):
                reading_points = False
                continue
            if reading_points:
                parts = line.split()
                if len(parts) >= 2:
                    points.append([float(parts[0]), float(parts[1])])
            else:
                parts = line.split()
                for p in parts:
                    val = int(p)
                    if val == -1:
                        if current_face:
                            faces.append(current_face)
                        current_face = []
                    else:
                        current_face.append(val)

    # Build CSR arrays
    face_pointer = np.zeros(len(faces) + 1, dtype=int)
    face_data = []
    for i, face in enumerate(faces):
        face_data.extend(face)
        face_pointer[i + 1] = len(face_data)

    points_arr = np.array(points) if points else np.zeros((0, 2))
    face_data_arr = np.array(face_data, dtype=int) if face_data else np.array([], dtype=int)

    return points_arr, face_pointer, face_data_arr


def xyf_data_write(filename: str, points: np.ndarray,
                   face_pointer: np.ndarray, face_data: np.ndarray,
                   header: str = ""):
    """Write face data to an XYF file."""
    n_faces = len(face_pointer) - 1
    with open(filename, 'w') as f:
        if header:
            f.write(f"# {header}\n")
        f.write(f"# {len(points)} points, {n_faces} faces\n")
        for i in range(len(points)):
            f.write(f"  {points[i, 0]:.15e}  {points[i, 1]:.15e}\n")
        f.write("---\n")
        for i in range(n_faces):
            start = face_pointer[i]
            end = face_pointer[i + 1]
            indices = " ".join(str(face_data[j]) for j in range(start, end))
            f.write(f"  {indices} -1\n")


# ---------------------------------------------------------------------------
# XYL line data I/O (polyline connectivity)
# ---------------------------------------------------------------------------
def xyl_data_read(filename: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Read line (polyline) data from an XYL file.
    Similar format to XYF but with line connectivity.
    """
    points = []
    lines = []
    current_line = []
    reading_points = True

    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if line.startswith('---'):
                reading_points = False
                continue
            if reading_points:
                parts = line.split()
                if len(parts) >= 2:
                    points.append([float(parts[0]), float(parts[1])])
            else:
                parts = line.split()
                for p in parts:
                    val = int(p)
                    if val == -1:
                        if current_line:
                            lines.append(current_line)
                        current_line = []
                    else:
                        current_line.append(val)

    line_pointer = np.zeros(len(lines) + 1, dtype=int)
    line_data = []
    for i, line in enumerate(lines):
        line_data.extend(line)
        line_pointer[i + 1] = len(line_data)

    points_arr = np.array(points) if points else np.zeros((0, 2))
    line_data_arr = np.array(line_data, dtype=int) if line_data else np.array([], dtype=int)

    return points_arr, line_pointer, line_data_arr


def xyl_data_write(filename: str, points: np.ndarray,
                   line_pointer: np.ndarray, line_data: np.ndarray,
                   header: str = ""):
    """Write line data to an XYL file."""
    n_lines = len(line_pointer) - 1
    with open(filename, 'w') as f:
        if header:
            f.write(f"# {header}\n")
        f.write(f"# {len(points)} points, {n_lines} lines\n")
        for i in range(len(points)):
            f.write(f"  {points[i, 0]:.15e}  {points[i, 1]:.15e}\n")
        f.write("---\n")
        for i in range(n_lines):
            start = line_pointer[i]
            end = line_pointer[i + 1]
            indices = " ".join(str(line_data[j]) for j in range(start, end))
            f.write(f"  {indices} -1\n")


# ---------------------------------------------------------------------------
# UQ results serialization
# ---------------------------------------------------------------------------
def save_uq_results(filename: str, results: dict):
    """
    Save UQ results to a structured text file.

    Parameters
    ----------
    filename : str
        Output file path.
    results : dict
        Dictionary of named arrays/values to save.
    """
    with open(filename, 'w') as f:
        f.write("# UQ Results Summary\n")
        f.write(f"# Keys: {list(results.keys())}\n")
        for key, val in results.items():
            if isinstance(val, np.ndarray):
                f.write(f"# {key}: ndarray shape={val.shape} dtype={val.dtype}\n")
                if val.ndim == 1:
                    for v in val[:20]:  # Limit output
                        f.write(f"  {v:.15e}\n")
                    if len(val) > 20:
                        f.write(f"  ... ({len(val) - 20} more values)\n")
                elif val.ndim == 2:
                    for row in val[:10]:
                        f.write("  " + " ".join(f"{v:.10e}" for v in row) + "\n")
                    if val.shape[0] > 10:
                        f.write(f"  ... ({val.shape[0] - 10} more rows)\n")
            elif isinstance(val, (int, float)):
                f.write(f"# {key}: {val}\n")
            elif isinstance(val, str):
                f.write(f"# {key}: {val}\n")
            else:
                f.write(f"# {key}: {type(val).__name__}\n")


# ---------------------------------------------------------------------------
# Environment detection
# ---------------------------------------------------------------------------
def is_octave() -> bool:
    """
    Detect if running in GNU Octave (always False in pure Python).
    Reference: is_octave from seed project 824.
    """
    return False


def get_runtime_info() -> dict:
    """Return runtime environment information."""
    import sys
    return {
        'language': 'Python',
        'version': sys.version,
        'numpy_version': np.__version__,
        'is_octave': is_octave()
    }
