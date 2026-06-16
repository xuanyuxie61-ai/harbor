"""
io_utils.py
===========
ASCII data I/O and snapshot management for the shearing-box MHD
simulation.

The I/O conventions follow the fd_to_tec tool (351_fd_to_tec):

    * node files carry the cell-centre coordinates (one per line)
    * value files carry one field per line (rho, p, alpha, etc.)
    * element connectivity is replaced by the structured-grid indices
      (structured grids need no explicit connectivity; we write the
      shape (Nx, Ny, Nz) in a metadata header instead)

No TECPLOT output is produced because the project rules forbid
visualisation artefacts.  The files are kept as plain ASCII so they
can be inspected and post-processed with any tool.

All output lives under a sub-directory ``output/`` created at the
project root.
"""

from __future__ import annotations
import os
import json
import math
import numpy as np
from typing import Dict, List

import boundary_conditions as bc
import grid_manager as gm
import mhd_equations as mhd
import mri_diagnostics as diag


# ---------------------------------------------------------------------------
#                    Directory / file management
# ---------------------------------------------------------------------------
def ensure_output_dir(base_dir: str) -> str:
    """Create and return the output directory path."""
    out = os.path.join(base_dir, "output")
    os.makedirs(out, exist_ok=True)
    return out


def write_grid_metadata(g, out_dir: str) -> str:
    """Write the grid metadata as JSON.

    The file mirrors the header conventions of fd_to_tec but uses
    JSON for machine-readability.
    """
    path = os.path.join(out_dir, "grid_metadata.json")
    meta = {
        "Nx": g.Nx, "Ny": g.Ny, "Nz": g.Nz,
        "Lx": g.Lx, "Ly": g.Ly, "Lz": g.Lz,
        "cells_per_mri": g.cells_per_mri,
        "refinement_level": g.refinement_level,
        "volume":        g.volume,
        "aspect_ratio":  gm.aspect_ratio(g),
        "min_cell_vol":  gm.min_cell_volume(g),
    }
    with open(path, "w") as fh:
        json.dump(meta, fh, indent=2)
    return path


def write_node_file(g, out_dir: str) -> str:
    """Write cell-centre coordinates to 'nodes.txt' (from 351)."""
    path = os.path.join(out_dir, "nodes.txt")
    X, Y, Z = np.meshgrid(g.xc, g.yc, g.zc, indexing="ij")
    data = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    header = f"# cell centres  (Nx*Ny*Nz = {g.Nx*g.Ny*g.Nz} rows)"
    np.savetxt(path, data, header=header, fmt="%14.6e")
    return path


def write_values_file(values: Dict[str, np.ndarray], out_dir: str,
                      tag: str = "snapshot") -> str:
    """Write a set of named scalar fields to 'values_<tag>.txt'."""
    path = os.path.join(out_dir, f"values_{tag}.txt")
    keys = sorted(values.keys())
    data = np.column_stack([values[k].ravel() for k in keys])
    header = "  ".join(keys)
    np.savetxt(path, data, header=header, fmt="%14.6e")
    return path


# ---------------------------------------------------------------------------
#                     Snapshot extraction
# ---------------------------------------------------------------------------
def snapshot_scalar_fields(U: np.ndarray, g) -> Dict[str, np.ndarray]:
    """Extract a dictionary of scalar fields from the conserved array
    for snapshot output."""
    ng = 2
    U_int = U[:, ng:-ng, ng:-ng, ng:-ng]
    rho = U_int[bc.ConsIdx.rho]
    p   = mhd.pressure(U_int)
    B2  = mhd.total_B2(U_int)
    vx, vy, vz = mhd.velocity(U_int)
    v2 = vx**2 + vy**2 + vz**2
    beta = pc.plasma_beta(p, B2) if False else \
           2.0 * p / (B2 / (8.0 * math.pi) + 1.0e-30)
    # Local Maxwell stress
    Bx = U_int[bc.ConsIdx.Bx]
    By = U_int[bc.ConsIdx.By]
    M_xy = - Bx * By / (4.0 * math.pi)
    return {
        "rho":   rho,
        "p":     p,
        "v2":    v2,
        "B2":    B2,
        "beta":  beta,
        "M_xy":  M_xy,
    }


# ---------------------------------------------------------------------------
#                     History file (time-series)
# ---------------------------------------------------------------------------
class HistoryWriter:
    """Append-only writer for the scalar time series."""
    def __init__(self, out_dir: str, keys: List[str]):
        self.path = os.path.join(out_dir, "history.txt")
        self.keys = keys
        with open(self.path, "w") as fh:
            fh.write("# " + "  ".join(keys) + "\n")

    def append(self, record: Dict[str, float]) -> None:
        with open(self.path, "a") as fh:
            fh.write("  ".join(f"{record[k]:+14.6e}" for k in self.keys) + "\n")


# ---------------------------------------------------------------------------
#                   Summary report writer
# ---------------------------------------------------------------------------
def write_summary(out_dir: str, lines: List[str]) -> str:
    """Write the textual summary produced by main.py."""
    path = os.path.join(out_dir, "summary.txt")
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    return path


# ---------------------------------------------------------------------------
#                    Import helpers (re-export)
# ---------------------------------------------------------------------------
import physical_constants as pc  # noqa: E402  (kept for snapshot use)
