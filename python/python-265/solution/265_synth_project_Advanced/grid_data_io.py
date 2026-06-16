# -*- coding: utf-8 -*-
"""
grid_data_io.py
---------------
Input / output utilities for the finite-difference cosmic-ray
transport data.  Provides

    * write_fd_data        -- dump FD node coordinates and values to
      text files;
    * read_fd_data         -- read them back;
    * fd_to_structured     -- convert unstructured FD output (in
      (r, mu) or (r, theta, phi)) to a structured Tecplot-like ASCII
      file;
    * summary_diagnostics  -- print a compact numerical report.
"""
from __future__ import annotations
import os
import math
import numpy as np
from typing import Dict, Optional, Tuple

import cosmic_ray_physics as crp


# =====================================================================
# write / read FD data
# =====================================================================
def write_fd_data(prefix: str, r: np.ndarray, mu: np.ndarray,
                  f: np.ndarray, t: float = 0.0) -> Tuple[str, str]:
    """Write  prefix_nodes.txt  and  prefix_values.txt  in the style
    of the Fortran FD model.

    Returns the two filenames.
    """
    node_path = prefix + "_nodes.txt"
    val_path = prefix + "_values.txt"
    Nr = r.size
    Nmu = mu.size
    with open(node_path, "w") as fh:
        fh.write(f"# nodes: Nr={Nr} Nmu={Nmu} t={t:.6e}\n")
        fh.write("#  i  j  r[m]  mu\n")
        for i, ri in enumerate(r):
            for j, muj in enumerate(mu):
                fh.write(f"{i:5d} {j:5d} {ri:18.10e} {muj:18.10e}\n")
    with open(val_path, "w") as fh:
        fh.write(f"# values: Nr={Nr} Nmu={Nmu} t={t:.6e}\n")
        fh.write("#  i  j  f[1/(m^3 sr GV)]\n")
        for i in range(Nr):
            for j in range(Nmu):
                fh.write(f"{i:5d} {j:5d} {f[i, j]:18.10e}\n")
    return node_path, val_path


def read_fd_data(prefix: str) -> Dict[str, np.ndarray]:
    """Read back the files written by  write_fd_data."""
    node_path = prefix + "_nodes.txt"
    val_path = prefix + "_values.txt"
    nodes = np.loadtxt(node_path)
    vals = np.loadtxt(val_path)
    return {"nodes": nodes, "values": vals}


# =====================================================================
# FD -> structured (Tecplot-style ASCII)
# =====================================================================
def fd_to_structured(prefix_in: str, prefix_out: str,
                     variable_names: Optional[list] = None) -> str:
    """Convert the pair  prefix_nodes.txt / prefix_values.txt  to a
    single structured Tecplot-like ASCII file  prefix_out.dat.

    This mirrors the fd_to_tec.m utility (Burkardt 2010) but writes a
    pure ASCII output that can be read by any text editor.
    """
    data = read_fd_data(prefix_in)
    nodes = data["nodes"]
    vals = data["values"]
    Nr = int(nodes[:, 0].max()) + 1
    Nmu = int(nodes[:, 1].max()) + 1
    r = np.unique(nodes[:, 2])
    mu = np.unique(nodes[:, 3])
    f = np.zeros((Nr, Nmu))
    for k in range(nodes.shape[0]):
        i = int(nodes[k, 0])
        j = int(nodes[k, 1])
        f[i, j] = vals[k, 2]
    path = prefix_out + ".dat"
    with open(path, "w") as fh:
        fh.write('TITLE = "Cosmic-Ray Transport FD Solution"\n')
        vars = variable_names or ["r", "mu", "f"]
        fh.write('VARIABLES = ' + ', '.join(f'"{v}"' for v in vars) + '\n')
        fh.write(f'ZONE T="time-slice", I={Nr}, J={Nmu}, '
                 f'F=POINT\n')
        for i in range(Nr):
            for j in range(Nmu):
                fh.write(f"{r[i]:18.10e} {mu[j]:18.10e} {f[i, j]:18.10e}\n")
    return path


# =====================================================================
# Summary diagnostics (printed, no plotting)
# =====================================================================
def summary_diagnostics(r: np.ndarray, mu: np.ndarray,
                        f: np.ndarray, Ek_GeV: float) -> Dict[str, float]:
    """Compute and print a compact set of diagnostics.

    * omnidirectional flux at each r;
    * anisotropy  A(r) = 3 <mu f> / <f>;
    * first angular moment (streaming flux);
    * pressure integral  P_cr ~ int f p^3 dp  (mono-energetic form).
    """
    dmu = np.diff(mu)
    dmu_full = np.concatenate([[dmu[0]], 0.5 * (dmu[:-1] + dmu[1:]),
                               [dmu[-1]]])
    omni = np.sum(f * dmu_full, axis=1)
    mu_f = np.sum(mu * f * dmu_full, axis=1)
    mu2_f = np.sum(mu * mu * f * dmu_full, axis=1)
    aniso = np.where(omni > 0.0, 3.0 * mu_f / omni, 0.0)
    stream = 4.0 * math.pi * mu_f
    Ek = Ek_GeV * 1.0e9 * crp.q_e
    v = crp.velocity_from_Ek(Ek)
    p = crp.m_p * crp.lorentz_factor(Ek) * v
    P_cr = (4.0 * math.pi / 3.0) * p * v * omni   # J / m^3
    out = {
        "r_min_AU": float(r[0] / crp.AU),
        "r_max_AU": float(r[-1] / crp.AU),
        "f_min": float(f.min()),
        "f_max": float(f.max()),
        "omni_max": float(omni.max()),
        "aniso_max": float(np.max(np.abs(aniso))),
        "stream_max": float(np.max(np.abs(stream))),
        "P_cr_max": float(P_cr.max()),
    }
    print(f"[grid_data_io] diagnostics at Ek = {Ek_GeV:.2f} GeV:")
    for k, v_ in out.items():
        print(f"    {k:16s} = {v_:.4e}")
    return out


# =====================================================================
# Demo
# =====================================================================
def _demo() -> None:
    import heliocentric_mesh as hm
    r = hm.logarithmic_radial_mesh(Nr=16)
    mu = hm.pitch_angle_mesh(Nmu=8)
    f = np.outer(np.exp(-(r - crp.AU) ** 2 / (0.5 * crp.AU) ** 2),
                 1.0 - mu * mu)
    node_p, val_p = write_fd_data("/tmp/cr_fd_demo", r, mu, f)
    out_p = fd_to_structured("/tmp/cr_fd_demo", "/tmp/cr_fd_structured",
                             variable_names=["r", "mu", "f"])
    print(f"[grid_data_io] wrote {out_p}")
    d = read_fd_data("/tmp/cr_fd_demo")
    print(f"[grid_data_io] read {d['values'].shape[0]} value rows")
    summary_diagnostics(r, mu, f, 1.0)


if __name__ == "__main__":
    _demo()
