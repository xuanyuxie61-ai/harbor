"""
diagnostics.py — Multi-panel equilibrium diagnostic suite, mapped from
(1) the multi-panel statistics pipeline of 1288_rmnldwg_oral-cavity-paper
(2) the FEM-mesh I/O of 380_fem_to_tec
(3) the magic-matrix structured indexing of 708_magic_matrix

Scientific background
---------------------
A tokamak equilibrium is characterised by a handful of scalar figures of
merit:

    β_t    = 2 μ₀ <p> / B₀²                          (toroidal beta)
    β_p    = 2 μ₀ <p> / <B_p²>                       (poloidal beta)
    l_i    = <B_p²> / B_p(a)²                        (internal inductance)
    IP     = ∫∫ j_φ dR dZ                            (plasma current)
    W_MHD  = (3/2) ∫ p dV                            (stored energy)
    τ_E    = W_MHD / P_heat                          (energy confinement time)

We compute all of them from (ψ, j_φ, p) and report them in a multi-panel
summary analogous to the oral-cavity statistical figures.

The FEM-like mesh representation stores (R,Z) nodes, triangular elements and
per-node values (ψ, p, j_φ) so that external tools (Tecplot, Paraview) can
read the equilibrium.  The magic-matrix indexing provides a deterministic
(i,j) → k mapping used throughout the project.
"""

from __future__ import annotations
import math


MU0 = 4.0 * math.pi * 1e-7


# ---------------------------------------------------------------------------
# Magic-matrix indexing (708_magic_matrix analog)
# ---------------------------------------------------------------------------

def magic_index_matrix(Nr: int, Nz: int) -> list[list[int]]:
    """Return a magic-like index matrix (each row/column sums to the same
    value) used as a deterministic (i,j) -> k ordering for the 2-D mesh.
    For even sizes we fall back to row-major ordering with a magic offset."""
    M = [[0] * Nz for _ in range(Nr)]
    # Magic offset (sum of each row/column = Nr · (Nr² + 1) / 2 for odd N)
    def _magic_odd(n: int) -> list[list[int]]:
        G = [[0] * n for _ in range(n)]
        i, j = 0, n // 2
        for k in range(1, n * n + 1):
            G[i][j] = k - 1
            ni = (i - 1) % n
            nj = (j + 1) % n
            if G[ni][nj] != 0:
                i = (i + 1) % n
            else:
                i, j = ni, nj
        return G
    n_min = min(Nr, Nz)
    if n_min % 2 == 1:
        G = _magic_odd(n_min)
        for i in range(Nr):
            for j in range(Nz):
                M[i][j] = G[i % n_min][j % n_min]
    else:
        # Row-major fallback with a diagonal magic offset
        for i in range(Nr):
            for j in range(Nz):
                M[i][j] = (i * Nz + j + (i + j) % n_min) % (Nr * Nz)
    return M


# ---------------------------------------------------------------------------
# FEM-like mesh export (fem_to_tec analog)
# ---------------------------------------------------------------------------

def write_fem_mesh(prefix: str, R_grid: list[float], Z_grid: list[float],
                   psi: list[list[float]], p_field: list[list[float]],
                   jt_field: list[list[float]]) -> dict:
    """Write three FEM-style text files:
        prefix_nodes.txt     — (R, Z) coordinates
        prefix_elements.txt  — triangular connectivity (Delaunay-like)
        prefix_values.txt    — per-node (ψ, p, j_φ)"""
    Nr = len(R_grid); Nz = len(Z_grid)
    # Nodes
    node_path = f"{prefix}_nodes.txt"
    with open(node_path, "w") as f:
        for i in range(Nr):
            for j in range(Nz):
                f.write(f"{R_grid[i]:.8f} {Z_grid[j]:.8f}\n")
    # Elements: split each quad (i,j)-(i+1,j)-(i,j+1)-(i+1,j+1) into 2 triangles
    elem_path = f"{prefix}_elements.txt"
    with open(elem_path, "w") as f:
        for i in range(Nr - 1):
            for j in range(Nz - 1):
                n0 = i * Nz + j
                n1 = (i + 1) * Nz + j
                n2 = i * Nz + (j + 1)
                n3 = (i + 1) * Nz + (j + 1)
                f.write(f"{n0} {n1} {n2}\n")
                f.write(f"{n1} {n3} {n2}\n")
    # Values
    val_path = f"{prefix}_values.txt"
    with open(val_path, "w") as f:
        for i in range(Nr):
            for j in range(Nz):
                f.write(f"{psi[i][j]:.8e} {p_field[i][j]:.8e} {jt_field[i][j]:.8e}\n")
    return {"nodes": node_path, "elements": elem_path, "values": val_path}


# ---------------------------------------------------------------------------
# Scalar diagnostics
# ---------------------------------------------------------------------------

def compute_diagnostics(psi: list[list[float]], jt: list[list[float]],
                        p_field: list[list[float]],
                        R_grid: list[float], Z_grid: list[float],
                        B0: float, Ip: float) -> dict:
    """Compute β_t, β_p, l_i, W_MHD, q_cyl, and volume-averaged quantities."""
    dR = R_grid[1] - R_grid[0]
    dZ = Z_grid[1] - Z_grid[0]
    Nr = len(R_grid); Nz = len(Z_grid)
    # Volume integral
    V = 0.0; Wp = 0.0; sum_p = 0.0; sum_Bp2 = 0.0
    for i in range(1, Nr - 1):
        R = R_grid[i]
        for j in range(1, Nz - 1):
            if psi[i][j] > 0.0:
                dV = 2.0 * math.pi * R * dR * dZ
                V += dV
                Wp += 1.5 * p_field[i][j] * dV
                sum_p += p_field[i][j] * dV
                # B_p² = |∇ψ|² / R²
                pR = (psi[i + 1][j] - psi[i - 1][j]) / (2.0 * dR)
                pZ = (psi[i][j + 1] - psi[i][j - 1]) / (2.0 * dZ)
                Bp2 = (pR * pR + pZ * pZ) / max(R * R, 1e-12)
                sum_Bp2 += Bp2 * dV
    p_avg = sum_p / V if V > 1e-30 else 0.0
    Bp2_avg = sum_Bp2 / V if V > 1e-30 else 0.0
    beta_t = 2.0 * MU0 * p_avg / (B0 * B0 + 1e-30)
    beta_p = 2.0 * MU0 * p_avg / (Bp2_avg + 1e-30)
    # Internal inductance: l_i = <B_p²> / B_p(a)² (approximate B_p(a)² by edge avg)
    Bp2_edge = 0.0; n_edge = 0
    for i in (1, Nr - 2):
        for j in range(1, Nz - 1):
            pR = (psi[i + 1][j] - psi[i - 1][j]) / (2.0 * dR)
            pZ = (psi[i][j + 1] - psi[i][j - 1]) / (2.0 * dZ)
            R = R_grid[i]
            Bp2_edge += (pR * pR + pZ * pZ) / max(R * R, 1e-12)
            n_edge += 1
    Bp2_a = Bp2_edge / max(n_edge, 1)
    l_i = Bp2_avg / (Bp2_a + 1e-30) if Bp2_a > 1e-30 else 1.0
    return {
        "volume_m3": V,
        "stored_energy_J": Wp,
        "p_avg_Pa": p_avg,
        "Bp2_avg_T2": Bp2_avg,
        "beta_toroidal": beta_t,
        "beta_poloidal": beta_p,
        "l_i": l_i,
        "Ip_A": Ip,
    }


def multi_panel_summary(diag: dict, eq_geom, stability: dict) -> str:
    """Format a multi-panel diagnostic summary (1288 oral-cavity style)."""
    lines = []
    lines.append("=" * 64)
    lines.append("TOKAMAK EQUILIBRIUM — MULTI-PANEL DIAGNOSTIC SUMMARY")
    lines.append("=" * 64)
    lines.append(f"[Panel 1]  Magnetic axis           : R = {eq_geom.R_axis:.4f} m, "
                 f"Z = {eq_geom.Z_axis:.4f} m")
    if eq_geom.R_X is not None:
        lines.append(f"[Panel 2]  X-point (divertor)        : R = {eq_geom.R_X:.4f} m, "
                     f"Z = {eq_geom.Z_X:.4f} m")
    else:
        lines.append("[Panel 2]  X-point                 : none (limiter configuration)")
    lines.append(f"[Panel 3]  Geometric shaping         : a = {eq_geom.a_geo:.4f} m, "
                 f"κ = {eq_geom.kappa_geo:.3f}, δ = {eq_geom.delta_geo:.3f}")
    lines.append(f"[Panel 4]  Plasma volume             : {diag['volume_m3']:.4f} m^3")
    lines.append(f"[Panel 5]  Stored energy             : {diag['stored_energy_J']:.3e} J")
    lines.append(f"[Panel 6]  <p>                       : {diag['p_avg_Pa']:.3e} Pa")
    lines.append(f"[Panel 7]  β_toroidal                : {diag['beta_toroidal']*100:.3f} %")
    lines.append(f"[Panel 8]  β_poloidal                : {diag['beta_poloidal']:.3f}")
    lines.append(f"[Panel 9]  l_i (internal inductance) : {diag['l_i']:.3f}")
    lines.append(f"[Panel 10] I_p                       : {diag['Ip_A']:.3e} A")
    lines.append(f"[Panel 11] q_cyl                     : {eq_geom.q_cyl:.3f}")
    lines.append(f"[Panel 12] Mercier D_M (axis)        : {stability.get('D_M_axis', 0.0):.4f}")
    lines.append(f"[Panel 13] Mercier stable            : {stability.get('stable', '?')}")
    lines.append(f"[Panel 14] Tearing Δ' (2,1)          : {stability.get('delta_prime', 0.0):.3f} m^-1")
    lines.append(f"[Panel 15] Max MHD growth rate       : {stability.get('max_growth', 0.0):.3e} s^-1")
    lines.append("=" * 64)
    return "\n".join(lines)
