"""
stability_analysis.py — Ideal-MHD and resistive stability analysis of the
computed equilibrium: Mercier criterion, tearing-mode Δ', and eigenmode
decomposition of the linearised MHD operator.  The eigenvalue decomposition
follows the strategy of the 1074_Akiraichi_Explicit-quantum-surrogates
project (diagonalise a matrix to extract modal structure).

Scientific background
---------------------
Mercier criterion (ideal MHD, interchange)
    D_Merc = (V'') / (V') · p' / (μ₀ ⟨1/R²⟩ - (V')² / (2π)²) - (p')² ⟨1/R²⟩ / ...
    Simplified (cylindrical limit, large aspect ratio):
        D_M = - (R₀ / B₀²) (dp/dψ) · (dq/dψ) / q²                 (1)
    D_M > 0 everywhere ⇒ stable to interchange.

Tearing-mode Δ' (copson-furth-rutherford)
    Δ' = [ψ̃₁' / ψ̃₁]_{r_s^+}^{r_s^-}                                (2)
    evaluated at the rational surface q(r_s) = m/n.

Eigenmode problem
    We form the matrix M_{ij} = -Δ* δ_{ij} + ∂j_φ/∂ψ|_{i=j}  and compute
    its eigenvalues to find the ideal-MHD growth rates γ² = -λ.
"""

from __future__ import annotations
import math


# ---------------------------------------------------------------------------
# Mercier criterion
# ---------------------------------------------------------------------------

def mercier_criterion(psi: list[list[float]], jt: list[list[float]],
                      R_grid: list[float], Z_grid: float,
                      R0: float, B0: float) -> dict:
    """Compute the Mercier parameter D_M on-axis and along a radial line.
    Returns {'D_M_axis', 'D_M_profile', 'stable'}."""
    dR = R_grid[1] - R_grid[0]
    Nr = len(R_grid); Nz = len(psi[0])
    j_mid = Nz // 2
    # dp/dψ approximated from ψ profile along Z=0
    dpdpsi = [0.0] * Nr
    for i in range(1, Nr - 1):
        dpdpsi[i] = (psi[i + 1][j_mid] - psi[i - 1][j_mid]) / (2.0 * dR)
    # Safety factor profile q(R) along Z=0 (simple estimate)
    q = [0.0] * Nr
    for i in range(2, Nr - 2):
        B_theta = abs(psi[i][j_mid + 1] - psi[i][j_mid - 1]) / (2.0 * dR) / max(R_grid[i], 1e-6)
        B_phi = B0 * R0 / max(R_grid[i], 1e-6)
        q[i] = (R_grid[i] * B_theta / max(B_phi, 1e-12)) if B_phi > 1e-12 else 1.0
    # D_M
    D_M = [0.0] * Nr
    for i in range(2, Nr - 2):
        dqdpsi = (q[i + 1] - q[i - 1]) / (2.0 * dR)
        denom = max(q[i] ** 2, 1e-6)
        D_M[i] = -(R0 / (B0 * B0 + 1e-12)) * dpdpsi[i] * dqdpsi / denom
    D_M_axis = D_M[Nr // 2]
    stable = all(d > -1e-6 for d in D_M)
    return {"D_M_axis": D_M_axis, "D_M_profile": D_M, "stable": stable}


# ---------------------------------------------------------------------------
# Tearing mode Δ' (single rational surface)
# ---------------------------------------------------------------------------

def tearing_delta_prime(psi: list[list[float]], q_profile: list[float],
                        R_grid: list[float], m: int = 2, n_tor: int = 1) -> float:
    """Estimate Δ' for the (m,n) tearing mode at the rational surface q = m/n.
    We locate r_s where q ≈ m/n and compute the logarithmic jump of a model
    helical flux perturbation ψ̃ ∝ exp(i m θ - i n φ)."""
    Nr = len(R_grid)
    q_target = m / n_tor
    # Find rational surface
    rs_idx = None
    for i in range(1, Nr - 1):
        if (q_profile[i] - q_target) * (q_profile[i + 1] - q_target) <= 0:
            rs_idx = i; break
    if rs_idx is None:
        return 0.0
    # Logarithmic derivative of ψ on each side
    dR = R_grid[1] - R_grid[0]
    j_mid = len(psi[0]) // 2
    psi_left = psi[rs_idx][j_mid]
    psi_right = psi[rs_idx + 1][j_mid]
    dpsi_left = (psi[rs_idx][j_mid] - psi[max(0, rs_idx - 1)][j_mid]) / dR
    dpsi_right = (psi[min(Nr - 1, rs_idx + 2)][j_mid] - psi[rs_idx + 1][j_mid]) / dR
    # Δ' = [ψ̃'/ψ̃]_+ - [ψ̃'/ψ̃]_-
    term_left = dpsi_left / max(abs(psi_left), 1e-12)
    term_right = dpsi_right / max(abs(psi_right), 1e-12)
    return term_right - term_left


# ---------------------------------------------------------------------------
# Eigenvalue decomposition of the linearised MHD operator
# (following 1074_Akiraichi: build the operator, diagonalise, extract modes)
# ---------------------------------------------------------------------------

def build_mhd_matrix(psi: list[list[float]], jt: list[list[float]],
                     dR: float, dZ: float, R_min: float,
                     Nr: int, Nz: int) -> list[list[float]]:
    """Build M = -Δ* + ∂j_φ/∂ψ  on interior nodes, as a dense matrix.
    For a small grid (Nr,Nz ~ 20) this is tractable."""
    n = Nr * Nz
    M = [[0.0] * n for _ in range(n)]
    for i in range(1, Nr - 1):
        R_i = R_min + i * dR
        for j in range(1, Nz - 1):
            k = i * Nz + j
            # -Δ* stencil
            cRR = 1.0 / (dR * dR)
            cR1 = 1.0 / (2.0 * R_i * dR)
            cZZ = 1.0 / (dZ * dZ)
            M[k][k - Nz] = -(cRR - cR1)
            M[k][k + Nz] = -(cRR + cR1)
            M[k][k] = 2.0 * cRR + 2.0 * cZZ
            M[k][k - 1] = -cZZ
            M[k][k + 1] = -cZZ
            # ∂j_φ/∂ψ diagonal contribution (small)
            M[k][k] += 0.01 * jt[i][j]
    return M


def symmetric_eigenvalues(M: list[list[float]], n_iter: int = 50) -> list[float]:
    """QR iteration for eigenvalues of a symmetric matrix (adapted from
    eigenvalue_decompose_util of the quantum surrogates project).
    For matrices up to ~100×100 this converges rapidly."""
    n = len(M)
    if n == 0:
        return []
    # Copy M
    A = [list(row) for row in M]
    for it in range(n_iter):
        # QR decomposition via Gram-Schmidt
        Q = [[0.0] * n for _ in range(n)]
        R = [[0.0] * n for _ in range(n)]
        for j in range(n):
            v = [A[i][j] for i in range(n)]
            for k in range(j):
                r = sum(Q[i][k] * v[i] for i in range(n))
                R[k][j] = r
                for i in range(n):
                    v[i] -= r * Q[i][k]
            norm = math.sqrt(sum(x * x for x in v))
            R[j][j] = norm
            if norm > 1e-12:
                for i in range(n):
                    Q[i][j] = v[i] / norm
        # A_new = R Q
        A_new = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                s = 0.0
                for k in range(n):
                    s += R[i][k] * Q[j][k]
                A_new[i][j] = s
        A = A_new
    return [A[i][i] for i in range(n)]


def mhd_growth_rates(eigenvalues: list[float]) -> list[float]:
    """γ² = -λ ⇒ γ = √(-λ) for λ < 0 (unstable modes).
    Returns the list of real growth rates for unstable eigenvalues."""
    rates = []
    for lam in eigenvalues:
        if lam < -1e-10:
            rates.append(math.sqrt(-lam))
        else:
            rates.append(0.0)
    return sorted(rates, reverse=True)


# ---------------------------------------------------------------------------
# Beta limit (Troyon-like scaling)
# ---------------------------------------------------------------------------

def troyon_beta_limit(Ip_MA: float, a_m: float, B0_T: float) -> float:
    """β_N,max = g · I / (a B)  with g ≈ 2.8 (Troyon).
    Returns β_max in percent."""
    g = 2.8
    return g * Ip_MA / (a_m * B0_T)


# ---------------------------------------------------------------------------
# Bootstrap current fraction (Sauter model, simplified)
# ---------------------------------------------------------------------------

def sauter_bootstrap(eps: float, nu_star: float, q: float) -> float:
    """Sauter-style f_bs ≈ C_bs · √ε / (1 + 0.5 ν_* + 0.3 q √ε).
    ε = r/R, ν_* = collisionality."""
    Cbs = 0.5
    sqrt_eps = math.sqrt(max(eps, 0.0))
    return Cbs * sqrt_eps / (1.0 + 0.5 * nu_star + 0.3 * q * sqrt_eps)
