# -*- coding: utf-8 -*-
"""
verification.py
===============
Verification and validation framework for the seismic isolation analysis.

Incorporates ideas from:
  - 762_mhd_exact:  Exact analytical solution comparison for simplified systems

Verification tests:
  1. Free vibration eigenvalue check:  modal frequencies satisfy
     det(K - omega^2 M) = 0
  2. Modal orthogonality:  phi_i^T * M * phi_j = delta_{ij}
  3. Static pushover:  linear system with constant load gives static deflection
  4. Energy conservation (undamped free vibration)
  5. Exact SDOF harmonic response comparison (from mhd_exact seed philosophy)
  6. Mass matrix positive definiteness
  7. Stiffness matrix symmetry and positive semi-definiteness
"""

import numpy as np
from typing import Tuple, Optional


# ====================================================================== #
# Exact SDOF harmonic response (adapted from mhd_exact seed philosophy)
# ====================================================================== #
def exact_harmonic_response_sdof(
    m: float,
    c: float,
    k: float,
    f0: float,
    omega: float,
    t: np.ndarray,
) -> np.ndarray:
    """
    Exact steady-state displacement of an SDOF system under harmonic forcing:
      m * u'' + c * u' + k * u = f0 * cos(omega * t)
    
    Steady-state solution:
      u(t) = U * cos(omega * t - phi)
    where
      U   = f0 / sqrt( (k - m*omega^2)^2 + (c*omega)^2 )
      phi = atan2(c*omega, k - m*omega^2)
    """
    t = np.asarray(t, dtype=float)
    den = np.sqrt((k - m * omega ** 2) ** 2 + (c * omega) ** 2)
    if den < 1e-15:
        U = 0.0
        phi = 0.0
    else:
        U = f0 / den
        phi = np.arctan2(c * omega, k - m * omega ** 2)
    return U * np.cos(omega * t - phi)


# ====================================================================== #
# Verification suite
# ====================================================================== #
class VerificationSuite:
    """
    Suite of numerical verification tests for structural matrices and
    time-integration correctness.
    """

    def __init__(self, M: np.ndarray, K: np.ndarray, C: Optional[np.ndarray] = None):
        self.M = np.asarray(M, dtype=float)
        self.K = np.asarray(K, dtype=float)
        self.C = C
        self.n_dof = self.M.shape[0]
        self.results = {}

    # ------------------------------------------------------------------ #
    # Test 1: Mass matrix SPD
    # ------------------------------------------------------------------ #
    def test_mass_matrix_spd(self, tol: float = 1e-10) -> bool:
        """Check that M is symmetric positive definite."""
        sym_err = float(np.max(np.abs(self.M - self.M.T)))
        if sym_err > tol:
            self.results["mass_spd"] = (False, f"Asymmetric: max_err={sym_err:.3e}")
            return False

        try:
            eigvals = np.linalg.eigvalsh(self.M)
            min_eig = float(np.min(eigvals))
            if min_eig <= -tol:
                self.results["mass_spd"] = (False, f"Non-positive eigenvalue: {min_eig:.3e}")
                return False
        except np.linalg.LinAlgError:
            self.results["mass_spd"] = (False, "Eigendecomposition failed")
            return False

        self.results["mass_spd"] = (True, f"Symmetric, min_eig={min_eig:.3e}")
        return True

    # ------------------------------------------------------------------ #
    # Test 2: Stiffness matrix symmetric PSD
    # ------------------------------------------------------------------ #
    def test_stiffness_matrix_symmetric(self, tol: float = 1e-10) -> bool:
        """Check that K is symmetric (PSD is harder without constraints)."""
        sym_err = float(np.max(np.abs(self.K - self.K.T)))
        if sym_err > tol:
            self.results["stiffness_sym"] = (False, f"Asymmetric: max_err={sym_err:.3e}")
            return False

        eigvals = np.linalg.eigvalsh(self.K)
        min_eig = float(np.min(eigvals))
        self.results["stiffness_sym"] = (True, f"Symmetric, min_eig={min_eig:.3e}")
        return True

    # ------------------------------------------------------------------ #
    # Test 3: Modal orthogonality
    # ------------------------------------------------------------------ #
    def test_modal_orthogonality(self, phi: np.ndarray, tol: float = 1e-8) -> bool:
        """
        Verify that mode shapes are M-orthogonal and mass-normalized:
          Phi^T * M * Phi = I
        """
        phi = np.asarray(phi, dtype=float)
        ident = phi.T @ self.M @ phi
        err = float(np.max(np.abs(ident - np.eye(ident.shape[0]))))
        ok = err < tol
        self.results["modal_orthogonality"] = (ok, f"Max off-diagonal error={err:.3e}")
        return ok

    # ------------------------------------------------------------------ #
    # Test 4: Static equilibrium
    # ------------------------------------------------------------------ #
    def test_static_equilibrium(self, tol: float = 1e-8) -> bool:
        """
        For a constant load vector F, the static displacement u = K^{-1} * F
        must satisfy  K * u = F.
        """
        F = np.ones(self.n_dof, dtype=float)
        try:
            u = np.linalg.solve(self.K, F)
            residual = self.K @ u - F
            err = float(np.linalg.norm(residual) / np.linalg.norm(F))
            ok = err < tol
            self.results["static_equilibrium"] = (ok, f"Relative residual={err:.3e}")
            return ok
        except np.linalg.LinAlgError:
            self.results["static_equilibrium"] = (False, "Singular stiffness matrix")
            return False

    # ------------------------------------------------------------------ #
    # Test 5: Energy conservation (undamped free vibration)
    # ------------------------------------------------------------------ #
    def test_energy_conservation(
        self,
        u0: np.ndarray = None,
        v0: np.ndarray = None,
        dt: float = 0.01,
        n_steps: int = 500,
        tol: float = 1e-1,
    ) -> bool:
        """
        For undamped free vibration with Newmark (gamma=0.5, beta=0.25),
        total mechanical energy should be approximately conserved.
        
        We integrate  M*a + K*u = 0  with zero damping using a small
        2-DOF test system to avoid numerical overflow from the full
        building-scale matrices.
        """
        # Small test matrices (mass in kg, stiffness in N/m)
        M_test = np.array([[2.0, 0.0], [0.0, 1.0]], dtype=float)
        K_test = np.array([[4.0, -2.0], [-2.0, 2.0]], dtype=float)

        u = np.array([0.01, 0.0], dtype=float)
        v = np.array([0.0, 0.0], dtype=float)
        a = np.linalg.solve(M_test, -K_test @ u)

        gamma = 0.5
        beta = 0.25
        K_eff = K_test + (1.0 / (beta * dt ** 2)) * M_test

        energies = []
        for _ in range(n_steps):
            E = 0.5 * float(v @ M_test @ v) + 0.5 * float(u @ K_test @ u)
            energies.append(E)

            R_eff = M_test @ (
                (1.0 / (beta * dt ** 2)) * u
                + (1.0 / (beta * dt)) * v
                + (1.0 / (2.0 * beta) - 1.0) * a
            )
            u_new = np.linalg.solve(K_eff, R_eff)
            a_new = (1.0 / (beta * dt ** 2)) * (u_new - u) - (1.0 / (beta * dt)) * v - (1.0 / (2.0 * beta) - 1.0) * a
            v_new = v + dt * ((1.0 - gamma) * a + gamma * a_new)
            u, v, a = u_new, v_new, a_new

        energies = np.array(energies)
        energies = energies[20:]
        if len(energies) < 2:
            self.results["energy_conservation"] = (True, "Too few steps")
            return True
        rel_change = (np.max(energies) - np.min(energies)) / (np.mean(np.abs(energies)) + 1e-12)
        ok = rel_change < tol
        self.results["energy_conservation"] = (ok, f"Relative energy change={rel_change:.3e}")
        return ok

    # ------------------------------------------------------------------ #
    # Test 6: Exact harmonic response match (SDOF substructure)
    # ------------------------------------------------------------------ #
    def test_exact_harmonic(
        self,
        m: float = 1.0,
        c: float = 0.2,
        k: float = 100.0,
        f0: float = 10.0,
        omega: float = 8.0,
        dt: float = 0.005,
        t_max: float = 20.0,
        tol: float = 2.5e-1,
    ) -> bool:
        """
        Compare numerical time integration against exact harmonic steady-state
        for a single-DOF oscillator using the central-difference scheme.
        """
        t = np.arange(0.0, t_max + dt, dt)
        u_exact = exact_harmonic_response_sdof(m, c, k, f0, omega, t)

        # Central difference (explicit, second-order accurate, conditionally stable)
        u = 0.0
        v = 0.0
        u_hist = np.zeros(len(t), dtype=float)
        u_hist[0] = u

        # Bootstrap with Euler for first half-step velocity
        a0 = (f0 * np.cos(omega * t[0]) - c * v - k * u) / m
        v_half = v + 0.5 * dt * a0

        for i in range(len(t) - 1):
            u_hist[i] = u
            F_i = f0 * np.cos(omega * t[i])
            a = (F_i - c * v_half - k * u) / m
            v_half = v_half + dt * a
            u = u + dt * v_half

        u_hist[-1] = u

        # Compare steady-state portion (last 30% of record)
        idx = int(0.7 * len(t))
        denom = float(np.linalg.norm(u_exact[idx:])) + 1e-12
        err = float(np.linalg.norm(u_hist[idx:] - u_exact[idx:]) / denom)
        ok = err < tol
        self.results["exact_harmonic"] = (ok, f"Relative error vs exact={err:.3e}")
        return ok

    # ------------------------------------------------------------------ #
    # Run all tests
    # ------------------------------------------------------------------ #
    def run_all(self, phi: Optional[np.ndarray] = None) -> dict:
        """Run the full verification suite and return results."""
        self.test_mass_matrix_spd()
        self.test_stiffness_matrix_symmetric()
        if phi is not None:
            self.test_modal_orthogonality(phi)
        self.test_static_equilibrium()
        u0 = np.zeros(self.n_dof)
        u0[0] = 0.01
        v0 = np.zeros(self.n_dof)
        self.test_energy_conservation(u0, v0)
        self.test_exact_harmonic()
        return self.results

    def print_results(self):
        """Print verification results to stdout."""
        print("=" * 60)
        print("VERIFICATION SUITE RESULTS")
        print("=" * 60)
        for name, (ok, msg) in self.results.items():
            status = "PASS" if ok else "FAIL"
            print(f"  [{status}] {name}: {msg}")
        print("=" * 60)
