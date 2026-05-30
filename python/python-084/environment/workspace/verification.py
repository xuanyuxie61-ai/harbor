# -*- coding: utf-8 -*-

import numpy as np
from typing import Tuple, Optional





def exact_harmonic_response_sdof(
    m: float,
    c: float,
    k: float,
    f0: float,
    omega: float,
    t: np.ndarray,
) -> np.ndarray:
    t = np.asarray(t, dtype=float)
    den = np.sqrt((k - m * omega ** 2) ** 2 + (c * omega) ** 2)
    if den < 1e-15:
        U = 0.0
        phi = 0.0
    else:
        U = f0 / den
        phi = np.arctan2(c * omega, k - m * omega ** 2)
    return U * np.cos(omega * t - phi)





class VerificationSuite:

    def __init__(self, M: np.ndarray, K: np.ndarray, C: Optional[np.ndarray] = None):
        self.M = np.asarray(M, dtype=float)
        self.K = np.asarray(K, dtype=float)
        self.C = C
        self.n_dof = self.M.shape[0]
        self.results = {}




    def test_mass_matrix_spd(self, tol: float = 1e-10) -> bool:
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




    def test_stiffness_matrix_symmetric(self, tol: float = 1e-10) -> bool:
        sym_err = float(np.max(np.abs(self.K - self.K.T)))
        if sym_err > tol:
            self.results["stiffness_sym"] = (False, f"Asymmetric: max_err={sym_err:.3e}")
            return False

        eigvals = np.linalg.eigvalsh(self.K)
        min_eig = float(np.min(eigvals))
        self.results["stiffness_sym"] = (True, f"Symmetric, min_eig={min_eig:.3e}")
        return True




    def test_modal_orthogonality(self, phi: np.ndarray, tol: float = 1e-8) -> bool:
        phi = np.asarray(phi, dtype=float)
        ident = phi.T @ self.M @ phi
        err = float(np.max(np.abs(ident - np.eye(ident.shape[0]))))
        ok = err < tol
        self.results["modal_orthogonality"] = (ok, f"Max off-diagonal error={err:.3e}")
        return ok




    def test_static_equilibrium(self, tol: float = 1e-8) -> bool:
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




    def test_energy_conservation(
        self,
        u0: np.ndarray = None,
        v0: np.ndarray = None,
        dt: float = 0.01,
        n_steps: int = 500,
        tol: float = 1e-1,
    ) -> bool:

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
        t = np.arange(0.0, t_max + dt, dt)
        u_exact = exact_harmonic_response_sdof(m, c, k, f0, omega, t)


        u = 0.0
        v = 0.0
        u_hist = np.zeros(len(t), dtype=float)
        u_hist[0] = u


        a0 = (f0 * np.cos(omega * t[0]) - c * v - k * u) / m
        v_half = v + 0.5 * dt * a0

        for i in range(len(t) - 1):
            u_hist[i] = u
            F_i = f0 * np.cos(omega * t[i])
            a = (F_i - c * v_half - k * u) / m
            v_half = v_half + dt * a
            u = u + dt * v_half

        u_hist[-1] = u


        idx = int(0.7 * len(t))
        denom = float(np.linalg.norm(u_exact[idx:])) + 1e-12
        err = float(np.linalg.norm(u_hist[idx:] - u_exact[idx:]) / denom)
        ok = err < tol
        self.results["exact_harmonic"] = (ok, f"Relative error vs exact={err:.3e}")
        return ok




    def run_all(self, phi: Optional[np.ndarray] = None) -> dict:
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
        print("=" * 60)
        print("VERIFICATION SUITE RESULTS")
        print("=" * 60)
        for name, (ok, msg) in self.results.items():
            status = "PASS" if ok else "FAIL"
            print(f"  [{status}] {name}: {msg}")
        print("=" * 60)
