
import numpy as np
from scipy.linalg import solve_triangular
from typing import Callable, Tuple, Optional, List


class ConstrainedHamiltonianSystem:
    def __init__(self, n_dof: int, n_constr: int,
                 mass_matrix: np.ndarray,
                 force_func: Callable[[np.ndarray, np.ndarray, float], np.ndarray],
                 constraint_func: Callable[[np.ndarray], np.ndarray],
                 constraint_jacobian: Callable[[np.ndarray], np.ndarray],
                 alpha_baumgarte: float = 5.0,
                 beta_baumgarte: float = 5.0,
                 potential_func: Optional[Callable[[np.ndarray], float]] = None):
        self.n_dof = n_dof
        self.n_constr = n_constr
        self.M = np.asarray(mass_matrix, dtype=np.float64)
        self.M_inv = np.linalg.inv(self.M)
        self.force_func = force_func
        self.potential_func = potential_func
        self.phi_func = constraint_func
        self.phi_q_func = constraint_jacobian
        self.alpha = alpha_baumgarte
        self.beta = beta_baumgarte

        self._M_chol = None
        try:
            self._M_chol = np.linalg.cholesky(self.M)
        except np.linalg.LinAlgError:
            pass

    def _solve_mass(self, b: np.ndarray) -> np.ndarray:
        if self._M_chol is not None:

            y = solve_triangular(self._M_chol, b, lower=True)
            x = solve_triangular(self._M_chol.T, y, lower=False)
            return x
        return self.M_inv @ b

    def potential_energy(self, q: np.ndarray) -> float:
        if self.potential_func is not None:
            return float(self.potential_func(q))
        return 0.0

    def kinetic_energy(self, p: np.ndarray) -> float:
        v = self._solve_mass(p)
        return 0.5 * float(p @ v)

    def total_energy(self, q: np.ndarray, p: np.ndarray) -> float:
        return self.kinetic_energy(p) + self.potential_energy(q)

    def compute_lagrange_multipliers(self, q: np.ndarray, p: np.ndarray,
                                     t: float, qdot: Optional[np.ndarray] = None) -> np.ndarray:
        phi_q = self.phi_q_func(q)
        Q = self.force_func(q, p, t)
        if qdot is None:
            qdot = self._solve_mass(p)
        phi = self.phi_func(q)
        phi_dot = phi_q @ qdot



        S = phi_q @ self._solve_mass(phi_q.T)
        rhs = phi_q @ self._solve_mass(Q) + 2.0 * self.alpha * phi_dot + self.beta ** 2 * phi

        reg = 1e-12 * np.eye(self.n_constr)
        lam = np.linalg.solve(S + reg, rhs)
        return lam

    def step_symplectic_euler(self, q: np.ndarray, p: np.ndarray,
                              t: float, h: float) -> Tuple[np.ndarray, np.ndarray]:

        raise NotImplementedError("Hole 2: step_symplectic_euler 待实现")

    def _project_position(self, q: np.ndarray, max_iter: int = 3) -> np.ndarray:
        q_proj = q.copy()
        for _ in range(max_iter):
            phi = self.phi_func(q_proj)
            if np.linalg.norm(phi) < 1e-12:
                break
            phi_q = self.phi_q_func(q_proj)
            S = phi_q @ self._solve_mass(phi_q.T)
            reg = 1e-12 * np.eye(self.n_constr)
            delta_lam = np.linalg.solve(S + reg, phi)
            q_proj -= self._solve_mass(phi_q.T @ delta_lam)
        return q_proj

    def integrate(self, q0: np.ndarray, p0: np.ndarray,
                  t_span: Tuple[float, float], h: float,
                  thinning_factor: int = 1) -> dict:
        t0, tf = t_span
        if h <= 0:
            raise ValueError("步长 h 必须为正")
        n_steps = int(np.ceil((tf - t0) / h))
        q = np.asarray(q0, dtype=np.float64).copy()
        p = np.asarray(p0, dtype=np.float64).copy()
        t = t0
        e0 = self.total_energy(q, p)

        save_every = max(1, thinning_factor)
        n_save = n_steps // save_every + 2
        ts = np.zeros(n_save)
        qs = np.zeros((n_save, self.n_dof))
        ps = np.zeros((n_save, self.n_dof))
        es = np.zeros(n_save)
        idx = 0
        ts[idx] = t
        qs[idx] = q
        ps[idx] = p
        es[idx] = e0
        idx += 1
        for step in range(n_steps):
            q, p = self.step_symplectic_euler(q, p, t, h)
            t = min(t + h, tf)
            if (step + 1) % save_every == 0 or step == n_steps - 1:
                if idx < n_save:
                    ts[idx] = t
                    qs[idx] = q
                    ps[idx] = p
                    es[idx] = self.total_energy(q, p)
                    idx += 1

        ts = ts[:idx]
        qs = qs[:idx]
        ps = ps[:idx]
        es = es[:idx]
        energy_scale = max(abs(e0), np.max(np.abs(es)), 1e-12)
        energy_drift = np.abs((es - e0) / energy_scale)
        return {
            "t": ts,
            "q": qs,
            "p": ps,
            "energy": es,
            "energy_drift": energy_drift,
            "max_drift": float(np.max(energy_drift)),
            "mean_drift": float(np.mean(energy_drift))
        }


def thin_state_vectors(states: np.ndarray, thin_factor: int,
                       method: str = "uniform") -> np.ndarray:
    if thin_factor <= 1:
        return states
    n = states.shape[0]
    if method == "uniform":
        keep = np.arange(0, n, thin_factor)
        return states[keep]
    elif method == "energy":
        if states.ndim == 1:
            return states[::thin_factor]

        diffs = np.linalg.norm(np.diff(states, axis=0), axis=1)

        n_bins = max(1, n // thin_factor)
        bin_size = n // n_bins
        keep = [0]
        for b in range(n_bins):
            start = b * bin_size
            end = min((b + 1) * bin_size, n - 1)
            if start >= end:
                continue
            local_max = start + np.argmax(diffs[start:end])
            keep.append(local_max)
        keep = sorted(set(keep))
        return states[keep]
    else:
        raise ValueError(f"未知稀疏化方法: {method}")
