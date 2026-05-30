
import numpy as np
from typing import Optional, Tuple
from sparse_matrix import SparseMatrix, conjugate_gradient
from utils import diff2_center


class PowerFlowSolver:

    def __init__(self, y_bus: np.ndarray, bus_types: np.ndarray,
                 v_magnitude: np.ndarray, v_angle: np.ndarray,
                 p_spec: np.ndarray, q_spec: np.ndarray):
        self.n = y_bus.shape[0]
        self.y_bus = np.array(y_bus, dtype=np.complex128)
        self.bus_types = np.array(bus_types, dtype=np.int32)
        self.vm = np.array(v_magnitude, dtype=np.float64)
        self.va = np.array(v_angle, dtype=np.float64)
        self.p_spec = np.array(p_spec, dtype=np.float64)
        self.q_spec = np.array(q_spec, dtype=np.float64)
        self.g = self.y_bus.real
        self.b = self.y_bus.imag

    def compute_power_mismatch(self) -> Tuple[np.ndarray, np.ndarray]:
        n = self.n
        p_calc = np.zeros(n, dtype=np.float64)
        q_calc = np.zeros(n, dtype=np.float64)





        raise NotImplementedError("Hole 1: compute_power_mismatch 待实现")
        dp = self.p_spec - p_calc
        dq = self.q_spec - q_calc
        return dp, dq

    def build_jacobian(self) -> np.ndarray:
        n = self.n
        j11 = np.zeros((n, n), dtype=np.float64)
        j12 = np.zeros((n, n), dtype=np.float64)
        j21 = np.zeros((n, n), dtype=np.float64)
        j22 = np.zeros((n, n), dtype=np.float64)

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                angle_diff = self.va[i] - self.va[j]
                j11[i, j] = self.vm[i] * self.vm[j] * (
                    self.g[i, j] * np.sin(angle_diff)
                    - self.b[i, j] * np.cos(angle_diff)
                )
                j12[i, j] = self.vm[i] * self.vm[j] * (
                    self.g[i, j] * np.cos(angle_diff)
                    + self.b[i, j] * np.sin(angle_diff)
                )
                j21[i, j] = -j12[i, j]
                j22[i, j] = j11[i, j]

            j11[i, i] = -self.q_calc_at_bus(i) - self.b[i, i] * self.vm[i]**2
            j12[i, i] = self.p_calc_at_bus(i) + self.g[i, i] * self.vm[i]**2
            j21[i, i] = self.p_calc_at_bus(i) - self.g[i, i] * self.vm[i]**2
            j22[i, i] = self.q_calc_at_bus(i) - self.b[i, i] * self.vm[i]**2


        pq_idx = np.where(self.bus_types == 0)[0]
        pv_idx = np.where(self.bus_types == 1)[0]
        slack_idx = np.where(self.bus_types == 2)[0]



        theta_idx = np.concatenate([pq_idx, pv_idx])
        v_idx = pq_idx

        rows = []
        cols = []

        for i in theta_idx:
            row = []
            for j in theta_idx:
                row.append(j11[i, j])
            for j in v_idx:
                row.append(j12[i, j])
            rows.append(row)

        for i in v_idx:
            row = []
            for j in theta_idx:
                row.append(j21[i, j])
            for j in v_idx:
                row.append(j22[i, j])
            rows.append(row)

        if len(rows) == 0:
            return np.zeros((0, 0), dtype=np.float64)
        return np.array(rows, dtype=np.float64)

    def p_calc_at_bus(self, i: int) -> float:
        p = 0.0
        for j in range(self.n):
            angle_diff = self.va[i] - self.va[j]
            p += self.vm[i] * self.vm[j] * (
                self.g[i, j] * np.cos(angle_diff)
                + self.b[i, j] * np.sin(angle_diff)
            )
        return p

    def q_calc_at_bus(self, i: int) -> float:
        q = 0.0
        for j in range(self.n):
            angle_diff = self.va[i] - self.va[j]
            q += self.vm[i] * self.vm[j] * (
                self.g[i, j] * np.sin(angle_diff)
                - self.b[i, j] * np.cos(angle_diff)
            )
        return q

    def solve(self, tol: float = 1e-8, max_iter: int = 50) -> dict:
        n = self.n
        pq_idx = np.where(self.bus_types == 0)[0]
        pv_idx = np.where(self.bus_types == 1)[0]
        slack_idx = np.where(self.bus_types == 2)[0]

        theta_idx = np.concatenate([pq_idx, pv_idx])
        v_idx = pq_idx

        for it in range(max_iter):
            dp, dq = self.compute_power_mismatch()

            dp[slack_idx] = 0.0
            dq[slack_idx] = 0.0
            dq[pv_idx] = 0.0

            mismatch = np.concatenate([dp[theta_idx], dq[v_idx]])
            norm_mis = np.linalg.norm(mismatch, ord=np.inf)
            if norm_mis < tol:
                return {
                    "converged": True,
                    "iterations": it,
                    "vm": self.vm.copy(),
                    "va": self.va.copy(),
                    "mismatch": norm_mis
                }


            jacobian = self.build_jacobian()
            if jacobian.size == 0:
                break
            try:
                dx = np.linalg.solve(jacobian, mismatch)
            except np.linalg.LinAlgError:

                dx = np.linalg.lstsq(jacobian, mismatch, rcond=None)[0]

            n_theta = len(theta_idx)
            dtheta = dx[:n_theta]
            dvm = dx[n_theta:]


            alpha = 1.0
            vm_old = self.vm.copy()
            va_old = self.va.copy()
            best_norm = norm_mis
            for _ in range(5):
                self.va = va_old.copy()
                self.vm = vm_old.copy()
                self.va[theta_idx] += alpha * dtheta
                self.vm[v_idx] *= (1.0 + alpha * dvm)
                self.vm = np.clip(self.vm, 0.8, 1.2)

                dp_t, dq_t = self.compute_power_mismatch()
                dp_t[slack_idx] = 0.0
                dq_t[slack_idx] = 0.0
                dq_t[pv_idx] = 0.0
                mis_t = np.concatenate([dp_t[theta_idx], dq_t[v_idx]])
                norm_t = np.linalg.norm(mis_t, ord=np.inf)
                if norm_t < best_norm:
                    best_norm = norm_t
                    break
                alpha *= 0.5
            else:

                self.va = va_old.copy()
                self.vm = vm_old.copy()
                self.va[theta_idx] += alpha * dtheta
                self.vm[v_idx] *= (1.0 + alpha * dvm)
                self.vm = np.clip(self.vm, 0.8, 1.2)

        dp, dq = self.compute_power_mismatch()
        dp[slack_idx] = 0.0
        dq[slack_idx] = 0.0
        dq[pv_idx] = 0.0
        mismatch = np.concatenate([dp[theta_idx], dq[v_idx]])
        norm_mis = np.linalg.norm(mismatch, ord=np.inf)
        return {
            "converged": False,
            "iterations": max_iter,
            "vm": self.vm.copy(),
            "va": self.va.copy(),
            "mismatch": norm_mis
        }


def build_y_bus(n: int, edges: np.ndarray,
                r_line: np.ndarray, x_line: np.ndarray,
                b_shunt: Optional[np.ndarray] = None) -> np.ndarray:
    y_bus = np.zeros((n, n), dtype=np.complex128)
    if b_shunt is None:
        b_shunt = np.zeros(len(edges), dtype=np.float64)

    for idx, (i, j) in enumerate(edges):
        i, j = int(i), int(j)
        z = complex(r_line[idx], x_line[idx])
        if abs(z) < 1e-12:
            z = 1e-12 + 1j * 1e-12
        y = 1.0 / z
        y_bus[i, i] += y + 1j * b_shunt[idx] * 0.5
        y_bus[j, j] += y + 1j * b_shunt[idx] * 0.5
        y_bus[i, j] -= y
        y_bus[j, i] -= y
    return y_bus
