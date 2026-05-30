
import numpy as np
from typing import Optional, Tuple
from utils import i4mat_rref
from sparse_matrix import SparseMatrix, conjugate_gradient


class WeightedLeastSquaresSE:

    def __init__(self, n_bus: int, y_bus: np.ndarray,
                 bus_types: np.ndarray):
        self.n_bus = n_bus
        self.y_bus = np.array(y_bus, dtype=np.complex128)
        self.g = self.y_bus.real
        self.b = self.y_bus.imag
        self.bus_types = np.array(bus_types, dtype=np.int32)

    def measurement_function(self, vm: np.ndarray, va: np.ndarray,
                             meas_type: str, idx: int) -> float:
        vm = np.array(vm, dtype=np.float64)
        va = np.array(va, dtype=np.float64)

        if meas_type == 'V_mag':
            return vm[idx]

        if meas_type in ('P_inj', 'Q_inj'):




            raise NotImplementedError("Hole 2: measurement_function P_inj/Q_inj 待实现")
            return val

        if meas_type in ('P_flow', 'Q_flow'):

            i, j = idx
            angle_diff = va[i] - va[j]
            if meas_type == 'P_flow':
                return vm[i]**2 * self.g[i, j] - vm[i] * vm[j] * (
                    self.g[i, j] * np.cos(angle_diff)
                    + self.b[i, j] * np.sin(angle_diff)
                )
            else:
                return -vm[i]**2 * self.b[i, j] - vm[i] * vm[j] * (
                    self.g[i, j] * np.sin(angle_diff)
                    - self.b[i, j] * np.cos(angle_diff)
                )
        return 0.0

    def build_measurement_jacobian(self, vm: np.ndarray, va: np.ndarray,
                                   measurements: list) -> np.ndarray:
        m = len(measurements)
        n_state = 2 * self.n_bus
        H = np.zeros((m, n_state), dtype=np.float64)
        h = 1e-4
        for mi, (mtype, midx, _, _) in enumerate(measurements):
            for si in range(n_state):
                vm_p = vm.copy()
                va_p = va.copy()
                vm_m = vm.copy()
                va_m = va.copy()
                if si < self.n_bus:
                    vm_p[si] += h
                    vm_m[si] -= h
                else:
                    idx_a = si - self.n_bus
                    va_p[idx_a] += h
                    va_m[idx_a] -= h
                hp = self.measurement_function(vm_p, va_p, mtype, midx)
                hm = self.measurement_function(vm_m, va_m, mtype, midx)
                H[mi, si] = (hp - hm) / (2.0 * h)
        return H

    def solve(self, measurements: list, vm0: np.ndarray, va0: np.ndarray,
              tol: float = 1e-6, max_iter: int = 20) -> dict:
        vm = np.array(vm0, dtype=np.float64)
        va = np.array(va0, dtype=np.float64)

        for it in range(max_iter):
            z = np.array([m[2] for m in measurements], dtype=np.float64)
            sigma = np.array([m[3] for m in measurements], dtype=np.float64)
            R_inv = np.diag(1.0 / (sigma**2 + 1e-15))

            h_est = np.array([self.measurement_function(vm, va, m[0], m[1])
                              for m in measurements], dtype=np.float64)
            r = z - h_est

            H = self.build_measurement_jacobian(vm, va, measurements)

            G = H.T @ R_inv @ H
            rhs = H.T @ R_inv @ r
            try:
                dx = np.linalg.solve(G, rhs)
            except np.linalg.LinAlgError:
                dx = np.linalg.lstsq(G, rhs, rcond=None)[0]


            dx_vm = dx[:self.n_bus].copy()
            dx_va = dx[self.n_bus:].copy()
            dx_vm[self.bus_types != 0] = 0.0
            dx_va[self.bus_types == 2] = 0.0

            vm = vm + dx_vm
            va = va + dx_va
            vm = np.clip(vm, 0.5, 1.5)

            if np.linalg.norm(dx_vm) + np.linalg.norm(dx_va) < tol:

                J = float(r.T @ R_inv @ r)
                return {
                    "converged": True,
                    "iterations": it + 1,
                    "vm": vm.copy(),
                    "va": va.copy(),
                    "J": J,
                    "residual": r.copy()
                }

        J = float(r.T @ R_inv @ r)
        return {
            "converged": False,
            "iterations": max_iter,
            "vm": vm.copy(),
            "va": va.copy(),
            "J": J,
            "residual": r.copy()
        }


class ObservabilityAnalysis:

    def __init__(self, n_bus: int):
        self.n_bus = n_bus

    def check_observability(self, H: np.ndarray) -> dict:
        H_int = np.round(H * 1e6).astype(np.int64)
        rref = i4mat_rref(H_int)
        rank = int(np.sum(np.any(rref != 0, axis=1)))
        n_states = H.shape[1]
        observable = rank >= n_states - 1
        return {
            "rank": rank,
            "n_states": n_states,
            "observable": observable,
            "deficiency": max(0, n_states - 1 - rank)
        }
