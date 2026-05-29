"""
state_estimation.py
电力系统状态估计（WLS）与可观测性分析
融合种子项目：i4mat_rref（整数矩阵行简化）
"""

import numpy as np
from typing import Optional, Tuple
from utils import i4mat_rref
from sparse_matrix import SparseMatrix, conjugate_gradient


class WeightedLeastSquaresSE:
    """
    加权最小二乘状态估计（WLS State Estimation）。

    量测方程：
        z = h(x) + ε
    其中：
        z ∈ R^m   为量测向量（节点注入功率、支路潮流、电压幅值）
        x ∈ R^n   为状态向量（节点电压幅值与相角）
        h(x)      为非线性量测函数
        ε ~ N(0, R) 为高斯白噪声，R = diag(σ_1^2, ..., σ_m^2)

    WLS 目标函数：
        J(x) = [z - h(x)]^T · R^{-1} · [z - h(x)]

    高斯-牛顿迭代：
        x_{k+1} = x_k + [H^T·R^{-1}·H]^{-1} · H^T·R^{-1}·[z - h(x_k)]
    其中 H = ∂h/∂x 为量测雅可比矩阵。

    在智能电网中，WLS 状态估计是能量管理系统（EMS）的核心模块，
    用于从冗余量测中估计系统真实运行状态，检测和辨识坏数据。
    """

    def __init__(self, n_bus: int, y_bus: np.ndarray,
                 bus_types: np.ndarray):
        self.n_bus = n_bus
        self.y_bus = np.array(y_bus, dtype=np.complex128)
        self.g = self.y_bus.real
        self.b = self.y_bus.imag
        self.bus_types = np.array(bus_types, dtype=np.int32)

    def measurement_function(self, vm: np.ndarray, va: np.ndarray,
                             meas_type: str, idx: int) -> float:
        """
        计算单个量测的估计值 h_i(x)。

        量测类型：
            'P_inj' : 节点注入有功
            'Q_inj' : 节点注入无功
            'P_flow': 支路有功潮流
            'Q_flow': 支路无功潮流
            'V_mag' : 电压幅值
        """
        vm = np.array(vm, dtype=np.float64)
        va = np.array(va, dtype=np.float64)

        if meas_type == 'V_mag':
            return vm[idx]

        if meas_type in ('P_inj', 'Q_inj'):
            i = idx
            val = 0.0
            for j in range(self.n_bus):
                angle_diff = va[i] - va[j]
                if meas_type == 'P_inj':
                    val += vm[i] * vm[j] * (self.g[i, j] * np.cos(angle_diff)
                                            + self.b[i, j] * np.sin(angle_diff))
                else:
                    val += vm[i] * vm[j] * (self.g[i, j] * np.sin(angle_diff)
                                            - self.b[i, j] * np.cos(angle_diff))
            return val

        if meas_type in ('P_flow', 'Q_flow'):
            # 简化：假设 idx 为 (i,j) 边的索引
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
        """
        构建量测雅可比矩阵 H（有限差分近似，用于中小规模系统）。
        大规模系统可采用解析导数或稀疏结构。
        """
        m = len(measurements)
        n_state = 2 * self.n_bus  # [vm_0, ..., vm_{n-1}, va_0, ..., va_{n-1}]
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
        """
        WLS 状态估计高斯-牛顿迭代求解。

        参数 measurements：[(type, idx, value, sigma), ...]
        """
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
            # 增益矩阵 G = H^T · R^{-1} · H
            G = H.T @ R_inv @ H
            rhs = H.T @ R_inv @ r
            try:
                dx = np.linalg.solve(G, rhs)
            except np.linalg.LinAlgError:
                dx = np.linalg.lstsq(G, rhs, rcond=None)[0]

            # 固定参考节点和 PV 节点电压，固定 slack 相角
            dx_vm = dx[:self.n_bus].copy()
            dx_va = dx[self.n_bus:].copy()
            dx_vm[self.bus_types != 0] = 0.0   # 只有 PQ 节点电压可变
            dx_va[self.bus_types == 2] = 0.0   # 只有非 slack 相角可变

            vm = vm + dx_vm
            va = va + dx_va
            vm = np.clip(vm, 0.5, 1.5)

            if np.linalg.norm(dx_vm) + np.linalg.norm(dx_va) < tol:
                # 计算目标函数值和坏数据检测
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
    """
    电网可观测性分析（基于 i4mat_rref 的整数矩阵秩分析）。

    量测矩阵 H 的秩 rank(H) 决定系统状态的可观测维度。
    若 rank(H) = n-1（忽略 slack 相角参考），则系统完全可观测。

    采用整数矩阵行简化（RREF）判断秩亏：
        将 H 的元素缩放为整数后做高斯-约当消元，
        非零行数即为矩阵的秩。
    """

    def __init__(self, n_bus: int):
        self.n_bus = n_bus

    def check_observability(self, H: np.ndarray) -> dict:
        """
        检查量测矩阵 H 的可观测性。
        """
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
