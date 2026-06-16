# -*- coding: utf-8 -*-
"""
validation.py
=============

数值鲁棒性与完整性验证模块.

融合的种子项目:
  - 1393_vin     -> checksum / validate 逻辑 (VIN 校验)
  - 763_middle_square -> seed 状态一致性校验

本模块在 SAA 计算流程中充当"守门人":
  1. 校验输入参数合法性 (边界, NaN, 正定性等)
  2. 对 SAA 样本集做完整性校验 (checksum)
  3. 对 PDE 求解器输出做物理一致性检验
  4. 对优化结果做最优性条件近似检验
"""

import math
import hashlib
from typing import List, Tuple, Optional


class ParameterValidator:
    """参数合法性校验 (复刻 seed project 1393_vin validate)."""

    @staticmethod
    def check_positive(value: float, name: str) -> None:
        if not math.isfinite(value):
            raise ValueError("参数 %s 非有限数: %s" % (name, value))
        if value <= 0:
            raise ValueError("参数 %s 必须为正: %s" % (name, value))

    @staticmethod
    def check_nonneg(value: float, name: str) -> None:
        if not math.isfinite(value):
            raise ValueError("参数 %s 非有限数: %s" % (name, value))
        if value < 0:
            raise ValueError("参数 %s 必须非负: %s" % (name, value))

    @staticmethod
    def check_range(value: float, lo: float, hi: float, name: str) -> None:
        if value < lo or value > hi:
            raise ValueError("参数 %s = %s 超出范围 [%s, %s]" %
                             (name, value, lo, hi))

    @staticmethod
    def check_integer_ge(value: int, threshold: int, name: str) -> None:
        if not isinstance(value, int):
            raise TypeError("参数 %s 必须是整数" % name)
        if value < threshold:
            raise ValueError("参数 %s = %d < %d" % (name, value, threshold))

    @staticmethod
    def check_vector_dim(vec: List[float], expected_dim: int,
                         name: str) -> None:
        if len(vec) != expected_dim:
            raise ValueError("向量 %s 维度 %d != 期望 %d" %
                             (name, len(vec), expected_dim))
        for i, v in enumerate(vec):
            if not math.isfinite(v):
                raise ValueError("向量 %s[%d] 非有限: %s" % (name, i, v))


class SampleIntegrityChecker:
    """样本完整性校验 (VIN checksum 思想 + middle_square 状态追溯).

    对 SAA 样本集计算 "指纹" (checksum), 保证:
      - 同一种子产生的样本集可复现;
      - 样本集未被篡改;
      - 维度一致性.
    """

    @staticmethod
    def compute_checksum(xi_batch: List[List[float]],
                         precision: int = 6) -> str:
        """对样本集计算 SHA-256 指纹.

        将每个浮点数四舍五入到 precision 位, 序列化后哈希.
        这避免了浮点表示微小差异导致的哈希变化.
        """
        h = hashlib.sha256()
        for row in xi_batch:
            for v in row:
                rounded = round(v, precision)
                h.update(str(rounded).encode('utf-8'))
                h.update(b",")
            h.update(b";")
        return h.hexdigest()[:16]

    @staticmethod
    def validate_dimensions(xi_batch: List[List[float]],
                            expected_K: int) -> bool:
        """检查样本集维度一致性."""
        for i, row in enumerate(xi_batch):
            if len(row) != expected_K:
                return False
            for v in row:
                if not math.isfinite(v):
                    return False
        return True

    @staticmethod
    def validate_rng_continuity(rng_state_before: int,
                                rng_state_after: int,
                                n_draws: int) -> bool:
        """验证 RNG 状态连续性 (middle_square 风格).
        简化: 仅检查前后状态不同且为正.
        """
        if rng_state_before <= 0 or rng_state_after <= 0:
            return False
        return rng_state_before != rng_state_after


class PDESolverValidator:
    """PDE 求解器输出校验."""

    @staticmethod
    def check_boundary_conditions(u: List[List[complex]],
                                  nx: int, ny: int,
                                  tol: float = 1e-8) -> bool:
        """检查 Dirichlet BC: u = 0 on boundary."""
        for i in range(nx):
            if abs(u[0][i]) > tol or abs(u[ny - 1][i]) > tol:
                return False
        for j in range(ny):
            if abs(u[j][0]) > tol or abs(u[j][nx - 1]) > tol:
                return False
        return True

    @staticmethod
    def check_solution_finiteness(u: List[List[complex]]) -> bool:
        """检查解的所有内部点有限."""
        ny = len(u)
        if ny == 0:
            return False
        nx = len(u[0])
        for j in range(ny):
            for i in range(nx):
                val = u[j][i]
                if not math.isfinite(val.real) or not math.isfinite(val.imag):
                    return False
        return True

    @staticmethod
    def check_objective_physical(J: float) -> bool:
        """目标函数物理合理性: J >= 0 (能量)."""
        return math.isfinite(J) and J >= 0.0

    @staticmethod
    def check_residual_bound(residual_norm: float,
                             rhs_norm: float,
                             tol: float = 0.1) -> bool:
        """相对残差检验: ||r|| / ||b|| < tol."""
        if rhs_norm < 1e-14:
            return residual_norm < tol
        return residual_norm / rhs_norm < tol


class OptimalityChecker:
    """优化结果最优性条件近似检验."""

    @staticmethod
    def check_stationarity(grad: List[float], tol: float = 0.1) -> bool:
        """近似平稳性: ||grad|| < tol."""
        g_norm = math.sqrt(sum(g * g for g in grad))
        return g_norm < tol

    @staticmethod
    def check_feasibility(x: List[float],
                          lower: float, upper: float) -> bool:
        """可行性: x in [lower, upper]^dim."""
        for xi in x:
            if xi < lower - 1e-8 or xi > upper + 1e-8:
                return False
        return True

    @staticmethod
    def check_improvement(f_history: List[float],
                          window: int = 5) -> bool:
        """检查最近 window 步是否有改进.
        用线性回归斜率判断.
        """
        if len(f_history) < window:
            return True
        recent = f_history[-window:]
        n = len(recent)
        x_mean = (n - 1) / 2.0
        y_mean = sum(recent) / n
        num = sum((i - x_mean) * (recent[i] - y_mean) for i in range(n))
        den = sum((i - x_mean) ** 2 for i in range(n))
        if abs(den) < 1e-14:
            return False
        slope = num / den
        return slope < -1e-6  # 仍在下降

    @staticmethod
    def compute_kkt_violation(x: List[float], grad: List[float],
                              lower: float, upper: float) -> float:
        """KKT 违反量 (box 约束):
            v = sum_k max(0, -grad_k if x_k > lower else 0)
              + sum_k max(0, grad_k if x_k < upper else 0)
              + sum_k |grad_k| if lower < x_k < upper
        简化版本.
        """
        v = 0.0
        for k in range(len(x)):
            if x[k] <= lower + 1e-8:
                v += max(0.0, -grad[k])
            elif x[k] >= upper - 1e-8:
                v += max(0.0, grad[k])
            else:
                v += abs(grad[k])
        return v


def validate_all_saa_inputs(nx: int, ny: int, Lx: float, Ly: float,
                            k_wave: float, damping: float,
                            K_kl: int, sigma: float, ell: float,
                            N_samples: int, batch_size: int) -> None:
    """集中校验所有 SAA 输入参数."""
    pv = ParameterValidator()
    pv.check_integer_ge(nx, 5, "nx")
    pv.check_integer_ge(ny, 5, "ny")
    pv.check_positive(Lx, "Lx")
    pv.check_positive(Ly, "Ly")
    pv.check_positive(k_wave, "k_wave")
    pv.check_nonneg(damping, "damping")
    pv.check_integer_ge(K_kl, 1, "K_kl")
    pv.check_nonneg(sigma, "sigma")
    pv.check_positive(ell, "ell")
    pv.check_integer_ge(N_samples, 1, "N_samples")
    pv.check_integer_ge(batch_size, 1, "batch_size")
