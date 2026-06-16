#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
implicit_solver.py  ——  径向隐式求解器 & Newton-Krylov 迭代

融合种子项目:
  - 963_r83_np    : 三对角无主元 LU 分解 & 求解 (r83_np_fa, r83_np_sl)
  - 1374_unstable_ode : 非稳定系统的隐式处理 (后向 Euler → 0)

核心: 对半隐式时间步, 径向方向用三对角隐式求解,
     解决 MHD  stiff 模式 (Alfvén 波/CFL 限制).

三对角系统:  a_i x_{i-1} + b_i x_i + c_i x_{i+1} = d_i
使用 Thomas 算法 (参考 963_r83_np_fa / r83_np_sl).
"""

from __future__ import annotations
import numpy as np
from typing import Tuple, Optional
from mhd_constants import MHDConfig


class ThomasSolver:
    """
    无主元 Thomas 算法 (复刻 963_r83_np 的核心).
    三对角矩阵 A 存储在 (3, N) 数组中:
      row 0: 超对角线 (c_i, i=0..N-2)
      row 1: 主对角线 (b_i, i=0..N-1)
      row 2: 次对角线 (a_i, i=1..N-1)
    """

    @staticmethod
    def r83_np_fa(a: np.ndarray) -> Tuple[np.ndarray, int]:
        """
        三对角 LU 分解 (无主元), 返回 (a_lu, info).
        info=0 成功, info>0 在第 info 步遇到零主元.
        """
        n = a.shape[1]
        a_lu = a.copy()
        for i in range(n - 1):
            if abs(a_lu[1, i]) < 1e-30:
                return a_lu, i + 1
            a_lu[2, i] = a_lu[2, i] / a_lu[1, i]
            a_lu[1, i + 1] -= a_lu[2, i] * a_lu[0, i + 1]
        if abs(a_lu[1, n - 1]) < 1e-30:
            return a_lu, n
        return a_lu, 0

    @staticmethod
    def r83_np_sl(a_lu: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        三对角 LU 回代求解, 返回 x.
        a_lu: LU 分解结果 (3, N).
        b: 右端项 (N,).
        """
        n = len(b)
        x = b.copy()
        # 前代 (L y = b)
        for i in range(1, n):
            x[i] -= a_lu[2, i - 1] * x[i - 1]
        # 回代 (U x = y)
        x[n - 1] /= a_lu[1, n - 1]
        for i in range(n - 2, -1, -1):
            x[i] = (x[i] - a_lu[0, i + 1] * x[i + 1]) / a_lu[1, i]
        return x

    @staticmethod
    def solve_tridiag(lower: np.ndarray, diag: np.ndarray,
                       upper: np.ndarray, rhs: np.ndarray) -> np.ndarray:
        """
        便捷接口: 传入 (lower, diag, upper, rhs), 返回解 x.
        """
        n = len(diag)
        a = np.zeros((3, n))
        a[0, 1:] = upper[:-1]     # 超对角线
        a[1, :]  = diag           # 主对角线
        a[2, :-1] = lower[1:]     # 次对角线
        a_lu, info = ThomasSolver.r83_np_fa(a)
        if info != 0:
            # 降级: 加正则化
            a[1, :] += 1e-12
            a_lu, info = ThomasSolver.r83_np_fa(a)
            if info != 0:
                raise ValueError(f"三对角求解失败, info={info}")
        return ThomasSolver.r83_np_sl(a_lu, rhs)


# ============================================================
# 径向隐式时间步 (后向 Euler / Crank-Nicolson)
# ============================================================
class RadialImplicitStepper:
    """
    对径向方向做隐式时间步, 解决 Alfvén 波的 stiff 限制.
    theta, phi 方向保持显式.

    后向 Euler:
      (I - dt L_r) U^{n+1} = U^n + dt R_explicit
    其中 L_r 是径向导数算子 (三对角).
    """

    def __init__(self, cfg: MHDConfig, nr: int) -> None:
        self.cfg = cfg
        self.nr = nr
        self.solver = ThomasSolver()

    def build_implicit_matrix(self, dr: np.ndarray,
                               wave_speed: np.ndarray,
                               dt: float,
                               theta: float = 0.5
                               ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        构建 (I - theta dt L_r) 的三对角矩阵.
        theta=0: 后向 Euler, theta=0.5: Crank-Nicolson.
        wave_speed: 每个单元的最大波速 (Nr,).
        """
        Nr = self.nr
        # 径向导数: (f_{i+1} - f_{i-1}) / (2 dr_i) → 三对角
        # 扩散型: (D_{i+1/2} (f_{i+1}-f_i)/dr - D_{i-1/2} (f_i-f_{i-1})/dr) / dr
        # 这里用一阶迎风: a_i = -dt * c_i / dr_i,  etc.
        lower = np.zeros(Nr)
        diag  = np.ones(Nr)
        upper = np.zeros(Nr)

        for i in range(Nr):
            c = wave_speed[i] if i < len(wave_speed) else 0.0
            if i == 0:
                # 内边界: 只向右
                upper[i] = -theta * dt * c / (dr[i] + 1e-30)
                diag[i]  = 1.0 - upper[i]
            elif i == Nr - 1:
                # 外边界: 只向左
                lower[i] = -theta * dt * c / (dr[i - 1] + 1e-30)
                diag[i]  = 1.0 - lower[i]
            else:
                # 内部: 中心差分
                lower[i] = -theta * dt * c / (2.0 * dr[i - 1] + 1e-30)
                upper[i] = -theta * dt * c / (2.0 * dr[i] + 1e-30)
                diag[i]  = 1.0 - lower[i] - upper[i]

        return lower, diag, upper

    def step(self, U: np.ndarray, rhs_explicit: np.ndarray,
             dr: np.ndarray, wave_speed: np.ndarray,
             dt: float) -> np.ndarray:
        """
        对一个 3D 场 U(Nr, Nt, Np) 做径向隐式步.
        对每个 (j, k) 列求解一个三对角系统.
        """
        Nr, Nt, Np = U.shape
        lower, diag, upper = self.build_implicit_matrix(dr, wave_speed, dt)
        result = np.zeros_like(U)
        for j in range(Nt):
            for k in range(Np):
                col_U = U[:, j, k]
                col_rhs = rhs_explicit[:, j, k]
                # 右端项: U^n + dt * R_explicit
                b = col_U + dt * col_rhs
                x = self.solver.solve_tridiag(lower, diag, upper, b)
                result[:, j, k] = x
        return result


# ============================================================
# Newton 迭代 (用于非线性隐式步)
# ============================================================
class NewtonSolver:
    """
    Newton 迭代求解 F(U) = 0.
    融合 1374_unstable_ode 的非稳定 ODE 隐式处理.
    """

    def __init__(self, tol: float = 1e-10, maxiter: int = 30) -> None:
        self.tol = tol
        self.maxiter = maxiter

    def solve(self, F, J, U0: np.ndarray) -> Tuple[np.ndarray, int, float]:
        """
        F: 残差函数 F(U) -> residual.
        J: Jacobian 函数 J(U) -> 三对角 (lower, diag, upper).
        U0: 初始猜测.
        返回 (U_sol, niter, residual_norm).
        """
        U = U0.copy()
        for it in range(self.maxiter):
            res = F(U)
            res_norm = np.linalg.norm(res)
            if res_norm < self.tol:
                return U, it, res_norm
            lower, diag, upper = J(U)
            dU = ThomasSolver.solve_tridiag(lower, diag, upper, -res)
            U = U + dU
        return U, self.maxiter, np.linalg.norm(F(U))
