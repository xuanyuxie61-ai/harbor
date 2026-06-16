#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
optimal_control.py  ——  Riccati 反馈控制 & 策略梯度优化 MHD 稳定性

融合种子项目:
  - 1246_paper-code-bridging-finite-and-infinite-horizon :
    耦合 Riccati 递推 & 策略梯度 Nash 均衡 → 喷流稳定性反馈控制

核心思想:
  将 MHD 稳定性控制建模为 LQR 问题:
    dU/dt = A U + B K U   (线性化 MHD + 反馈控制)
  目标: 最小化 J = sum (U^T Q U + K^T R K).
  最优反馈: K = R^{-1} B^T P,  P 满足离散 Riccati 方程.

  耦合 Riccati 递推 (多玩家):
    P_i(t) = Q_i + K_i^T R_i K_i + A_cl^T P_i(t+1) A_cl
  其中 A_cl = A - sum_j B_j K_j.
"""

from __future__ import annotations
import numpy as np
from typing import Tuple
from mhd_constants import MHDConfig


# ============================================================
# Riccati 求解器 (复刻 1246 的 compute_riccati_update)
# ============================================================
def riccati_step(A: np.ndarray, B: np.ndarray, Q: np.ndarray,
                  R: np.ndarray, P_next: np.ndarray
                  ) -> np.ndarray:
    """
    离散 Riccati 一步递推:
      P = Q + A^T P_{next} A - A^T P_{next} B (R + B^T P_{next} B)^{-1} B^T P_{next} A
    简化形式 (K 已知):
      P = Q + K^T R K + A_cl^T P_next A_cl
    """
    n = A.shape[0]
    # 计算最优增益 K = (R + B^T P B)^{-1} B^T P A
    BtP = B.T @ P_next
    BtPB = BtP @ B
    inv_term = np.linalg.inv(R + BtPB + 1e-12 * np.eye(R.shape[0]))
    K = inv_term @ BtP @ A
    A_cl = A - B @ K
    P = Q + K.T @ R @ K + A_cl.T @ P_next @ A_cl
    return P, K


def solve_discrete_riccati(A: np.ndarray, B: np.ndarray,
                             Q: np.ndarray, R: np.ndarray,
                             max_iter: int = 200, tol: float = 1e-10
                             ) -> Tuple[np.ndarray, np.ndarray]:
    """
    迭代求解离散代数 Riccati 方程 (DARE).
    返回 (P_inf, K_opt).
    """
    n = A.shape[0]
    P = Q.copy()
    K = np.zeros((R.shape[0], n))
    for it in range(max_iter):
        P_new, K_new = riccati_step(A, B, Q, R, P)
        if np.linalg.norm(P_new - P) < tol:
            return P_new, K_new
        P = P_new
        K = K_new
    return P, K


# ============================================================
# 策略梯度 (复刻 1246 的 compute_gradient_descent)
# ============================================================
def policy_gradient_step(K: np.ndarray, A: np.ndarray, B: np.ndarray,
                           Q: np.ndarray, R: np.ndarray,
                           eta: float = 0.01) -> np.ndarray:
    """
    策略梯度一步:
      K <- K - eta * grad_K J
    grad_K J = 2 (R K - B^T P A_cl) Sigma
    其中 Sigma 是状态协方差.
    简化: 用有限差分近似梯度.
    """
    n = A.shape[0]
    m = K.shape[0]
    eps = 1e-5
    grad = np.zeros_like(K)
    J0 = compute_cost(K, A, B, Q, R)
    for i in range(m):
        for j in range(n):
            K_eps = K.copy()
            K_eps[i, j] += eps
            J1 = compute_cost(K_eps, A, B, Q, R)
            grad[i, j] = (J1 - J0) / eps
    return K - eta * grad


def compute_cost(K: np.ndarray, A: np.ndarray, B: np.ndarray,
                   Q: np.ndarray, R: np.ndarray, horizon: int = 100
                   ) -> float:
    """计算给定 K 的有限时域代价."""
    n = A.shape[0]
    A_cl = A - B @ K
    P = np.zeros((n, n))
    x = np.random.randn(n)
    total_cost = 0.0
    for t in range(horizon):
        total_cost += x.T @ Q @ x + (K @ x).T @ R @ (K @ x)
        x = A_cl @ x
        if np.linalg.norm(x) > 1e10:
            return 1e20  # 发散
    return total_cost


# ============================================================
# MHD 稳定性控制器
# ============================================================
class MHDStabilityController:
    """
    对线性化 MHD 系统设计最优反馈控制, 抑制最不稳定模态.
    """

    def __init__(self, cfg: MHDConfig) -> None:
        self.cfg = cfg

    def design_controller(self, A_lin: np.ndarray, B_act: np.ndarray,
                            Q_weight: np.ndarray, R_weight: np.ndarray
                            ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        A_lin: 线性化 MHD 矩阵 (n, n).
        B_act: 控制输入矩阵 (n, m).
        Q_weight: 状态权重 (n, n).
        R_weight: 控制权重 (m, m).
        返回 (K_opt, P_inf, closed_loop_spectral_radius).
        """
        P_inf, K_opt = solve_discrete_riccati(A_lin, B_act, Q_weight, R_weight)
        A_cl = A_lin - B_act @ K_opt
        eigvals = np.linalg.eigvals(A_cl)
        rho = np.max(np.abs(eigvals))
        return K_opt, P_inf, rho
