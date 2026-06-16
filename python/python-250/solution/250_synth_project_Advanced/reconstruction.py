"""
WENO5 高阶有限差分重构 (from 高阶 FD 核心).

超新星辐射流体方程 ∂U/∂t + ∂F(U)/∂r = S(U) 的数值离散
要求激波处无振荡、光滑区高精度。WENO5 (Jiang-Shu 1996)
是三阶精度的本质无振荡重构。

对一维函数 q_i (i = 0..N-1)，在界面 i+1/2 处的左/右值：
  候选模板 (3 个，每个含 3 点)：
    S_0 = {i-2, i-1, i},   S_1 = {i-1, i, i+1},   S_2 = {i, i+1, i+2}
  各模板上的二阶重构：
    q^{(0)}_{i+1/2} = (1/3) q_{i-2} - (7/6) q_{i-1} + (11/6) q_i
    q^{(1)}_{i+1/2} = -(1/6) q_{i-1} + (5/6) q_i + (1/3) q_{i+1}
    q^{(2)}_{i+1/2} = (1/3) q_i + (5/6) q_{i+1} - (1/6) q_{i+2}
  光滑指标 (Jiang-Shu):
    β_0 = (13/12)(q_{i-2} - 2 q_{i-1} + q_i)^2 + (1/4)(q_{i-2} - 4 q_{i-1} + 3 q_i)^2
    β_1 = (13/12)(q_{i-1} - 2 q_i + q_{i+1})^2 + (1/4)(q_{i-1} - q_{i+1})^2
    β_2 = (13/12)(q_i - 2 q_{i+1} + q_{i+2})^2 + (1/4)(3 q_i - 4 q_{i+1} + q_{i+2})^2
  理想权重：d_0 = 1/10, d_1 = 6/10, d_2 = 3/10
  非线性权重：α_k = d_k / (ε + β_k)^2,   ω_k = α_k / Σ α_j
  重构值：q_{i+1/2}^L = Σ ω_k q^{(k)}_{i+1/2}

右值 q_{i+1/2}^R 通过对称得到。

球坐标下的散度：
  ∇·F = (1/r^2) ∂(r^2 F_r) / ∂r
用几何守恒形式 (Ritter 2003)：
  (d/dt)(r_i^3 U_i) - (d/dt)(r_{i-1}^3 U_i) = 3 (r_{i+1/2}^2 F_{i+1/2} - r_{i-1/2}^2 F_{i-1/2})
"""
from __future__ import annotations
import numpy as np


class WENO5Reconstructor:
    """WENO5 重构器 (有限差分版本，守恒律界面通量)."""

    def __init__(self, epsilon: float = 1.0e-6):
        self.epsilon = float(epsilon)
        # 理想权重 (Jiang-Shu 1996)
        self.d = np.array([1.0 / 10.0, 6.0 / 10.0, 3.0 / 10.0], dtype=np.float64)

    def reconstruct_interface(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """在 i+1/2 界面计算左值 qL 和右值 qR.

        输入 q 长度 N，输出 qL, qR 长度 N-1 (内部界面).
        边界使用三阶外推/镜像.
        """
        N = len(q)
        # 扩展：镜像边界 (零梯度)
        q_ext = np.empty(N + 4, dtype=np.float64)
        q_ext[2:2 + N] = q
        q_ext[0] = q_ext[1] = q[0]
        q_ext[N + 2] = q_ext[N + 3] = q[-1]
        # 计算 N-1 个内部界面 i+1/2 (i = 1..N-1 物理索引)
        n_iface = N - 1
        qL = np.empty(n_iface, dtype=np.float64)
        qR = np.empty(n_iface, dtype=np.float64)
        for iface in range(n_iface):
            # 物理界面 i + 1/2，i = iface
            # 扩展数组中对应位置：左重构中心 = iface + 2 (在扩展数组)
            i = iface + 2
            # S_0: i-2, i-1, i;  S_1: i-1, i, i+1;  S_2: i, i+1, i+2
            qm2 = q_ext[i - 2]
            qm1 = q_ext[i - 1]
            q0 = q_ext[i]
            qp1 = q_ext[i + 1]
            qp2 = q_ext[i + 2]
            # 左值重构
            q0_k = np.array([
                (1.0 / 3.0) * qm2 - (7.0 / 6.0) * qm1 + (11.0 / 6.0) * q0,
                -(1.0 / 6.0) * qm1 + (5.0 / 6.0) * q0 + (1.0 / 3.0) * qp1,
                (1.0 / 3.0) * q0 + (5.0 / 6.0) * qp1 - (1.0 / 6.0) * qp2,
            ])
            beta = np.array([
                (13.0 / 12.0) * (qm2 - 2.0 * qm1 + q0) ** 2
                + 0.25 * (qm2 - 4.0 * qm1 + 3.0 * q0) ** 2,
                (13.0 / 12.0) * (qm1 - 2.0 * q0 + qp1) ** 2
                + 0.25 * (qm1 - qp1) ** 2,
                (13.0 / 12.0) * (q0 - 2.0 * qp1 + qp2) ** 2
                + 0.25 * (3.0 * q0 - 4.0 * qp1 + qp2) ** 2,
            ])
            alpha = self.d / ((self.epsilon + beta) ** 2)
            omega = alpha / alpha.sum()
            qL[iface] = float(np.sum(omega * q0_k))
            # 右值重构：镜像 (i+1 为中心, 反向)
            # S_0': i+3, i+2, i+1;  S_1': i+2, i+1, i;  S_2': i+1, i, i-1
            # 等价于把 q 反转后做左重构
            qm2_r = qp2
            qm1_r = qp1
            q0_r = q0
            qp1_r = qm1
            qp2_r = qm2
            q0_k_r = np.array([
                (1.0 / 3.0) * qm2_r - (7.0 / 6.0) * qm1_r + (11.0 / 6.0) * q0_r,
                -(1.0 / 6.0) * qm1_r + (5.0 / 6.0) * q0_r + (1.0 / 3.0) * qp1_r,
                (1.0 / 3.0) * q0_r + (5.0 / 6.0) * qp1_r - (1.0 / 6.0) * qp2_r,
            ])
            beta_r = np.array([
                (13.0 / 12.0) * (qm2_r - 2.0 * qm1_r + q0_r) ** 2
                + 0.25 * (qm2_r - 4.0 * qm1_r + 3.0 * q0_r) ** 2,
                (13.0 / 12.0) * (qm1_r - 2.0 * q0_r + qp1_r) ** 2
                + 0.25 * (qm1_r - qp1_r) ** 2,
                (13.0 / 12.0) * (q0_r - 2.0 * qp1_r + qp2_r) ** 2
                + 0.25 * (3.0 * q0_r - 4.0 * qp1_r + qp2_r) ** 2,
            ])
            alpha_r = self.d / ((self.epsilon + beta_r) ** 2)
            omega_r = alpha_r / alpha_r.sum()
            qR[iface] = float(np.sum(omega_r * q0_k_r))
        return qL, qR


def hll_flux(fL, fR, qL, qR, wave_speeds):
    """HLL 近似黎曼通量 (Harten-Lax-van Leer 1983).

    F_{i+1/2} = (S_R F_L - S_L F_R + S_L S_R (U_R - U_L)) / (S_R - S_L)
  其中 S_L, S_R 为左右波速估计 (Davis 1988):
    S_L = min(λ(U_L), λ(U_R)),  S_R = max(λ(U_L), λ(U_R))
  """
    SL, SR = wave_speeds
    denom = SR - SL
    if abs(denom) < 1.0e-30:
        return 0.5 * (fL + fR)
    return (SR * fL - SL * fR + SL * SR * (qR - qL)) / denom


def global_lax_friedrichs_flux(fL, fR, qL, qR, alpha):
    """全局 Lax-Friedrichs 通量 (Rusanov).

    F_{i+1/2} = (1/2) (F_L + F_R) - (α/2) (U_R - U_L)
  其中 α = max |λ(U)|.
  """
    return 0.5 * (fL + fR) - 0.5 * alpha * (qR - qL)
