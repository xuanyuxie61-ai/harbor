"""
moment_tensor.py
震源矩张量表示与辐射花样分析模块

原项目映射:
    1381_vandermonde -> Vandermonde 矩阵用于多项式插值辐射花样
    132_caesar       -> 循环置换用于矩张量主轴坐标系旋转

微地震事件的震源机制由矩张量 M_{ij} 完全描述。
本模块实现矩张量的参数化、坐标旋转（Caesar 型循环置换模拟主轴轮换）、
以及基于 Vandermonde 多项式插值的辐射花样拟合。

核心公式:
1. 矩张量基本形式:
   M = μ A [ s n^T + n s^T ]
   其中 μ 为剪切模量，A 为破裂面积，
   s 为滑移方向单位向量，n 为破裂面法向单位向量。

2. 矩张量特征值分解:
   M = Q Λ Q^T
   Λ = diag(λ1, λ2, λ3)，通常排序 λ1 >= λ2 >= λ3。

3. 地震矩与矩震级:
   M_0 = (1/√2) sqrt(Σ_{i,j} M_{ij}²) = (1/√2) ||M||_F
   M_w = (2/3) log10(M_0) - 6.07  (Hanks-Kanamori)

4. 辐射花样（远场 P 波振幅）:
   A_P(θ,φ) = γ_i γ_j M_{ij}
   其中 γ = [sinθ cosφ, sinθ sinφ, cosθ] 为射线方向。

5. 利用 Vandermonde 矩阵进行辐射花样的多项式插值:
   给定方位角样本 {φ_k} 和振幅样本 {A_k}，构造线性系统:
   V a = A
   V_{k,j} = φ_k^{j-1}  (Vandermonde)
   解出系数 a 后，可重构任意方位角上的辐射花样估计。

6. 坐标循环置换（Caesar 型旋转）:
   对主轴坐标系进行 (x,y,z) -> (y,z,x) 的循环置换，
   等价于将矩张量特征向量矩阵右乘置换矩阵 P_c:
   M' = Q P_c Λ P_c^T Q^T
   用于研究不同主轴假设下的震源机制分类。
"""

import numpy as np
from typing import Tuple


def vandermonde_matrix(n: int, x: np.ndarray) -> np.ndarray:
    """
    构造 Vandermonde 矩阵 V_{ij} = x_j^{i-1}。

    参数:
        n: 矩阵阶数。
        x: 节点向量，长度 n。

    返回:
        V: (n,n) Vandermonde 矩阵。
    """
    x = np.asarray(x, dtype=float)
    if x.size != n:
        raise ValueError("x 长度必须等于 n")
    V = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == 0 and abs(x[j]) < 1.0e-14:
                V[i, j] = 1.0
            else:
                V[i, j] = x[j] ** i
    return V


def caesar_cycle_matrix_3d() -> np.ndarray:
    """
    三维循环置换矩阵 P，实现 (x,y,z) -> (y,z,x)。

    P = [[0, 1, 0],
         [0, 0, 1],
         [1, 0, 0]]
    """
    return np.array([[0.0, 1.0, 0.0],
                     [0.0, 0.0, 1.0],
                     [1.0, 0.0, 0.0]])


def apply_caesar_rotation(M: np.ndarray, k: int = 1) -> np.ndarray:
    """
    对矩张量应用 k 次 Caesar 型循环置换。

    公式:
        M' = P^k M (P^T)^k
    其中 P 为三维循环置换矩阵。

    参数:
        M: (3,3) 矩张量。
        k: 置换次数，可为正、负或零。

    返回:
        置换后的矩张量。
    """
    if M.shape != (3, 3):
        raise ValueError("M 必须是 3x3 矩阵")
    P = caesar_cycle_matrix_3d()
    # 计算 P^k
    Pk = np.linalg.matrix_power(P, k % 3)
    return Pk @ M @ Pk.T


class MomentTensor:
    """
    震源矩张量对象，支持多种参数化方式。
    """

    def __init__(self, M: np.ndarray):
        M = np.asarray(M, dtype=float)
        if M.shape != (3, 3):
            raise ValueError("矩张量必须是 3x3 矩阵")
        # TODO Hole 3a: 矩张量必须是对称的，请实现对称化
        self.M = M  # 占位，需补充对称化

    @classmethod
    def from_strike_dip_rake(cls, strike_deg: float, dip_deg: float,
                              rake_deg: float, M0: float = 1.0e12):
        """
        由走向-倾角-滑动角（strike/dip/rake）构造双力偶矩张量。

        公式（Aki & Richards, 2002）:
            s = [cos(λ) cos(δ) cos(σ) + sin(λ) sin(σ),
                 cos(λ) cos(δ) sin(σ) - sin(λ) cos(σ),
                 -cos(λ) sin(δ)]
            n = [-sin(δ) cos(σ), -sin(δ) sin(σ), -cos(δ)]
            M = M0 (s n^T + n s^T)
        """
        sigma = np.deg2rad(strike_deg)
        delta = np.deg2rad(dip_deg)
        lam = np.deg2rad(rake_deg)

        s = np.array([
            np.cos(lam) * np.cos(delta) * np.cos(sigma) + np.sin(lam) * np.sin(sigma),
            np.cos(lam) * np.cos(delta) * np.sin(sigma) - np.sin(lam) * np.cos(sigma),
            -np.cos(lam) * np.sin(delta)
        ])
        n = np.array([
            -np.sin(delta) * np.cos(sigma),
            -np.sin(delta) * np.sin(sigma),
            -np.cos(delta)
        ])
        M = M0 * (np.outer(s, n) + np.outer(n, s))
        return cls(M)

    @property
    def eigenvalues(self) -> np.ndarray:
        """返回排序后的特征值（降序）。"""
        w = np.linalg.eigvalsh(self.M)
        return np.sort(w)[::-1]

    @property
    def seismic_moment(self) -> float:
        """
        地震矩 M_0 = (1/√2) sqrt(Σ M_{ij}²)。
        """
        # TODO Hole 3b: 地震矩 M_0 = (1/√2) sqrt(Σ M_{ij}²)
        raise NotImplementedError("Hole 3: 请实现地震矩公式")

    @property
    def moment_magnitude(self) -> float:
        """
        矩震级 M_w = (2/3) log10(M_0) - 6.07。
        """
        M0 = self.seismic_moment
        if M0 <= 0:
            return -np.inf
        return (2.0 / 3.0) * np.log10(M0) - 6.07

    def radiation_pattern_p(self, theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
        """
        计算远场 P 波辐射花样 A_P(θ,φ) = γ_i γ_j M_{ij}。

        参数:
            theta: 极角数组 (rad)。
            phi: 方位角数组 (rad)。

        返回:
            A_P 数组。
        """
        theta = np.asarray(theta)
        phi = np.asarray(phi)
        gamma1 = np.sin(theta) * np.cos(phi)
        gamma2 = np.sin(theta) * np.sin(phi)
        gamma3 = np.cos(theta)
        # γ_i γ_j M_{ij}
        A = (gamma1 ** 2 * self.M[0, 0]
             + gamma2 ** 2 * self.M[1, 1]
             + gamma3 ** 2 * self.M[2, 2]
             + 2.0 * gamma1 * gamma2 * self.M[0, 1]
             + 2.0 * gamma1 * gamma3 * self.M[0, 2]
             + 2.0 * gamma2 * gamma3 * self.M[1, 2])
        return A

    def radiation_pattern_s(self, theta: np.ndarray, phi: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算远场 S 波辐射花样的两个正交分量（SV, SH）。

        公式:
            A_SV = γ_k M_{ik} - γ_i (γ_j γ_k M_{jk})
            在球坐标下投影。
        """
        theta = np.asarray(theta)
        phi = np.asarray(phi)
        gamma1 = np.sin(theta) * np.cos(phi)
        gamma2 = np.sin(theta) * np.sin(phi)
        gamma3 = np.cos(theta)

        # M @ gamma
        Mgk = self.M @ np.array([gamma1, gamma2, gamma3])
        # P 波投影
        Mproj = (gamma1 * Mgk[0] + gamma2 * Mgk[1] + gamma3 * Mgk[2])
        # S 波矢量 = Mgk - gamma * Mproj
        S1 = Mgk[0] - gamma1 * Mproj
        S2 = Mgk[1] - gamma2 * Mproj
        S3 = Mgk[2] - gamma3 * Mproj

        # 投影到 SV (theta 增加方向) 和 SH (phi 增加方向)
        # e_theta = [cosθ cosφ, cosθ sinφ, -sinθ]
        # e_phi   = [-sinφ, cosφ, 0]
        e_th1 = np.cos(theta) * np.cos(phi)
        e_th2 = np.cos(theta) * np.sin(phi)
        e_th3 = -np.sin(theta)
        e_ph1 = -np.sin(phi)
        e_ph2 = np.cos(phi)
        e_ph3 = 0.0

        Asv = S1 * e_th1 + S2 * e_th2 + S3 * e_th3
        Ash = S1 * e_ph1 + S2 * e_ph2 + S3 * e_ph3
        return Asv, Ash

    def interpolate_radiation_vandermonde(self, n_samples: int = 8) -> np.ndarray:
        """
        使用 Vandermonde 插值在方位角方向拟合辐射花样。

        在固定极角 θ=π/2（水平面）下，对 φ∈[0,2π) 采样，
        构造 Vandermonde 系统求解多项式系数。

        返回:
            poly_coeffs: 多项式系数 [a0, a1, ..., a_{n-1}]。
        """
        phi_samples = np.linspace(0.0, 2.0 * np.pi, n_samples, endpoint=False)
        theta_fixed = np.full_like(phi_samples, np.pi / 2.0)
        A_samples = self.radiation_pattern_p(theta_fixed, phi_samples)

        # 构造 Vandermonde 矩阵（注意使用 cos/sin 基或单项式基）
        # 这里使用单项式基: 1, φ, φ^2, ..., φ^{n-1}
        V = vandermonde_matrix(n_samples, phi_samples)
        # 用最小二乘求解（Vandermonde 通常病态，加正则化）
        coeffs = np.linalg.lstsq(V, A_samples, rcond=None)[0]
        return coeffs
