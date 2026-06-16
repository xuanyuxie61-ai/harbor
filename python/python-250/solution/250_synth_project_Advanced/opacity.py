"""
不透明度表的切比雪夫插值 (from 591_interp_chebyshev).

超新星物质不透明度 κ(ρ,T) 来自 OPAL/Iron 数据库。在数值模拟中
通常预先以二维表形式储存，运行时做双线性/双三次插值。
这里改用**双二维切比雪夫插值**：
  - logT ∈ [logT_min, logT_max] 方向 N_T 个切比雪夫节点
  - logρ ∈ [logρ_min, logρ_max] 方向 N_ρ 个切比雪夫节点
  - 在每一个 (T,ρ) 格点储存 log κ

对 (logT, logρ) 上的函数 f, 切比雪夫插值：
  f(x) ≈ Σ_{k=0}^{N-1} c_k T_k(ξ(x)) - c_0/2
其中 ξ(x) ∈ [-1, 1] 为线性映射，T_k 为切比雪夫多项式。
系数 c_k 由离散余弦变换给出：
  c_k = (2/N) Σ_{j=0}^{N-1} f(x_j) cos(π k (j + 0.5) / N)

Kramers 不透明度 (自由-自由吸收):
  κ_ff = 4.0e25 Z(1+X) ρ T^{-3.5}  [cm^2/g]
电子散射 (Thomson):
  κ_es = 0.2 (1+X)  [cm^2/g]
总不透明度 Rosseland 平均采用加法合成：
  κ_R = κ_ff + κ_es
"""
from __future__ import annotations
import math
import numpy as np


class ChebyshevInterpolator1D:
    """一维切比雪夫插值器 (Clenshaw 递推求值)."""

    def __init__(self, a: float, b: float, n_order: int, f_values: np.ndarray):
        if n_order < 2:
            raise ValueError("切比雪夫阶数至少为 2")
        if len(f_values) != n_order:
            raise ValueError("f_values 长度必须等于 n_order")
        self.a = float(a)
        self.b = float(b)
        self.N = int(n_order)
        # 切比雪夫系数 (离散余弦变换)
        k = np.arange(self.N)
        j = np.arange(self.N)
        # T_{kj} = cos(π k (j + 0.5) / N)
        cos_mat = np.cos(np.pi * np.outer(k, j + 0.5) / self.N)
        self.coeffs = (2.0 / self.N) * cos_mat @ np.asarray(f_values, dtype=np.float64)
        self.coeffs[0] *= 0.5  # c_0/2 修正

    def x_to_xi(self, x: np.ndarray) -> np.ndarray:
        """从物理域 [a,b] 映射到 [-1, 1]."""
        return (2.0 * x - (self.a + self.b)) / (self.b - self.a)

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """Clenshaw 算法：
          b_{N+1} = b_{N+2} = 0
          b_k = 2 ξ b_{k+1} - b_{k+2} + c_k
          f(ξ) = b_0 ξ - b_1 + c_0 / 2  (已并入系数)
        标准 Clenshaw 对 T_k: f = b_0 - ξ b_1, b_{N+1}=b_{N+2}=0
        """
        xi = self.x_to_xi(np.asarray(x, dtype=np.float64))
        b_kp2 = np.zeros_like(xi)
        b_kp1 = np.zeros_like(xi)
        for k in range(self.N - 1, -1, -1):
            b_k = 2.0 * xi * b_kp1 - b_kp2 + self.coeffs[k]
            b_kp2 = b_kp1
            b_kp1 = b_k
        # 最终值 f = b_0 - ξ b_1
        # b_kp1 现在是 b_0, b_kp2 是 b_1
        return b_kp1 - xi * b_kp2


class OpacityTable:
    """双二维切比雪夫不透明度表 κ(log10(T), log10(ρ))."""

    def __init__(self, logT_range=(6.0, 11.0), logRho_range=(0.0, 13.0),
                 n_T: int = 12, n_rho: int = 10):
        self.logT_min, self.logT_max = logT_range
        self.logRho_min, self.logRho_max = logRho_range
        self.n_T = n_T
        self.n_rho = n_rho
        # 切比雪夫节点 (物理域)
        j_T = np.arange(n_T)
        j_rho = np.arange(n_rho)
        xi_T = -np.cos(np.pi * (j_T + 0.5) / n_T)
        xi_rho = -np.cos(np.pi * (j_rho + 0.5) / n_rho)
        self.logT_nodes = 0.5 * ((self.logT_max - self.logT_min) * xi_T
                                  + (self.logT_max + self.logT_min))
        self.logRho_nodes = 0.5 * ((self.logRho_max - self.logRho_min) * xi_rho
                                    + (self.logRho_max + self.logRho_min))
        # 构造 log κ 表 (Kramers + Thomson + e± 对)
        self.logkappa_grid = self._build_table()
        # 在 ρ 方向做切比雪夫插值族 (每个 logT 节点一条)
        self.interps_rho = []
        for i in range(n_T):
            self.interps_rho.append(
                ChebyshevInterpolator1D(self.logRho_min, self.logRho_max,
                                        n_rho, self.logkappa_grid[i, :])
            )
        # 外层对 logT 方向做切比雪夫插值 (对每个 logρ 节点)
        self.interps_T = []
        for j in range(n_rho):
            self.interps_T.append(
                ChebyshevInterpolator1D(self.logT_min, self.logT_max,
                                        n_T, self.logkappa_grid[:, j])
            )

    def _build_table(self) -> np.ndarray:
        """Kramers 自由-自由 + Thomson 电子散射 + e± 对修正."""
        logT_grid, logRho_grid = np.meshgrid(self.logT_nodes, self.logRho_nodes,
                                              indexing='ij')
        T = 10.0 ** logT_grid
        rho = 10.0 ** logRho_grid
        X = 0.70  # 氢丰度
        Z = 0.02  # 金属丰度
        # Kramers
        kappa_ff = 4.0e25 * Z * (1.0 + X) * rho * T ** (-3.5)
        # Thomson
        kappa_es = 0.2 * (1.0 + X) * np.ones_like(rho)
        # 光子-电子散射的相对论修正 (Suleimanov 2006)
        T_kev = C.K_BOLTZMANN * T / 1.6022e-9  # 转换为 keV
        rel_corr = 1.0 + 2.0 * T_kev / 511.0  # 一级修正
        kappa_es_corr = kappa_es * rel_corr
        # e± 对吸收 (Blinnikov 1999): κ_pair ≈ 5.0e15 T^9 ρ^{-1} (极高温主导)
        # 限制在 T > 1e9
        kappa_pair = np.where(
            T > 1.0e9,
            5.0e15 * (T / 1.0e9) ** 9.0 / np.maximum(rho, 1.0e-20),
            0.0
        )
        # Rosseland 平均近似：直接加和 (准确需用 1/κ 加权积分)
        kappa_total = kappa_ff + kappa_es_corr + kappa_pair
        logkappa = np.log10(np.maximum(kappa_total, 1.0e-30))
        return logkappa

    def evaluate(self, T: np.ndarray, rho: np.ndarray) -> np.ndarray:
        """给定 T(K), ρ(g/cm^3), 返回 κ(cm^2/g)."""
        T = np.asarray(T, dtype=np.float64)
        rho = np.asarray(rho, dtype=np.float64)
        logT = np.log10(np.maximum(T, 1.0))
        logRho = np.log10(np.maximum(rho, 1.0e-30))
        # 截断
        logT = np.clip(logT, self.logT_min, self.logT_max)
        logRho = np.clip(logRho, self.logRho_min, self.logRho_max)
        # 双插值：先沿 ρ 方向在每个 logT 节点求值，再沿 logT 方向插值
        # 简化：对每个网格点直接线性混合 (避免高阶张量积复杂性)
        # 采用张量积双切比雪夫
        xi_T = (2.0 * logT - (self.logT_min + self.logT_max)) / (self.logT_max - self.logT_min)
        xi_rho = (2.0 * logRho - (self.logRho_min + self.logRho_max)) / (self.logRho_max - self.logRho_min)
        # 简化：对每个元素做双线性插值 (在两个 logT 节点之间，每个节点用 ρ 方向切比雪夫)
        logT_clamped = np.clip(logT, self.logT_min + 1e-6, self.logT_max - 1e-6)
        idx_arr = np.searchsorted(self.logT_nodes, logT_clamped) - 1
        idx_arr = np.clip(idx_arr, 0, self.n_T - 2)
        logkappa = np.zeros_like(logT)
        for flat_i in range(logkappa.size):
            idx = int(idx_arr.flat[flat_i])
            frac = (logT_clamped.flat[flat_i] - self.logT_nodes[idx]) / max(
                self.logT_nodes[idx + 1] - self.logT_nodes[idx], 1e-30)
            frac = max(0.0, min(1.0, float(frac)))
            lr = float(logRho.flat[flat_i])
            v0 = float(self.interps_rho[idx].evaluate(lr))
            v1 = float(self.interps_rho[idx + 1].evaluate(lr))
            logkappa.flat[flat_i] = (1.0 - frac) * v0 + frac * v1
        return 10.0 ** logkappa


# 为常量引用
import constants as C  # noqa: E402
