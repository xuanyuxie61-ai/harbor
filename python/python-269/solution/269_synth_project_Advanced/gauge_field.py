"""
gauge_field.py — 矢量势与磁场规范
============================================================

量子霍尔效应中磁场的引入需要选择规范 (gauge).
不同规范在数学上等价, 但数值实现各有优势.

本模块实现三种常用规范:

1. Landau 规范: A = (0, Bx, 0)
   - 平移不变性沿 y 方向
   - 波函数形式: ψ(x,y) = e^{iky} · φ(x)
   - 动量 k_y 是好量子数

2. 对称规范: A = (-By/2, Bx/2, 0)
   - 旋转对称性
   - 角动量是好量子数
   - 朗道能级波函数有确定的轨道角动量

3. Hofstadter 规范 (六角格点):
   - A_y(x) = Bx (Landau-like)
   - Peierls 相位: φ_{ij} = B · (x_i + x_j)/2 · (y_j - y_i)

规范变换: A → A + ∇χ, ψ → ψ·exp(iχ)
物理可观测量不依赖于规范选择 (规范不变性).

参考文献:
    [1] Harper, P. G. Proc. Phys. Soc. A 68, 874 (1955)
    [2] Hofstadter, D. R. Phys. Rev. B 14, 2239 (1976)
"""

import numpy as np
from typing import Callable, Tuple


class GaugeField:
    """矢量势的基类

    矢量势 A(r) 与磁场的关系: B = ∇ × A
    在二维系统中, B = (0, 0, B_z), 所以:
        B_z = ∂_x A_y - ∂_y A_x
    """
    def __init__(self, B: float):
        self.B = B
        self.name = "base"

    def A(self, x: float, y: float) -> Tuple[float, float]:
        raise NotImplementedError

    def curl_A(self, x: float, y: float) -> float:
        """通过有限差分计算 ∇×A, 验证规范的正确性"""
        eps = 1e-7
        Ax_py, _ = self.A(x, y + eps)
        Ax_my, _ = self.A(x, y - eps)
        _, Ay_px = self.A(x + eps, y)
        _, Ay_mx = self.A(x - eps, y)
        dAx_dy = (Ax_py - Ax_my) / (2 * eps)
        dAy_dx = (Ay_px - Ay_mx) / (2 * eps)
        return dAy_dx - dAx_dy


class LandauGauge(GaugeField):
    """Landau 规范: A = (0, Bx, 0)

    性质:
    - A_x = 0, A_y = Bx
    - ∇·A = ∂_x(0) + ∂_y(Bx) = 0 (Coulomb 规范)
    - 沿 y 方向平移不变: H 与 p_y 对易

    Peierls 相位 (格点模型):
        x-方向: θ_{i→i+ŷ} = 0 (因为 A_x = 0)
        y-方向: θ_{j→j+ŷ} = B·x_j · a_y

    在有限差分中, 矢势 A_y = Bx 直接出现在交叉项 -2i·A_y·∂_y 中.
    """
    def __init__(self, B: float):
        super().__init__(B)
        self.name = "Landau"

    def A(self, x: float, y: float) -> Tuple[float, float]:
        return (0.0, self.B * x)

    def peierls_phase_x(self, x: float, y: float, dx: float) -> complex:
        """x-方向跃迁的 Peierls 相位
        ∫_{(x,y)}^{(x+dx,y)} A·dl = ∫ 0 dx' = 0
        """
        return 1.0 + 0.0j

    def peierls_phase_y(self, x: float, y: float, dy: float) -> complex:
        """y-方向跃迁的 Peierls 相位
        ∫_{(x,y)}^{(x,y+dy)} A·dl = ∫ Bx dy' = Bx·dy
        """
        return np.exp(1j * self.B * x * dy)


class SymmetricGauge(GaugeField):
    """对称规范: A = (-By/2, Bx/2, 0)

    性质:
    - A_x = -By/2, A_y = Bx/2
    - ∇·A = ∂_x(-By/2) + ∂_y(Bx/2) = 0 (Coulomb 规范)
    - 旋转对称: [H, L_z] = 0
    - 朗道能级轨道角动量: l_z = -n (对于最低朗道能级)

    Peierls 相位:
        θ_{r→r'} = (B/2)(x·y' - y·x') (叉积形式)

    对称规范下的波函数:
        ψ_{n,m}(r,θ) ∝ r^|m| · L_n^|m|(r²/2l_B²) · exp(-r²/4l_B²) · exp(imθ)
    其中 L_n^m 是关联拉盖尔多项式.
    """
    def __init__(self, B: float):
        super().__init__(B)
        self.name = "symmetric"

    def A(self, x: float, y: float) -> Tuple[float, float]:
        return (-self.B * y / 2.0, self.B * x / 2.0)

    def peierls_phase(self, x1: float, y1: float,
                      x2: float, y2: float) -> complex:
        """对称规范的 Peierls 相位 (直线路径积分)
        ∫A·dl = (B/2)(x₁y₂ - y₁x₂)
        """
        return np.exp(1j * self.B / 2.0 * (x1 * y2 - y1 * x2))


class HofstadterGauge(GaugeField):
    """Hofstadter 规范 (用于六角格点): A = (0, Bx, 0)

    与 Landau 规范相同, 但在六角格点上使用离散的 Peierls 相位.
    对于六角格点的三个近邻键方向:
        δ₁ = (1, 0),    δ₂ = (-1/2, √3/2),   δ₃ = (1/2, √3/2)

    Peierls 相位 = B · x_mid · Δy
    其中 x_mid = (x_i + x_j)/2, Δy = y_j - y_i
    """
    def __init__(self, B: float):
        super().__init__(B)
        self.name = "Hofstadter"
        self.honeycomb_vectors = np.array([
            [1.0, 0.0],
            [-0.5, np.sqrt(3.0) / 2.0],
            [0.5, np.sqrt(3.0) / 2.0]
        ])

    def A(self, x: float, y: float) -> Tuple[float, float]:
        return (0.0, self.B * x)

    def peierls_honeycomb(self, r_i: np.ndarray, r_j: np.ndarray) -> complex:
        """六角格点近邻跃迁的 Peierls 相位

        使用对称积分路径 (中点规则):
            θ_{ij} = B · (x_i + x_j)/2 · (y_j - y_i)
        这是 ∫A·dl 的一阶近似, 对于直线跳跃精确.
        """
        x_mid = (r_i[0] + r_j[0]) / 2.0
        dy = r_j[1] - r_i[1]
        return np.exp(1j * self.B * x_mid * dy)

    def flux_per_plaquette(self, lattice_constant: float = 1.0) -> float:
        """每个六角形元胞的磁通量 Φ (单位: 磁通量子 Φ₀ = 2π)

        对于 Hofstadter 问题, 关键参数是:
            φ = Φ/Φ₀ = B·A_hex/(2π)
        其中 A_hex = (3√3/2)·a² 是六角形面积.

        当 φ = p/q (有理数) 时, 能谱分裂为 q 个子带.
        """
        A_hex = 3.0 * np.sqrt(3.0) / 2.0 * lattice_constant ** 2
        return self.B * A_hex / (2.0 * np.pi)


def gauge_transform(psi: np.ndarray, chi: Callable, x_grid: np.ndarray,
                    y_grid: np.ndarray) -> np.ndarray:
    """规范变换: ψ → ψ' = ψ·exp(iχ(r))

    规范变换保持所有物理可观测量不变:
        A' = A + ∇χ
        ψ' = ψ · exp(iχ)
        ⟨ψ'|Ô'|ψ'⟩ = ⟨ψ|Ô|ψ⟩

    Args:
        psi: 原始波函数 (一维展开)
        chi: 规范函数 χ(x,y)
        x_grid: x 坐标网格
        y_grid: y 坐标网格
    Returns:
        变换后的波函数
    """
    phase = np.zeros_like(psi, dtype=complex)
    for idx, (xi, yi) in enumerate(zip(x_grid, y_grid)):
        phase[idx] = np.exp(1j * chi(xi, yi))
    return psi * phase


def verify_gauge_invariance(H1: np.ndarray, H2: np.ndarray,
                            U: np.ndarray, tol: float = 1e-10) -> bool:
    """验证两个哈密顿量通过规范变换等价: H₂ = U†H₁U

    Args:
        H1: 原始哈密顿量
        H2: 变换后哈密顿量
        U: 规范变换矩阵 (对角幺正矩阵)
        tol: 容差
    Returns:
        是否等价
    """
    U_dag = U.conj().T
    H2_computed = U_dag @ H1 @ U
    error = np.linalg.norm(H2 - H2_computed, 'fro')
    return error < tol
