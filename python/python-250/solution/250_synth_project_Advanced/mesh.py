"""
球对称拉格朗日网格.

径向 Lagrangian 质量坐标 m_i = (4/3) pi r_i^3 rho_i，但采用
Eulerian 半径 r_i 作为基本几何变量。网格质量坐标守恒，
而 r_i 随演化更新：
    dr_i/dt = v_i,
    dm_i = 4 pi r_i^2 rho_i dr_i   (壳层质量).

超新星爆发中，激波从内向外传播，激波前后的网格尺度比
可达 1e3；因此需采用几何递进网格：
    r_{i+1}/r_i = q  (q > 1).

稳定性要求 Courant 条件:
    dt <= C_CFL * min_i (dr_i / (|v_i| + c_s,i)).
"""
from __future__ import annotations
import math
import numpy as np

import constants as C


class SphericalMesh:
    """一维球对称拉格朗日网格."""

    def __init__(self, n_cells: int, r_inner: float, r_outer: float,
                 q_ratio: float = 1.02, cfl: float = 0.4):
        if n_cells < 4:
            raise ValueError("n_cells must be >= 4 for WENO5 stencils")
        if r_inner <= 0 or r_outer <= r_inner:
            raise ValueError("invalid radial range")
        if q_ratio <= 0.999 or q_ratio >= 1.2:
            raise ValueError("q_ratio should be in (0.999, 1.2) for stability")

        self.n_cells = int(n_cells)
        self.r_inner = float(r_inner)
        self.r_outer = float(r_outer)
        self.q_ratio = float(q_ratio)
        self.cfl = float(cfl)

        # 节点数 = n_cells + 1 (含两个边界)
        n_node = n_cells + 1
        if abs(q_ratio - 1.0) < 1.0e-10:
            # 均匀网格
            r_nodes = np.linspace(r_inner, r_outer, n_node)
        else:
            # 几何递进
            # r_i = r_inner * q^i ; 令 r_{n} = r_outer 求步长
            log_q = math.log(q_ratio)
            # r_i = r_inner * exp(i * log_q), i = 0..n_cells
            # 调整使末端精确等于 r_outer
            idx = np.arange(n_node, dtype=np.float64)
            log_r = np.log(r_inner) + idx * (math.log(r_outer) - math.log(r_inner)) / (n_node - 1)
            r_nodes = np.exp(log_r)
            # 轻微几何拉伸叠加，模拟真实超新星网格
            stretch = 1.0 + 0.002 * (q_ratio - 1.0) * idx
            r_nodes = r_nodes * stretch
            # 重新归一化使端点精确
            r_nodes *= r_outer / r_nodes[-1]

        self.r_nodes = r_nodes                       # 节点半径 (n_node,)
        self.r_centers = 0.5 * (r_nodes[:-1] + r_nodes[1:])   # 单元中心
        self.dr = np.diff(r_nodes)                   # 单元宽度 (n_cells,)
        self.n_cells = len(self.dr)
        # 预计算体积
        self.volumes = (4.0 / 3.0) * math.pi * (r_nodes[1:] ** 3 - r_nodes[:-1] ** 3)
        # 表面积
        self.area_inner = 4.0 * math.pi * r_nodes[:-1] ** 2
        self.area_outer = 4.0 * math.pi * r_nodes[1:] ** 2
        # 单元中心间距 (用于有限差分)
        self.dr_centers = 0.5 * (self.dr[:-1] + self.dr[1:])

    def courant_dt(self, v: np.ndarray, cs: np.ndarray) -> float:
        """Courant-Friedrichs-Lewy 时间步限制.

        dt_CFL = C_CFL * min_i ( dr_i / (|v_i| + c_s,i) )
        对辐射扩散还需考虑辐射声速 c_r = c / sqrt(3).
        """
        v_abs = np.abs(v) + 1.0e-30
        c_rad = C.C_LIGHT / math.sqrt(3.0)
        signal = v_abs + cs + c_rad * 0.0  # 显式流体部分不含辐射
        signal = np.maximum(signal, 1.0e-10)
        dt = self.cfl * float(np.min(self.dr / signal))
        return max(dt, 1.0e-14)

    def shock_radius_index(self, rho: np.ndarray, p: np.ndarray,
                           gamma: float = 5.0 / 3.0,
                           threshold: float = 2.5) -> int:
        """定位激波位置：熵梯度跳跃最大的单元.

        比熵 s = p / rho^gamma, 激波满足 ds/dr >> 0 (向外).
        返回满足 d(ln s)/d(ln r) > threshold 的最外层索引.
        """
        s = p / np.maximum(rho, C.TINY_RHO) ** gamma
        ln_s = np.log(np.maximum(s, C.TINY_P))
        ln_r = np.log(np.maximum(self.r_centers, 1.0))
        # 中心差分
        grad = np.gradient(ln_s, ln_r)
        candidates = np.where(grad > threshold)[0]
        if len(candidates) == 0:
            return self.n_cells // 2
        # 取最外层的激波 (超新星主激波)
        return int(candidates[-1])

    def shock_radius(self, rho: np.ndarray, p: np.ndarray,
                     gamma: float = 5.0 / 3.0,
                     threshold: float = 2.5) -> float:
        idx = self.shock_radius_index(rho, p, gamma, threshold)
        return float(self.r_centers[idx])
