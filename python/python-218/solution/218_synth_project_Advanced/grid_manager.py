"""
grid_manager.py
===============
网格管理与多网格 VI 求解器框架.

数学背景
--------
在 PDE 约束的 VI 中 (如障碍问题), 解的精度依赖于网格分辨率.
多网格方法 (Multigrid) 通过在不同分辨率的网格间传递信息,
实现 O(N) 复杂度的求解 (N = 总未知数).

V-循环多网格:
    1. 在细网格上预光滑 ν_1 次
    2. 限制残差到粗网格
    3. 在粗网格上求解 (或递归)
    4. 延拓校正到细网格
    5. 在细网格上后光滑 ν_2 次

限制算子 (restriction):  R = 平均插值 (细 → 粗)
延拓算子 (prolongation): P = 线性插值 (粗 → 细)

网格层次: Ω_h ⊃ Ω_{2h} ⊃ Ω_{4h} ⊃ ... ⊃ Ω_H

在 VI 中的应用:
    多网格投影方法 (MGP):
        在每层网格上求解 VI 的子问题,
        通过活动集传递保证跨层一致性.

作者: DA synthesis project
"""

from __future__ import annotations

import numpy as np
from typing import List, Tuple, Optional, Callable
from scipy import sparse


class Grid1D:
    """
    一维均匀网格.

    网格点: x_i = a + i·h, i = 0, ..., N+1
    内部点: x_1, ..., x_N (边界 x_0, x_{N+1} 由 BC 给出)
    网格间距: h = (b - a) / (N + 1)
    """

    def __init__(self, a: float, b: float, n_interior: int):
        self.a = a
        self.b = b
        self.n_interior = n_interior
        self.h = (b - a) / (n_interior + 1)
        self.n_total = n_interior + 2
        self.nodes = np.linspace(a, b, self.n_total)
        self.interior_nodes = self.nodes[1:-1]

    def coarsen(self) -> 'Grid1D':
        """生成粗化网格 (隔点取样)."""
        n_coarse = (self.n_interior - 1) // 2
        if n_coarse < 1:
            n_coarse = 1
        return Grid1D(self.a, self.b, n_coarse)

    def restrict_vector(self, v_fine: np.ndarray) -> np.ndarray:
        """
        限制算子: 细网格 → 粗网格 (全权重).

        R_{ij} = 1/4 if j = 2i-1 or 2i+1
               = 1/2 if j = 2i
               = 0   otherwise
        """
        n_c = (self.n_interior - 1) // 2
        if n_c < 1:
            n_c = 1
        v_coarse = np.zeros(n_c)
        for i in range(n_c):
            j_fine = 2 * i + 1  # 0-indexed in interior
            if j_fine < len(v_fine):
                v_coarse[i] = 0.5 * v_fine[j_fine]
                if j_fine > 0:
                    v_coarse[i] += 0.25 * v_fine[j_fine - 1]
                if j_fine + 1 < len(v_fine):
                    v_coarse[i] += 0.25 * v_fine[j_fine + 1]
        return v_coarse

    def prolongate_vector(self, v_coarse: np.ndarray) -> np.ndarray:
        """
        延拓算子: 粗网格 → 细网格 (线性插值).

        P: 粗网格点直接传递, 细网格新点取相邻粗点的平均.
        """
        n_fine = self.n_interior
        v_fine = np.zeros(n_fine)
        n_c = len(v_coarse)
        for i in range(n_fine):
            if i % 2 == 0:
                # 偶数索引: 对应粗网格点
                j_coarse = i // 2
                if j_coarse < n_c:
                    v_fine[i] = v_coarse[j_coarse]
            else:
                # 奇数索引: 相邻粗点的平均
                j_left = (i - 1) // 2
                j_right = (i + 1) // 2
                left_val = v_coarse[j_left] if j_left < n_c else 0.0
                right_val = v_coarse[j_right] if j_right < n_c else 0.0
                v_fine[i] = 0.5 * (left_val + right_val)
        return v_fine


class Grid2D:
    """
    二维均匀网格.

    内部点: (nx × ny), 边界由 Dirichlet BC 处理.
    编号: (i,j) → i + j*nx
    """

    def __init__(self, a: float, b: float, c: float, d: float,
                 nx: int, ny: int):
        self.a, self.b = a, b
        self.c, self.d = c, d
        self.nx = nx
        self.ny = ny
        self.hx = (b - a) / (nx + 1)
        self.hy = (d - c) / (ny + 1)
        self.n_interior = nx * ny

        # 生成网格点
        x = np.linspace(a, b, nx + 2)
        y = np.linspace(c, d, ny + 2)
        self.xx, self.yy = np.meshgrid(x, y, indexing='ij')
        self.positions = np.column_stack([
            self.xx[1:-1, 1:-1].flatten(),
            self.yy[1:-1, 1:-1].flatten(),
        ])

    def coarsen(self) -> 'Grid2D':
        """生成粗化网格."""
        nx_c = max(1, (self.nx - 1) // 2)
        ny_c = max(1, (self.ny - 1) // 2)
        return Grid2D(self.a, self.b, self.c, self.d, nx_c, ny_c)


class MultigridHierarchy:
    """
    多网格层次结构.

    构建从细到粗的网格序列, 以及相应的限制/延拓算子.
    """

    def __init__(self, fine_grid: Grid1D, n_levels: int = 3):
        self.grids: List[Grid1D] = [fine_grid]
        current = fine_grid
        for _ in range(n_levels - 1):
            coarse = current.coarsen()
            self.grids.append(coarse)
            current = coarse
        self.n_levels = len(self.grids)

    def v_cycle(
        self,
        A_list: List[np.ndarray],
        b_list: List[np.ndarray],
        x_list: List[np.ndarray],
        nu1: int = 2,
        nu2: int = 2,
    ) -> List[np.ndarray]:
        """
        V-循环多网格迭代.

        Parameters
        ----------
        A_list : list of ndarray
            各层网格上的算子矩阵
        b_list : list of ndarray
            各层网格上的右端项
        x_list : list of ndarray
            各层网格上的初始猜测 (将被修改)
        nu1, nu2 : int
            预/后光滑次数

        Returns
        -------
        x_list : 更新后的各层解
        """
        return self._v_cycle_recursive(A_list, b_list, x_list, 0, nu1, nu2)

    def _v_cycle_recursive(
        self, A_list, b_list, x_list, level, nu1, nu2
    ) -> List[np.ndarray]:
        """递归 V-循环."""
        if level == self.n_levels - 1:
            # 最粗层: 直接求解
            try:
                x_list[level] = np.linalg.solve(A_list[level], b_list[level])
            except np.linalg.LinAlgError:
                pass
            return x_list

        # 预光滑
        for _ in range(nu1):
            x_list[level] = self._gauss_seidel_step(A_list[level], b_list[level], x_list[level])

        # 计算残差
        r = b_list[level] - A_list[level] @ x_list[level]

        # 限制到粗网格
        grid_fine = self.grids[level]
        r_coarse = grid_fine.restrict_vector(r)

        # 粗网格校正 (递归)
        e_coarse = np.zeros(len(r_coarse))
        b_list[level + 1] = r_coarse
        x_list[level + 1] = e_coarse
        x_list = self._v_cycle_recursive(A_list, b_list, x_list, level + 1, nu1, nu2)
        e_coarse = x_list[level + 1]

        # 延拓到细网格
        e_fine = grid_fine.prolongate_vector(e_coarse)
        x_list[level] = x_list[level] + e_fine

        # 后光滑
        for _ in range(nu2):
            x_list[level] = self._gauss_seidel_step(A_list[level], b_list[level], x_list[level])

        return x_list

    def _gauss_seidel_step(self, A: np.ndarray, b: np.ndarray, x: np.ndarray) -> np.ndarray:
        """一次 Gauss-Seidel 光滑步."""
        n = len(x)
        x_new = x.copy()
        for i in range(n):
            s = b[i]
            for j in range(n):
                if j != i:
                    s -= A[i, j] * x_new[j]
            if abs(A[i, i]) > 1e-14:
                x_new[i] = s / A[i, i]
        return x_new


class GridConvergenceAnalyzer:
    """
    网格收敛性分析器.

    通过在不同分辨率的网格上求解 VI, 估计收敛阶.

    收敛阶 p 的估计 (Richardson 外推):
        p ≈ log(||u_h - u_{2h}|| / ||u_{2h} - u_{4h}||) / log(2)

    误差估计:
        ||u - u_h|| ≈ C · h^p
    """

    def __init__(self):
        self.solutions: List[Tuple[float, np.ndarray]] = []

    def add_solution(self, h: float, u: np.ndarray) -> None:
        """添加 (网格间距, 解) 对."""
        self.solutions.append((h, u.copy()))
        self.solutions.sort(key=lambda x: x[0])

    def estimate_convergence_order(self) -> Optional[float]:
        """Richardson 外推估计收敛阶."""
        if len(self.solutions) < 3:
            return None

        # 取最细的三个网格
        h_vals = [s[0] for s in self.solutions[-3:]]
        u_vals = [s[1] for s in self.solutions[-3:]]

        # 需要相同长度, 否则插值
        min_len = min(len(u) for u in u_vals)
        u_vals = [u[:min_len] for u in u_vals]

        diff_1 = np.linalg.norm(u_vals[1] - u_vals[0])
        diff_2 = np.linalg.norm(u_vals[2] - u_vals[1])

        if diff_2 < 1e-14 or diff_1 < 1e-14:
            return None

        p = np.log(diff_1 / diff_2) / np.log(h_vals[0] / h_vals[1])
        return float(p)
