"""
multichannel_cvt.py
-------------------
多通道衰变振幅的 Cell 数组 (ragged array) 存储与基于空间变化度规的
Centroidal Voronoi Tessellation (CVT) 探测器接受度优化。
映射自种子项目 147_cell (向量向量的 cell 操作) 和 258_cvt_metric (metric-dependent CVT)。

物理背景:
  在三体 B 衰变 B → h1 h2 h3 中, 末态可能包含多个共振子结构:
      B → R_1(→ h1 h2) h3
      B → R_2(→ h1 h3) h2
      B → R_3(→ h2 h3) h1
  每个共振道有各自的角动量 L_r、自旋 J_r、Blatt-Weisskopf 因子,
  因此不同通道的数据长度和结构不同, 需要用 ragged array (cell) 存储。

  探测器接受度 acceptance(s12, s13) 在 Dalitz 图上不均匀,
  优化探测器几何使得每个 Voronoi 单元的 acceptance 方差最小,
  等价于在度量矩阵 M(s) 下的 CVT:
      d(x, y) = sqrt( (x-y)^T M((x+y)/2) (x-y) )
  其中 M(s) 与局部 acceptance 梯度相关:
      M(s) = det(A(s))^{-1/2} A(s),
      A(s) = Jacobian of acceptance w.r.t. (s12, s13)

本模块实现:
  1) CellArray: 轻量级 ragged array, 支持按通道索引、拼接、统计 (仿 147);
  2) MetricCVT: 在空间变化度量下计算 Centroidal Voronoi 分割 (仿 258);
  3) 将 CVT 用于优化 Dalitz 图上的探测器接受度采样。
"""

from __future__ import annotations
import math
import random
from typing import Callable, List, Tuple

import numpy as np


# ========================================================================== #
#                 Cell 数组 (ragged array, 仿 147_cell)                    #
# ========================================================================== #
class CellArray:
    """
    向量向量的 ragged 数组。每个 cell 可存放长度不同的数据序列。
    实现 (仿 147 的 r8cvv_* 系列):
      - size:   返回 cell 总数
      - get(i): 返回第 i 个 cell 的数据 (numpy 数组)
      - set(i, v): 设置第 i 个 cell
      - append(v): 追加新 cell
      - total_length(): 所有 cell 的总元素数
      - offset(): 返回 cell 起始位置的数组 (仿 147.r8cvv_offset)
    """
    def __init__(self, cells: List[np.ndarray] = None):
        self.cells: List[np.ndarray] = []
        if cells is not None:
            for c in cells:
                self.cells.append(np.asarray(c, dtype=float))

    def size(self) -> int:
        return len(self.cells)

    def get(self, i: int) -> np.ndarray:
        if i < 0 or i >= len(self.cells):
            raise IndexError(f"Cell 索引 {i} 越界, size={len(self.cells)}")
        return self.cells[i]

    def set(self, i: int, v: np.ndarray) -> None:
        if i < 0 or i >= len(self.cells):
            raise IndexError(f"Cell 索引 {i} 越界")
        self.cells[i] = np.asarray(v, dtype=float)

    def append(self, v: np.ndarray) -> None:
        self.cells.append(np.asarray(v, dtype=float))

    def total_length(self) -> int:
        return sum(len(c) for c in self.cells)

    def offset(self) -> np.ndarray:
        """返回每个 cell 起始的全局索引, 长度 size+1。"""
        offs = [0]
        for c in self.cells:
            offs.append(offs[-1] + len(c))
        return np.array(offs, dtype=int)

    def flatten(self) -> np.ndarray:
        """将所有 cell 拼接为 1D 数组。"""
        if not self.cells:
            return np.array([], dtype=float)
        return np.concatenate(self.cells)

    def print_summary(self, title: str = "CellArray") -> str:
        """生成字符串摘要 (仿 147.r8cvv_print)。"""
        lines = [f"\n{title} (size={self.size()}, total_length={self.total_length()})"]
        for i, c in enumerate(self.cells):
            if len(c) == 0:
                lines.append(f"  cell[{i}]: []")
            elif len(c) <= 4:
                lines.append(f"  cell[{i}]: {c.tolist()}")
            else:
                lines.append(
                    f"  cell[{i}]: [{c[0]:.4g}, ..., {c[-1]:.4g}] "
                    f"(len={len(c)})"
                )
        return "\n".join(lines)


# ========================================================================== #
#            多通道共振振幅的 Cell 存储                                      #
# ========================================================================== #
def build_multichannel_resonance_cells(
        resonance_channels: List[dict],
        n_points: int = 20,
) -> CellArray:
    """
    对每个共振通道, 在子不变质量 s 区间 [s_min, s_max] 上均匀采样 n_points,
    计算 |BW(s)|^2 (Breit-Wigner 线形), 存入 cell 数组。
    resonance_channels: 列表, 每项为 dict:
        { "name": str, "m_r": float, "gamma_r": float,
          "s_min": float, "s_max": float }
    """
    cells = CellArray()
    for ch in resonance_channels:
        name = ch["name"]
        m_r = ch["m_r"]
        gamma_r = ch["gamma_r"]
        s_min = ch["s_min"]
        s_max = ch["s_max"]
        if s_max <= s_min:
            raise ValueError(f"通道 {name}: s_max <= s_min")
        s_vals = np.linspace(s_min, s_max, n_points)
        bw_mod2 = np.zeros(n_points, dtype=float)
        for k, s in enumerate(s_vals):
            bw = 1.0 / (m_r * m_r - s - 1j * m_r * gamma_r)
            bw_mod2[k] = abs(bw) ** 2
        # 归一化: 除以最大值
        max_val = bw_mod2.max()
        if max_val > 1.0e-30:
            bw_mod2 = bw_mod2 / max_val
        cells.append(bw_mod2)
    return cells


# ========================================================================== #
#           度量相关的 CVT (仿 258_cvt_metric)                             #
# ========================================================================== #
def metric_matrix_acceptance(
        s12: float,
        s13: float,
        m_parent: float,
        m1: float,
        m2: float,
        m3: float,
) -> np.ndarray:
    """
    计算在点 (s12, s13) 处的度量矩阵 M:
        M = det(A)^{-1/2} A
    其中 A 为 acceptance 的 Fisher 信息矩阵近似:
        A_{ab} = δ_{ab} + eps * d^2 log(acc) / ds_a ds_b
    简化实现: 取 A = diag(f(s12), f(s13)), 其中 f 与到 Dalitz 边界
    的距离成反比 (边界附近 acceptance 下降快, 需要更密采样)。
    """
    # 到 Dalitz 边界的简化距离度量 (在归一化参考坐标下)
    s12_min = (m1 + m2) ** 2
    s12_max = (m_parent - m3) ** 2
    s13_min = (m1 + m3) ** 2
    s13_max = (m_parent - m2) ** 2
    L12 = s12_max - s12_min
    L13 = s13_max - s13_min
    if L12 <= 0 or L13 <= 0:
        return np.eye(2)
    xi  = (s12 - s12_min) / L12
    eta = (s13 - s13_min) / L13
    # 到参考三角形三边的最小距离
    d1 = xi              # 到 xi=0 边
    d2 = eta             # 到 eta=0 边
    d3 = 1.0 - xi - eta  # 到 xi+eta=1 边
    d_min = max(min(d1, d2, d3), 1.0e-6)
    # acceptance 梯度的近似: 越接近边界, A 越大 → 更密的 Voronoi 网格
    f = 1.0 + 0.5 / d_min
    A = np.array([[f, 0.0], [0.0, f]])
    detA = max(np.linalg.det(A), 1.0e-30)
    M = A / math.sqrt(detA)
    return M


def metric_distance(
        p: np.ndarray,
        q: np.ndarray,
        metric_func: Callable,
) -> float:
    """
    计算点 p, q 在空间变化度量下的距离:
        d(p, q) = sqrt( (p-q)^T M((p+q)/2) (p-q) )
    M 在中点 (p+q)/2 处计算 (仿 258_cvt_metric)。
    """
    mid = 0.5 * (p + q)
    M = metric_func(mid[0], mid[1])
    diff = p - q
    return math.sqrt(max(float(diff @ M @ diff), 0.0))


def cvt_lloyd_iteration(
        generators: np.ndarray,
        sample_points: np.ndarray,
        metric_func: Callable,
        n_iter: int = 20,
) -> np.ndarray:
    """
    Lloyd 算法的度量相关版本 (仿 258_cvt_metric):
      1) 对每个样本点, 找到最近的 generator (按度量距离);
      2) 将每个 generator 移动到其 Voronoi 单元的度量质心;
      3) 重复直到收敛。
    generators: shape (N, 2), 初始 generator 位置
    sample_points: shape (M, 2), 用于近似 Voronoi 单元的采样点
    返回: 迭代后的 generator 位置
    """
    n_gen = len(generators)
    n_sample = len(sample_points)
    if n_gen == 0 or n_sample == 0:
        return generators.copy()
    gen = generators.copy().astype(float)
    for it in range(n_iter):
        # 分配: 每个样本点属于最近的 generator
        assignment = np.zeros(n_sample, dtype=int)
        for k in range(n_sample):
            best_j = 0
            best_d = float("inf")
            sp = sample_points[k]
            for j in range(n_gen):
                d = metric_distance(sp, gen[j], metric_func)
                if d < best_d:
                    best_d = d
                    best_j = j
            assignment[k] = best_j
        # 更新: 每个 generator 移动到其 Voronoi 单元的 Euclidean 质心
        # (度量质心需解非线性方程, 此处简化为 Euclidean 质心)
        for j in range(n_gen):
            mask = (assignment == j)
            if not np.any(mask):
                continue
            pts = sample_points[mask]
            gen[j] = pts.mean(axis=0)
    return gen


def cvt_energy(
        generators: np.ndarray,
        sample_points: np.ndarray,
        metric_func: Callable,
) -> float:
    """
    CVT 能量泛函 (量化接受度采样质量):
        E = (1/M) Σ_{k=1}^{M} d(x_k, gen(x_k))^2
    其中 gen(x_k) 是样本点 x_k 的最近 generator, d 为度量距离。
    能量越小, 表示 generator 分布越均匀地覆盖采样空间。
    """
    n_gen = len(generators)
    n_sample = len(sample_points)
    if n_gen == 0 or n_sample == 0:
        return 0.0
    total = 0.0
    for k in range(n_sample):
        sp = sample_points[k]
        best_d = float("inf")
        for j in range(n_gen):
            d = metric_distance(sp, generators[j], metric_func)
            if d < best_d:
                best_d = d
        total += best_d * best_d
    return total / max(n_sample, 1)


# ========================================================================== #
#                    多通道数据集成                                          #
# ========================================================================== #
def integrate_channel_cells(
        cells: CellArray,
        weights_per_channel: List[float] = None,
) -> float:
    """
    对多通道 cell 数据进行加权积分:
        I = Σ_r w_r Σ_k |BW_r(s_k)|^2
    权重 w_r 默认为 1。用于估计各共振道的相对贡献。
    """
    n_ch = cells.size()
    if weights_per_channel is None:
        weights_per_channel = [1.0] * n_ch
    if len(weights_per_channel) != n_ch:
        raise ValueError("权重数目与通道数不一致")
    total = 0.0
    for r in range(n_ch):
        data = cells.get(r)
        total += weights_per_channel[r] * float(np.sum(data))
    return total
