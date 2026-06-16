"""
jet_boundary.py — 喷注边界与区域拓扑模块

融合种子项目:
  - 106_boundary_word_drafter : 边界词追踪 (boundary word tracing),
    用方向序列描述形状边界的通用框架
  - 112_box_display : 二维网格区域逻辑判定

物理背景:
  喷注的边界定义在 η-φ 平面上并非简单圆形:
  - anti-kT 喷注的边界可能是不规则多边形
  - 重叠喷注的共享区域需要精确的边界追踪
  - 喷注面积 (jet area) 的计算依赖边界的离散化

关键概念:
  1. 喷注边界词 (jet boundary word):
     类似于 106 中的 eternity grid word,
     用一系列方向步长描述喷注边界.
     在 η-φ 平面上, 12 个方向对应正六边形网格.

  2. Voronoi cell (源自 192_closest_point_brute):
     每个粒子在 η-φ 平面上有一个 Voronoi cell,
     喷注边界是 Voronoi cell 的并集.

  3. 喷注面积:
     A_jet = ∫_{boundary} dη dφ
     通过 Monte Carlo 或解析方法计算.
"""

import numpy as np
import math
from typing import List, Tuple
from jet_fourvector import FourVector


# ─────────────────────────────────────────────────────────────────────────────
# 12 方向边界词系统 (源自 106_boundary_word_drafter)
# ─────────────────────────────────────────────────────────────────────────────

# 12 个方向对应的 (Δη, Δφ) 单位步长
# 方向命名: E=东, ENE=东北偏东, N=北, ... (顺时针)
BOUNDARY_DIRECTIONS = {
    'E':   (1.0, 0.0),
    'ENE': (math.sqrt(3) / 2, 0.5),
    'NE':  (0.5, math.sqrt(3) / 2),
    'NNE': (0.5, math.sqrt(3) / 2),  # 同上
    'N':   (0.0, 1.0),
    'NNW': (-0.5, math.sqrt(3) / 2),
    'NW':  (-0.5, math.sqrt(3) / 2),
    'WNW': (-math.sqrt(3) / 2, 0.5),
    'W':   (-1.0, 0.0),
    'WSW': (-math.sqrt(3) / 2, -0.5),
    'SW':  (-0.5, -math.sqrt(3) / 2),
    'SSW': (-0.5, -math.sqrt(3) / 2),
    'S':   (0.0, -1.0),
    'SSE': (0.5, -math.sqrt(3) / 2),
    'SE':  (0.5, -math.sqrt(3) / 2),
    'ESE': (math.sqrt(3) / 2, -0.5),
}

# 简化为 8 个主方向
CARDINAL_DIRECTIONS = {
    'E':  (1.0, 0.0),
    'NE': (math.sqrt(2) / 2, math.sqrt(2) / 2),
    'N':  (0.0, 1.0),
    'NW': (-math.sqrt(2) / 2, math.sqrt(2) / 2),
    'W':  (-1.0, 0.0),
    'SW': (-math.sqrt(2) / 2, -math.sqrt(2) / 2),
    'S':  (0.0, -1.0),
    'SE': (math.sqrt(2) / 2, -math.sqrt(2) / 2),
}


class BoundaryWord:
    """喷注边界词.

    边界词是一个方向-步长序列:
      W = [(d₁, s₁), (d₂, s₂), ..., (dₙ, sₙ)]
    其中 d_i 是方向, s_i 是步长.

    从基点 P₀ 出发, 依次执行:
      P_{k+1} = P_k + s_k · d_k
    最终回到 P₀ (封闭边界).
    """

    def __init__(self, start_point: Tuple[float, float] = (0.0, 0.0)):
        self.start = start_point
        self.steps: List[Tuple[str, float]] = []
        self.vertices: List[Tuple[float, float]] = [start_point]

    def add_step(self, direction: str, step_size: float = 1.0):
        """添加一个边界步."""
        if direction not in CARDINAL_DIRECTIONS:
            raise ValueError(f"Unknown direction: {direction}")
        self.steps.append((direction, step_size))
        last = self.vertices[-1]
        d_eta, d_phi = CARDINAL_DIRECTIONS[direction]
        new_eta = last[0] + step_size * d_eta
        new_phi = last[1] + step_size * d_phi
        self.vertices.append((new_eta, new_phi))

    def is_closed(self, tol: float = 1e-10) -> bool:
        """检查边界词是否封闭 (起点 = 终点)."""
        if len(self.vertices) < 2:
            return False
        start = np.array(self.vertices[0])
        end = np.array(self.vertices[-1])
        return np.linalg.norm(start - end) < tol

    def perimeter(self) -> float:
        """边界周长."""
        return sum(s for _, s in self.steps)

    def area(self) -> float:
        """用 Shoelace 公式计算边界围成的面积.

        A = 0.5 |Σ_{i} (x_i y_{i+1} - x_{i+1} y_i)|
        """
        n = len(self.vertices)
        if n < 3:
            return 0.0
        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            area += self.vertices[i][0] * self.vertices[j][1]
            area -= self.vertices[j][0] * self.vertices[i][1]
        return abs(area) / 2.0

    def centroid(self) -> Tuple[float, float]:
        """边界围成区域的质心."""
        n = len(self.vertices)
        if n < 3:
            return self.start
        cx, cy, a_signed = 0.0, 0.0, 0.0
        for i in range(n):
            j = (i + 1) % n
            cross = (self.vertices[i][0] * self.vertices[j][1] -
                     self.vertices[j][0] * self.vertices[i][1])
            cx += (self.vertices[i][0] + self.vertices[j][0]) * cross
            cy += (self.vertices[i][1] + self.vertices[j][1]) * cross
            a_signed += cross
        a_signed *= 0.5
        if abs(a_signed) < 1e-15:
            return self.start
        return (cx / (6 * a_signed), cy / (6 * a_signed))


def circular_boundary_word(center_eta: float, center_phi: float,
                           R: float, n_segments: int = 32) -> BoundaryWord:
    """用多边形逼近圆形喷注边界.

    将半径 R 的圆离散为 n_segments 段,
    每段用最近的方向近似.

    Returns:
        BoundaryWord 对象
    """
    bw = BoundaryWord(start_point=(center_eta + R, center_phi))

    angles = np.linspace(0, 2 * math.pi, n_segments, endpoint=False)
    for i in range(n_segments):
        theta1 = angles[i]
        theta2 = angles[(i + 1) % n_segments]
        # 中点方向
        theta_mid = 0.5 * (theta1 + theta2)
        d_eta = math.cos(theta_mid)
        d_phi = math.sin(theta_mid)

        # 找到最近的 cardinal 方向
        best_dir = 'E'
        best_dot = -1.0
        for dir_name, (de, dp) in CARDINAL_DIRECTIONS.items():
            dot = d_eta * de + d_phi * dp
            if dot > best_dot:
                best_dot = dot
                best_dir = dir_name

        # 步长 = 弧长 ≈ R · Δθ
        step_size = R * abs(theta2 - theta1)
        bw.add_step(best_dir, step_size)

    return bw


# ─────────────────────────────────────────────────────────────────────────────
# 喷注面积计算 (Voronoi + Monte Carlo)
# ─────────────────────────────────────────────────────────────────────────────

def jet_area_monte_carlo(jet_axis: FourVector,
                         constituents: List[FourVector],
                         R: float = 0.4,
                         n_samples: int = 5000,
                         rng: np.random.Generator = None) -> float:
    """用 Monte Carlo 方法计算喷注的有效面积.

    在 (η, φ) 平面上以 jet_axis 为中心、R 为半径的圆内
    随机采样 n_samples 个点, 计算有多少点
    属于该喷注的 Voronoi cell (即最近粒子是该 jet 的 constituent).

    A_jet ≈ π R² · (N_inside / N_total)
    """
    if rng is None:
        rng = np.random.default_rng(42)

    eta_c = jet_axis.eta
    phi_c = jet_axis.phi

    # 在圆内均匀采样 (用 rejection sampling)
    n_inside = 0
    for _ in range(n_samples):
        # 均匀采样 in disk
        r = R * math.sqrt(rng.uniform(0, 1))
        theta = rng.uniform(0, 2 * math.pi)
        eta_sample = eta_c + r * math.cos(theta)
        phi_sample = phi_c + r * math.sin(theta)

        # 检查该点是否被喷注 constituents 的 Voronoi cell 覆盖
        # (简化: 检查是否在任何 constituent 的 ΔR < R 内)
        # 更精确: 检查最近 constituent 是否属于该 jet
        sample_fv = FourVector(
            1.0,  # 虚拟能量
            math.cos(phi_sample) / math.cosh(eta_sample),
            math.sin(phi_sample) / math.cosh(eta_sample),
            math.tanh(eta_sample)
        )

        min_dr = float('inf')
        for c in constituents:
            dr = sample_fv.delta_R(c)
            if dr < min_dr:
                min_dr = dr

        if min_dr < R:
            n_inside += 1

    area = math.pi * R * R * n_inside / n_samples
    return area


def voronoi_cell_area(constituents: List[FourVector],
                      jet_axis: FourVector,
                      R: float = 0.4) -> List[float]:
    """计算每个 constituent 在喷注内的 Voronoi cell 面积.

    使用 Monte Carlo 方法近似.

    Returns:
        面积列表, 与 constituents 顺序对应
    """
    n = len(constituents)
    if n == 0:
        return []

    rng = np.random.default_rng(42)
    n_samples = 2000
    eta_c = jet_axis.eta
    phi_c = jet_axis.phi

    counts = [0] * n
    total_valid = 0

    for _ in range(n_samples):
        r = R * math.sqrt(rng.uniform(0, 1))
        theta = rng.uniform(0, 2 * math.pi)
        eta_s = eta_c + r * math.cos(theta)
        phi_s = phi_c + r * math.sin(theta)

        # 找最近的 constituent
        best_idx = -1
        best_dr = float('inf')
        for i, c in enumerate(constituents):
            deta = eta_s - c.eta
            dphi = phi_s - c.phi
            while dphi > math.pi:
                dphi -= 2 * math.pi
            while dphi < -math.pi:
                dphi += 2 * math.pi
            dr2 = deta ** 2 + dphi ** 2
            if dr2 < best_dr:
                best_dr = dr2
                best_idx = i

        if best_idx >= 0 and best_dr < R * R:
            counts[best_idx] += 1
            total_valid += 1

    if total_valid == 0:
        return [0.0] * n

    total_area = math.pi * R * R
    areas = [total_area * counts[i] / n_samples for i in range(n)]
    return areas


# ─────────────────────────────────────────────────────────────────────────────
# 边界反射与旋转对称性 (源自 106_boundary_word_drafter)
# ─────────────────────────────────────────────────────────────────────────────

def word_reflect(bw: BoundaryWord, axis: str = 'eta') -> BoundaryWord:
    """将边界词关于 η 或 φ 轴反射.

    用于分析喷注的对称性: 如果喷注关于某个轴对称,
    则其边界词在反射操作下不变 (或等价).
    """
    new_bw = BoundaryWord(start_point=bw.start)
    for direction, step_size in bw.steps:
        # 反射方向
        d_eta, d_phi = CARDINAL_DIRECTIONS[direction]
        if axis == 'eta':
            d_phi = -d_phi
        else:
            d_eta = -d_eta

        # 找最近 cardinal 方向
        best_dir = 'E'
        best_dot = -1.0
        for dir_name, (de, dp) in CARDINAL_DIRECTIONS.items():
            dot = d_eta * de + d_phi * dp
            if dot > best_dot:
                best_dot = dot
                best_dir = dir_name

        new_bw.add_step(best_dir, step_size)

    return new_bw


def word_rotate(bw: BoundaryWord, n_steps: int = 1) -> BoundaryWord:
    """将边界词循环旋转 n_steps 步.

    用于测试边界词的周期性.
    """
    if len(bw.steps) == 0:
        return BoundaryWord(start_point=bw.start)

    rotated_steps = bw.steps[n_steps:] + bw.steps[:n_steps]
    new_bw = BoundaryWord(start_point=bw.start)
    for direction, step_size in rotated_steps:
        new_bw.add_step(direction, step_size)
    return new_bw
