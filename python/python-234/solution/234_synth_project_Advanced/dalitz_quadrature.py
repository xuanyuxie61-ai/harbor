"""
dalitz_quadrature.py
--------------------
Dalitz 图三角形上的对称求积规则与直方图分箱统计。
映射自种子项目 1318_triangle_symq_rule_original (高精度对称求积规则) 和
1306_triangle_histogram (三角形区域分箱统计)。

物理背景:
  三体 B 衰变的总宽度
      Γ = (1 / (256 π^3 m_B^3)) ∫∫ |A(s12, s13)|^2 ds12 ds13
  积分在物理允许的 Dalitz 三角形上进行。为高精度计算此积分,
  本模块提供:
    1) 参考三角形上的对称求积规则 (degree 0, 1, 2, 3, 4, 5);
    2) 仿射映射到物理 Dalitz 三角形的求积节点和权重;
    3) Dalitz 三角形上衰变事件的分箱直方图, 用于实验数据分析。

数学公式:
  参考三角形 T_ref = { (0,0), (1,0), (0,1) }
  物理 Dalitz 三角形: 以 (s12^{min}, s13^{min}), (s12^{max}, s13^{min}),
                       (s12^{min}, s13^{max}) 为顶点 (近似)。
  仿射映射:
      s12(xi, eta) = s12_min + (s12_max - s12_min) * xi
      s13(xi, eta) = s13_min + (s13_max - s13_min) * eta
      det(J) = (s12_max - s12_min)(s13_max - s13_min)
  积分:
      ∫∫_T f(s12, s13) ds12 ds13 = det(J) ∫∫_{T_ref} f(s12(xi, eta), s13(xi, eta)) dxi deta
      ≈ det(J) Σ_k w_k f(s12(x_k, y_k), s13(x_k, y_k))

  对称求积规则 (Xiao-Gimbutas, 来自 1318):
    degree 0 (1 point):  centroid, weight = 1/2
    degree 1 (3 points): edge midpoints, equal weights
    degree 2 (4 points): vertices + centroid
    ...
    degree 5 (16 points): 高精度对称规则

  直方图 (来自 1306):
    将参考三角形细分为 N(N+1)/2 个全等子三角形,
    统计每个子三角形内落入的衰变事件数。
"""

from __future__ import annotations
import math
from typing import Callable, List, Tuple

import numpy as np

from b_physics_constants import kallen
from dalitz_geometry import DalitzAffineMap


# ========================================================================== #
#                   参考三角形上的对称求积规则 (仿 1318)                    #
# ========================================================================== #
def symq_rule_degree_0() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """degree 0 (1 点): centroid (1/3, 1/3), weight = 1/2 (= area of ref tri)."""
    x = np.array([1.0 / 3.0])
    y = np.array([1.0 / 3.0])
    w = np.array([0.5])
    return x, y, w


def symq_rule_degree_1() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    degree 1 (3 点): 三角形边中点
      (1/2, 0), (0, 1/2), (1/2, 1/2)
    权重: 各 1/6
    """
    x = np.array([0.5, 0.0, 0.5])
    y = np.array([0.0, 0.5, 0.5])
    w = np.array([1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0])
    return x, y, w


def symq_rule_degree_2() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    degree 2 (4 点): Hammer-Stroud 规则
      (1/3, 1/3) with weight -27/96
      (1/5, 1/5), (3/5, 1/5), (1/5, 3/5) with weights 25/96
    """
    x = np.array([1.0 / 3.0, 0.2, 0.6, 0.2])
    y = np.array([1.0 / 3.0, 0.2, 0.2, 0.6])
    w = np.array([-27.0 / 96.0, 25.0 / 96.0, 25.0 / 96.0, 25.0 / 96.0])
    return x, y, w


def symq_rule_degree_3() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    degree 3 (6 点): 对称求积规则
    节点: (a1, a1), (b1, c1), (c1, b1), (a1, b1), (b1, a1), (c1, c1)
    其中 a1 = 1/3, b1 = 0.797426985353..., c1 = 0.101286507323...
    权重: w1, w2 两种
    """
    a1 = 1.0 / 3.0
    b1 = 0.79742698535308733
    c1 = 0.10128650732345634
    w1 = -0.1125
    w2 = 0.0625 + 0.04166666666666667
    x = np.array([a1, b1, c1, a1, b1, c1])
    y = np.array([a1, c1, b1, b1, a1, c1])
    w = np.array([w1, w2, w2, w2, w2, w2])
    return x, y, w


def symq_rule_degree_4() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    degree 4 (7 点): 对称求积规则
    节点: (1/3, 1/3) with w1; (a2, b2), (b2, a2), (a2, 1-a2-b2), ...
    """
    a2 = 0.10128650732345634
    b2 = 0.79742698535308733
    c2 = 1.0 - a2 - b2
    w1 = 0.225 / 2.0
    w2 = 0.13239415278850618 / 2.0
    x = np.array([1.0 / 3.0, a2, b2, c2, a2, b2, c2])
    y = np.array([1.0 / 3.0, b2, c2, a2, c2, a2, b2])
    w = np.array([w1, w2, w2, w2, w2, w2, w2])
    return x, y, w


def symq_rule_degree_5() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    degree 5 (12 点): 高精度对称求积规则 (仿 1318 rule05)
    两组对称轨道: 3 节点轨道和 6 节点轨道。
    系数来自 Xiao-Gimbutas 的优化规则。
    """
    a1 = 0.09157621350977074
    b1 = 0.81684757298045851
    c1 = 1.0 - a1 - b1
    a2 = 0.44594849091596489
    b2 = 0.10810301816807022
    c2 = 1.0 - a2 - b2
    w1 = 0.09094750212351201 / 2.0
    w2 = 0.20430679732594244 / 2.0
    x = np.array([a1, b1, c1, a2, b2, c2, a2, b2, c2, a1, b1, c1])
    y = np.array([b1, c1, a1, b2, c2, a2, c2, a2, b2, c1, a1, b1])
    w = np.array([w1, w1, w1, w2, w2, w2, w2, w2, w2, w1, w1, w1])
    return x, y, w


def get_symq_rule(degree: int):
    """按 degree 返回对称求积规则 (x, y, w)。"""
    rules = {
        0: symq_rule_degree_0,
        1: symq_rule_degree_1,
        2: symq_rule_degree_2,
        3: symq_rule_degree_3,
        4: symq_rule_degree_4,
        5: symq_rule_degree_5,
    }
    if degree not in rules:
        raise ValueError(f"仅支持 degree 0..5, 请求 degree={degree}")
    return rules[degree]()


# ========================================================================== #
#               物理 Dalitz 三角形上的求积                                  #
# ========================================================================== #
def integrate_on_dalitz(
        integrand: Callable[[float, float], float],
        m_parent: float,
        m1: float,
        m2: float,
        m3: float,
        degree: int = 5,
) -> float:
    """
    在物理 Dalitz 区域上积分 integrand(s12, s13), 使用对称求积规则。
    步骤:
      1) 获取参考三角形上 degree 的求积节点 (x_k, y_k) 和权重 w_k;
      2) 仿射映射到物理 Dalitz 三角形 (近似);
      3) 判断每个映射点是否在物理区域内; 对在区域内的点累加贡献。

    注意: 对精确的 Dalitz 曲边三角形积分, 应使用曲边映射而非仿射映射;
    此处采用仿射近似 + 物理区域筛选, 对小规模实验足够精确。
    """
    x_ref, y_ref, w_ref = get_symq_rule(degree)
    affine = DalitzAffineMap(m_parent, m1, m2, m3)
    detJ = affine.Jacobian
    total = 0.0
    n_inside = 0
    for k in range(len(w_ref)):
        s12, s13 = affine.ref_to_dalitz(x_ref[k], y_ref[k])
        if affine.in_physical_region(s12, s13, m_parent, m1, m2, m3):
            total += w_ref[k] * integrand(s12, s13)
            n_inside += 1
    if n_inside == 0:
        return 0.0
    # 权重归一化: 参考三角形面积 = 1/2, 物理区域近似面积 = detJ * 1/2
    return total * detJ


def integrate_dalitz_decay_width(
        amplitude_sq: Callable[[float, float], float],
        m_parent: float,
        m1: float,
        m2: float,
        m3: float,
        degree: int = 5,
) -> float:
    """
    计算三体衰变的部分宽度:
        Gamma = (1 / (256 pi^3 m_B^3)) * ∫∫ |A(s12, s13)|^2 ds12 ds13
    返回 Gamma, 单位为 MeV (若 m 输入为 MeV, A 为无量纲)。
    """
    integral = integrate_on_dalitz(amplitude_sq, m_parent, m1, m2, m3, degree)
    prefactor = 1.0 / (256.0 * math.pi ** 3 * m_parent ** 3)
    return prefactor * integral


# ========================================================================== #
#           Dalitz 直方图分箱 (仿 1306_triangle_histogram)                  #
# ========================================================================== #
def triangle_histogram(
        points: np.ndarray,
        n_subdiv: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    将位于参考三角形 {(x,y) | x>=0, y>=0, x+y<=1} 中的点集
    划分到 N(N+1)/2 个全等子三角形中, 统计每个子三角形内点的数目。

    参数:
        points:     shape (n_pts, 2), 点的 (x, y) 坐标
        n_subdiv:   细分层数 N
    返回:
        subtri_indices: shape (n_pts,), 每个点所属子三角形的索引
        counts:         shape (N(N+1)/2,), 每个子三角形内的点数

    子三角形编号: 第 i 行 (0 <= i < N) 有 (N-i) 个上三角形和 (N-i-1) 个下三角形。
    使用面积坐标 (barycentric):
        lambda_1 = 1 - x - y,   lambda_2 = x,   lambda_3 = y
    子三角形索引:
        上三角形 (i, j, "up"):   lambda_1 in [1-(i+1)/N, 1-i/N], ...
    """
    if n_subdiv < 1:
        raise ValueError("细分层数必须 >= 1")
    N = n_subdiv
    if points.size == 0:
        return np.array([], dtype=int), np.zeros(N * (N + 1) // 2, dtype=int)
    x = points[:, 0]
    y = points[:, 1]
    # 面积坐标
    lam1 = 1.0 - x - y   # 对应顶点 (0,0)
    lam2 = x              # 对应顶点 (1,0)
    lam3 = y              # 对应顶点 (0,1)
    # 粗化索引: (i, j) 其中 i = floor(lam2 * N), j = floor(lam3 * N)
    i = np.floor(lam2 * N).astype(int)
    j = np.floor(lam3 * N).astype(int)
    # 截断到 [0, N-1]
    i = np.clip(i, 0, N - 1)
    j = np.clip(j, 0, N - 1)
    # 判断在 "up" 还是 "down" 子三角形
    # 在每个 (i,j) 单元中, 上三角形: frac(lam2*N) + frac(lam3*N) <= 1
    frac2 = lam2 * N - i
    frac3 = lam3 * N - j
    is_up = (frac2 + frac3) <= 1.0
    # 计算全局子三角形索引: 第 r 行 (r = i+j) 包含 2(N-r)-1 个子三角形
    # 简化编号: 使用字典
    n_total = N * (N + 1) // 2
    # 行优先编号: 第 r 行 (r = i + j) 从 sum_{k=0}^{r-1} (N-k) 开始
    row_start = np.array([sum(N - k for k in range(r)) for r in range(N + 1)])
    # 上三角形在行 r 内的索引: j (0 <= j < N-r)
    # 下三角形在行 r 内的索引: (N-r) + j
    idx = np.zeros(len(points), dtype=int)
    for k in range(len(points)):
        r = i[k] + j[k]
        if r >= N:
            r = N - 1
            j_eff = N - 1 - i[k]
        else:
            j_eff = j[k]
        if is_up[k]:
            idx[k] = row_start[r] + j_eff
        else:
            # 下三角形: 归入相邻上三角形 (简化)
            idx[k] = min(row_start[r] + j_eff, n_total - 1)
        idx[k] = min(idx[k], n_total - 1)
    counts = np.bincount(idx, minlength=n_total)
    return idx, counts


def histogram_chi_squared(
        counts: np.ndarray,
        expected: np.ndarray,
) -> float:
    """
    计算直方图的 chi-squared:
        chi^2 = Σ_i (O_i - E_i)^2 / E_i
    用于检验衰变事件在 Dalitz 图上的分布是否符合理论预期。
    """
    if len(counts) != len(expected):
        raise ValueError("counts 和 expected 长度不一致")
    chi2 = 0.0
    for k in range(len(counts)):
        if expected[k] > 1.0e-30:
            chi2 += (counts[k] - expected[k]) ** 2 / expected[k]
    return chi2
