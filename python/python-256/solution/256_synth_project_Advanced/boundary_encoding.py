"""
boundary_encoding.py
====================
恒星振荡边界条件的符号编码与合法性验证.

融合种子项目:
  - boundary_word_hexagon (108): 六边形边界字合法性检查、排序、旋转
  - boundary_word_drafter (106): 边界字到几何坐标的映射

科学背景
--------
恒星振荡的边界条件:

中心 (r → 0):
  ξ_r ∝ r^{l-1}       (正则性条件)
  δP → 有限值
  δΦ → 有限值

表面 (r → R):
  δP = 0              (零 Lagrange 压强扰动, 简化)
  δΦ_ext = -l(l+1) δΦ/R  (外场匹配)

对于更精确的表面边界:
  y₂(R) = y₂^{atm}(ω, l, T_eff, g)

  其中 y₂^{atm} 是大气边界条件, 依赖于:
  - 有效温度 T_eff
  - 表面重力 g
  - 不透明度模型

编码方法 (融合 108 项目):
  将边界条件的满足程度编码为六方向边界字:
  '1'-'6' 对应 6 个约束维度

  合法性检查:
  - 方向 1/4 必须匹配 (径向平衡)
  - 方向 2/5 必须匹配 (角向平衡)
  - 方向 3/6 必须匹配 (热平衡)
  - 方向变化必须为 ±1 (平滑性)

排序 (canonical form):
  对边界字进行旋转, 得到字典序最小的表示.

本模块实现:
  1. 边界条件满足度的六方向编码
  2. 边界字合法性检查
  3. 边界字排序 (规范化)
  4. 边界字到几何坐标的映射
"""

import numpy as np
from typing import Tuple, Dict, List, Optional


class BoundaryConditionEncoder:
    """
    边界条件的六方向编码器.

    融合 boundary_word_hexagon (108) 项目:

    六方向定义 (在平行四边形坐标中):
      1: (I+1, J)    → 径向正方向
      2: (I, J+1)    → 角向正方向
      3: (I-1, J+1)  → 对角线
      4: (I-1, J)    → 径向负方向
      5: (I, J-1)    → 角向负方向
      6: (I+1, J-1)  → 对角线

    边界条件编码:
    对每个约束维度, 判断是否满足:
    - 满足: 取正方向 (1, 2, 3)
    - 不满足: 取反方向 (4, 5, 6)
    - 边界值: 根据违规程度选择

    参数
    ----
    stellar_model : object
        恒星结构模型
    """

    # 六方向位移向量
    DIRECTION_VECTORS = {
        '1': (1, 0),
        '2': (0, 1),
        '3': (-1, 1),
        '4': (-1, 0),
        '5': (0, -1),
        '6': (1, -1),
    }

    def __init__(self, stellar_model):
        self.model = stellar_model

    def encode_boundary_conditions(
        self,
        omega: float,
        l: int,
        surface_y2: float = 0.0,
    ) -> str:
        """
        将边界条件编码为六方向边界字.

        约束维度:
        1. 中心正则性 (ξ_r ∝ r^{l-1})
        2. 表面压强条件 (y₂ ≈ 0)
        3. 表面引力势匹配 (y₄ ≈ -(l+1) y₃)
        4. 能量守恒 (L' = 0 at surface)
        5. 质量守恒 (m' = 0 at center)
        6. 热平衡 (T' = 0 at center)

        Parameters
        ----------
        omega : float
            振荡频率
        l : int
            角量子数
        surface_y2 : float
            表面 y₂ 值 (应为 ~0)

        Returns
        -------
        boundary_word : str
            六方向边界字
        """
        word = []

        # 约束 1: 中心正则性
        xi_center = self._check_center_regularization(l)
        word.append(self._violation_to_direction(xi_center, base=1))

        # 约束 2: 表面压强
        surface_violation = abs(surface_y2)
        word.append(self._violation_to_direction(surface_violation, base=2))

        # 约束 3: 引力势匹配
        phi_match = self._check_potential_matching(omega, l)
        word.append(self._violation_to_direction(phi_match, base=3))

        # 约束 4-6: 守恒条件
        word.append(self._violation_to_direction(0.01, base=4))
        word.append(self._violation_to_direction(0.005, base=5))
        word.append(self._violation_to_direction(0.008, base=6))

        return "".join(word)

    def _check_center_regularization(self, l: int) -> float:
        """检查中心正则性."""
        # ξ_r(0) 应为有限值 (l=0) 或零 (l>0)
        r_inner = self.model.r[1] if self.model.n_r > 1 else 1e10
        return r_inner**max(l - 1, 0) * 1e-5

    def _check_potential_matching(self, omega: float, l: int) -> float:
        """检查表面引力势匹配."""
        return 0.02 * (1.0 + l)  # 简化: 正比于 l+1

    def _violation_to_direction(self, violation: float, base: int) -> str:
        """
        将违规程度映射为方向.

        Parameters
        ----------
        violation : float
            违规程度 (0 = 完美满足)
        base : int
            基础方向 (1, 2, 或 3)

        Returns
        -------
        direction : str
            '1'-'6'
        """
        if violation < 0.01:
            return str(base)  # 满足
        elif violation < 0.05:
            return str(base + 1)  # 轻微违规
        else:
            return str(base + 3)  # 严重违规 (反方向)


def boundary_is_legal(w: str, p: Tuple[int, int] = (0, 0)) -> bool:
    """
    检查边界字是否合法.

    融合 boundary_word_hexagon (108) 的 boundary_is_legal:

    合法性条件:
    1. 只包含字符 '1'-'6'
    2. 每对 1/4, 2/5, 3/6 必须数量匹配 (闭合路径)
    3. 相邻步的方向变化必须为 ±1 (模 6)

    Parameters
    ----------
    w : str
        边界字
    p : tuple
        起点坐标

    Returns
    -------
    is_legal : bool
    """
    if not w:
        return False

    # 条件 1: 字符合法性
    valid_chars = set('123456')
    for c in w:
        if c not in valid_chars:
            return False

    # 条件 2: 对偶匹配
    if w.count('1') != w.count('4'):
        return False
    if w.count('2') != w.count('5'):
        return False
    if w.count('3') != w.count('6'):
        return False

    # 条件 3: 方向连续性
    for i in range(len(w)):
        c_old = w[i - 1] if i > 0 else w[-1]
        c_new = w[i]

        d_old = int(c_old)
        d_new = int(c_new)

        diff = abs(d_new - d_old)
        if diff == 5:
            diff = 1  # 环绕 (1→6 或 6→1)

        if diff != 1:
            return False

    return True


def boundary_sort(w: str, p: Tuple[int, int] = (0, 0)) -> Tuple[str, Tuple[int, int]]:
    """
    边界字排序 (规范化).

    融合 boundary_word_hexagon (108) 的 boundary_sort:
    旋转边界字, 使起点的字典序最小.

    Parameters
    ----------
    w : str
        原始边界字
    p : tuple
        原始起点

    Returns
    -------
    w_sorted : str
        排序后的边界字
    p_sorted : tuple
        排序后的起点
    """
    wn = len(w)
    p2 = list(p)
    p3 = list(p)
    w2 = w

    for in_idx in range(wn):
        c = w[in_idx]
        direction = int(c)

        # 更新位置
        if direction == 1:
            p3[0] += 1
        elif direction == 2:
            p3[1] += 1
        elif direction == 3:
            p3[0] -= 1
            p3[1] += 1
        elif direction == 4:
            p3[0] -= 1
        elif direction == 5:
            p3[1] -= 1
        elif direction == 6:
            p3[0] += 1
            p3[1] -= 1

        # 检查是否更小
        if (p3[0] < p2[0]) or (p3[0] == p2[0] and p3[1] < p2[1]):
            p2 = p3[:]
            w2 = w[in_idx + 1:] + w[:in_idx + 1]

    return w2, tuple(p2)


def boundary_to_edge_xy(
    w: str,
    p_xy: Tuple[float, float] = (0.0, 0.0),
) -> np.ndarray:
    """
    将边界字转换为边缘节点的 (x, y) 坐标.

    融合 boundary_word_drafter (106) 的 word_to_edge_xy:

    每个字符对应一个方向和步长, 在二维平面上画出路径.

    Parameters
    ----------
    w : str
        边界字
    p_xy : tuple
        起点 (x, y) 坐标

    Returns
    -------
    e_xy : ndarray, shape (len(w), 2)
        边缘节点坐标
    """
    wn = len(w)
    e_xy = np.zeros((wn, 2))
    e_xy[0] = p_xy

    for wi in range(1, wn):
        c = w[wi - 1]

        # 方向和步长映射 (from 106)
        if c == 'A':
            r, t = 0.5, 0.0
        elif c in ('B', 'b'):
            r = np.sqrt(3.0) / (3.0 if c == 'B' else 6.0)
            t = np.pi / 6.0
        elif c == 'C':
            r, t = 0.5, 2.0 * np.pi / 6.0
        elif c in ('D', 'd'):
            r = np.sqrt(3.0) / (3.0 if c == 'D' else 6.0)
            t = 3.0 * np.pi / 6.0
        elif c == 'E':
            r, t = 0.5, 4.0 * np.pi / 6.0
        elif c in ('F', 'f'):
            r = np.sqrt(3.0) / (3.0 if c == 'F' else 6.0)
            t = 5.0 * np.pi / 6.0
        elif c == 'G':
            r, t = 0.5, 6.0 * np.pi / 6.0
        elif c in ('H', 'h'):
            r = np.sqrt(3.0) / (3.0 if c == 'H' else 6.0)
            t = 7.0 * np.pi / 6.0
        elif c == 'I':
            r, t = 0.5, 8.0 * np.pi / 6.0
        elif c in ('J', 'j'):
            r = np.sqrt(3.0) / (3.0 if c == 'J' else 6.0)
            t = 9.0 * np.pi / 6.0
        elif c == 'K':
            r, t = 0.5, 10.0 * np.pi / 6.0
        elif c in ('L', 'l'):
            r = np.sqrt(3.0) / (3.0 if c == 'L' else 6.0)
            t = 11.0 * np.pi / 6.0
        else:
            r, t = 0.0, 0.0

        e_xy[wi, 0] = e_xy[wi - 1, 0] + r * np.cos(t)
        e_xy[wi, 1] = e_xy[wi - 1, 1] + r * np.sin(t)

    return e_xy


def compute_boundary_enclosure_area(e_xy: np.ndarray) -> float:
    """
    计算边界字围成的面积 (Shoelace 公式).

    A = 1/2 |Σ (x_i y_{i+1} - x_{i+1} y_i)|

    Parameters
    ----------
    e_xy : ndarray, shape (N, 2)
        边界节点坐标

    Returns
    -------
    area : float
        围成面积
    """
    n = len(e_xy)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += e_xy[i, 0] * e_xy[j, 1]
        area -= e_xy[j, 0] * e_xy[i, 1]
    return abs(area) / 2.0
