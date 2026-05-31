"""
lattice_geometry.py
===================
晶格几何描述、边界追踪与 Diophantine 约束处理。

融合 boundary_word_hexagon、pram、mcnuggets 等项目的核心算法，
应用于晶体结构描述和周期性边界条件的数学处理。

核心数学原理
------------
六方密堆积（HCP）的基矢（2D）：
    a₁ = a(1, 0)
    a₂ = a(1/2, √3/2)

晶格平移不变性要求：
    r_i → r_i + n₁a₁ + n₂a₂    (n₁, n₂ ∈ ℤ)

Diophantine 约束（源自 mcnuggets 项目）：
    给定系数 a = (a₁, ..., a_n) 和右端 b，求所有非负整数解：
        a₁x₁ + a₂x₂ + ... + a_n x_n = b

    解空间维度：当 gcd(a₁,...,a_n) | b 时存在解。
    解的数量随 b 增长呈多项式增长。

边界词（源自 boundary_word_hexagon 和 pram）：
    用字母序列描述多边形边界，每个字母对应一个方向的步进。
    对于六方网格，6 个方向对应 6 个字母。

PRAM 问题中的边界词使用 12 个方向（考虑步长变化）。

周期多边形面积（Pick 定理）：
    A = I + B/2 - 1
    其中 I 为内部格点数，B 为边界格点数。
"""

import numpy as np
from typing import List, Tuple, Optional


# ----------------------------------------------------------------------
# 六方网格边界追踪
# ----------------------------------------------------------------------

HEX_DIRECTIONS = {
    '1': np.array([1, 0]),
    '2': np.array([0, 1]),
    '3': np.array([-1, 1]),
    '4': np.array([-1, 0]),
    '5': np.array([0, -1]),
    '6': np.array([1, -1]),
}


def boundary_range_hex(word: str, start: np.ndarray = None) -> Tuple[int, int, int, int]:
    """
    计算六方边界词描述的多边形的坐标范围。
    
    源自 boundary_word_hexagon 的 boundary_range 算法。
    
    参数:
        word: 由 '1'-'6' 组成的边界词
        start: 起始点 (i,j)，默认 (0,0)
    
    返回:
        (imin, imax, jmin, jmax)
    """
    if start is None:
        start = np.array([0, 0])
    i, j = int(start[0]), int(start[1])
    imin = imax = i
    jmin = jmax = j

    for ch in word:
        if ch in HEX_DIRECTIONS:
            di, dj = HEX_DIRECTIONS[ch]
            i += di
            j += dj
            imin = min(imin, i)
            imax = max(imax, i)
            jmin = min(jmin, j)
            jmax = max(jmax, j)

    return imin, imax, jmin, jmax


def boundary_trace_hex(word: str, start: np.ndarray = None) -> np.ndarray:
    """
    追踪六方边界词，返回边界顶点序列。
    """
    if start is None:
        start = np.array([0, 0])
    pts = [start.copy()]
    p = start.copy()
    for ch in word:
        if ch in HEX_DIRECTIONS:
            p = p + HEX_DIRECTIONS[ch]
            pts.append(p.copy())
    return np.array(pts)


def pram_boundary_word() -> Tuple[str, np.ndarray]:
    """
    PRAM 问题的边界词（源自 pram_grid_word）。
    
    返回边界词字符串和起始点。
    """
    w = (
        "AAAAAAAA"
        "CCCCCCCC"
        "EEEEEEEEEEEEEE"
        "CCCCCCCCCCCCCCCCCCCCCCCC"
        "FffFFffF"
        "GGGGGGGGGGGG"
        "JjjJJjjJ"
        "IIIIIIIIIIIIIIIIIIII"
        "KKKKKKKKKKKKKKKKKKKKKK"
    )
    p = np.array([0, 0])
    return w, p


# ----------------------------------------------------------------------
# Diophantine 方程求解
# ----------------------------------------------------------------------

def diophantine_nd_nonnegative(a: List[int], b: int) -> np.ndarray:
    """
    求解非负整数 Diophantine 方程的所有解。
    
    方程: a₁x₁ + a₂x₂ + ... + a_n x_n = b
    
    源自 diophantine_nd_nonnegative 项目。
    
    参数:
        a: 正整数系数列表
        b: 非负整数右端项
    
    返回:
        k×n 解矩阵，每行是一个解
    """
    a = np.array(a, dtype=int)
    a = a[a > 0]  # 只保留正系数
    n = len(a)
    if n == 0:
        return np.array([]).reshape(0, 0)
    if b < 0:
        return np.array([]).reshape(0, n)

    solutions = []
    y = np.zeros(n, dtype=int)
    j = 0
    r = b

    while True:
        # 计算剩余量
        r = b
        for idx in range(j):
            r -= a[idx] * y[idx]

        if j < n:
            y[j] = r // a[j]
            j += 1
        else:
            if r == 0:
                solutions.append(y.copy())
            # 回溯
            while j > 0:
                j -= 1
                if y[j] > 0:
                    y[j] -= 1
                    j += 1
                    break
            else:
                break

    if len(solutions) == 0:
        return np.array([]).reshape(0, n)
    sol = np.array(solutions)
    # 排序
    sol = sol[np.lexsort(sol.T[::-1])]
    return sol


def frobenius_number_2d(a: int, b: int) -> int:
    """
    两个互素正整数的 Frobenius 数：
        g(a,b) = ab - a - b
    
    即最大的不能表示为 ax + by（x,y≥0）的整数。
    """
    if np.gcd(a, b) != 1:
        return -1
    return a * b - a - b


# ----------------------------------------------------------------------
# 晶格构造与对称性
# ----------------------------------------------------------------------

def generate_hcp_lattice_2d(nx: int, ny: int,
                             lattice_constant: float = 1.0) -> np.ndarray:
    """
    生成二维六方密堆积（HCP）晶格点。
    
    基矢:
        a₁ = a(1, 0)
        a₂ = a(1/2, √3/2)
    """
    a = lattice_constant
    pts = []
    for j in range(ny):
        for i in range(nx):
            x = a * (i + 0.5 * j)
            y = a * (np.sqrt(3.0) / 2.0) * j
            pts.append([x, y])
    return np.array(pts)


def generate_square_lattice_2d(nx: int, ny: int,
                                lattice_constant: float = 1.0) -> np.ndarray:
    """生成二维正方晶格点。"""
    a = lattice_constant
    x = np.arange(nx) * a
    y = np.arange(ny) * a
    xv, yv = np.meshgrid(x, y)
    return np.column_stack([xv.ravel(), yv.ravel()])


def apply_periodic_boundary_hex(points: np.ndarray,
                                 lattice_constant: float = 1.0) -> np.ndarray:
    """
    对六方晶格应用周期性边界条件。
    
    使用六方原胞的斜坐标：
        r = n₁ a₁ + n₂ a₂
    """
    a = lattice_constant
    a1 = np.array([a, 0.0])
    a2 = np.array([0.5 * a, np.sqrt(3.0) / 2.0 * a])
    
    # 转换到斜坐标 (n1, n2)
    # [x, y] = n1*a1 + n2*a2
    # x = n1*a + 0.5*n2*a
    # y = n2*(sqrt(3)/2)*a
    # => n2 = 2y/(sqrt(3)*a), n1 = x/a - 0.5*n2
    n2 = 2.0 * points[:, 1] / (np.sqrt(3.0) * a)
    n1 = points[:, 0] / a - 0.5 * n2
    
    # 取整回绕
    n1 = n1 - np.floor(n1)
    n2 = n2 - np.floor(n2)
    
    # 转换回笛卡尔坐标
    new_x = n1 * a + 0.5 * n2 * a
    new_y = n2 * np.sqrt(3.0) / 2.0 * a
    return np.column_stack([new_x, new_y])


def lattice_miller_index_to_direction(h: int, k: int, l: int = 0,
                                       crystal_system: str = "hexagonal") -> np.ndarray:
    """
    将密勒指数转换为晶向向量。
    
    对于六方晶系 [hkil]（i = -(h+k)）：
        v = h·a₁ + k·a₂ + l·c
    """
    if crystal_system == "hexagonal":
        # 使用四指数 [uvtw] 到三指数 [UVW] 的转换
        # 这里简化为二维
        U = 2 * h + k
        V = 2 * k + h
        return np.array([U, V])
    else:
        return np.array([h, k, l])


def voronoi_cell_area_hex(lattice_constant: float = 1.0) -> float:
    """
    六方晶格的 Voronoi 单元面积（Wigner-Seitz 原胞）。
    
    A = (√3 / 2) a²
    """
    return np.sqrt(3.0) / 2.0 * lattice_constant ** 2


def coordination_number(crystal_structure: str = "fcc") -> int:
    """
    常见晶体结构的配位数。
    
    fcc: 12, bcc: 8, hcp: 12, sc: 6, diamond: 4
    """
    mapping = {
        "fcc": 12,
        "bcc": 8,
        "hcp": 12,
        "sc": 6,
        "diamond": 4,
        "hexagonal": 6,  # 2D
        "square": 4,     # 2D
    }
    return mapping.get(crystal_structure.lower(), 6)
