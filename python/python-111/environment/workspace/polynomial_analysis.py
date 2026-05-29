"""
多项式结式分析与构象空间约束求解模块
基于 polynomial_resultant 核心算法：
- Sylvester 矩阵构造
- 结式 (Resultant) 计算
- 公共根检测

在蛋白质折叠中的应用：
- 检测势能面 (PES) 多项式约束的交点/临界点
- 蛋白质结构判定中的代数约束（loop closure）
- 二面角势能面交叉点分析
- 构象空间约束流形的分岔点识别

数学基础:
    结式定义:
        设 p(x) = a_m ∏_{i=1}^{m}(x - α_i)
            q(x) = b_n ∏_{j=1}^{n}(x - β_j)
        Res(p, q) = a_m^n b_n^m ∏_{i=1}^{m}∏_{j=1}^{n}(α_i - β_j)
    
    Sylvester 矩阵:
        S 为 (m+n) × (m+n) 矩阵，其前 n 行由 p 的系数平移填充，
        后 m 行由 q 的系数平移填充。
        Res(p, q) = det(S)
    
    性质: Res(p, q) = 0  ⟺  p 和 q 有公共根。
"""

import numpy as np
from typing import Tuple


def sylvester_matrix(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """
    构造两个多项式的 Sylvester 矩阵。
    
    设 p(x) = p_m x^m + ... + p_0
        q(x) = q_n x^n + ... + q_0
    
    Sylvester 矩阵 S 的结构:
        前 n 行: p_m, p_{m-1}, ..., p_0, 0, ..., 0  (逐行右移)
        后 m 行: q_n, q_{n-1}, ..., q_0, 0, ..., 0  (逐行右移)
    
    Parameters
    ----------
    p : np.ndarray
        多项式 p 的系数，从高次到低次 [p_m, ..., p_0]。
    q : np.ndarray
        多项式 q 的系数，从高次到低次 [q_n, ..., q_0]。
    
    Returns
    -------
    S : np.ndarray, shape (m+n, m+n)
        Sylvester 矩阵。
    """
    m = len(p) - 1
    n = len(q) - 1
    if m < 0 or n < 0:
        raise ValueError("Polynomial degrees must be non-negative")
    
    S = np.zeros((m + n, m + n))
    # 前 n 行: p 的系数
    for i in range(n):
        S[i, i:i + m + 1] = p
    # 后 m 行: q 的系数
    for i in range(m):
        S[n + i, i:i + n + 1] = q
    return S


def polynomial_resultant_sylvester(p: np.ndarray, q: np.ndarray) -> float:
    """
    通过 Sylvester 矩阵行列式计算结式。
    
    Parameters
    ----------
    p, q : np.ndarray
        多项式系数。
    
    Returns
    -------
    resultant : float
        结式值。
    """
    S = sylvester_matrix(p, q)
    return float(np.linalg.det(S))


def polynomial_resultant_roots(p: np.ndarray, q: np.ndarray) -> float:
    """
    通过求根和乘积公式计算结式。
    
    公式:
        Res(p, q) = p_m^n * q_n^m * ∏_{i=1}^{m}∏_{j=1}^{n}(α_i - β_j)
    
    Parameters
    ----------
    p, q : np.ndarray
        多项式系数。
    
    Returns
    -------
    resultant : float
        结式值。
    """
    m = len(p) - 1
    n = len(q) - 1
    
    if m == 0 or n == 0:
        # 常数多项式
        return p[0] ** n * q[0] ** m if len(p) > 0 and len(q) > 0 else 0.0
    
    roots_p = np.roots(p)
    roots_q = np.roots(q)
    
    lead_p = p[0]
    lead_q = q[0]
    
    prod = 1.0
    for alpha in roots_p:
        for beta in roots_q:
            prod *= (alpha - beta)
    
    resultant = (lead_p ** n) * (lead_q ** m) * prod
    return float(np.real(resultant))


def find_critical_points_polynomial(potential_poly: np.ndarray) -> np.ndarray:
    """
    通过结式方法找到多项式势能的临界点（导数为零的点）。
    
    对于一维势能 V(x)，临界点满足 dV/dx = 0。
    若 V 为多项式，则 dV/dx 也为多项式，求根即可。
    
    对于二维势能 V(x, y)，临界点满足 ∂V/∂x = 0 和 ∂V/∂y = 0。
    可用结式消元法：将两个方程视为关于 y 的多项式，计算结式消去 y，
    得到关于 x 的单变量方程。
    
    本函数实现一维情况。
    
    Parameters
    ----------
    potential_poly : np.ndarray
        势能多项式系数 [a_n, ..., a_0]。
    
    Returns
    -------
    critical_points : np.ndarray
        临界点坐标。
    """
    # 导数多项式: dV/dx = n*a_n x^{n-1} + ... + a_1
    n = len(potential_poly) - 1
    if n < 1:
        return np.array([])
    
    deriv = np.array([potential_poly[i] * (n - i) for i in range(n)])
    roots = np.roots(deriv)
    # 只保留实根
    real_roots = np.real(roots[np.abs(np.imag(roots)) < 1e-8])
    return np.sort(real_roots)


def detect_bifurcation_points(poly1: np.ndarray, poly2: np.ndarray,
                               x_range: Tuple[float, float] = (-2.0, 2.0)) -> np.ndarray:
    """
    检测两个势能多项式的交点（分岔点）。
    
    交点满足 p1(x) = p2(x)，即 p1(x) - p2(x) = 0。
    若两多项式次数不同，先补齐系数。
    
    Parameters
    ----------
    poly1, poly2 : np.ndarray
        两个多项式的系数。
    x_range : tuple
        搜索范围。
    
    Returns
    -------
    intersections : np.ndarray
        交点 x 坐标。
    """
    max_len = max(len(poly1), len(poly2))
    p1 = np.zeros(max_len)
    p2 = np.zeros(max_len)
    p1[max_len - len(poly1):] = poly1
    p2[max_len - len(poly2):] = poly2
    
    diff = p1 - p2
    # 去除前导零
    while len(diff) > 1 and abs(diff[0]) < 1e-14:
        diff = diff[1:]
    
    roots = np.roots(diff)
    real_roots = np.real(roots[np.abs(np.imag(roots)) < 1e-8])
    # 筛选在范围内的根
    intersections = real_roots[(real_roots >= x_range[0]) & (real_roots <= x_range[1])]
    return np.sort(intersections)


def construct_dihedral_potential_polynomial(coeffs: np.ndarray) -> np.ndarray:
    """
    构造 Ryckaert-Bellemans 型二面角势能的多项式表示。
    
    Ryckaert-Bellemans 势能:
        V(φ) = Σ_{n=0}^{5} C_n * cos^n(φ)
    
    通过变量替换 x = cos(φ)，转化为关于 x 的多项式:
        V(x) = Σ C_n * x^n
    
    Parameters
    ----------
    coeffs : np.ndarray, shape (6,)
        系数 [C_0, C_1, ..., C_5]。
    
    Returns
    -------
    poly : np.ndarray
        多项式系数（低次在前）。
    """
    # coeffs[i] = C_i
    # 返回标准多项式系数 [C_5, C_4, ..., C_0]（高次在前）
    poly = coeffs[::-1].copy()
    return poly


def analyze_potential_landscape_criticality(coeffs: np.ndarray) -> dict:
    """
    分析一维势能景观的临界点性质（极小值/极大值/拐点）。
    
    通过二阶导数判断:
        V''(x) > 0 → 局部极小值 (稳定态，对应折叠/未折叠)
        V''(x) < 0 → 局部极大值 (过渡态)
        V''(x) = 0 → 高阶临界点
    
    Parameters
    ----------
    coeffs : np.ndarray
        势能多项式系数（高次在前）。
    
    Returns
    -------
    result : dict
        包含 critical_points, types, barrier_heights。
    """
    cp = find_critical_points_polynomial(coeffs)
    
    n = len(coeffs) - 1
    # 一阶导数
    d1 = np.array([coeffs[i] * (n - i) for i in range(n)])
    # 二阶导数
    d2 = np.array([d1[i] * (n - 1 - i) for i in range(n - 1)])
    
    types = []
    energies = []
    for x in cp:
        v = np.polyval(coeffs, x)
        v2 = np.polyval(d2, x)
        energies.append(v)
        if v2 > 1e-6:
            types.append("minimum")
        elif v2 < -1e-6:
            types.append("maximum")
        else:
            types.append("degenerate")
    
    # 计算势垒高度
    barriers = []
    minima_indices = [i for i, t in enumerate(types) if t == "minimum"]
    for i in range(len(minima_indices) - 1):
        idx1 = minima_indices[i]
        idx2 = minima_indices[i + 1]
        # 找到两者之间的极大值
        max_energy = -np.inf
        for j in range(idx1 + 1, idx2):
            if types[j] == "maximum" and energies[j] > max_energy:
                max_energy = energies[j]
        if max_energy > -np.inf:
            barriers.append(max_energy - energies[idx1])
    
    return {
        "critical_points": cp,
        "types": types,
        "energies": energies,
        "barrier_heights": barriers,
    }
