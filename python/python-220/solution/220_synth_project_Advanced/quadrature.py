"""
quadrature.py — 高阶数值积分规则
============================================
来源项目: 931_pyramid_felippa_rule (金字塔Felippa积分规则),
          396_fem1d_pmethod (Gauss-Legendre 求积)

本模块实现:
  1. Gauss-Legendre 求积 (任意阶)
  2. 三角形区域积分规则 (Dunavant 规则)
  3. 金字塔区域 Felippa 积分规则 (1, 5, 8 点)
  4. 单项式精确积分验证

数学基础:
  积分近似: ∫_Ω f(x) dx ≈ Σ_{i=1}^{n} w_i f(x_i)
  代数精度: n 点 Gauss 规则精确积分 2n-1 次多项式
"""

import numpy as np
from typing import Tuple, List, Callable
from config import EPS_NUM, PI


# ============================================================
#  Gauss-Legendre 求积 (来源: 396_fem1d_pmethod)
# ============================================================
def gauss_legendre_rule(n: int, a: float = -1.0, b: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """n 点 Gauss-Legendre 求积规则

    在区间 [a, b] 上计算: ∫_a^b f(x) dx ≈ Σ w_i f(x_i)

    算法: 利用 Legendre 多项式 P_n(x) 的零点作为节点,
    权重由 w_i = 2 / ((1-x_i^2) [P'_n(x_i)]^2) 给出.

    使用 Golub-Welsch 算法 (对称三对角矩阵的特征值问题):
      J 为 n×n 对称三对角矩阵, 其中
      J[i, i+1] = J[i+1, i] = (i+1)/sqrt((2i+1)(2i+3))
      节点 = J 的特征值
      权重 = 2 * (特征向量的第一个分量)^2

    Args:
        n: 求积点数
        a, b: 积分区间

    Returns:
        (points, weights): 节点和权重
    """
    if n < 1:
        raise ValueError(f"求积点数必须 >= 1, got {n}")
    if n == 1:
        pts = np.array([0.5 * (a + b)])
        wts = np.array([b - a])
        return pts, wts

    # Golub-Welsch 算法
    i = np.arange(1, n, dtype=np.float64)
    beta = i / np.sqrt(4.0 * i * i - 1.0)

    # 构造对称三对角矩阵
    J = np.diag(beta, -1) + np.diag(beta, 1)
    eigenvalues, eigenvectors = np.linalg.eigh(J)

    # 节点 (从 [-1,1] 映射到 [a,b])
    pts_ref = eigenvalues
    wts_ref = 2.0 * eigenvectors[0, :] ** 2

    # 仿射变换: x = (b-a)/2 * xi + (a+b)/2
    pts = 0.5 * (b - a) * pts_ref + 0.5 * (a + b)
    wts = 0.5 * (b - a) * wts_ref

    return pts, wts


# ============================================================
#  三角形积分规则 (来源: 931 扩展)
# ============================================================
def triangle_quadrature_1pt() -> Tuple[np.ndarray, np.ndarray]:
    """1 点三角形积分规则 (精度 1)
    重心坐标: (1/3, 1/3, 1/3), 权重 = 面积
    """
    pts = np.array([[1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]])
    wts = np.array([1.0])  # 归一化 (乘以面积使用)
    return pts, wts


def triangle_quadrature_3pt() -> Tuple[np.ndarray, np.ndarray]:
    """3 点三角形积分规则 (精度 2)
    重心坐标:
      (1/6, 1/6, 2/3), (1/6, 2/3, 1/6), (2/3, 1/6, 1/6)
    权重: 1/3 每个
    """
    pts = np.array([
        [1.0 / 6.0, 1.0 / 6.0, 2.0 / 3.0],
        [1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0],
        [2.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0],
    ])
    wts = np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])
    return pts, wts


def triangle_quadrature_7pt() -> Tuple[np.ndarray, np.ndarray]:
    """7 点三角形积分规则 (精度 5, Dunavant)
    包括重心点和 6 个对称分布点
    """
    a1 = 0.059715871789770
    b1 = 0.470142064105115
    a2 = 0.797426985353087
    b2 = 0.101286507323456

    w0 = 0.225000000000000
    w1 = 0.132394152788506
    w2 = 0.125939180544827

    pts = np.array([
        [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
        [a1, b1, b1], [b1, a1, b1], [b1, b1, a1],
        [a2, b2, b2], [b2, a2, b2], [b2, b2, a2],
    ])
    wts = np.array([w0, w1, w1, w1, w2, w2, w2])
    return pts, wts


# ============================================================
#  金字塔区域 Felippa 积分规则 (来源: 931_pyramid_felippa_rule)
#  单位金字塔: -(1-z) <= x,y <= (1-z), 0 <= z <= 1
#  体积 = 4/3
# ============================================================
def pyramid_volume() -> float:
    """单位金字塔体积: V = (1/3) * base_area * height = 4/3"""
    return 4.0 / 3.0


def pyramid_quadrature_1pt() -> Tuple[np.ndarray, np.ndarray]:
    """1 点金字塔积分规则 (精度 1)
    质心位置: (0, 0, 1/4), 权重 = 4/3
    """
    pts = np.array([[0.0, 0.0, 0.25]])
    wts = np.array([4.0 / 3.0])
    return pts, wts


def pyramid_quadrature_5pt() -> Tuple[np.ndarray, np.ndarray]:
    """5 点金字塔积分规则 (Felippa, 精度 2)
    1 个顶点规则 + 4 个底部规则
    """
    pts = np.array([
        [0.0, 0.0, 0.617142857142857],   # 顶部附近
        [0.471428571428571, 0.0, 0.114285714285714],
        [-0.471428571428571, 0.0, 0.114285714285714],
        [0.0, 0.471428571428571, 0.114285714285714],
        [0.0, -0.471428571428571, 0.114285714285714],
    ])
    wts = np.array([
        0.551111111111111,
        0.346111111111111,
        0.346111111111111,
        0.346111111111111,
        0.346111111111111,
    ])
    return pts, wts


def pyramid_quadrature_8pt() -> Tuple[np.ndarray, np.ndarray]:
    """8 点金字塔积分规则 (Felippa, 精度 3)"""
    a = 0.527472805334735
    b = 0.170411775939426
    c = 0.666666666666667
    d = 0.261900144263918
    w_ab = 0.380920117412870
    w_cd = 0.285746549253796

    pts = np.array([
        [a, 0.0, d], [-a, 0.0, d],
        [0.0, a, d], [0.0, -a, d],
        [b, 0.0, c], [-b, 0.0, c],
        [0.0, b, c], [0.0, -b, c],
    ])
    wts = np.array([w_ab, w_ab, w_ab, w_ab, w_cd, w_cd, w_cd, w_cd])
    return pts, wts


# ============================================================
#  单项式精确积分 (来源: 931 验证工具)
# ============================================================
def pyramid_monomial_exact(a: int, b: int, c: int) -> float:
    """单位金字塔上单项式 x^a * y^b * z^c 的精确积分

    解析公式:
      ∫∫∫_P x^a y^b z^c dV = 0 (若 a 或 b 为奇数)
      = (2/(a+1))(2/(b+1)) * Σ_{i=0}^{2+a+b} (-1)^i C(2+a+b,i) / (i+c+1)
        (若 a, b 均为偶数)

    其中 C(n,k) 为二项式系数.
    """
    if a % 2 != 0 or b % 2 != 0:
        return 0.0  # 奇对称性

    from math import comb
    n = 2 + a + b
    result = 0.0
    for i in range(n + 1):
        sign = (-1) ** i
        binom = comb(n, i)
        denom = i + c + 1
        if denom > EPS_NUM:
            result += sign * binom / denom
    result *= (2.0 / (a + 1)) * (2.0 / (b + 1))
    return result


# ============================================================
#  数值积分执行器
# ============================================================
def integrate_function_2d(f: Callable, x_range: Tuple[float, float],
                          y_range: Tuple[float, float], n_quad: int = 5) -> float:
    """矩形域上的 2D Gauss 积分

    ∫∫ f(x,y) dx dy ≈ Σ_i Σ_j w_i w_j f(x_i, y_j)

    Args:
        f: 被积函数 f(x, y) -> float
        x_range: x 积分区间
        y_range: y 积分区间
        n_quad: 每维求积点数

    Returns:
        积分近似值
    """
    pts_x, wts_x = gauss_legendre_rule(n_quad, x_range[0], x_range[1])
    pts_y, wts_y = gauss_legendre_rule(n_quad, y_range[0], y_range[1])

    result = 0.0
    for i in range(n_quad):
        for j in range(n_quad):
            result += wts_x[i] * wts_y[j] * f(pts_x[i], pts_y[j])
    return result


def integrate_on_triangle(f: Callable, vertices: np.ndarray, order: int = 2) -> float:
    """三角形区域上的数值积分

    ∫∫_T f(x,y) dA = |T| * Σ_k w_k f(x_k, y_k)

    其中 (x_k, y_k) 由重心坐标转换为物理坐标:
      x_k = λ1*x1 + λ2*x2 + λ3*x3
      y_k = λ1*y1 + λ2*y2 + λ3*y3

    Args:
        f: 被积函数
        vertices: 三角形顶点 (3, 2)
        order: 精度阶数 (1, 2, 或 5)

    Returns:
        积分值
    """
    # 三角形面积
    v1, v2, v3 = vertices[0], vertices[1], vertices[2]
    area = 0.5 * abs((v2[0] - v1[0]) * (v3[1] - v1[1]) - (v3[0] - v1[0]) * (v2[1] - v1[1]))

    if area < EPS_NUM:
        return 0.0

    # 选择积分规则
    if order <= 1:
        bary_pts, bary_wts = triangle_quadrature_1pt()
    elif order <= 2:
        bary_pts, bary_wts = triangle_quadrature_3pt()
    else:
        bary_pts, bary_wts = triangle_quadrature_7pt()

    result = 0.0
    for k in range(len(bary_wts)):
        lam1, lam2, lam3 = bary_pts[k]
        # 重心 → 物理坐标
        x = lam1 * v1[0] + lam2 * v2[0] + lam3 * v3[0]
        y = lam1 * v1[1] + lam2 * v2[1] + lam3 * v3[1]
        result += bary_wts[k] * f(x, y)

    return area * result


# ============================================================
#  积分规则精度验证
# ============================================================
def verify_quadrature_accuracy() -> dict:
    """验证各积分规则的代数精度

    对每种规则, 测试递增阶数的单项式, 找到最大精确阶数.

    Returns:
        dict: 各规则的验证结果
    """
    results = {}

    # Gauss-Legendre 验证
    for n in [2, 4, 8]:
        pts, wts = gauss_legendre_rule(n, -1.0, 1.0)
        max_exact = 0
        for degree in range(2 * n + 2):
            # 精确值: ∫_{-1}^{1} x^degree dx
            if degree % 2 == 0:
                exact = 2.0 / (degree + 1)
            else:
                exact = 0.0
            # 数值值
            numeric = sum(w * p ** degree for p, w in zip(pts, wts))
            if abs(numeric - exact) < 1e-12:
                max_exact = degree
            else:
                break
        results[f"gauss_legendre_{n}pt"] = {
            "max_exact_degree": max_exact,
            "expected": 2 * n - 1
        }

    # 金字塔规则验证
    for name, rule_fn in [("1pt", pyramid_quadrature_1pt),
                          ("5pt", pyramid_quadrature_5pt),
                          ("8pt", pyramid_quadrature_8pt)]:
        pts, wts = rule_fn()
        max_err = 0.0
        for a_exp in range(4):
            for b_exp in range(4):
                for c_exp in range(4):
                    if a_exp + b_exp + c_exp > 3:
                        continue
                    numeric = sum(
                        w * (p[0]**a_exp) * (p[1]**b_exp) * (p[2]**c_exp)
                        for p, w in zip(pts, wts)
                    )
                    exact = pyramid_monomial_exact(a_exp, b_exp, c_exp)
                    max_err = max(max_err, abs(numeric - exact))
        results[f"pyramid_{name}"] = {"max_error": max_err}

    return results
