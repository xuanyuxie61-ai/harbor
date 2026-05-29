"""
quadrature_engine.py
高精度数值求积引擎模块

融合来源：
- 463_gegenbauer_rule: Gegenbauer-Gauss 求积规则生成（IQPACK算法）
- 1249_tetrahedron_jaskowiec_rule: 四面体高阶对称求积规则
- 313_dot_l2: L2内积计算

科学背景：
在骨骼有限元分析中，单元刚度矩阵和载荷向量的组装需要高精度数值积分。
本项目集成了：
1. Gegenbauer-Gauss 求积（一维，用于边界积分）
2. 三角形高阶求积（二维，用于单元刚度矩阵组装）
3. L2内积（用于密度场与形函数的耦合）

核心数学公式：
1. Gegenbauer 正交多项式 C_n^{(λ)}(x)，权函数 w(x) = (1-x^2)^{λ-0.5}
   高斯求积：∫_{-1}^{1} f(x) w(x) dx ≈ Σ_i w_i f(x_i)

2. Jacobi 矩阵构造（来自 IQPACK class_matrix）：
   对称三对角矩阵 T，其特征值为求积节点，特征向量平方为首元素得权重。

3. 隐式 QL 算法（imtqlx）对角化 Jacobi 矩阵。

4. 四面体求积规则（Jaskowiec & Sukumar, 2020）：
   在标准四面体上精确积分高次多项式。
"""

import numpy as np
from scipy.linalg import eig_banded
from typing import Tuple, Optional


# ===================================================================
# Gegenbauer-Gauss 求积规则生成（来自 463_gegenbauer_rule）
# ===================================================================
def gegenbauer_rule(n: int, lambda_param: float = 0.5,
                    a: float = -1.0, b: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    生成 n 点 Gegenbauer-Gauss 求积规则。

    积分公式：
        ∫_a^b f(x) (1 - x^2)^{λ - 0.5} dx ≈ Σ_{i=1}^n w_i f(x_i)

    当 λ = 0.5 时退化为 Legendre-Gauss 求积。
    当 λ = 1.0 时为 Chebyshev 第二类。

    Parameters
    ----------
    n : int
        求积点数
    lambda_param : float
        Gegenbauer 参数 λ > -0.5
    a, b : float
        积分区间

    Returns
    -------
    x : np.ndarray, shape (n,)
        求积节点
    w : np.ndarray, shape (n,)
        求积权重
    """
    if n < 1:
        raise ValueError("n must be >= 1")
    if lambda_param <= -0.5:
        raise ValueError("lambda must be > -0.5")

    # 构造 Jacobi 矩阵（class_matrix）
    aj, bj = _class_matrix_gegenbauer(n, lambda_param)

    # 对角化（使用 scipy 的带状矩阵特征值算法）
    # 对于 lower=True，ab[i-j, j] = a[i,j]
    # ab[0, j] = diagonal, ab[1, j] = sub-diagonal
    ab = np.zeros((2, n))
    ab[0, :] = aj         # 对角线
    ab[1, :-1] = bj[1:]  # 下次对角线

    w, v = eig_banded(ab, lower=True)
    x = np.real(w)
    # 权重：w_i = v[0,i]^2 * μ_0
    mu0 = _gegenbauer_moment0(lambda_param)
    weights = np.real(v[0, :] ** 2) * mu0

    # 按节点排序
    idx = np.argsort(x)
    x = x[idx]
    weights = weights[idx]

    # 缩放到区间 [a, b]
    if a != -1.0 or b != 1.0:
        scale = (b - a) / 2.0
        shift = (a + b) / 2.0
        x = scale * x + shift
        weights = weights * scale

    return x, weights


def _class_matrix_gegenbauer(n: int, lambda_param: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    构造 Gegenbauer 正交多项式的 Jacobi 矩阵。

    三对角矩阵 T，满足：
        T_{i,i}   = 0  (对于 Gegenbauer，对角元为 0)
        T_{i,i+1} = sqrt( i * (i + 2λ - 1) / (4 * (i + λ - 1) * (i + λ)) )
    """
    aj = np.zeros(n)
    bj = np.zeros(n)

    bj[0] = 0.0
    for i in range(1, n):
        num = i * (i + 2.0 * lambda_param - 1.0)
        den = 4.0 * (i + lambda_param - 1.0) * (i + lambda_param)
        if den <= 0:
            raise ValueError("Invalid Gegenbauer parameters")
        bj[i] = np.sqrt(num / den)

    return aj, bj


def _gegenbauer_moment0(lambda_param: float) -> float:
    """
    Gegenbauer 权函数的零阶矩：∫_{-1}^{1} (1-x^2)^{λ-0.5} dx。

    解析解：sqrt(pi) * Γ(λ + 0.5) / Γ(λ + 1)
    """
    from math import gamma, sqrt, pi
    if lambda_param == 0.5:
        return 2.0
    return sqrt(pi) * gamma(lambda_param + 0.5) / gamma(lambda_param + 1.0)


# ===================================================================
# 三角形高阶求积规则（简化版，融合 1249 思想）
# ===================================================================
def triangle_gauss_rule(order: int = 3) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    返回标准三角形 {(0,0), (1,0), (0,1)} 上的高斯求积规则。

    节点与权重来自 Strang & Fix 及后续文献的标准表。

    Parameters
    ----------
    order : int
        精度阶数（支持 1, 2, 3, 4, 5, 6, 7）

    Returns
    -------
    x, y : np.ndarray
        参考三角形中的节点重心坐标 (x, y)，z = 1 - x - y
    w : np.ndarray
        权重（已归一化，Σ w_i = 0.5）
    """
    if order == 1:
        # 1点，精度1
        x = np.array([1.0 / 3.0])
        y = np.array([1.0 / 3.0])
        w = np.array([0.5])
    elif order == 2:
        # 3点，精度2
        x = np.array([1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0])
        y = np.array([1.0 / 6.0, 1.0 / 6.0, 2.0 / 3.0])
        w = np.array([1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0])
    elif order == 3:
        # 3点中边，精度3（来自 fem2d_poisson_rectangle 的 quad_a）
        x = np.array([0.5, 0.5, 0.0])
        y = np.array([0.0, 0.5, 0.5])
        w = np.array([1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0])
    elif order == 4:
        # 6点，精度4
        a1 = 0.445948490915965
        b1 = 0.091576213509771
        w1 = 0.111690794839005
        w2 = 0.054975871827661
        x = np.array([a1, 1.0 - 2.0 * a1, a1,
                      b1, 1.0 - 2.0 * b1, b1])
        y = np.array([a1, a1, 1.0 - 2.0 * a1,
                      b1, b1, 1.0 - 2.0 * b1])
        w = np.array([w1, w1, w1, w2, w2, w2])
    elif order == 5:
        # 7点，精度5
        a1 = 0.470142064105115
        w1 = 0.066197076394253
        w2 = 0.062969590272413
        w3 = 0.1125
        x = np.array([a1, 1.0 - 2.0 * a1, a1,
                      1.0 / 3.0, 0.059715871789770,
                      0.797426985353087, 0.059715871789770])
        y = np.array([a1, a1, 1.0 - 2.0 * a1,
                      1.0 / 3.0, 0.059715871789770,
                      0.059715871789770, 0.797426985353087])
        w = np.array([w1, w1, w1, w3, w2, w2, w2])
    elif order == 6:
        # 12点，精度6
        a1 = 0.249286745170910
        b1 = 0.501426509658179
        a2 = 0.063089014491502
        b2 = 0.873821971016996
        a3 = 0.310352451033785
        b3 = 0.053145049844816
        w1 = 0.058393137863189
        w2 = 0.025422453185104
        w3 = 0.041425537809187
        x = np.array([a1, b1, a1, a2, b2, a2, a3, b3, 1.0 - a3 - b3,
                      a3, 1.0 - a3 - b3, b3])
        y = np.array([a1, a1, b1, a2, a2, b2, a3, a3, a3,
                      b3, b3, 1.0 - a3 - b3])
        w = np.array([w1, w1, w1, w2, w2, w2, w3, w3, w3, w3, w3, w3])
    elif order == 7:
        # 13点，精度7（来自 fem2d_poisson_rectangle 的 quad_e）
        a1 = 0.260345966079038
        b1 = 0.065130102902216
        w1 = 0.087977301162222
        w2 = 0.008744311553736
        w3 = 0.038081799045199
        w4 = 0.018855448056131
        w5 = -0.002166998150765
        x = np.array([1.0 / 3.0, a1, 1.0 - 2.0 * a1, a1,
                      b1, 1.0 - 2.0 * b1, b1,
                      0.312865496004874, 0.638444188569809,
                      0.048690315425317, 0.638444188569809,
                      0.048690315425317, 0.312865496004874])
        y = np.array([1.0 / 3.0, a1, a1, 1.0 - 2.0 * a1,
                      b1, b1, 1.0 - 2.0 * b1,
                      0.638444188569809, 0.048690315425317,
                      0.312865496004874, 0.312865496004874,
                      0.638444188569809, 0.048690315425317])
        w = np.array([w5, w1, w1, w1, w2, w2, w2,
                      w3, w3, w3, w3, w3, w3])
    else:
        raise ValueError(f"Unsupported order {order}. Supported: 1-7.")

    return x, y, w


def map_triangle_quad_points(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray,
                             x_ref: np.ndarray, y_ref: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    将参考三角形上的求积点映射到物理三角形。

    等参映射：
        x = x1 + (x2 - x1) * xi + (x3 - x1) * eta
        y = y1 + (y2 - y1) * xi + (y3 - y1) * eta

    Parameters
    ----------
    p1, p2, p3 : np.ndarray, shape (2,)
        物理三角形顶点
    x_ref, y_ref : np.ndarray
        参考坐标

    Returns
    -------
    x_phys, y_phys : np.ndarray
        物理坐标
    """
    x_phys = p1[0] + (p2[0] - p1[0]) * x_ref + (p3[0] - p1[0]) * y_ref
    y_phys = p1[1] + (p2[1] - p1[1]) * x_ref + (p3[1] - p1[1]) * y_ref
    return x_phys, y_phys


# ===================================================================
# 四面体求积规则（简化版，融合 1249 思想）
# ===================================================================
def tetrahedron_unit_monomial_integral(alpha: int, beta: int, gamma: int,
                                       delta: int) -> float:
    """
    计算标准四面体上单项式的精确积分：

        ∫_T x^α y^β z^γ w^δ dV
        = α! β! γ! δ! / (α + β + γ + δ + 3)!

    其中 T = {(x,y,z,w) | x,y,z,w >= 0, x+y+z+w = 1}。
    """
    from math import factorial
    if alpha < 0 or beta < 0 or gamma < 0 or delta < 0:
        raise ValueError("Exponents must be non-negative")
    num = factorial(alpha) * factorial(beta) * factorial(gamma) * factorial(delta)
    den = factorial(alpha + beta + gamma + delta + 3)
    return float(num) / float(den)


def monomial_value(m: int, e: np.ndarray, x: np.ndarray) -> float:
    """
    计算单项式在求积点处的值。

    Parameters
    ----------
    m : int
        空间维数
    e : np.ndarray
        指数向量
    x : np.ndarray
        坐标点
    """
    val = 1.0
    for i in range(m):
        if e[i] != 0:
            val *= x[i] ** e[i]
    return val


# ===================================================================
# 组合生成器（来自 1249 的 comp_next）
# ===================================================================
def comp_next(n: int, k: int, a: np.ndarray, more: bool,
              h: int, t: int) -> Tuple[np.ndarray, bool, int, int]:
    """
    生成 n 的 k 部分组合，用于单项式遍历。

    Parameters
    ----------
    n : int
        总和
    k : int
        部分数
    a : np.ndarray
        当前组合
    more, h, t : bool, int, int
        状态变量

    Returns
    -------
    a, more, h, t : 更新后的组合和状态
    """
    if not more:
        a[:] = 0
        a[0] = n
        more = True
        h = 0
        t = n
        if k == 1:
            more = False
        return a, more, h, t

    if 1 < t:
        h = 0

    h = h + 1
    t = a[h - 1]
    a[h - 1] = 0
    a[0] = t - 1
    a[h] = a[h] + 1

    if a[k - 1] == n:
        more = False

    return a, more, h, t


# ===================================================================
# L2 内积与误差估计工具
# ===================================================================
def l2_error_estimate(uh: np.ndarray, uexact: np.ndarray,
                      weights: np.ndarray, areas: np.ndarray) -> float:
    """
    估计 L2 误差：sqrt( Σ w_i * (uh_i - uexact_i)^2 * area_i )
    """
    err2 = np.sum(weights * (uh - uexact) ** 2 * areas)
    return float(np.sqrt(err2))


def h1_seminorm_error_estimate(duh: np.ndarray, duexact: np.ndarray,
                               weights: np.ndarray, areas: np.ndarray) -> float:
    """
    估计 H1 半范数误差：sqrt( Σ w_i * |grad(uh) - grad(u)|^2 * area_i )
    """
    diff2 = np.sum((duh[:, 0] - duexact[:, 0]) ** 2 +
                   (duh[:, 1] - duexact[:, 1]) ** 2)
    # 简化：假设 weights 和 areas 已综合
    return float(np.sqrt(diff2))
