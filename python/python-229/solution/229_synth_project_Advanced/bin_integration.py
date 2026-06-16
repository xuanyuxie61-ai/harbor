"""
bin_integration.py
==================

来源:
  - 466_gen_laguerre_exactness :  广义 Gauss-Laguerre 求积精确度检验
  - 1302_triangle_exactness   :  三角形上求积规则的多项式精确度测试

物理重构
--------
在 unfold 中, 我们需要计算两类积分:

1. 半无限区间上的谱积分 (来自 466):
       I_n = ∫_0^∞ E^n f(E) dE
   当 f(E) ∝ exp(-E/T) 时, 精确值为 T^{n+1} Γ(n+1) = T^{n+1} n!.
   Gauss-Laguerre 求积:
       I_n ≈ Σ_{k=1}^N w_k E_k^n,  E_k, w_k 为 generalized Laguerre 节点与权重.
   验证: 比较数值 vs 解析, 确认 exactness degree = 2N-1.

2. 二维 bin 上的积分 (来自 1302):
       ∫∫_{triangle} E_x^m E_y^n dE_x dE_y = m! n! / (m + n + 2)!
   用于响应矩阵列的 2D 校准 (例如双喷注能量响应).

本模块提供:
  - Gauss-Laguerre 节点/权重生成 (Golub-Welsch 算法)
  - 精确度检验函数 (exactness_degree)
  - 三角形上 Duffy 变换求积
  - bin 上的谱矩计算
"""

from __future__ import annotations
from typing import List, Tuple
import math
import special_functions as sf


# ===========================================================================
#        Gauss-Laguerre 求积 (广义, 参数 α > -1)
# ===========================================================================

def gauss_laguerre_rule(n: int, alpha: float = 0.0) -> Tuple[List[float], List[float]]:
    """
    广义 Gauss-Laguerre 求积节点与权重.

    求积公式:
        ∫_0^∞ x^α exp(-x) f(x) dx ≈ Σ_{k=1}^n w_k f(x_k)

    精确度: 对任意 f ∈ P_{2n-1} 精确成立.

    算法 (Golub-Welsch):
        广义 Laguerre 多项式满足三项递推:
            x L_k^α = -(k+1) L_{k+1}^α + (2k + α + 1) L_k^α - (k + α) L_{k-1}^α
        ⇒  Jacobi 矩阵 J 为三对角:
            J_{k,k} = 2k + α + 1
            J_{k,k+1} = J_{k+1,k} = -sqrt((k+1)(k+1+α))
        节点 = J 的特征值, 权重 ∝ (特征向量第 1 分量)^2 × Γ(α+1).
    """
    if n < 1:
        return [], []
    if alpha <= -1.0:
        raise ValueError(f"alpha={alpha} 必须 > -1")

    # 构造 Jacobi 矩阵
    diag = [2.0 * k + alpha + 1.0 for k in range(n)]
    offdiag = [-math.sqrt((k + 1.0) * (k + 1.0 + alpha)) for k in range(n - 1)]

    # QR 算法求特征值/特征向量 (简单实现, n 较小)
    evecs = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    evals = list(diag)
    e = list(offdiag) + [0.0]

    # QL with implicit shifts
    for l in range(n):
        iter_count = 0
        while True:
            # 找小的 subdiag
            m = l
            while m < n - 1:
                dd = abs(evals[m]) + abs(evals[m + 1])
                if dd == 0.0:
                    dd = 1e-300
                if abs(e[m]) <= 1e-14 * dd:
                    break
                m += 1
            if m == l:
                break
            iter_count += 1
            if iter_count > 100:
                break
            g = (evals[l + 1] - evals[l]) / (2.0 * e[l])
            r = math.sqrt(g * g + 1.0)
            g = evals[m] - evals[l] + e[l] / (g + r * (1.0 if g >= 0 else -1.0))
            s = c = 1.0
            p = 0.0
            for i in range(m - 1, l - 1, -1):
                f = s * e[i]
                b = c * e[i]
                if abs(f) >= abs(g):
                    c = g / f
                    r = math.sqrt(c * c + 1.0)
                    e[i + 1] = f * r
                    s = 1.0 / r
                    c *= s
                else:
                    s = f / g
                    r = math.sqrt(s * s + 1.0)
                    e[i + 1] = g * r
                    c = 1.0 / r
                    s *= c
                g = evals[i + 1] - p
                r = (evals[i] - g) * s + 2.0 * c * b
                p = s * r
                evals[i + 1] = g + p
                g = c * r - b
                for k in range(n):
                    f = evecs[k][i + 1]
                    evecs[k][i + 1] = s * evecs[k][i] + c * f
                    evecs[k][i] = c * evecs[k][i] - s * f
            evals[l] -= p
            e[l] = g
            if m < n - 1:
                e[m] = 0.0

    # 排序
    order = sorted(range(n), key=lambda i: evals[i])
    nodes = [evals[i] for i in order]

    # 权重: w_k = Γ(α+1) (v_{k,0})^2
    gamma_alpha1 = sf.r8_gamma(alpha + 1.0)
    weights = [gamma_alpha1 * evecs[0][order[k]] ** 2 for k in range(n)]

    return nodes, weights


def gen_laguerre_exactness(
    n: int, alpha: float, degree_max: int,
) -> List[float]:
    """
    测试 Gauss-Laguerre 规则的 exactness degree.

    对每个 m = 0, 1, ..., degree_max:
        解析: I_m = Γ(α + m + 1)
        数值: Q_m = Σ_{k=1}^n w_k x_k^m
        相对误差: |Q_m - I_m| / |I_m|

    返回相对误差列表.
    """
    nodes, weights = gauss_laguerre_rule(n, alpha)
    errors = []
    for m in range(degree_max + 1):
        # 解析
        exact = sf.r8_gamma(alpha + m + 1.0)
        # 数值
        num = sum(w * x ** m for w, x in zip(weights, nodes))
        rel = abs(num - exact) / (abs(exact) + 1e-300)
        errors.append(rel)
    return errors


# ===========================================================================
#           三角形求积 (Duffy 变换)
# ===========================================================================

def triangle_quadrature_duffy(
    v1: Tuple[float, float],
    v2: Tuple[float, float],
    v3: Tuple[float, float],
    f,
    order: int = 5,
) -> float:
    """
    三角形 (v1, v2, v3) 上的函数 f(x, y) 积分.

    Duffy 变换: 将三角形映射到单位正方形:
        x = v1_x + (v2_x - v1_x) u + (v3_x - v1_x) u v
        y = v1_y + (v2_y - v1_y) u + (v3_y - v1_y) u v
        Jacobian = 2 A u,  A = 三角形面积

    积分:
        ∫∫_T f(x,y) dx dy = 2A ∫_0^1 ∫_0^1 f(x(u,v), y(u,v)) u du dv

    使用 Gauss-Legendre 求积在 [0,1]^2 上.
    """
    # 面积
    A = 0.5 * abs(
        (v2[0] - v1[0]) * (v3[1] - v1[1]) - (v3[0] - v1[0]) * (v2[1] - v1[1])
    )
    if A < 1e-300:
        return 0.0

    import phase_space_grid as psg
    nodes, weights = psg.gauss_legendre_cos_theta(order)
    # 映射 [-1,1] → [0,1]
    u_nodes = [0.5 * (xi + 1.0) for xi in nodes]
    u_weights = [0.5 * wi for wi in weights]

    s = 0.0
    for i in range(order):
        for j in range(order):
            u = u_nodes[i]
            v = u_nodes[j]
            x = v1[0] + (v2[0] - v1[0]) * u + (v3[0] - v1[0]) * u * v
            y = v1[1] + (v2[1] - v1[1]) * u + (v3[1] - v1[1]) * u * v
            s += u_weights[i] * u_weights[j] * u * f(x, y)
    return 2.0 * A * s


def triangle_exactness(
    v1: Tuple[float, float],
    v2: Tuple[float, float],
    v3: Tuple[float, float],
    degree_max: int,
    quadrature_order: int = 7,
) -> List[float]:
    """
    测试三角形求积规则的 exactness degree.

    对每个 (m, n) 满足 m + n ≤ degree_max:
        解析: ∫∫_T x^m y^n dx dy (通过仿射变换到参考三角形后计算)
        数值: Duffy 求积
        相对误差

    参考三角形 (0,0), (1,0), (0,1) 上的精确值:
        ∫∫ x^m y^n dx dy = m! n! / (m + n + 2)!
    """
    errors = []
    for total in range(degree_max + 1):
        for m in range(total + 1):
            n_exp = total - m
            # 解析 (参考三角形)
            exact = sf.r8_gamma(m + 1) * sf.r8_gamma(n_exp + 1) / sf.r8_gamma(m + n_exp + 3)
            # 数值
            num = triangle_quadrature_duffy(
                v1, v2, v3, lambda x, y, _m=m, _n=n_exp: x ** _m * y ** _n,
                order=quadrature_order,
            )
            # 缩放 (参考三角形面积 = 0.5)
            A = 0.5 * abs(
                (v2[0] - v1[0]) * (v3[1] - v1[1]) - (v3[0] - v1[0]) * (v2[1] - v1[1])
            )
            # 精确值需要乘以 2A (仿射变换的 Jacobian 常数)
            exact_scaled = exact * 2.0 * A
            rel = abs(num - exact_scaled) / (abs(exact_scaled) + 1e-300)
            errors.append(rel)
    return errors


# ===========================================================================
#        谱矩计算 (使用 Gauss-Laguerre)
# ===========================================================================

def spectral_moments(
    spectrum,
    n_moments: int = 6,
    alpha: float = 0.0,
    quadrature_order: int = 12,
) -> List[float]:
    """
    计算能谱 f(E) 的矩:
        M_n = ∫_0^∞ E^n f(E) dE

    通过 Gauss-Laguerre 变换:
        令 f(E) = exp(-E) g(E),  则 M_n = ∫_0^∞ E^n exp(-E) g(E) dE
        ≈ Σ_k w_k E_k^n g(E_k)

    若 f 不含 exp(-E) 因子, 则 g(E) = f(E) exp(E) (可能数值不稳).
    对衰减快的谱, 直接截断区间并用 Gauss-Legendre 更稳.
    这里返回 Gauss-Laguerre 估计.
    """
    nodes, weights = gauss_laguerre_rule(quadrature_order, alpha)
    moments = []
    for n in range(n_moments):
        s = 0.0
        for k in range(quadrature_order):
            E = nodes[k]
            # f(E) 需额外除以 exp(-E) 权重
            # 但 f 本身可能很小; 安全处理
            g_val = spectrum(E) * math.exp(E) if E < 500 else 0.0
            s += weights[k] * (E ** n) * g_val
        moments.append(s)
    return moments


__all__ = [
    "gauss_laguerre_rule",
    "gen_laguerre_exactness",
    "triangle_quadrature_duffy",
    "triangle_exactness",
    "spectral_moments",
]
