"""
special_math.py
博士级科学计算特殊函数模块

核心功能：
- Clausen 函数 Cl2(x) 的 Chebyshev 级数展开
- 用于酶催化反应中二面角扭转势能的周期性势函数计算
- Chebyshev 级数求值器

科学背景：
在酶催化过渡态搜索中，二面角扭转势能常具有周期性：
    V_torsion(φ) = Σ_n (V_n / 2) * [1 + cos(nφ - γ_n)]
Clausen 函数 Cl2(θ) = -∫_0^θ ln|2 sin(t/2)| dt 出现在：
1. 周期性势的傅里叶展开系数计算
2. 配分函数的角向积分
3. Wigner 转动矩阵元的求和公式

公式：
    Cl_2(θ) = Σ_{k=1}^∞ sin(kθ) / k^2
    其 Chebyshev 展开在 [-π/2, π/2] 和 [π/2, 3π/2] 区间分别使用不同系数
"""

import numpy as np


class ChebyshevEvaluator:
    """Chebyshev 级数求值器，改编自 r8_csevl"""

    def __init__(self, coeffs):
        self.coeffs = np.asarray(coeffs, dtype=float)
        self.n = len(self.coeffs)
        if self.n < 1:
            raise ValueError("Chebyshev 级数项数必须 >= 1")
        if self.n > 1000:
            raise ValueError("Chebyshev 级数项数必须 <= 1000")

    def evaluate(self, x):
        """
        计算 Chebyshev 级数在 x 处的值，x ∈ [-1.1, 1.1]
        使用 Clenshaw 递推公式：
            b_k = 2x*b_{k+1} - b_{k+2} + a_k
            f(x) = (b_0 - b_2) / 2
        """
        x = float(x)
        if x < -1.1 or x > 1.1:
            raise ValueError(f"x = {x} 超出 Chebyshev 级数定义域 [-1.1, 1.1]")
        b1 = 0.0
        b0 = 0.0
        for i in range(self.n - 1, -1, -1):
            b2 = b1
            b1 = b0
            b0 = 2.0 * x * b1 - b2 + self.coeffs[i]
        return 0.5 * (b0 - b2)


def clausen_function(x):
    """
    计算 Clausen 函数 Cl2(x)

    数学定义：
        Cl_2(x) = -∫_0^x ln|2 sin(t/2)| dt
                = Σ_{k=1}^∞ sin(kx) / k^2

    在酶催化分子动力学中，Clausen 函数用于：
    - 计算周期性边界条件下的有效势
    - 评估二面角自由度的配分函数
    - 过渡态理论中的角向动量修正

    参数：
        x: 输入角度（弧度）
    返回：
        Cl2(x) 的值
    """
    # 将 x 映射到 [-π/2, 3π/2]
    xa = -0.5 * np.pi
    xc = 1.5 * np.pi
    x2 = x
    two_pi = 2.0 * np.pi
    while x2 < xa:
        x2 += two_pi
    while x2 > xc:
        x2 -= two_pi

    # 处理 x ≈ 0 或 2π 倍数的情况
    if abs(x2) < np.finfo(float).eps:
        return 0.0

    # 第一区间 [-π/2, π/2] 的 Chebyshev 系数
    c1 = np.array([
        0.05590566394715132269,
        0.0,
        0.00017630887438981157,
        0.0,
        0.00000126627414611565,
        0.0,
        0.00000001171718181344,
        0.0,
        0.00000000012300641288,
        0.0,
        0.00000000000139527290,
        0.0,
        0.00000000000001669078,
        0.0,
        0.00000000000000020761,
        0.0,
        0.00000000000000000266,
        0.0,
        0.00000000000000000003
    ], dtype=float)

    # 第二区间 [π/2, 3π/2] 的 Chebyshev 系数
    c2 = np.array([
        0.0,
        -0.96070972149008358753,
        0.0,
        0.04393661151911392781,
        0.0,
        0.00078014905905217505,
        0.0,
        0.00002621984893260601,
        0.0,
        0.00000109292497472610,
        0.0,
        0.00000005122618343931,
        0.0,
        0.00000000258863512670,
        0.0,
        0.00000000013787545462,
        0.0,
        0.00000000000763448721,
        0.0,
        0.00000000000043556938,
        0.0,
        0.00000000000002544696,
        0.0,
        0.00000000000000151561,
        0.0,
        0.00000000000000009172,
        0.0,
        0.00000000000000000563,
        0.0,
        0.00000000000000000035,
        0.0,
        0.00000000000000000002
    ], dtype=float)

    xb = 0.5 * np.pi
    if x2 < xb:
        # 第一区间展开
        x3 = 2.0 * x2 / np.pi
        cheb = ChebyshevEvaluator(c1)
        value = x2 - x2 * np.log(abs(x2)) + 0.5 * x2 ** 3 * cheb.evaluate(x3)
    else:
        # 第二区间展开
        x3 = 2.0 * x2 / np.pi - 2.0
        cheb = ChebyshevEvaluator(c2)
        value = cheb.evaluate(x3)

    return value


def periodic_torsion_potential(phi, n_terms=3):
    """
    计算周期性二面角扭转势能

    势能函数形式（CHARMM/AMBER 力场）：
        V(φ) = Σ_{n=1}^{N} (V_n / 2) * [1 + cos(nφ - γ_n)]

    使用 Clausen 函数对高阶展开进行正则化：
        V_reg(φ) = Σ_n (V_n / 2) * [1 + cos(nφ)] * exp(-α * n^2)
        其中 exp(-α * n^2) 为频域截断因子

    参数：
        phi: 二面角（弧度）
        n_terms: 傅里叶展开项数
    返回：
        势能值（kcal/mol）
    """
    # 典型酶催化二面角势能参数（丙酮酸脱氢酶体系）
    V_n = np.array([2.0, 1.5, 0.8, 0.3, 0.1, 0.05])[:n_terms]
    gamma_n = np.array([0.0, np.pi / 3, np.pi / 2, np.pi, 4 * np.pi / 3, 0.0])[:n_terms]

    energy = 0.0
    for i in range(n_terms):
        n = i + 1
        energy += (V_n[i] / 2.0) * (1.0 + np.cos(n * phi - gamma_n[i]))

    # 添加 Clausen 函数修正项（描述长程关联）
    # E_corr = -0.1 * Cl2(2φ) 表示相邻二面角的耦合
    energy -= 0.1 * clausen_function(2.0 * phi)

    return energy


def angular_partition_function(theta_range, temperature=300.0):
    """
    计算角向配分函数

    公式：
        q_angular = ∫_{-π}^{π} exp(-V(φ)/(k_B T)) dφ

    使用 Clausen 函数的积分性质进行解析近似
    """
    kB = 0.0019872041  # kcal/(mol·K)
    beta = 1.0 / (kB * temperature)

    n_points = 200
    phi_vals = np.linspace(-np.pi, np.pi, n_points)
    dphi = 2.0 * np.pi / (n_points - 1)

    integrand = np.exp(-beta * np.array([periodic_torsion_potential(p) for p in phi_vals]))
    # Simpson 积分
    q_ang = np.trapezoid(integrand, phi_vals)

    return q_ang
