"""
special_functions.py — 特殊函数库
===================================
种子项目映射:
  443_fn (FNLIB/SLATEC特殊函数) → 散射振幅与截面计算所需特殊函数
  470_gl_fast_rule (Gauss-Legendre) → 高精度数值积分

实现:
  - Gamma函数, 对数Gamma
  - 误差函数 erf/erfc
  - Bessel函数近似
  - Breit-Wigner共振
  - Gauss-Legendre正交节点与权重
  - 相空间体积
"""
import numpy as np
import math


def gamma_func(x):
    """Gamma函数 Γ(x), 使用Lanczos近似."""
    if x <= 0 and x == int(x):
        return float('inf')
    return math.exp(log_gamma(x)) * (1.0 if x > 0 else -1.0)


def log_gamma(x):
    """对数Gamma函数 ln Γ(x), Stirling+Lanczos."""
    if x <= 0:
        return float('inf')
    # Lanczos近似 (g=7, n=9)
    if x < 0.5:
        return math.log(math.pi / math.sin(math.pi * x)) - log_gamma(1.0 - x)

    x -= 1.0
    g = 7
    c = [
        0.99999999999980993, 676.5203681218851, -1259.1392167224028,
        771.32342877765313, -176.61502916214059, 12.507343278686905,
        -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7,
    ]
    s = c[0]
    for i in range(1, g + 2):
        s += c[i] / (x + i)
    t = x + g + 0.5
    return 0.5 * math.log(2 * math.pi) + (x + 0.5) * math.log(t) - t + math.log(s)


def erf_func(x):
    """误差函数 erf(x), Abramowitz-Stegun近似."""
    sign = 1 if x >= 0 else -1
    x = abs(x)
    t = 1.0 / (1.0 + 0.3275911 * x)
    y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t
                - 0.284496736) * t + 0.254829592) * t * math.exp(-x * x)
    return sign * y


def erfc_func(x):
    """互补误差函数 erfc(x) = 1 - erf(x)."""
    return 1.0 - erf_func(x)


def breit_wigner(M, M0, Gamma):
    """
    relativistic Breit-Wigner分布:
      BW(M; M0, Γ) = (2M0Γ/π) / [(M²-M0²)² + M0²Γ²]
    """
    if Gamma <= 0:
        return 0.0
    num = 2.0 * M0 * Gamma / math.pi
    den = (M**2 - M0**2)**2 + M0**2 * Gamma**2
    return num / max(den, 1e-30)


def breit_wigner_pdf(M, M0, Gamma):
    """归一化Breit-Wigner PDF."""
    if Gamma <= 0 or M0 <= 0:
        return 0.0
    # 归一化: ∫ BW dM = 1
    norm = math.atan(M0 / Gamma)
    if norm < 1e-10:
        norm = 1e-10
    return breit_wigner(M, M0, Gamma) / (2.0 * norm / Gamma)


def alpha_s_running(Q, MZ=91.1876, alpha_s_MZ=0.1179):
    """
    QCD跑动耦合常数 (1-loop):
      α_s(Q) = α_s(MZ) / [1 + b0 * α_s(MZ) * ln(Q²/MZ²) / (2π)]
    b0 = 11 - 2*n_f/3, n_f = 5 (对 Q > mb)
    """
    if Q <= 0:
        return 0.0
    nf = 5 if Q > 4.18 else 4 if Q > 1.3 else 3
    b0 = 11.0 - 2.0 * nf / 3.0
    log_ratio = math.log(Q**2 / MZ**2)
    denom = 1.0 + b0 * alpha_s_MZ * log_ratio / (2.0 * math.pi)
    if denom <= 0:
        return 10.0  # 非微扰区域
    return alpha_s_MZ / denom


def legendre_P(l, x):
    """Legendre多项式 P_l(x), 递推关系."""
    if l == 0:
        return 1.0
    if l == 1:
        return x
    P_prev2 = 1.0
    P_prev1 = x
    for n in range(2, l + 1):
        P = ((2 * n - 1) * x * P_prev1 - (n - 1) * P_prev2) / n
        P_prev2 = P_prev1
        P_prev1 = P
    return P


def gauss_legendre_nodes_weights(n):
    """
    Gauss-Legendre正交的节点与权重.
    种子项目 470_gl_fast_rule 映射: 高精度数值积分.
    使用Newton迭代求解P_n(x)=0的根.
    """
    if n <= 0:
        raise ValueError("n 必须为正")
    if n == 1:
        return np.array([0.0]), np.array([2.0])

    nodes = np.zeros(n)
    weights = np.zeros(n)

    for i in range((n + 1) // 2):
        # 初始猜测 (Chebyshev近似)
        x = math.cos(math.pi * (i + 0.75) / (n + 0.5))

        for _ in range(100):
            # 计算 P_n(x) 和 P_n'(x)
            P0, P1 = 1.0, x
            for j in range(2, n + 1):
                P2 = ((2 * j - 1) * x * P1 - (j - 1) * P0) / j
                P0, P1 = P1, P2
            # P_n'(x) = n*(x*P_n - P_{n-1})/(x²-1)
            dP = n * (x * P1 - P0) / (x**2 - 1.0) if abs(x**2 - 1) > 1e-30 else 0.0
            dx = -P1 / dP if abs(dP) > 1e-30 else 0.0
            x += dx
            if abs(dx) < 1e-15:
                break

        nodes[i] = -x
        nodes[n - 1 - i] = x
        w = 2.0 / ((1 - x**2) * dP**2) if abs(dP) > 1e-30 else 0.0
        weights[i] = w
        weights[n - 1 - i] = w

    return nodes, weights


def phase_space_2body(s, m1, m2):
    """
    二体相空间体积:
      Φ₂(s; m1, m2) = π/(2s) * λ^{1/2}(s, m1², m2²)
    λ(a,b,c) = a² + b² + c² - 2ab - 2ac - 2bc (Källén函数)
    """
    if s < (m1 + m2)**2:
        return 0.0
    lam = s**2 + m1**4 + m2**4 - 2*s*m1**2 - 2*s*m2**2 - 2*m1**2*m2**2
    if lam < 0:
        return 0.0
    return math.pi / (2.0 * s) * math.sqrt(lam)


def phase_space_3body(s, m1, m2, m3, n_mc=10000):
    """
    三体相空间体积 (Monte Carlo).
    种子项目 501_hand_area 映射: Monte Carlo积分.
    """
    if s < (m1 + m2 + m3)**2:
        return 0.0

    rng = np.random.RandomState(42)
    count = 0
    for _ in range(n_mc):
        # 随机采样 Dalitz 图
        E1_max = (s + m1**2 - (m2 + m3)**2) / (2 * math.sqrt(s))
        E1_min = m1
        E1 = rng.uniform(E1_min, E1_max)
        E2_max = (s + m2**2 - (m1 + m3)**2) / (2 * math.sqrt(s))
        E2_min = m2
        E2 = rng.uniform(E2_min, E2_max)
        # 运动学检查
        cos12_max = 1.0
        p1 = math.sqrt(max(E1**2 - m1**2, 0))
        p2 = math.sqrt(max(E2**2 - m2**2, 0))
        E3 = math.sqrt(s) - E1 - E2
        if E3 < m3:
            continue
        m12_sq = s + m1**2 + m2**2 - 2*math.sqrt(s)*(E1+E2) + 2*E1*E2 - 2*p1*p2*cos12_max
        if m12_sq >= (m1+m2)**2:
            count += 1

    vol_range = (E1_max - E1_min) * (E2_max - E2_min)
    return vol_range * count / n_mc
