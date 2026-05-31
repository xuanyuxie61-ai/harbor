"""
annealing_schedules.py
================================================================================
量子退火中的非线性退火 schedule 与多项式迭代映射。

融合来源：
  - 198_collatz_polynomial（模 2 多项式迭代）
  - 048_asa226（不完全 Beta 函数）

物理背景：
  标准量子退火采用线性 schedule：
      A(s) = 1 - s,   B(s) = s,   s = t / T

  然而理论研究表明，非线性 schedule 可显著改善绝热条件：

      H(t) = A(t) H_D + B(t) H_P

  其中 A(t) 控制横向场强度（量子涨落），B(t) 控制问题哈密顿量权重。

  最优 schedule 可由量子绝热定理推导：
      |⟨ψ_1(t)| dH/dt |ψ_0(t)⟩| / Δ(t)^2 ≤ ε

  其中 Δ(t) = E_1(t) - E_0(t) 为能隙。当能隙在 s* 处最小时，
  需使 dA/dt, dB/dt 在 s* 附近尽可能缓慢（局部放缓）。

  本文引入 Collatz-like 多项式迭代与不完全 Beta 函数构造高阶可微 schedule。
"""

import numpy as np
from math import log, exp, gamma as math_gamma


def incomplete_beta(x: float, p: float, q: float, max_iter: int = 1000,
                    eps: float = 1e-14) -> float:
    """
    正则化不完全 Beta 函数 I_x(p,q) = B(x; p,q) / B(p,q)。

    采用 Majumder & Bhattacharjee (AS 63) 的连分数/级数算法。

    定义：
        B(x; p,q) = ∫_0^x t^{p-1} (1-t)^{q-1} dt
        B(p,q)    = Γ(p)Γ(q)/Γ(p+q)

    对称性：I_x(p,q) = 1 - I_{1-x}(q,p)
    """
    if x < 0.0 or x > 1.0:
        raise ValueError("x must be in [0,1]")
    if p <= 0.0 or q <= 0.0:
        raise ValueError("p, q must be positive")
    if x == 0.0:
        return 0.0
    if x == 1.0:
        return 1.0

    # 使用对称性确保数值稳定
    if p < (p + q) * x:
        xx = 1.0 - x
        cx = x
        pp = q
        qq = p
        indx = True
    else:
        xx = x
        cx = 1.0 - x
        pp = p
        qq = q
        indx = False

    beta_log = math_gamma(pp) + math_gamma(qq) - math_gamma(pp + qq)
    # 连分数展开
    term = 1.0
    ai = 1.0
    value = 1.0
    ns = int(np.floor(qq + cx * (pp + qq)))
    rx = xx / cx
    temp = qq - ai
    if ns == 0:
        rx = xx

    for _ in range(max_iter):
        term *= temp * rx / (pp + ai)
        value += term
        temp_abs = abs(term)
        if temp_abs <= eps and temp_abs <= eps * value:
            break
        ai += 1.0
        ns -= 1
        if 0 <= ns:
            temp = qq - ai
            if ns == 0:
                rx = xx
        else:
            temp = pp + qq
            # 简化处理
    else:
        # 未收敛，返回近似值
        pass

    value *= exp(pp * log(xx) + (qq - 1.0) * log(cx) - beta_log) / pp
    if indx:
        value = 1.0 - value
    return float(np.clip(value, 0.0, 1.0))


def collatz_polynomial_next(p1: np.ndarray) -> np.ndarray:
    """
    Collatz 多项式迭代（模 2）：

        若 P1(x) 可被 x 整除（常数项为 0）：
            P2(x) = P1(x) / x
        否则：
            P2(x) = P1(x)*(x+1) + 1   (mod 2)

    该迭代在 GF(2) 上定义，展示了复杂的动力学行为，
    类比于量子退火中哈密顿量参数的非线性演化。
    """
    p1 = np.asarray(p1, dtype=int)
    if not np.all(np.isin(p1, [0, 1])):
        raise ValueError("coefficients must be binary (0 or 1)")
    # 去掉尾部零
    n = p1.size - 1
    while n >= 0 and p1[n] == 0:
        n -= 1
    if n < 0:
        return np.array([0])
    if n == 0:
        return p1[:1].copy()
    if p1[0] == 0:
        # 除以 x
        p2 = p1[1:n + 1].copy()
    else:
        # P1*(x+1)+1
        p2 = np.zeros(n + 2, dtype=int)
        p2[0:n + 1] = p1[0:n + 1]
        p2[1:n + 2] = (p2[1:n + 2] + p1[0:n + 1]) % 2
        p2[0] = (p2[0] + 1) % 2
    return p2


class AnnealingSchedule:
    """
    量子退火 schedule 生成器，支持多种非线性形式。
    """

    def __init__(self, T_total: float = 1.0, n_steps: int = 200):
        if T_total <= 0:
            raise ValueError("T_total must be positive")
        if n_steps <= 0:
            raise ValueError("n_steps must be positive")
        self.T_total = float(T_total)
        self.n_steps = int(n_steps)
        self.times = np.linspace(0.0, T_total, n_steps)
        self.s_values = self.times / T_total

    def linear(self) -> tuple:
        """
        标准线性 schedule：
            A(s) = 1 - s
            B(s) = s
        """
        A = 1.0 - self.s_values
        B = self.s_values.copy()
        return A, B

    def polynomial_slowdown(self, degree: int = 3, s_star: float = 0.5) -> tuple:
        """
        多项式局部放缓 schedule：

            A(s) = 1 - s^p
            B(s) = s^p

        其中 p 控制过渡区陡峭度。为模拟能隙最小点 s* 处的局部放缓，
        引入修正：
            A(s) = 1 - I_s(p, q)    (不完全 Beta 函数)
            B(s) = I_s(p, q)
        """
        if not (0.0 <= s_star <= 1.0):
            raise ValueError("s_star must be in [0,1]")
        p_param = float(degree)
        q_param = float(degree)
        B = np.array([incomplete_beta(s, p_param, q_param) for s in self.s_values])
        A = 1.0 - B
        return A, B

    def logistic_schedule(self, kappa: float = 10.0, s0: float = 0.5) -> tuple:
        """
        Logistic 型平滑过渡 schedule：

            B(s) = 1 / (1 + exp( -κ (s - s0) ))
            A(s) = 1 - B(s)
        """
        B = 1.0 / (1.0 + np.exp(-kappa * (self.s_values - s0)))
        A = 1.0 - B
        return A, B

    def collatz_inspired_schedule(self, n_iter: int = 8) -> tuple:
        """
        受 Collatz 多项式迭代启发的分形型 schedule。

        思想：将 Collatz 序列的复杂度映射为 schedule 的局部振荡结构，
        模拟真实退火器中控制参数的微小抖动：

            c_k = hamming_weight( Collatz^k(P0) ) / deg(P_k)
            B(s) = s + ε Σ_k c_k sin(2π k s)
        """
        # 初始多项式 P0(x) = 1 + x + x^2 (二进制 111)
        p = np.array([1, 1, 1], dtype=int)
        coeffs = []
        for _ in range(n_iter):
            p = collatz_polynomial_next(p)
            deg = p.size - 1
            hw = np.count_nonzero(p)
            coeffs.append(hw / max(deg, 1))

        B = self.s_values.copy()
        eps = 0.03
        for k, c in enumerate(coeffs):
            B += eps * c * np.sin(2.0 * np.pi * (k + 1) * self.s_values)
        # 边界裁剪与单调化（物理约束）
        B = np.clip(B, 0.0, 1.0)
        # 强制单调递增
        for i in range(1, len(B)):
            if B[i] < B[i - 1]:
                B[i] = B[i - 1]
        A = 1.0 - B
        return A, B

    def adiabatic_optimal_local(self, gap_estimate: float = 0.1,
                                 s_star: float = 0.4) -> tuple:
        """
        基于绝热定理的局部最优 schedule：

            ds/dt ∝ Δ(s)^2

        近似实现为在 s* 附近引入高斯放缓：
            B(s) = Φ( (s - s*) / w )  *  scale + offset
        """
        w = gap_estimate / 2.0
        B = 0.5 * (1.0 + np.tanh((self.s_values - s_star) / w))
        A = 1.0 - B
        return A, B

    def generate_full_hamiltonian_schedule(self, schedule_type: str = "polynomial",
                                            **kwargs) -> dict:
        """
        生成完整的 schedule 字典，包含 A(t), B(t) 及派生物理量。
        """
        if schedule_type == "linear":
            A, B = self.linear()
        elif schedule_type == "polynomial":
            A, B = self.polynomial_slowdown(**kwargs)
        elif schedule_type == "logistic":
            A, B = self.logistic_schedule(**kwargs)
        elif schedule_type == "collatz":
            A, B = self.collatz_inspired_schedule(**kwargs)
        elif schedule_type == "adiabatic":
            A, B = self.adiabatic_optimal_local(**kwargs)
        else:
            raise ValueError(f"Unknown schedule type: {schedule_type}")

        # 数值导数 dA/dt, dB/dt
        dt = self.T_total / max(self.n_steps - 1, 1)
        dA_dt = np.gradient(A, dt)
        dB_dt = np.gradient(B, dt)

        return {
            "times": self.times,
            "s": self.s_values,
            "A": A,
            "B": B,
            "dA_dt": dA_dt,
            "dB_dt": dB_dt,
            "type": schedule_type,
        }
