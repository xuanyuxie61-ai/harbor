"""
root_finding.py — Chandrupatla 混合二次/二分法求根 (映射自 zero_chandrupatla)
=============================================================
本模块实现 Chandrupatla 求根算法及其在 PDF 全局拟合中的应用:
  (1) Chandrupatla 混合二次/二分法求根;
  (2) α_s(Q²) 的反求 (给定 α_s 求 Q²);
  (3) PDF 演化特征值的根 (用于稳定性分析);
  (4) χ² 最小值附近的抛物线拟合根.

核心公式 (Chandrupatla 算法):
    初始化: 给定变号区间 [a, b], f(a)f(b) < 0
    迭代:
        1) 计算 t ∈ [0,1] 使得 x = a + t(b−a)
        2) 评估 f(x)
        3) 若 f(x)f(a) < 0, 则 b ← x; 否则 a ← x
        4) 计算二次插值参数 Φ
        5) 若 Φ 在 [0,1] 内, 使用二次插值; 否则二分
        6) 收敛判据: |f(xm)| < ε 或 |b−a| < δ

核心公式 (逆二次插值):
    Φ = (f_1/(f_2 f_3)) × (f_3−f_1)(f_2−f_1)/(f_2−f_3)²
        − (f_1−f_2)/(f_3−f_2) × (f_2/(f_3))

核心公式 (α_s 反求):
    给定 α_s_target, 求 Q² 使 α_s(Q²) = α_s_target
    → Q² = Λ² × exp[4π / (β_0 α_s)]
"""
from __future__ import annotations
import math
from typing import Callable, Tuple, Optional

from phys_consts import (
    alpha_s_lo, alpha_s_nlo, beta_coefficient, LAMBDA_QCD_LO,
    EPS_NUMERICAL
)


# ============================================================
# 1. Chandrupatla 求根算法
# ============================================================
def chandrupatla_root(
    f: Callable[[float], float],
    x1: float, x2: float,
    epsilon: float = 1.0e-10,
    delta: float = 1.0e-8,
    max_iter: int = 100
) -> Tuple[float, float, int]:
    """
    Chandrupatla 混合二次/二分法求根 (映射自 zero_chandrupatla):

    算法核心:
        在变号区间 [x1, x2] 内寻找 f(x) = 0 的根.
        混合使用逆二次插值 (快速收敛) 和二分法 (保证收敛).

    判别条件:
        设 xm 为 |f| 最小的端点, f_t = (f_2−f_1)/(f_3−f_1)
        若 f_t < Φ_t (二次插值阈值), 使用插值; 否则二分.

    参数:
        f:        目标函数
        x1, x2:   变号区间端点 (f(x1)×f(x2) < 0)
        epsilon:  函数值收敛阈值
        delta:    区间宽度收敛阈值
        max_iter: 最大迭代次数

    返回:
        (xm, fm, calls): 根估计, 函数值, 函数调用次数
    """
    f1 = f(x1)
    f2 = f(x2)
    calls = 2

    if f1 * f2 > 0:
        # 非变号区间, 尝试缩小
        return 0.5 * (x1 + x2), 0.5 * (f1 + f2), calls

    t = 0.5
    for _ in range(max_iter):
        x0 = x1 + t * (x2 - x1)
        f0 = f(x0)
        calls += 1

        # 排列: x2 (最小 |f|), x1 (次小), x3 (被丢弃)
        if (f0 > 0) == (f1 > 0):
            x3, f3 = x1, f1
            x1, f1 = x0, f0
        else:
            x3, f3 = x2, f2
            x2, f1 = x1, f0
            x1, f1 = x0, f0

        # 找 xm (最小 |f| 的点)
        if abs(f2) < abs(f1):
            xm, fm = x2, f2
        else:
            xm, fm = x1, f1

        # 收敛检查
        if abs(fm) < epsilon or abs(x2 - x1) < delta:
            return xm, fm, calls

        # 计算二次插值参数
        if abs(f1 - f3) > EPS_NUMERICAL and abs(f2 - f3) > EPS_NUMERICAL:
            phi = ((f1 / (f2 * f3)) * ((f3 - f1) * (f2 - f1) / ((f2 - f3) ** 2 + EPS_NUMERICAL))
                   - (f1 - f2) / (f3 - f2 + EPS_NUMERICAL) * (f2 / (f3 + EPS_NUMERICAL)))
            phi = max(0.0, min(1.0, abs(phi)))
        else:
            phi = 0.5

        # 阈值: phi_t = 0.5 × sqrt(|f1/f2|)
        phi_t = 0.5 * math.sqrt(abs(f1 / max(abs(f2), EPS_NUMERICAL)))

        if phi < phi_t:
            t = phi  # 使用二次插值
        else:
            t = 0.5  # 使用二分

    return xm, fm, calls


# ============================================================
# 2. α_s 反求
# ============================================================
def invert_alpha_s(alpha_s_target: float, nf: int = 5,
                   q2_range: Tuple[float, float] = (1.0, 1.0e6)
                   ) -> Tuple[float, float]:
    """
    给定 α_s_target, 反求 Q² 使 α_s(Q²) = α_s_target.

    解析解 (LO):
        Q² = Λ² × exp[4π / (β_0 α_s)]

    此处也使用 Chandrupatla 方法验证.
    """
    # 解析解
    b0 = beta_coefficient(nf, 0)
    q2_analytic = LAMBDA_QCD_LO ** 2 * math.exp(4.0 * math.pi / (b0 * alpha_s_target))

    # Chandrupatla 验证
    def residual(log_q2: float) -> float:
        q2 = math.exp(log_q2)
        return alpha_s_lo(q2, nf) - alpha_s_target

    log_q2_min = math.log(max(q2_range[0], LAMBDA_QCD_LO ** 2 * 1.1))
    log_q2_max = math.log(q2_range[1])
    q2_chandra, _, _ = chandrupatla_root(residual, log_q2_min, log_q2_max)
    q2_chandra = math.exp(q2_chandra)

    return q2_analytic, q2_chandra


# ============================================================
# 3. DGLAP 特征值求根
# ============================================================
def dglap_eigenvalue_root(
    N: int, alpha_s: float, target_eigenvalue: float = 1.0,
    nf: int = 3
) -> Tuple[float, float]:
    """
    求 DGLAP 演化核的特征值:
        λ(N, Q²) = exp[(α_s/2π) P(N) ln(Q²/Q0²)]

    给定 λ_target, 求 N 使 λ(N) = λ_target.
    使用 Chandrupatla 方法.

    LO P_qq(N) 矩:
        P_{qq}(N) = C_F [−2 S_1(N) + 1/(N(N+1)) + 3/2]
    """
    CF = 4.0 / 3.0
    as2pi = alpha_s / (2.0 * math.pi)

    def harmonic(n: float) -> float:
        # 连续推广的调和数 (digamma 函数近似)
        n_int = max(int(n), 1)
        return sum(1.0 / k for k in range(1, n_int + 1))

    def eigenvalue_residual(log_q2_ratio: float) -> float:
        # 在固定 N 下, 求 Q²/Q0² 使特征值 = target
        P_N = CF * (-2.0 * harmonic(N) + 1.0 / (N * (N + 1)) + 1.5)
        lam = math.exp(as2pi * P_N * log_q2_ratio)
        return lam - target_eigenvalue

    # 搜索区间
    log_r_min = 0.01
    log_r_max = 20.0
    try:
        root, fval, calls = chandrupatla_root(
            eigenvalue_residual, log_r_min, log_r_max)
        return math.exp(root), calls
    except Exception:
        return 1.0, 0


# ============================================================
# 4. χ² 抛物线根
# ============================================================
def chi2_parabola_minimum(
    params_1d: list, chi2_values: list
) -> Tuple[float, float]:
    """
    通过三点抛物线拟合求 χ² 最小值位置:
        χ²(p) ≈ a p² + b p + c
        p_min = −b / (2a)
        χ²_min = c − b² / (4a)

    参数:
        params_1d: 三个参数值 [p1, p2, p3]
        chi2_values: 对应的 χ² 值 [χ1, χ2, χ3]
    返回:
        (p_min, chi2_min)
    """
    if len(params_1d) != 3 or len(chi2_values) != 3:
        return params_1d[1], chi2_values[1]

    p1, p2, p3 = params_1d
    c1, c2, c3 = chi2_values

    # 拉格朗日插值系数
    denom = (p1 - p2) * (p1 - p3) * (p2 - p3)
    if abs(denom) < EPS_NUMERICAL:
        return p2, c2

    a = (p3 * (c2 - c1) + p2 * (c1 - c3) + p1 * (c3 - c2)) / denom
    b = (p3 * p3 * (c1 - c2) + p2 * p2 * (c3 - c1) + p1 * p1 * (c2 - c3)) / denom

    if abs(a) < EPS_NUMERICAL:
        return p2, c2

    p_min = -b / (2.0 * a)
    c_coeff = c1 - a * p1 * p1 - b * p1
    chi2_min = c_coeff - b * b / (4.0 * a)
    return p_min, chi2_min
