# -*- coding: utf-8 -*-
"""
error_analysis.py
=================
数值误差分析: 浮点误差, 截断误差, Horner 多项式求值.

核心算法 (来自 338_errors):
----------------------------
1) 浮点灾难性抵消 (Exercise 1):
   R = P^2 - 2Q^2  理论值 = 1
   展示浮点精度损失.

2) Horner 多项式求值 (来自 338_errors/r8poly_value_horner, f8, f11):
   p(x) = c_0 + x(c_1 + x(c_2 + ... + x c_n))

3) 矩阵指数 (来自 338_errors/r8mat_expm1):
   验证不同算法的数值误差.

在 fast ignition 中的应用:
- 非线性项 u^{5/2} 的舍入误差传播
- FD 截断误差估计
- 条件数与数值稳定性
"""

import math


# ============================================================
# Horner 多项式求值 (来自 338_errors/r8poly_value_horner)
# ============================================================
def poly_value_horner(n, c, x):
    """
    Horner 方法多项式求值 (来自 338_errors/r8poly_value_horner).

    p(x) = c[0] + c[1]*x + c[2]*x^2 + ... + c[n]*x^n
         = c[0] + x*(c[1] + x*(c[2] + ... + x*c[n]))

    参数:
        n: 多项式阶数
        c: 系数列表 (长度 n+1)
        x: 求值点
    返回:
        p(x)
    """
    value = c[n]
    for i in range(n - 1, -1, -1):
        value = value * x + c[i]
    return value


# ============================================================
# 浮点误差测试 (来自 338_errors/errors_test01)
# ============================================================
def catastrophic_cancellation_test():
    """
    灾难性抵消测试 (来自 338_errors/errors_test01).

    P = 665857, Q = 470832
    R = P^2 - 2Q^2  理论值 = 1

    展示浮点运算中的精度损失.
    """
    P = 665857.0
    Q = 470832.0
    R_computed = P ** 2 - 2.0 * Q ** 2
    R_exact = 1.0
    abs_error = abs(R_computed - R_exact)
    rel_error = abs_error / max(abs(R_exact), 1.0e-300)

    return {
        "P": P,
        "Q": Q,
        "R_computed": R_computed,
        "R_exact": R_exact,
        "abs_error": abs_error,
        "rel_error": rel_error,
    }


def polynomial_error_test():
    """
    多项式求值误差测试 (来自 338_errors/f8, f11).

    f(x) = 18.601 + 168.97x - 356.41x^2 + 170.4x^3
    在 x = 1 - 1/(2^12) 处有灾难性抵消.
    """
    # f8: c = [18.601, 168.97, -356.41, 170.4]
    c8 = [18.601, 168.97, -356.41, 170.4]
    x_test = 1.0 - 1.0 / (2 ** 12)

    # Horner 求值
    val_horner = poly_value_horner(3, c8, x_test)

    # 直接求值
    val_direct = c8[0] + c8[1] * x_test + c8[2] * x_test ** 2 + c8[3] * x_test ** 3

    # 精确值 (使用高精度)
    # 在 x = 1 - ε: f(x) ≈ f(1) - f'(1)ε
    f1 = sum(c8)  # f(1) = 18.601 + 168.97 - 356.41 + 170.4 = 1.561
    fp1 = c8[1] + 2 * c8[2] + 3 * c8[3]  # f'(1)
    val_approx = f1 - fp1 / (2 ** 12)

    return {
        "x_test": x_test,
        "val_horner": val_horner,
        "val_direct": val_direct,
        "val_approx": val_approx,
        "f1": f1,
        "fp1": fp1,
    }


# ============================================================
# 矩阵指数 (来自 338_errors/r8mat_expm1)
# ============================================================
def matrix_expm_power_series(A, n_terms=20):
    """
    矩阵指数幂级数 (来自 338_errors/r8mat_expm1):
        exp(A) = I + A + A^2/2! + A^3/3! + ...

    参数:
        A      : n×n 矩阵 (list of lists)
        n_terms: 展开项数
    返回:
        exp(A): n×n 矩阵
    """
    n = len(A)
    # 初始化结果 = I
    result = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    # 当前幂: A^k / k!
    term = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]

    for k in range(1, n_terms + 1):
        # term = term @ A / k
        new_term = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                s = 0.0
                for l in range(n):
                    s += term[i][l] * A[l][j]
                new_term[i][j] = s / k
        term = new_term
        for i in range(n):
            for j in range(n):
                result[i][j] += term[i][j]

    return result


# ============================================================
# FD 截断误差估计
# ============================================================
def truncation_error_estimate_2nd(f, x0, dx):
    """
    2阶 FD 截断误差估计:
        f''(x) ≈ (f(x+h) - 2f(x) + f(x-h)) / h^2
        截断误差 ≈ h^2/12 · f''''(x)

    通过 Richardson 外推估计:
        f''_exact ≈ (4 D(h/2) - D(h)) / 3
    其中 D(h) = (f(x+h) - 2f(x) + f(x-h)) / h^2
    """
    D_h = (f(x0 + dx) - 2.0 * f(x0) + f(x0 - dx)) / (dx ** 2)
    D_h2 = (f(x0 + dx / 2) - 2.0 * f(x0) + f(x0 - dx / 2)) / ((dx / 2) ** 2)
    # Richardson 外推
    D_extrap = (4.0 * D_h2 - D_h) / 3.0
    error_est = abs(D_h2 - D_h) / 3.0
    return {
        "D_h": D_h,
        "D_h2": D_h2,
        "D_extrap": D_extrap,
        "error_estimate": error_est,
        "order": 2.0,
    }


def truncation_error_estimate_4th(f, x0, dx):
    """
    4阶 FD 截断误差估计.

    使用 5 点模板:
        D_4(h) = (-f(x+2h) + 16f(x+h) - 30f(x) + 16f(x-h) - f(x-2h)) / (12h^2)

    Richardson 外推:
        D_extrap = (16 D_4(h/2) - D_4(h)) / 15
    """
    def D4(h):
        return (-f(x0 + 2 * h) + 16.0 * f(x0 + h) - 30.0 * f(x0)
                + 16.0 * f(x0 - h) - f(x0 - 2 * h)) / (12.0 * h ** 2)

    D_h = D4(dx)
    D_h2 = D4(dx / 2)
    D_extrap = (16.0 * D_h2 - D_h) / 15.0
    error_est = abs(D_h2 - D_h) / 15.0

    return {
        "D_h": D_h,
        "D_h2": D_h2,
        "D_extrap": D_extrap,
        "error_estimate": error_est,
        "order": 4.0,
    }


# ============================================================
# 条件数与误差放大
# ============================================================
def condition_number_error_amplification(cond, machine_eps=2.22e-16):
    """
    条件数导致的误差放大:
        relative_error ≤ cond · machine_eps

    参数:
        cond      : 矩阵条件数
        machine_eps: 机器精度
    返回:
        估计的相对误差上界
    """
    return cond * machine_eps


# ============================================================
# 完整误差分析
# ============================================================
def full_error_analysis():
    """
    执行完整误差分析.
    """
    results = {}

    # 1. 灾难性抵消
    results["cancellation"] = catastrophic_cancellation_test()

    # 2. 多项式求值误差
    results["polynomial"] = polynomial_error_test()

    # 3. 矩阵指数
    A = [[0.0, 1.0], [-1.0, 0.0]]  # 旋转矩阵
    results["matrix_exp"] = matrix_expm_power_series(A, n_terms=15)

    # 4. 截断误差 (测试函数: f(x) = sin(x), f''(0) = 0)
    results["truncation_2nd"] = truncation_error_estimate_2nd(
        math.sin, 0.5, 0.01)
    results["truncation_4th"] = truncation_error_estimate_4th(
        math.sin, 0.5, 0.01)

    return results


def print_error_summary(results):
    """打印误差分析摘要."""
    print("\n" + "=" * 72)
    print("数值误差分析")
    print("=" * 72)

    # 灾难性抵消
    c = results["cancellation"]
    print("  灾难性抵消测试:")
    print("    P = {:.0f}, Q = {:.0f}".format(c["P"], c["Q"]))
    print("    R = P^2 - 2Q^2 = {:.1f} (理论: {:.1f})".format(
        c["R_computed"], c["R_exact"]))
    print("    绝对误差: {:.4e}".format(c["abs_error"]))

    # 多项式
    p = results["polynomial"]
    print("  多项式求值 (Horner):")
    print("    x = {:.8f}".format(p["x_test"]))
    print("    f_horner = {:.8e}".format(p["val_horner"]))
    print("    f_direct = {:.8e}".format(p["val_direct"]))
    print("    f(1) = {:.4e}, f'(1) = {:.4e}".format(p["f1"], p["fp1"]))

    # 矩阵指数
    expA = results["matrix_exp"]
    print("  矩阵指数 exp([0,1;-1,0]):")
    print("    [0] = [{:+.6f}, {:+.6f}]".format(expA[0][0], expA[0][1]))
    print("    [1] = [{:+.6f}, {:+.6f}]".format(expA[1][0], expA[1][1]))
    print("    理论: [[cos1, sin1], [-sin1, cos1]]")
    print("         = [[{:+.6f}, {:+.6f}], [{:+.6f}, {:+.6f}]]".format(
        math.cos(1), math.sin(1), -math.sin(1), math.cos(1)))

    # 截断误差
    t2 = results["truncation_2nd"]
    t4 = results["truncation_4th"]
    print("  截断误差估计 (f=sin, x=0.5, h=0.01):")
    print("    2阶 D(h)   = {:.8e}".format(t2["D_h"]))
    print("    2阶 D(h/2) = {:.8e}".format(t2["D_h2"]))
    print("    2阶外推    = {:.8e}, 误差 ≈ {:.2e}".format(
        t2["D_extrap"], t2["error_estimate"]))
    print("    4阶 D(h)   = {:.8e}".format(t4["D_h"]))
    print("    4阶 D(h/2) = {:.8e}".format(t4["D_h2"]))
    print("    4阶外推    = {:.8e}, 误差 ≈ {:.2e}".format(
        t4["D_extrap"], t4["error_estimate"]))
    print("    理论 f''(0.5) = {:.8e}".format(-math.sin(0.5)))

    print("=" * 72)
