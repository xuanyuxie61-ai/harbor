"""
local_stress_return.py
博士级大变形非线性有限元分析 — 局部应力回归与求根模块

融合原项目:
  - 1229_test_zero: Brent 法、Newton 法、Muller 法非线性求根

核心数学:
  在大变形弹塑性/超弹性分析中，常需在积分点级别求解非线性方程。
  本模块实现多种求根算法，用于:
    1. 等效塑性应变 Δε_p 的求解（径向返回算法）
    2. 体积比 J 的约束投影
    3. 损伤阈值判定后的应力松弛

  1. Newton 法:
     求解 f(x) = 0
     x_{k+1} = x_k - f(x_k) / f'(x_k)
     局部二阶收敛，要求初值足够接近真解

  2. Brent 法:
     结合二分法、割线法和反二次插值
     要求初始区间 [a,b] 满足 f(a)f(b) < 0
     收敛阶超线性 (≈1.324)，且保证全局收敛

  3. Muller 法:
     使用二次插值，可处理复根（实数版本）

  应用示例 — 超弹性体积约束:
    在某些约束格式中，需要求解:
      g(J) = K * ln(J) + p = 0
    其中 p 为拉格朗日乘子（静水压力）
"""

import numpy as np


class RootFinderError(Exception):
    pass


def newton_method(f, df, x0, fatol=1e-12, xatol=1e-12, step_max=50, xmin=-1e6, xmax=1e6):
    """
    Newton 法求根

    源自原项目 1229_test_zero (newton)

    数学:
      x_{k+1} = x_k - f(x_k) / f'(x_k)

    输入:
        f: 目标函数
        df: 导数函数
        x0: 初始猜测
        fatol: 函数值绝对容差
        xatol: 步长绝对容差
        step_max: 最大迭代步数
        xmin, xmax: 允许搜索区间
    """
    xa = float(x0)
    fxa = float(f(xa))
    step = 0.0

    for step_num in range(1, step_max + 1):
        if xa < xmin or xa > xmax:
            raise RootFinderError(f"Iterate left region [{xmin}, {xmax}]: x={xa}")

        if abs(fxa) <= fatol:
            return xa

        if step_num > 1 and abs(step) <= xatol:
            return xa

        fp = float(df(xa))
        if abs(fp) < 1e-15:
            raise RootFinderError("Derivative vanishes in Newton method")

        step = fxa / fp
        xa = xa - step
        fxa = float(f(xa))

    # 达到最大迭代次数，返回当前最优值
    return xa


def brent_method(f, xa, xb, fatol=1e-12, xatol=1e-12, xrtol=1e-12, step_max=100):
    """
    Brent 法求根

    源自原项目 1229_test_zero (brent)

    数学:
      结合二分法的稳健性和反二次插值的快速收敛性
      在每一步选择更安全的方法（二分或插值）

    输入:
        f: 目标函数
        xa, xb: 初始区间端点，要求 f(xa)*f(xb) <= 0
        fatol: 函数值容差
        xatol: 绝对误差容差
        xrtol: 相对误差容差
        step_max: 最大迭代步数
    """
    fxa = float(f(xa))
    fxb = float(f(xb))

    # 检查符号变化
    if fxa * fxb > 0:
        raise RootFinderError("Brent method requires f(xa)*f(xb) <= 0")

    xc = xa
    fxc = fxa
    d = xb - xa
    e = d

    for step_num in range(step_max + 1):
        if abs(fxc) < abs(fxb):
            xa = xb
            xb = xc
            xc = xa
            fxa = fxb
            fxb = fxc
            fxc = fxa

        xtol = 2.0 * xrtol * abs(xb) + 0.5 * xatol
        xm = 0.5 * (xc - xb)

        if abs(xm) <= xtol:
            return xb
        if abs(fxb) <= fatol:
            return xb

        # 判断是否强制二分
        if abs(e) < xtol or abs(fxa) <= abs(fxb):
            d = xm
            e = d
        else:
            s = fxb / fxa
            if xa == xc:
                # 线性插值（割线法）
                p = 2.0 * xm * s
                q = 1.0 - s
            else:
                # 反二次插值
                q = fxa / fxc
                r = fxb / fxc
                p = s * (2.0 * xm * q * (q - r) - (xb - xa) * (r - 1.0))
                q = (q - 1.0) * (r - 1.0) * (s - 1.0)

            if p > 0:
                q = -q
            else:
                p = -p

            s = e
            e = d

            cond1 = (3.0 * xm * q - abs(xtol * q) <= 2.0 * p)
            cond2 = (abs(0.5 * s * q) <= p)
            if cond1 or cond2:
                d = xm
                e = d
            else:
                d = p / q

        xa = xb
        fxa = fxb

        if abs(d) > xtol:
            xb = xb + d
        elif xm > 0:
            xb = xb + xtol
        else:
            xb = xb - xtol

        fxb = float(f(xb))

        # 更新 xc 以保持符号变化区间
        if fxb * fxc > 0:
            xc = xa
            fxc = fxa
            d = xb - xa
            e = d

    return xb


def muller_method(f, x0, x1, x2, fatol=1e-12, step_max=50):
    """
    Muller 法求根（实数版本）

    源自原项目 1229_test_zero (muller)

    数学:
      通过三点 (x0,f0), (x1,f1), (x2,f2) 构造二次插值多项式
      选择距离 x2 更近的根作为下一步迭代点
    """
    f0 = float(f(x0))
    f1 = float(f(x1))
    f2 = float(f(x2))

    for step_num in range(step_max):
        if abs(f2) <= fatol:
            return x2

        # 计算差分
        h0 = x0 - x2
        h1 = x1 - x2
        d0 = f0 - f2
        d1 = f1 - f2

        denom = h0 * h1 * (h0 - h1)
        if abs(denom) < 1e-15:
            break

        a = (h1 * d0 - h0 * d1) / denom
        b = (h0 ** 2 * d1 - h1 ** 2 * d0) / denom
        c = f2

        disc = b ** 2 - 4.0 * a * c
        if disc < 0:
            disc = 0.0

        sqrt_disc = np.sqrt(disc)
        if b >= 0:
            den = b + sqrt_disc
        else:
            den = b - sqrt_disc

        if abs(den) < 1e-15:
            break

        dx = -2.0 * c / den
        x_new = x2 + dx
        f_new = float(f(x_new))

        # 轮换点
        x0, f0 = x1, f1
        x1, f1 = x2, f2
        x2, f2 = x_new, f_new

    return x2


def solve_equivalent_plastic_strain(yield_stress, hardening_modulus, mu, trial_stress_norm,
                                    tol=1e-12, max_iter=50):
    """
    径向返回算法中求解等效塑性应变 Δε_p

    数学:
      考虑各向同性线性硬化，von Mises 屈服函数:
        f = ||s_{n+1}^{trial}|| - sqrt(2/3) (σ_y + H * Δε_p) - 2μ Δε_p = 0

      其中:
        s_{n+1}^{trial}: 试算偏应力
        σ_y: 初始屈服应力
        H: 硬化模量
        μ: 剪切模量

      令 σ_eq = ||s_{trial}||，需解:
        g(Δε_p) = σ_eq - sqrt(2/3) (σ_y + H Δε_p) - 2μ Δε_p = 0
    """
    sqrt23 = np.sqrt(2.0 / 3.0)

    def g(dep):
        return trial_stress_norm - sqrt23 * (yield_stress + hardening_modulus * dep) - 2.0 * mu * dep

    def dg(dep):
        return -sqrt23 * hardening_modulus - 2.0 * mu

    # 检查弹性/塑性
    f_trial = trial_stress_norm - sqrt23 * yield_stress
    if f_trial <= 0:
        return 0.0  # 弹性步

    # Newton 法求解
    dep = newton_method(g, dg, 0.0, fatol=tol, step_max=max_iter)
    return max(0.0, dep)


def solve_volume_constraint(K_modulus, pressure_target, J_guess=1.0):
    """
    求解体积约束方程: K * ln(J) + p = 0

    即求 J 使得静水压力等于目标值
    """
    def h(J):
        if J <= 0:
            J = 1e-12
        return K_modulus * np.log(J) + pressure_target

    def dh(J):
        if J <= 0:
            J = 1e-12
        return K_modulus / J

    try:
        J_sol = newton_method(h, dh, J_guess, fatol=1e-12, step_max=30)
    except RootFinderError:
        try:
            J_sol = brent_method(h, 0.1, 10.0, fatol=1e-12, step_max=50)
        except RootFinderError:
            J_sol = J_guess

    return J_sol
