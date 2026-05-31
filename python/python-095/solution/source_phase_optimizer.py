"""
source_phase_optimizer.py
次级声源相位角非线性优化

融合原始项目:
  - 1428_zero_chandrupatla (Chandrupatla求根算法)

科学背景:
  在单频主动噪声控制中,对于给定的次级声源振幅,
  最优相位角 phi_n 使得总声场能量最小.

  对于第m个传感器,总声压为:
      p_m(phi) = d_m + A * H_m * exp(j phi)

  总能量:
      J(phi) = \sum_m |p_m(phi)|^2

  最优条件 dJ/dphi = 0 导出一个超越方程:
      \sum_m Im[ p_m^*(phi) * A * H_m * exp(j phi) ] = 0

  该方程具有多根特性,可用 Chandrupatla 算法在局部区间求根.
"""

import numpy as np
import math


def zero_chandrupatla(f, x1, x2, epsilon=1.0e-10, delta=0.00001, max_iter=200):
    """
    Chandrupatla混合二次/二分求根算法.

    算法特点:
        结合逆二次插值(IQI)和二分法,
        当IQI条件不满足时回退到二分,
        收敛速度介于二分法和割线法之间,鲁棒性极强.

    参数:
        f: 目标函数
        x1, x2: 初始含根区间 (f(x1)*f(x2)<0)
        epsilon, delta: 容差参数
        max_iter: 最大迭代次数

    返回:
        xm: 近似根
        fm: 函数值
        calls: 函数调用次数
    """
    f1 = f(x1)
    f2 = f(x2)
    calls = 2

    if f1 * f2 > 0:
        raise ValueError("zero_chandrupatla: f(x1) and f(x2) must have opposite signs")

    t = 0.5
    iter_count = 0

    while iter_count < max_iter:
        x0 = x1 + t * (x2 - x1)
        f0 = f(x0)
        calls += 1

        # 重新排列点: x1为最接近根的, x2为另一端
        if np.sign(f0) == np.sign(f1):
            x3 = x1
            f3 = f1
            x1 = x0
            f1 = f0
        else:
            x3 = x2
            f3 = f2
            x2 = x1
            f2 = f1
            x1 = x0
            f1 = f0

        # 识别最佳近似
        if abs(f2) < abs(f1):
            xm = x2
            fm = f2
        else:
            xm = x1
            fm = f1

        tol = 2.0 * epsilon * abs(xm) + 0.5 * delta
        tl = tol / abs(x2 - x1)

        if tl >= 0.5 or abs(fm) < epsilon:
            break

        # 逆二次插值条件检查
        xi = (x1 - x2) / (x3 - x2)
        ph = (f1 - f2) / (f3 - f2)
        fl = 1.0 - math.sqrt(1.0 - xi)
        fh = math.sqrt(xi)

        if fl < ph < fh:
            al = (x3 - x1) / (x2 - x1)
            a = f1 / (f2 - f1)
            b = f3 / (f2 - f3)
            c = f1 / (f3 - f1)
            d = f2 / (f3 - f2)
            t = a * b + c * d * al
        else:
            t = 0.5

        t = max(t, tl)
        t = min(t, 1.0 - tl)
        iter_count += 1

    return xm, fm, calls


def optimize_source_phase(H_col, d, amplitude, phi_bounds=(0.0, 2.0 * math.pi)):
    """
    对单个次级声源优化其相位角.

    参数:
        H_col: (M,) 该源到所有传感器的传递函数
        d: (M,) 初级噪声
        amplitude: 源振幅
        phi_bounds: 相位搜索区间

    返回:
        phi_opt: 最优相位 [rad]
        min_energy: 最小声能量
    """
    H_col = np.asarray(H_col, dtype=complex)
    d = np.asarray(d, dtype=complex)

    def energy_gradient(phi):
        s = amplitude * np.exp(1j * phi)
        p = d + H_col * s
        # dJ/dphi = 2 Re[ sum p* * j * H_col * s ]
        grad = 2.0 * np.real(np.vdot(p, 1j * H_col * s))
        return grad

    # 在区间内搜索多个子区间寻找变号点
    n_brackets = 36
    phis = np.linspace(phi_bounds[0], phi_bounds[1], n_brackets + 1)
    vals = np.array([energy_gradient(p) for p in phis])

    best_phi = phis[0]
    best_energy = np.inf

    for i in range(n_brackets):
        if vals[i] == 0.0:
            candidate = phis[i]
        elif vals[i] * vals[i + 1] < 0:
            try:
                candidate, _, _ = zero_chandrupatla(energy_gradient, phis[i], phis[i + 1])
            except ValueError:
                continue
        else:
            continue

        s = amplitude * np.exp(1j * candidate)
        p = d + H_col * s
        energy = np.vdot(p, p).real
        if energy < best_energy:
            best_energy = energy
            best_phi = candidate

    # 边界检查
    for p in phi_bounds:
        s = amplitude * np.exp(1j * p)
        p_ = d + H_col * s
        energy = np.vdot(p_, p_).real
        if energy < best_energy:
            best_energy = energy
            best_phi = p

    return best_phi, best_energy
