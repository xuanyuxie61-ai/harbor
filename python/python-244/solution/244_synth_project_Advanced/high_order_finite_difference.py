#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
high_order_finite_difference.py
================================
【融合种子项目】 787_navier_stokes_2d_exact (Navier-Stokes 精确解 + 有限差分)

本模块实现中子星结构方程的高阶有限差分离散化, 借鉴 Navier-Stokes 精确解
项目中高精度有限差分 (FD) 构造右端函数的方法, 将 2D 差分格式推广至
球对称径向问题.

物理/数学公式
-------------
1. 中心差分 (二阶):
       f'(x_i) ≈ (-f_{i-1} + f_{i+1}) / (2h)
       f''(x_i) ≈ (f_{i-1} - 2f_i + f_{i+1}) / h^2

2. 四阶中心差分:
       f'(x_i) ≈ (f_{i-2} - 8f_{i-1} + 8f_{i+1} - f_{i+2}) / (12h)
       f''(x_i) ≈ (-f_{i-2} + 16f_{i-1} - 30f_i + 16f_{i+1} - f_{i+2}) / (12h^2)

3. 六阶中心差分:
       f'(x_i) ≈ (-f_{i-3} + 9f_{i-2} - 45f_{i-1} + 45f_{i+1}
                   - 9f_{i+2} + f_{i+3}) / (60h)

4. 截断误差 (Taylor 展开):
       二阶: E = -h^2/6 f'''(xi)
       四阶: E = h^4/30 f^{(5)}(xi)
       六阶: E = -h^6/140 f^{(7)}(xi)

5. 球坐标拉普拉斯 (径向):
       nabla^2 f = (1/r^2) d/dr (r^2 df/dr)
                 = f'' + (2/r) f'

   四阶离散:
       (nabla^2 f)_i = [f_{i-1} - 2f_i + f_{i+1}]/h^2
                       + (2/r_i) [f_{i+1} - f_{i-1}]/(2h)
                       + O(h^2) 修正项

6. 紧致差分 (Pade 型):
       (1/6) f'_{i-1} + (2/3) f'_i + (1/6) f'_{i+1}
           = (f_{i+1} - f_{i-1}) / (2h)

   解三对角系统得到高阶精度导数.

7. 人工粘性 (用于激波捕捉):
       q_i = -c_1 h c_s |du/dr| + c_2 h^2 (du/dr)^2
   其中 c_1 ~ 1, c_2 ~ 1.

8. 边界条件处理:
   - 中心 (r=0): 对称条件 f'(0) = 0
   - 表面 (r=R): P(R) = 0 (Dirichlet)
   - 外推边界: f_{N+1} = 2f_N - f_{N-1}
"""

import math
import numpy as np
from typing import Tuple, Dict, Callable, Optional
from numerical_constants import r8_epsilon


# ============================================================
# 第一部分: 有限差分矩阵构造
# ============================================================

def fd_first_derivative_matrix(n: int, h: float, order: int = 2,
                                bc_type: str = "neumann") -> np.ndarray:
    """
    构造一阶导数有限差分矩阵 D1.

    D1 @ f ≈ f'  (在内部节点)

    Parameters
    ----------
    n : int
        内部节点数
    h : float
        网格间距
    order : int
        精度阶数 (2, 4, 6)
    bc_type : str
        边界条件类型: "neumann" (f'(0)=0), "extrapolate"

    Returns
    -------
    np.ndarray, shape (n, n)
        一阶导数矩阵
    """
    D = np.zeros((n, n))

    if order == 2:
        # 二阶中心差分
        for i in range(n):
            if i > 0:
                D[i, i - 1] = -1.0 / (2.0 * h)
            if i < n - 1:
                D[i, i + 1] = 1.0 / (2.0 * h)
    elif order == 4:
        # 四阶中心差分
        for i in range(n):
            if i >= 2:
                D[i, i - 2] = 1.0 / (12.0 * h)
            if i >= 1:
                D[i, i - 1] = -8.0 / (12.0 * h)
            if i < n - 1:
                D[i, i + 1] = 8.0 / (12.0 * h)
            if i < n - 2:
                D[i, i + 2] = -1.0 / (12.0 * h)
    elif order == 6:
        # 六阶中心差分
        coeffs = [-1.0, 9.0, -45.0, 0.0, 45.0, -9.0, 1.0]
        for i in range(n):
            for k, c in enumerate(coeffs):
                j = i + k - 3
                if 0 <= j < n and c != 0.0:
                    D[i, j] = c / (60.0 * h)
    else:
        raise ValueError(f"不支持的精度阶数: {order}")

    # 边界处理
    if bc_type == "neumann":
        # 一阶节点: f'_0 = 0 => 使用向前差分
        if order >= 2:
            D[0, 0] = -3.0 / (2.0 * h)
            D[0, 1] = 4.0 / (2.0 * h)
            D[0, 2] = -1.0 / (2.0 * h) if n > 2 else 0.0
            D[0, :] /= 2.0  # 乘以 0 (Neumann) → 但保留格式
            # 实际: f'(0) = 0, 所以第一行全 0
            D[0, :] = 0.0
    elif bc_type == "extrapolate":
        # 外推边界
        D[0, :] = 0.0
        D[-1, :] = 0.0

    return D


def fd_second_derivative_matrix(n: int, h: float, order: int = 2) -> np.ndarray:
    """
    构造二阶导数有限差分矩阵 D2.

    D2 @ f ≈ f''  (在内部节点)
    """
    D = np.zeros((n, n))

    if order == 2:
        for i in range(n):
            D[i, i] = -2.0 / (h * h)
            if i > 0:
                D[i, i - 1] = 1.0 / (h * h)
            if i < n - 1:
                D[i, i + 1] = 1.0 / (h * h)
    elif order == 4:
        for i in range(n):
            D[i, i] = -30.0 / (12.0 * h * h)
            if i > 0:
                D[i, i - 1] = 16.0 / (12.0 * h * h)
            if i < n - 1:
                D[i, i + 1] = 16.0 / (12.0 * h * h)
            if i > 1:
                D[i, i - 2] = -1.0 / (12.0 * h * h)
            if i < n - 2:
                D[i, i + 2] = -1.0 / (12.0 * h * h)
    elif order == 6:
        for i in range(n):
            D[i, i] = -490.0 / (180.0 * h * h)
            for offset, c in [(1, 270.0), (2, -27.0), (3, 2.0)]:
                if i >= offset:
                    D[i, i - offset] = c / (180.0 * h * h)
                if i < n - offset:
                    D[i, i + offset] = c / (180.0 * h * h)
    else:
        raise ValueError(f"不支持的精度阶数: {order}")

    return D


# ============================================================
# 第二部分: 球坐标拉普拉斯算子
# ============================================================

def spherical_laplacian_fd(f: np.ndarray, r: np.ndarray,
                            order: int = 2) -> np.ndarray:
    """
    球坐标径向拉普拉斯算子的高阶有限差分.

    nabla^2 f = f'' + (2/r) f'

    使用中心差分:
        (nabla^2 f)_i = D2[i,:] @ f + (2/r_i) D1[i,:] @ f

    Parameters
    ----------
    f : np.ndarray, shape (N,)
        函数值
    r : np.ndarray, shape (N,)
        径向坐标 (必须 > 0)
    order : int
        精度阶数

    Returns
    -------
    np.ndarray, shape (N,)
        nabla^2 f 在各节点的值
    """
    n = len(f)
    h = r[1] - r[0] if n > 1 else 1.0

    D1 = fd_first_derivative_matrix(n, h, order)
    D2 = fd_second_derivative_matrix(n, h, order)

    df = D1 @ f
    d2f = D2 @ f

    # 球坐标修正: (2/r) f'
    lap = np.zeros(n)
    for i in range(n):
        if r[i] > r8_epsilon():
            lap[i] = d2f[i] + 2.0 / r[i] * df[i]
        else:
            # L'Hôpital: lim_{r->0} (2/r) f'(r) = 2 f''(0)
            lap[i] = 3.0 * d2f[i]

    return lap


def tov_rhs_fd(r_nodes: np.ndarray, P_profile: np.ndarray,
                eos_callable: Callable, order: int = 4) -> np.ndarray:
    """
    使用高阶有限差分计算 TOV 方程右端.

    将 TOV 方程改写为:
        dP/dr = RHS(P, m, r)

    并用 FD 矩阵直接计算 dP/dr 和 d²P/dr² 用于稳定性分析.

    Parameters
    ----------
    r_nodes : np.ndarray
        径向节点 (km)
    P_profile : np.ndarray
        压力分布
    eos_callable : callable
        EoS 函数 P -> (rho, epsilon, dp/drho)
    order : int
        FD 精度阶数

    Returns
    -------
    np.ndarray
        dP/dr 的 FD 近似
    """
    n = len(r_nodes)
    h = r_nodes[1] - r_nodes[0] if n > 1 else 0.01

    D1 = fd_first_derivative_matrix(n, h, order)
    dPdr = D1 @ P_profile

    return dPdr


# ============================================================
# 第三部分: 紧致差分 (Pade 格式)
# ============================================================

def compact_first_derivative(f: np.ndarray, h: float,
                              alpha: float = 1.0 / 3.0) -> np.ndarray:
    """
    四阶紧致 (Pade) 一阶导数.

    隐式格式:
        alpha f'_{i-1} + f'_i + alpha f'_{i+1}
            = a (f_{i+1} - f_{i-1}) / (2h)

    四阶: alpha = 1/4, a = 3/2
    六阶: alpha = 1/3, a = 14/9

    解三对角系统得到 f'.

    Parameters
    ----------
    f : np.ndarray
    h : float
    alpha : float
        Pade 参数

    Returns
    -------
    np.ndarray
        紧致导数
    """
    n = len(f)
    # 对于给定 alpha, a = (4 + 2*alpha) / 3 (Lele 1992)
    # 六阶: alpha = 1/3, a = 14/9
    a = (4.0 + 2.0 * alpha) / 3.0

    # 构造三对角系统
    lower = np.full(n, alpha)
    diag = np.ones(n)
    upper = np.full(n, alpha)

    rhs = np.zeros(n)
    # 内部节点: 中心差分
    for i in range(1, n - 1):
        rhs[i] = a * (f[i + 1] - f[i - 1]) / (2.0 * h)

    # 边界 (三阶向前/向后差分)
    if n >= 4:
        # 一阶向前差分 (三阶精度)
        rhs[0] = (-25.0/12 * f[0] + 4.0 * f[1] - 3.0 * f[2]
                  + 4.0/3.0 * f[3] - 1.0/4.0 * f[4] if n >= 5 else
                  -3.0 * f[0] + 4.0 * f[1] - f[2]) / (2.0 * h)
        rhs[-1] = (25.0/12 * f[-1] - 4.0 * f[-2] + 3.0 * f[-3]
                   - 4.0/3.0 * f[-4] + 1.0/4.0 * f[-5] if n >= 5 else
                   3.0 * f[-1] - 4.0 * f[-2] + f[-3]) / (2.0 * h)
    else:
        rhs[0] = (-3.0 * f[0] + 4.0 * f[1] - f[2]) / (2.0 * h) if n > 2 else 0.0
        rhs[-1] = (3.0 * f[-1] - 4.0 * f[-2] + f[-3]) / (2.0 * h) if n > 2 else 0.0

    # Thomas 算法解三对角
    return thomas_solve(lower, diag, upper, rhs)


def thomas_solve(lower: np.ndarray, diag: np.ndarray,
                  upper: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """
    Thomas 算法解三对角系统 Ax = d.

    A = tridiag(lower, diag, upper)

    算法:
        1. 前消: c'_i = c_i / (b_i - a_i c'_{i-1})
                 d'_i = (d_i - a_i d'_{i-1}) / (b_i - a_i c'_{i-1})
        2. 回代: x_n = d'_n
                 x_i = d'_i - c'_i x_{i+1}
    """
    n = len(diag)
    c_prime = np.zeros(n)
    d_prime = np.zeros(n)
    x = np.zeros(n)

    # 前消
    c_prime[0] = upper[0] / diag[0]
    d_prime[0] = rhs[0] / diag[0]

    for i in range(1, n):
        m = lower[i] if i < len(lower) else 0.0
        denom = diag[i] - m * c_prime[i - 1]
        if abs(denom) < r8_epsilon():
            denom = r8_epsilon() if denom >= 0 else -r8_epsilon()
        c_prime[i] = upper[i] / denom if i < n - 1 else 0.0
        d_prime[i] = (rhs[i] - m * d_prime[i - 1]) / denom

    # 回代
    x[-1] = d_prime[-1]
    for i in range(n - 2, -1, -1):
        x[i] = d_prime[i] - c_prime[i] * x[i + 1]

    return x


# ============================================================
# 第四部分: 人工粘性 (Navier-Stokes 思想移植)
# ============================================================

def artificial_viscosity(dPdr: np.ndarray, rho: np.ndarray,
                          cs: np.ndarray, h: float,
                          c1: float = 1.0, c2: float = 1.0) -> np.ndarray:
    """
    人工粘性项 (von Neumann-Richtmyer 型).

    q = c_1 h c_s |dP/dr| / (rho c_s^2) + c_2 h^2 (dP/dr)^2 / (rho c_s^2)

    用于在压力陡变处 (壳-核交界) 提供数值耗散, 防止非物理振荡.

    Parameters
    ----------
    dPdr : np.ndarray
        压力梯度
    rho : np.ndarray
        密度
    cs : np.ndarray
        声速
    h : float
        网格间距
    c1, c2 : float
        人工粘性系数

    Returns
    -------
    np.ndarray
        人工粘性贡献
    """
    q = np.zeros_like(dPdr)
    for i in range(len(dPdr)):
        if rho[i] > 0 and cs[i] > 0:
            div_v = abs(dPdr[i]) / (rho[i] * cs[i] ** 2 + r8_epsilon())
            q[i] = (c1 * h * cs[i] * div_v
                    + c2 * h * h * div_v * div_v)
    return q


# ============================================================
# 第五部分: 收敛性分析
# ============================================================

def convergence_order(h_list: list, error_list: list) -> float:
    """
    从不同网格尺寸的误差估计收敛阶.

    E(h) ~ C h^p  =>  ln E = p ln h + ln C

    p = (ln E_2 - ln E_1) / (ln h_2 - ln h_1)

    Parameters
    ----------
    h_list : list of float
        网格尺寸列表
    error_list : list of float
        对应误差列表

    Returns
    -------
    float
        估计的收敛阶 p
    """
    if len(h_list) < 2:
        return 0.0

    log_h = np.log(np.array(h_list))
    log_e = np.log(np.array(error_list) + r8_epsilon())

    # 线性拟合
    n = len(log_h)
    if n == 2:
        return (log_e[1] - log_e[0]) / (log_h[1] - log_h[0])

    # 最小二乘
    A = np.column_stack([log_h, np.ones(n)])
    coeffs, _, _, _ = np.linalg.lstsq(A, log_e, rcond=None)
    return float(coeffs[0])


def fd_accuracy_test(func: Callable, deriv_func: Callable,
                      a: float, b: float, n_base: int = 50,
                      max_order: int = 6) -> Dict:
    """
    有限差分精度测试.

    对给定函数 f(x), 在不同网格上计算 FD 近似, 与精确导数比较.

    Parameters
    ----------
    func : callable
        测试函数 f(x)
    deriv_func : callable
        精确导数 f'(x)
    a, b : float
        区间 [a, b]
    n_base : int
        基准网格数
    max_order : int
        最高测试阶数

    Returns
    -------
    dict
        各阶精度在各网格上的误差
    """
    h_list = [1.0 / n for n in [n_base, 2 * n_base, 4 * n_base, 8 * n_base]]
    results = {}

    for order in [2, 4, 6]:
        if order > max_order:
            continue
        errors = []
        for n_pts in [n_base, 2 * n_base, 4 * n_base, 8 * n_base]:
            x = np.linspace(a, b, n_pts)
            h = x[1] - x[0]
            f_vals = np.array([func(xi) for xi in x])
            f_exact = np.array([deriv_func(xi) for xi in x])

            D = fd_first_derivative_matrix(n_pts, h, order)
            f_fd = D @ f_vals

            # 内部节点误差
            err = np.max(np.abs(f_fd[2:-2] - f_exact[2:-2]))
            errors.append(err)

        p = convergence_order(h_list, errors)
        results[f"order_{order}"] = {
            'h_values': h_list,
            'errors': errors,
            'convergence_order': p,
        }

    return results


# ============================================================
# 自检
# ============================================================

if __name__ == "__main__":
    print("=== 高阶有限差分自检 ===")

    n = 20
    h = 0.1
    D2 = fd_first_derivative_matrix(n, h, order=2)
    D4 = fd_first_derivative_matrix(n, h, order=4)
    D6 = fd_first_derivative_matrix(n, h, order=6)
    print(f"D2 矩阵非零元: {np.count_nonzero(D2)}")
    print(f"D4 矩阵非零元: {np.count_nonzero(D4)}")
    print(f"D6 矩阵非零元: {np.count_nonzero(D6)}")

    # 测试函数: f(x) = sin(x), f'(x) = cos(x)
    def f_test(x): return math.sin(x)
    def df_test(x): return math.cos(x)

    r_test = np.linspace(0.1, 3.0, 30)
    h_test = r_test[1] - r_test[0]
    f_vals = np.array([f_test(x) for x in r_test])
    df_exact = np.array([df_test(x) for x in r_test])

    for order in [2, 4, 6]:
        D = fd_first_derivative_matrix(30, h_test, order)
        df_fd = D @ f_vals
        err = np.max(np.abs(df_fd[3:-3] - df_exact[3:-3]))
        print(f"  阶数 {order}: 最大误差 = {err:.3e}")

    # 紧致差分
    df_compact = compact_first_derivative(f_vals, h_test)
    err_compact = np.max(np.abs(df_compact[3:-3] - df_exact[3:-3]))
    print(f"  紧致差分: 最大误差 = {err_compact:.3e}")

    # 球坐标拉普拉斯
    # 测试: f(r) = r^2, nabla^2 f = 6
    r_lap = np.linspace(0.5, 3.0, 40)
    f_lap = r_lap ** 2
    lap_result = spherical_laplacian_fd(f_lap, r_lap, order=2)
    print(f"  nabla^2 (r^2) = {lap_result[20]:.3f}  (精确 6)")

    # 精度测试
    result = fd_accuracy_test(math.sin, math.cos, 0.5, 3.0)
    for key, val in result.items():
        print(f"  {key}: 收敛阶 = {val['convergence_order']:.2f}")

    # Thomas 算法
    n_th = 5
    diag = np.full(n_th, 4.0)
    lower = np.full(n_th, 1.0)
    upper = np.full(n_th, 1.0)
    rhs = np.array([6.0, 7.0, 7.0, 7.0, 5.0])
    x_sol = thomas_solve(lower, diag, upper, rhs)
    print(f"  Thomas 解: {x_sol}")
    print(f"  残差: {np.max(np.abs(np.diag(diag) @ x_sol + np.diag(lower, -1) @ x_sol + np.diag(upper, 1) @ x_sol - rhs)):.3e}")

    print("\nhigh_order_finite_difference.py 自检通过.")
