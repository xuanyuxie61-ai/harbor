"""
radial_solver.py
================
径向薛定谔方程求解模块

本模块实现：
1. Numerov 算法求解径向薛定谔方程
2. Brent 寻根法确定束缚态本征能量
3. 波函数归一化与边界条件匹配

数学基础：
径向薛定谔方程（对于 u(r) = r R(r)）：
  d²u/dr² = f(r) u(r)
其中 f(r) = (2M/ħ²)[V(r) - E] + l(l+1)/r²

Numerov 算法（适用于 y'' = g(x) y 形式的方程）：
  y_{n+1} = [2(1 - 5h²g_n/12)y_n - (1 + h²g_{n-1}/12)y_{n-1}] / (1 + h²g_{n+1}/12)

Brent 寻根法：结合二分法、割线法与逆二次插值，
用于在已知变号区间内寻找使波函数在远边界趋于零的能量 E。
"""

import numpy as np
from math import sqrt, exp, fabs

# 物理常数
HBARC = 197.3269804
M_NUCLEON = 939.0


def numerov_integrate(r_grid, f_values, u0, u1):
    """
    使用 Numerov 算法积分二阶微分方程 u'' = f(r) u。

    参数
    ----
    r_grid : ndarray
        均匀格点
    f_values : ndarray
        每个格点上的 f(r) 值
    u0, u1 : float
        前两个点的初值

    返回
    ----
    u : ndarray
        解向量
    """
    N = len(r_grid)
    h = r_grid[1] - r_grid[0]
    h2 = h * h
    h12 = h2 / 12.0

    u = np.zeros(N)
    u[0] = u0
    u[1] = u1

    for n in range(1, N - 1):
        denom = 1.0 + h12 * f_values[n + 1]
        if abs(denom) < 1e-15:
            denom = 1e-15
        u[n + 1] = ((2.0 * (1.0 - 5.0 * h12 * f_values[n]) * u[n] -
                     (1.0 + h12 * f_values[n - 1]) * u[n - 1]) / denom)
    return u


def compute_radial_wavefunction(r_grid, potential, E, l, mass=M_NUCLEON):
    """
    计算给定能量 E 下的径向波函数 u(r)。

    f(r) = (2M/ħ²)[V(r) - E] + l(l+1)/r²

    参数
    ----
    r_grid : ndarray
        径向格点 (fm)
    potential : ndarray
        势值 V(r) (MeV)
    E : float
        试探能量 (MeV)
    l : int
        轨道角动量

    返回
    ----
    u : ndarray
        波函数 u(r) = r R(r)
    f_values : ndarray
        f(r) 值
    """
    N = len(r_grid)
    f_values = np.zeros(N)
    prefactor = 2.0 * mass / (HBARC ** 2)

    for i in range(N):
        r = r_grid[i]
        # TODO [Hole 2]: 填入离心势项的离散表达式
        # f(r) = (2M/ħ²)[V(r) - E] + l(l+1)/r²
        # 注意 r → 0 时的正则化处理
        centrifugal = 0.0  # 占位符，需要正确实现
        f_values[i] = prefactor * (potential[i] - E) + centrifugal

    # 边界条件：u(0) = 0，u(h) ~ h^{l+1}
    h = r_grid[1] - r_grid[0]
    u0 = 0.0
    u1 = h ** (l + 1)

    u = numerov_integrate(r_grid, f_values, u0, u1)
    return u, f_values


def wavefunction_logarithmic_derivative(u, r_grid):
    """
    计算波函数在最右端格点的对数导数 (1/u) du/dr。

    用于匹配边界条件判断束缚态。
    """
    N = len(r_grid)
    if abs(u[N - 1]) < 1e-30:
        return 1e30
    # 三点微商
    h = r_grid[1] - r_grid[0]
    dudr = (3.0 * u[N - 1] - 4.0 * u[N - 2] + u[N - 3]) / (2.0 * h)
    return dudr / u[N - 1]


def brent_root_find(a, b, t, func):
    """
    Brent 寻根法（基于 zero_brent 的 Python 实现）。

    在已知变号区间 [a, b] 内寻找 func(x) = 0 的根。
    结合二分法的鲁棒性与割线法/逆二次插值的收敛速度。

    参数
    ----
    a, b : float
        变号区间端点，满足 func(a)·func(b) < 0
    t : float
        误差容限
    func : callable
        目标函数

    返回
    ----
    root : float
        根的近似值
    calls : int
        函数调用次数
    """
    calls = 0
    sa, sb = a, b
    fa = func(sa)
    calls += 1
    fb = func(sb)
    calls += 1

    if fa * fb > 0:
        raise ValueError("区间端点必须变号")

    c, fc = sa, fa
    e = sb - sa
    d = e

    while True:
        if abs(fc) < abs(fb):
            sa, sb, c = sb, c, sa
            fa, fb, fc = fb, fc, fa

        tol = 2.0 * np.finfo(float).eps * abs(sb) + t
        m = 0.5 * (c - sb)

        if abs(m) <= tol or fb == 0.0:
            break

        if abs(e) < tol or abs(fa) <= abs(fb):
            e = m
            d = e
        else:
            s = fb / fa
            if sa == c:
                p = 2.0 * m * s
                q = 1.0 - s
            else:
                q = fa / fc
                r = fb / fc
                p = s * (2.0 * m * q * (q - r) - (sb - sa) * (r - 1.0))
                q = (q - 1.0) * (r - 1.0) * (s - 1.0)

            if p > 0.0:
                q = -q
            else:
                p = -p

            s = e
            e = d

            if 2.0 * p < 3.0 * m * q - abs(tol * q) and p < abs(0.5 * s * q):
                d = p / q
            else:
                e = m
                d = e

        sa = sb
        fa = fb

        if abs(d) > tol:
            sb = sb + d
        elif m > 0.0:
            sb = sb + tol
        else:
            sb = sb - tol

        fb = func(sb)
        calls += 1

        if (fb > 0.0 and fc > 0.0) or (fb <= 0.0 and fc <= 0.0):
            c = sa
            fc = fa
            e = sb - sa
            d = e

    return sb, calls


def find_bound_state_energy(r_grid, potential, l, E_min, E_max, tol=1e-6):
    """
    使用 Brent 法寻找束缚态能量。

    核心思想：对于正确的束缚态能量 E，波函数 u(r) 在 r → ∞ 时必须指数衰减。
    通过匹配对数导数或简单地观察 u(r_max) 的符号变化来定位本征值。

    参数
    ----
    r_grid : ndarray
        径向格点
    potential : ndarray
        势值
    l : int
        角动量量子数
    E_min, E_max : float
        能量搜索区间 (MeV)
    tol : float
        能量容差

    返回
    ----
    E_bound : float
        束缚态能量
    u_bound : ndarray
        归一化波函数
    n_calls : int
        函数调用次数
    """
    def mismatch(E):
        u, _ = compute_radial_wavefunction(r_grid, potential, E, l)
        # 使用右端点值作为判据：正确的 E 使 u 趋于 0
        return u[-1]

    # 检查变号
    f_min = mismatch(E_min)
    f_max = mismatch(E_max)

    if f_min * f_max > 0:
        # 尝试扩展区间或返回最佳近似
        # 在区间内细搜索
        E_test = np.linspace(E_min, E_max, 200)
        best_E = E_min
        best_val = abs(f_min)
        for Et in E_test:
            val = abs(mismatch(Et))
            if val < best_val:
                best_val = val
                best_E = Et
        u_best, _ = compute_radial_wavefunction(r_grid, potential, best_E, l)
        return best_E, u_best, 200

    E_bound, n_calls = brent_root_find(E_min, E_max, tol, mismatch)
    u_bound, _ = compute_radial_wavefunction(r_grid, potential, E_bound, l)

    # 归一化：∫ u² dr = 1
    h = r_grid[1] - r_grid[0]
    norm = sqrt(np.trapezoid(u_bound ** 2, r_grid))
    if norm > 0:
        u_bound = u_bound / norm

    return E_bound, u_bound, n_calls


def solve_all_bound_states(r_grid, potential, l, n_max_states=5,
                           E_search_min=-60.0, E_search_max=-1.0):
    """
    求解给定角动量 l 下的所有束缚态。

    策略：在能量区间内等距取样，通过波函数节点数确定主量子数 n。
    节点数 = n - 1（u(r) 的零点个数，不计 r=0）。
    """
    energies = []
    wavefunctions = []

    n_probe = 300
    E_probe = np.linspace(E_search_min, E_search_max, n_probe)

    # 先计算所有试探能量的波函数末值
    u_ends = []
    for E in E_probe:
        u, _ = compute_radial_wavefunction(r_grid, potential, E, l)
        u_ends.append(u[-1])
    u_ends = np.array(u_ends)

    # 寻找变号点（本征能量近似位置）
    sign_changes = []
    for i in range(n_probe - 1):
        if u_ends[i] * u_ends[i + 1] < 0:
            sign_changes.append((E_probe[i], E_probe[i + 1]))

    for (E_a, E_b) in sign_changes[:n_max_states]:
        try:
            E_bnd, u_bnd, _ = find_bound_state_energy(r_grid, potential, l,
                                                        E_a, E_b, tol=1e-5)
            energies.append(E_bnd)
            wavefunctions.append(u_bnd)
        except ValueError:
            continue

    return energies, wavefunctions
