"""
stability.py - 冯·诺伊曼稳定性分析

本模块对激光等离子体仿真中使用的高阶有限差分格式进行
冯·诺伊曼 (Von Neumann) 稳定性分析。

冯·诺伊曼稳定性分析的基本原理:

  将数值解展开为 Fourier 模式:
      u_j^n = G^n * exp(i * k * j * h)

  其中 G 为放大因子 (amplification factor), k 为波数, h 为网格间距。

  稳定性条件 (Von Neumann 条件):
      |G(k*h)| <= 1 + C*dt  对所有可分辨波数 k

  对于无耗散格式, 严格要求:
      |G| <= 1

稳定性分析的目标方程:

  1. 对流方程 u_t + c * u_x = 0 的前Euler格式:
     G = 1 - i * nu * sin(kh)     (2阶FD)
     其中 nu = c*dt/h (Courant 数)
     稳定性: nu <= 0 (无条件不稳定)

  2. FTCS (Forward-Time Centered-Space) 格式:
     G = 1 - i * nu * (keff*h)
     |G|^2 = 1 + nu^2 * (keff*h)^2 > 1 → 无条件不稳定

  3. Leapfrog 格式:
     G^2 - 2i*nu*(keff*h)*G - 1 = 0
     稳定性: nu * max|keff*h| <= 1

  4. 对流方程的 Lax-Friedrichs 格式:
     G = cos(kh) - i*nu*sin(kh)
     稳定性: |nu| <= 1

  5. 扩散方程 du/dt = D * u_xx 的 FTCS:
     G = 1 - 4*r*sin^2(kh/2)
     其中 r = D*dt/h^2
     稳定性: r <= 1/2

  6. 高阶FD + RK4 组合:
     先计算空间半离散化的最大特征值 lambda_max,
     然后检查 RK4 稳定域是否包含 dt*lambda_max。
     RK4 稳定域: |1 + z + z^2/2 + z^3/6 + z^4/24| <= 1
     其中 z = dt * lambda

CFL 条件:
  对于 p 阶空间差分 + q 阶时间积分:
    dt <= C_pq * h / max|v|
  其中 C_pq 取决于具体格式组合。
"""

import numpy as np
from fd_operators import modified_wavenumber, fd_coefficients_first_deriv


def von_neumann_advection(nu_array, kh_array=None, scheme='leapfrog', fd_order=4):
    """对流方程 u_t + c*u_x = 0 的 Von Neumann 稳定性分析。

    分析不同时间积分格式结合高阶空间差分的稳定性。

    Parameters
    ----------
    nu_array : ndarray
        Courant 数 nu = c*dt/dx 的数组
    kh_array : ndarray, optional
        无量纲波数数组。默认为 [0, pi]。
    scheme : str
        时间积分格式 ('forward_euler', 'leapfrog', 'lax_friedrichs', 'rk4')
    fd_order : int
        空间差分阶数

    Returns
    -------
    results : dict
        包含各 (nu, kh) 组合下放大因子模 |G| 的字典
    """
    if kh_array is None:
        kh_array = np.linspace(0, np.pi, 500)

    keff_h = modified_wavenumber(kh_array, fd_order)
    results = {
        'kh': kh_array,
        'keff_h': keff_h,
        'nu': nu_array,
    }

    for nu in nu_array:
        G = np.zeros_like(kh_array, dtype=complex)

        if scheme == 'forward_euler':
            # Forward Euler: G = 1 - i*nu*(keff*h)
            G = 1.0 - 1j * nu * keff_h

        elif scheme == 'leapfrog':
            # Leapfrog: G^2 - 2i*nu*(keff*h)*G - 1 = 0
            # G = i*nu*(keff*h) +/- sqrt(1 - nu^2*(keff*h)^2)
            discriminant = 1.0 - (nu * keff_h) ** 2
            sqrt_disc = np.where(
                discriminant >= 0,
                np.sqrt(np.maximum(discriminant, 0)),
                1j * np.sqrt(np.maximum(-discriminant, 0))
            )
            G = 1j * nu * keff_h + sqrt_disc

        elif scheme == 'lax_friedrichs':
            # Lax-Friedrichs: G = cos(kh) - i*nu*sin(kh)
            # (注意: Lax-Friedrichs 不使用修正波数)
            G = np.cos(kh_array) - 1j * nu * np.sin(kh_array)

        elif scheme == 'rk4':
            # RK4 应用于半离散化 du/dt = -c * D_h * u
            # 对于 Fourier 模式: du_k/dt = -i*c*keff*u_k
            # lambda = -i*c*keff, z = dt*lambda = -i*nu*keff*h
            z = -1j * nu * keff_h
            G = 1.0 + z + z ** 2 / 2.0 + z ** 3 / 6.0 + z ** 4 / 24.0

        else:
            raise ValueError(f"未知格式: {scheme}")

        results[f'G_nu{nu:.3f}'] = G
        results[f'modG_nu{nu:.3f}'] = np.abs(G)

    return results


def von_neumann_diffusion(r_array, kh_array=None):
    """扩散方程 u_t = D*u_xx 的 Von Neumann 稳定性分析。

    FTCS (Forward-Time Centered-Space) 格式:
        u_j^{n+1} = u_j^n + r*(u_{j-1}^n - 2u_j^n + u_{j+1}^n)

    放大因子:
        G = 1 - 4*r*sin^2(kh/2)

    稳定性条件: r <= 1/2
    其中 r = D*dt/dx^2

    Parameters
    ----------
    r_array : ndarray
        扩散数 r = D*dt/dx^2 的数组
    kh_array : ndarray, optional
        无量纲波数数组

    Returns
    -------
    results : dict
        各 r 值下的放大因子
    """
    if kh_array is None:
        kh_array = np.linspace(0, np.pi, 500)

    results = {'kh': kh_array, 'r': r_array}

    for r in r_array:
        G = 1.0 - 4.0 * r * np.sin(kh_array / 2.0) ** 2
        results[f'G_r{r:.4f}'] = G
        results[f'modG_r{r:.4f}'] = np.abs(G)

    return results


def cfl_condition(fd_order, scheme='rk4', c=1.0):
    """计算 CFL 稳定条件限制。

    对于 p 阶空间差分，修正波数的最大值为:
        max|keff*h| = max_k |sum_{j=1}^p a_j * sin(j*kh)|

    CFL 条件依赖于时间积分格式:
    - Forward Euler: 无条件不稳定 (对流方程)
    - Leapfrog: nu * max|keff*h| <= 1
    - Lax-Friedrichs: nu <= 1
    - RK4: |z| <= 2.83 (近似), z = -i*nu*keff*h
      -> nu * max|keff*h| <= 2.83

    Parameters
    ----------
    fd_order : int
        空间差分精度阶数
    scheme : str
        时间积分格式
    c : float
        波速

    Returns
    -------
    cfl_info : dict
        CFL 条件信息
    """
    # 计算修正波数最大值
    kh_fine = np.linspace(0, np.pi, 10000)
    keff_h = modified_wavenumber(kh_fine, fd_order)
    max_keff_h = np.max(np.abs(keff_h))

    if scheme == 'forward_euler':
        cfl_max = 0.0
        condition = "无条件不稳定"
    elif scheme == 'leapfrog':
        cfl_max = 1.0 / max_keff_h
        condition = f"nu <= 1/max|keff*h| = {cfl_max:.6f}"
    elif scheme == 'lax_friedrichs':
        cfl_max = 1.0
        condition = "nu <= 1"
    elif scheme == 'rk4':
        # RK4 稳定性边界: |exp(z) 的 4 阶 Taylor 近似| <= 1
        # 对于纯虚数 z = i*y, 稳定域约为 |y| <= 2.83
        cfl_max = 2.828 / max_keff_h
        condition = f"nu * max|keff*h| <= 2.828 -> nu <= {cfl_max:.6f}"
    else:
        raise ValueError(f"未知格式: {scheme}")

    return {
        'fd_order': fd_order,
        'scheme': scheme,
        'max_keff_h': max_keff_h,
        'cfl_max_nu': cfl_max,
        'cfl_max_dt': cfl_max / c if c > 0 else float('inf'),
        'condition': condition,
    }


def stability_diagram(fd_orders=None, schemes=None, n_nu=50, n_kh=200):
    """生成完整的稳定性图数据。

    对每个 (fd_order, scheme) 组合，计算放大因子随波数的变化,
    并确定稳定性边界。

    Parameters
    ----------
    fd_orders : list of int
        空间差分阶数列表
    schemes : list of str
        时间积分格式列表
    n_nu : int
        Courant 数采样数
    n_kh : int
        波数采样数

    Returns
    -------
    diagram : dict
        完整的稳定性图数据
    """
    if fd_orders is None:
        fd_orders = [2, 4, 6, 8]
    if schemes is None:
        schemes = ['leapfrog', 'lax_friedrichs', 'rk4']

    kh = np.linspace(0, np.pi, n_kh)
    nu_values = np.linspace(0.01, 2.0, n_nu)

    diagram = {}
    for order in fd_orders:
        diagram[order] = {}
        cfl = cfl_condition(order, 'rk4')
        diagram[order]['cfl_rk4'] = cfl

        for scheme in schemes:
            # 找出最大稳定 nu
            max_stable_nu = 0.0
            for nu in nu_values:
                result = von_neumann_advection(
                    np.array([nu]), kh, scheme, order
                )
                key = f'modG_nu{nu:.3f}'
                if key in result and np.all(result[key] <= 1.0 + 1e-14):
                    max_stable_nu = nu
                else:
                    break

            diagram[order][scheme] = {
                'max_stable_nu': max_stable_nu,
                'cfl_satisfied': max_stable_nu > 0,
            }

    return diagram


def rk4_stability_boundary(n_points=500):
    """计算 RK4 方法的稳定性边界。

    RK4 的放大函数为:
        R(z) = 1 + z + z^2/2 + z^3/6 + z^4/24

    稳定域为 |R(z)| <= 1 的 z 区域。

    对于纯虚数 z = iy (对流方程的空间半离散化):
        R(iy) = 1 + iy - y^2/2 - iy^3/6 + y^4/24
        |R(iy)|^2 = (1 - y^2/2 + y^4/24)^2 + (y - y^3/6)^2

    Parameters
    ----------
    n_points : int
        边界采样点数

    Returns
    -------
    boundary : dict
        稳定性边界数据
    """
    theta = np.linspace(0, 2 * np.pi, n_points)

    # 纯虚轴上的稳定域
    y_max = np.linspace(0, 4, 1000)
    R_values = (1 + 1j * y_max - y_max ** 2 / 2
                - 1j * y_max ** 3 / 6 + y_max ** 4 / 24)
    mod_R = np.abs(R_values)
    stable_mask = mod_R <= 1.0 + 1e-14
    if np.any(stable_mask):
        y_stable_max = y_max[stable_mask][-1]
    else:
        y_stable_max = 0.0

    # 实轴上的稳定域 (扩散方程)
    x_values = np.linspace(-5, 0, 1000)
    R_real = 1 + x_values + x_values ** 2 / 2 + x_values ** 3 / 6 + x_values ** 4 / 24
    stable_real = x_values[R_real >= -1.0]
    x_stable_min = stable_real[0] if len(stable_real) > 0 else 0.0

    return {
        'imag_axis_stable_max': y_stable_max,
        'real_axis_stable_min': x_stable_min,
        'theta': theta,
    }
