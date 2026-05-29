"""
quadrature_rules.py
================================================================================
高斯-帕特森求积规则模块 —— 基于种子项目 851_patterson_rule

在 LES 的湍流统计量计算中，需要高精度数值积分来评估：
- 平均场量：⟨u⟩, ⟨v⟩, ⟨w⟩, ⟨θ⟩
- 二阶矩：⟨u'u'⟩, ⟨u'w'⟩, ⟨w'θ'⟩ 等
- 高阶结构函数：S_p(r) = ⟨(u(x+r) - u(x))^p⟩

Gauss-Patterson 规则是嵌套高斯求积规则，具有最优的代数精度。

核心物理公式
--------------------------------------------------------------------------------
一维积分：
    ∫_{-1}^{1} f(x) dx ≈ Σ_{i=1}^{n} w_i f(x_i)

对于截断到 [-1,1] 的 Gauss-Patterson 规则，节点数 n 为：
    n ∈ {1, 3, 7, 15, 31, 63, 127, 255, 511}

三维积分（用于统计量体积平均）：
    ∫_V f(x,y,z) dV ≈ Σ_{i,j,k} w_i w_j w_k f(x_i, y_j, z_k)  ΔV

能量耗散率的空间平均：
    ε = 2 ν ⟨s_{ij} s_{ij}⟩

其中 s_{ij} = 1/2 (∂u_i/∂x_j + ∂u_j/∂x_i) 为应变率张量。
"""

import numpy as np


# Gauss-Patterson 节点与权重（预计算到 n=31）
_PATTERSON_TABLE = {
    1: {
        'x': np.array([0.0]),
        'w': np.array([2.0])
    },
    3: {
        'x': np.array([-0.7745966692414834, 0.0, 0.7745966692414834]),
        'w': np.array([0.5555555555555556, 0.8888888888888889, 0.5555555555555556])
    },
    7: {
        'x': np.array([-0.9604912687080203, -0.7745966692414834, -0.43424374934680256,
                        0.0, 0.43424374934680256, 0.7745966692414834, 0.9604912687080203]),
        'w': np.array([0.10465622602646727, 0.26848808986833344, 0.40139741477596224,
                       0.45091653865847414, 0.40139741477596224, 0.26848808986833344,
                       0.10465622602646727])
    },
    15: {
        'x': np.array([-0.993831963212755, -0.9604912687080203, -0.888459232872257,
                       -0.7745966692414834, -0.6211029467372264, -0.43424374934680256,
                       -0.22338668642896688, 0.0, 0.22338668642896688,
                       0.43424374934680256, 0.6211029467372264, 0.7745966692414834,
                       0.888459232872257, 0.9604912687080203, 0.993831963212755]),
        'w': np.array([0.01700171962994026, 0.05160328299707974, 0.09292719531512454,
                       0.13441525524378423, 0.17151190913639138, 0.20062852937698903,
                       0.2191568584015875, 0.2255104997982067, 0.2191568584015875,
                       0.20062852937698903, 0.17151190913639138, 0.13441525524378423,
                       0.09292719531512454, 0.05160328299707974, 0.01700171962994026])
    },
    31: {
        'x': np.array([
            -0.9990981249676676, -0.993831963212755, -0.9815311495537401,
            -0.9604912687080203, -0.9296548574297401, -0.888459232872257,
            -0.8367259381688687, -0.7745966692414834, -0.7024962064915271,
            -0.6211029467372264, -0.5313197436443756, -0.43424374934680256,
            -0.33113539325797684, -0.22338668642896688, -0.11248894313318663,
            0.0, 0.11248894313318663, 0.22338668642896688, 0.33113539325797684,
            0.43424374934680256, 0.5313197436443756, 0.6211029467372264,
            0.7024962064915271, 0.7745966692414834, 0.8367259381688687,
            0.888459232872257, 0.9296548574297401, 0.9604912687080203,
            0.9815311495537401, 0.993831963212755, 0.9990981249676676
        ]),
        'w': np.array([
            0.0025447807915618745, 0.008434565739321106, 0.01644604985438781,
            0.025807598096176654, 0.03595710330712932, 0.046462893261757986,
            0.05697950949412336, 0.0672077542959907, 0.07687962049900353,
            0.08575592004999035, 0.09362710998126447, 0.10031427861179558,
            0.10566989358023481, 0.10957842105592464, 0.11195687302095346,
            0.11275525672076869, 0.11195687302095346, 0.10957842105592464,
            0.10566989358023481, 0.10031427861179558, 0.09362710998126447,
            0.08575592004999035, 0.07687962049900353, 0.0672077542959907,
            0.05697950949412336, 0.046462893261757986, 0.03595710330712932,
            0.025807598096176654, 0.01644604985438781, 0.008434565739321106,
            0.0025447807915618745
        ])
    }
}


def get_patterson_rule(n):
    """
    获取 Gauss-Patterson 求积规则。

    参数
    ----------
    n : int
        节点数，必须是 1, 3, 7, 15, 31 之一

    返回
    -------
    x, w : np.ndarray
        节点与权重（在 [-1,1] 区间）
    """
    if n not in _PATTERSON_TABLE:
        raise ValueError(f"get_patterson_rule: 不支持的阶数 {n}，支持 {list(_PATTERSON_TABLE.keys())}")
    return _PATTERSON_TABLE[n]['x'].copy(), _PATTERSON_TABLE[n]['w'].copy()


def rescale_rule(x, w, a, b):
    """
    将 [-1, 1] 上的求积规则缩放到 [a, b]。

    参数
    ----------
    x, w : np.ndarray
        标准区间上的节点与权重
    a, b : float
        目标区间

    返回
    -------
    x_new, w_new : np.ndarray
    """
    scale = (b - a) / 2.0
    shift = (a + b) / 2.0
    return x * scale + shift, w * scale


def patterson_integrate_1d(f_func, a, b, order=15):
    """
    使用 Gauss-Patterson 规则计算一维积分。

    参数
    ----------
    f_func : callable
        被积函数
    a, b : float
        积分上下限
    order : int
        求积阶数

    返回
    -------
    result : float
    """
    x_std, w_std = get_patterson_rule(order)
    x, w = rescale_rule(x_std, w_std, a, b)
    return np.sum(w * f_func(x))


def patterson_integrate_3d(f_func, xlim, ylim, zlim, order=7):
    """
    使用张量积 Gauss-Patterson 规则计算三维积分。

    参数
    ----------
    f_func : callable
        f_func(x, y, z) → scalar or array
    xlim, ylim, zlim : tuple
        (min, max) 区间
    order : int
        每个维度的求积阶数

    返回
    -------
    result : float
    """
    x_std, w_x = get_patterson_rule(order)
    y_std, w_y = get_patterson_rule(order)
    z_std, w_z = get_patterson_rule(order)

    x, wx = rescale_rule(x_std, w_x, xlim[0], xlim[1])
    y, wy = rescale_rule(y_std, w_y, ylim[0], ylim[1])
    z, wz = rescale_rule(z_std, w_z, zlim[0], zlim[1])

    result = 0.0
    for i in range(len(x)):
        for j in range(len(y)):
            for k in range(len(z)):
                result += wx[i] * wy[j] * wz[k] * f_func(x[i], y[j], z[k])

    return result


def compute_energy_dissipation_rate(u, v, w, dx, dy, dz, nu):
    """
    计算体积平均湍动能耗散率 ε = 2 ν ⟨s_{ij} s_{ij}⟩。

    参数
    ----------
    u, v, w : np.ndarray, shape (nx, ny, nz)
        速度分量
    dx, dy, dz : float
        网格间距
    nu : float
        运动粘性系数

    返回
    -------
    epsilon : float
        平均耗散率（m²/s³）
    """
    # 中心差分计算速度梯度
    def central_diff(f, axis, h):
        df = np.zeros_like(f)
        slc_p = [slice(None)] * 3
        slc_m = [slice(None)] * 3
        slc_c = [slice(None)] * 3
        slc_p[axis] = slice(2, None)
        slc_m[axis] = slice(None, -2)
        slc_c[axis] = slice(1, -1)
        df[tuple(slc_c)] = (f[tuple(slc_p)] - f[tuple(slc_m)]) / (2 * h)
        return df

    dudx = central_diff(u, 0, dx)
    dudy = central_diff(u, 1, dy)
    dudz = central_diff(u, 2, dz)

    dvdx = central_diff(v, 0, dx)
    dvdy = central_diff(v, 1, dy)
    dvdz = central_diff(v, 2, dz)

    dwdx = central_diff(w, 0, dx)
    dwdy = central_diff(w, 1, dy)
    dwdz = central_diff(w, 2, dz)

    # 应变率张量 s_ij = 0.5 (∂u_i/∂x_j + ∂u_j/∂x_i)
    s11 = dudx
    s22 = dvdy
    s33 = dwdz
    s12 = 0.5 * (dudy + dvdx)
    s13 = 0.5 * (dudz + dwdx)
    s23 = 0.5 * (dvdz + dwdy)

    # 2 s_ij s_ij
    s2 = 2.0 * (s11**2 + s22**2 + s33**2 + 2.0 * s12**2 + 2.0 * s13**2 + 2.0 * s23**2)

    # 体积平均（仅内部点）
    nx, ny, nz = u.shape
    n_inner = (nx - 2) * (ny - 2) * (nz - 2)
    if n_inner <= 0:
        n_inner = nx * ny * nz
        epsilon = 2.0 * nu * np.mean(s2)
    else:
        epsilon = 2.0 * nu * np.sum(s2[1:-1, 1:-1, 1:-1]) / n_inner

    return epsilon
