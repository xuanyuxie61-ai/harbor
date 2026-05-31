# -*- coding: utf-8 -*-
"""
quadrature_engine.py

基于 quad_rule (Gauss-Legendre) 与 sparse_grid_hermite 的
高维数值积分引擎。

原项目 943_quad_rule 提供了多达 33 点的 Gauss-Legendre 求积法则；
原项目 1105_sparse_grid_hermite 提供了基于 Gauss-Hermite 的稀疏网格。
二者融合后用于:
    1. 沿激光射线进行高精度一维积分（能量沉积、光程）。
    2. 对激光参数空间进行高维稀疏网格积分（不确定性量化）。

核心公式:
    Gauss-Legendre 积分:
        ∫_{-1}^{1} f(x) dx ≈ Σ_i w_i * f(x_i)
    区间变换:
        ∫_{a}^{b} f(x) dx = (b-a)/2 * Σ_i w_i * f((b-a)/2 * x_i + (b+a)/2)

    稀疏网格 (Smolyak) 对于 d 维积分:
        Q_d^{L} = Σ_{max(0, L+1-d) <= |l| <= L} (-1)^{L-|l|} * C(d-1, L-|l|) * (Q_{l1} ⊗ ... ⊗ Q_{ld})
    其中 Q_{li} 为第 i 维的 1D 法则。
"""

import numpy as np
from itertools import product


# Gauss-Legendre 节点与权重的预计算表 (支持 1 到 10 点)
_LEGENDRE_TABLE = {
    1: {
        'x': np.array([0.0]),
        'w': np.array([2.0])
    },
    2: {
        'x': np.array([-0.5773502691896258, 0.5773502691896258]),
        'w': np.array([1.0, 1.0])
    },
    3: {
        'x': np.array([-0.7745966692414834, 0.0, 0.7745966692414834]),
        'w': np.array([0.5555555555555556, 0.8888888888888889, 0.5555555555555556])
    },
    4: {
        'x': np.array([-0.8611363115940526, -0.3399810435848563,
                       0.3399810435848563, 0.8611363115940526]),
        'w': np.array([0.3478548451374539, 0.6521451548625461,
                       0.6521451548625461, 0.3478548451374539])
    },
    5: {
        'x': np.array([-0.9061798459386640, -0.5384693101056831, 0.0,
                       0.5384693101056831, 0.9061798459386640]),
        'w': np.array([0.2369268850561891, 0.4786286704993665, 0.5688888888888889,
                       0.4786286704993665, 0.2369268850561891])
    },
    6: {
        'x': np.array([-0.9324695142031520, -0.6612093864662645, -0.2386191860831969,
                       0.2386191860831969, 0.6612093864662645, 0.9324695142031520]),
        'w': np.array([0.1713244923791703, 0.3607615730481386, 0.4679139345726910,
                       0.4679139345726910, 0.3607615730481386, 0.1713244923791703])
    },
    7: {
        'x': np.array([-0.9491079123427585, -0.7415311855993945, -0.4058451513773972, 0.0,
                       0.4058451513773972, 0.7415311855993945, 0.9491079123427585]),
        'w': np.array([0.1294849661688697, 0.2797053914892767, 0.3818300505051189, 0.4179591836734694,
                       0.3818300505051189, 0.2797053914892767, 0.1294849661688697])
    },
    8: {
        'x': np.array([-0.9602898564975362, -0.7966664774136267, -0.5255324099163290, -0.1834346424956498,
                       0.1834346424956498, 0.5255324099163290, 0.7966664774136267, 0.9602898564975362]),
        'w': np.array([0.1012285362903763, 0.2223810344533745, 0.3137066458778873, 0.3626837833783620,
                       0.3626837833783620, 0.3137066458778873, 0.2223810344533745, 0.1012285362903763])
    },
    9: {
        'x': np.array([-0.9681602395076261, -0.8360311073266358, -0.6133714327005904,
                       -0.3242534234038089, 0.0, 0.3242534234038089,
                       0.6133714327005904, 0.8360311073266358, 0.9681602395076261]),
        'w': np.array([0.0812743883615744, 0.1806481606948574, 0.2606106964029355,
                       0.3123470770400028, 0.3302393550012598, 0.3123470770400028,
                       0.2606106964029355, 0.1806481606948574, 0.0812743883615744])
    },
    10: {
        'x': np.array([-0.9739065285171717, -0.8650633666889845, -0.6794095682990244,
                       -0.4333953941292472, -0.1488743389816312, 0.1488743389816312,
                       0.4333953941292472, 0.6794095682990244, 0.8650633666889845, 0.9739065285171717]),
        'w': np.array([0.0666713443086881, 0.1494513491505806, 0.2190863625159820,
                       0.2692667193099964, 0.2955242247147529, 0.2955242247147529,
                       0.2692667193099964, 0.2190863625159820, 0.1494513491505806, 0.0666713443086881])
    }
}


def gauss_legendre_rule(n):
    """
    获取 n 点 Gauss-Legendre 求积法则。

    Parameters
    ----------
    n : int
        点数，支持 1 到 10。

    Returns
    -------
    x, w : ndarray
        节点和权重。
    """
    if n not in _LEGENDRE_TABLE:
        raise ValueError(f"Gauss-Legendre 规则仅支持 n=1..10，收到 n={n}。")
    return _LEGENDRE_TABLE[n]['x'].copy(), _LEGENDRE_TABLE[n]['w'].copy()


def integrate_1d_gauss_legendre(f, a, b, n=8):
    """
    使用 Gauss-Legendre 求积计算一维积分 ∫_a^b f(x) dx。

    变换公式:
        x = (b-a)/2 * t + (b+a)/2
        dx = (b-a)/2 * dt

    Parameters
    ----------
    f : callable
        被积函数，接受 ndarray 输入返回 ndarray。
    a, b : float
        积分区间。
    n : int, optional
        求积点数，默认 8。

    Returns
    -------
    result : float
        积分近似值。
    """
    if a >= b:
        return 0.0
    t, w = gauss_legendre_rule(n)
    x = 0.5 * (b - a) * t + 0.5 * (b + a)
    fx = f(x)
    fx = np.asarray(fx, dtype=float)
    result = 0.5 * (b - a) * np.sum(w * fx)
    return float(result)


def integrate_along_ray_gauss(f_s, s_nodes, n_quad=8):
    """
    对给定路径节点上的函数值进行分段 Gauss-Legendre 积分。

    Parameters
    ----------
    f_s : callable or ndarray
        若 callable: 接受光程 s 的函数；
        若 ndarray: 与 s_nodes 等长的函数采样值（使用梯形法则）。
    s_nodes : ndarray
        光程节点序列（已排序）。
    n_quad : int, optional
        每段的 Gauss-Legendre 点数。

    Returns
    -------
    integral : float
        积分值。
    """
    s_nodes = np.asarray(s_nodes, dtype=float)
    if len(s_nodes) < 2:
        return 0.0
    if np.any(np.diff(s_nodes) <= 0):
        raise ValueError("s_nodes 必须严格递增。")

    if callable(f_s):
        t, w = gauss_legendre_rule(n_quad)
        integral = 0.0
        for i in range(len(s_nodes) - 1):
            a = s_nodes[i]
            b = s_nodes[i + 1]
            if b <= a:
                continue
            s_quad = 0.5 * (b - a) * t + 0.5 * (b + a)
            fs = f_s(s_quad)
            integral += 0.5 * (b - a) * np.sum(w * fs)
    else:
        f_vals = np.asarray(f_s, dtype=float)
        if len(f_vals) != len(s_nodes):
            raise ValueError("f_s 作为数组时长度必须与 s_nodes 一致。")
        integral = np.trapezoid(f_vals, s_nodes)

    return float(integral)


# Gauss-Hermite 节点与权重 (标准权重 exp(-x^2), 支持 1 到 7 点)
_HERMITE_TABLE = {
    1: {
        'x': np.array([0.0]),
        'w': np.array([np.sqrt(np.pi)])
    },
    2: {
        'x': np.array([-0.7071067811865475, 0.7071067811865475]),
        'w': np.array([0.8862269254527580, 0.8862269254527580])
    },
    3: {
        'x': np.array([-1.224744871391589, 0.0, 1.224744871391589]),
        'w': np.array([0.2954089751509193, 1.181635900603677, 0.2954089751509193])
    },
    4: {
        'x': np.array([-1.650680123885785, -0.5246476232752900,
                       0.5246476232752900, 1.650680123885785]),
        'w': np.array([0.0813128354472452, 0.8049140900055135,
                       0.8049140900055135, 0.0813128354472452])
    },
    5: {
        'x': np.array([-2.020182870456086, -0.9585724646138185, 0.0,
                       0.9585724646138185, 2.020182870456086]),
        'w': np.array([0.0199532420590459, 0.3936193231522402, 0.9453087204829419,
                       0.3936193231522402, 0.0199532420590459])
    },
    6: {
        'x': np.array([-2.350604973674492, -1.335849074013697, -0.4360774119276165,
                       0.4360774119276165, 1.335849074013697, 2.350604973674492]),
        'w': np.array([0.0045300099055088, 0.1570673203228566, 0.7246295952243924,
                       0.7246295952243924, 0.1570673203228566, 0.0045300099055088])
    },
    7: {
        'x': np.array([-2.651961356835233, -1.673551628767471, -0.8162878828589647, 0.0,
                       0.8162878828589647, 1.673551628767471, 2.651961356835233]),
        'w': np.array([0.0009717812450995, 0.0545155828191270, 0.4256072526101278,
                       0.8102646175568073, 0.4256072526101278, 0.0545155828191270, 0.0009717812450995])
    }
}


def hermite_rule(order):
    """
    获取 order 点 Gauss-Hermite 求积法则（标准权重 exp(-x^2)）。

    Parameters
    ----------
    order : int
        点数，支持 1 到 7。

    Returns
    -------
    x, w : ndarray
        节点和权重。
    """
    if order not in _HERMITE_TABLE:
        raise ValueError(f"Gauss-Hermite 规则仅支持 order=1..7，收到 order={order}。")
    return _HERMITE_TABLE[order]['x'].copy(), _HERMITE_TABLE[order]['w'].copy()


def level_to_order_open(level):
    """
    将一维稀疏网格层级映射为 Gauss-Hermite 求积阶数。

    规则: order = 2^{level+1} - 1，但被截断到可用表的最大值。

    Parameters
    ----------
    level : int
        层级 (>= 0)。

    Returns
    -------
    order : int
        求积阶数。
    """
    if level < 0:
        return 1
    order = 2 ** (level + 1) - 1
    max_avail = max(_HERMITE_TABLE.keys())
    if order > max_avail:
        order = max_avail
    return order


def comp_next(n, k, a, more, h, t):
    """
    基于原项目 1105_sparse_grid_hermite 的 comp_next 算法:
    生成整数 n 的 k 部分下一个组合（composition）。

    Parameters
    ----------
    n : int
        目标和。
    k : int
        部分数。
    a : list or ndarray
        当前组合。
    more : bool
        是否还有更多组合。
    h, t : int
        内部状态。

    Returns
    -------
    a : ndarray
        下一个组合。
    more : bool
        是否还有更多。
    h, t : int
        更新后的状态。
    """
    a = np.asarray(a, dtype=int)
    if not more:
        a[:] = 0
        a[0] = n
        more = True
        h = 0
        t = n if k > 1 else 0
    else:
        if 1 < t:
            h = 0
        h = h + 1
        t = a[h - 1]
        a[h - 1] = 0
        a[0] = t - 1
        a[h] = a[h] + 1
        if t - 1 != 0:
            h = 1
            t = a[0]
        more = (a[k - 1] != n)
    return a, more, h, t


def sparse_grid_hermite_size(dim_num, level_max):
    """
    计算稀疏网格的总点数。

    基于原 sparse_grid_herm_size 算法。

    Parameters
    ----------
    dim_num : int
        空间维度。
    level_max : int
        最大层级。

    Returns
    -------
    point_num : int
        总点数。
    """
    if level_max == 0:
        return 1
    level_min = max(0, level_max + 1 - dim_num)
    point_num = 0
    for level in range(level_min, level_max + 1):
        level_1d = np.zeros(dim_num, dtype=int)
        more = False
        h = 0
        t = 0
        while True:
            level_1d, more, h, t = comp_next(level, dim_num, level_1d, more, h, t)
            order_1d = np.array([level_to_order_open(l) for l in level_1d])
            for dim in range(dim_num):
                if level_min < level and 1 < order_1d[dim]:
                    order_1d[dim] -= 1
            point_num += int(np.prod(order_1d))
            if not more:
                break
    return point_num


def sparse_grid_hermite_index(dim_num, level_max):
    """
    生成稀疏网格的索引。

    返回每个维度上对应的一维规则索引。

    Parameters
    ----------
    dim_num : int
        空间维度。
    level_max : int
        最大层级。

    Returns
    -------
    indices : list of tuple
        每个点对应的 (dim_num,) 维度的 1D 节点索引组合。
    weights : list of float
        每个点对应的组合权重。
    """
    if level_max == 0:
        return [(tuple([0] * dim_num),)], [1.0]

    level_min = max(0, level_max + 1 - dim_num)
    points_data = []

    for level in range(level_min, level_max + 1):
        level_1d = np.zeros(dim_num, dtype=int)
        more = False
        h = 0
        t = 0
        while True:
            level_1d, more, h, t = comp_next(level, dim_num, level_1d, more, h, t)
            order_1d = np.array([level_to_order_open(l) for l in level_1d])
            for dim in range(dim_num):
                if level_min < level and 1 < order_1d[dim]:
                    order_1d[dim] -= 1

            # 组合系数
            coeff = ((-1) ** (level_max - level)) * \
                    comb(dim_num - 1, level_max - level)

            # 生成所有笛卡尔积组合
            ranges = [range(o) for o in order_1d]
            for idx_tuple in product(*ranges):
                points_data.append((idx_tuple, order_1d, coeff))

            if not more:
                break

    # 去重: 相同的 idx_tuple 权重相加
    unique = {}
    for idx_tuple, order_1d, coeff in points_data:
        key = tuple(int(i) for i in idx_tuple)
        if key not in unique:
            unique[key] = {'orders': order_1d, 'weight': 0.0}
        # 权重由 1D Hermite 权重乘积再乘以组合系数得到
        # 这里先累积组合系数，后续乘 1D 权重
        unique[key]['weight'] += float(coeff)

    indices = []
    weights = []
    for key, val in unique.items():
        indices.append(key)
        weights.append(val['weight'])

    return indices, weights


def comb(n, k):
    """
    计算组合数 C(n, k)。
    """
    if k < 0 or k > n:
        return 0
    if k == 0 or k == n:
        return 1
    k = min(k, n - k)
    result = 1
    for i in range(1, k + 1):
        result = result * (n - k + i) // i
    return result


def integrate_nd_sparse_hermite(f, dim_num, level_max):
    """
    使用稀疏 Gauss-Hermite 网格计算 d 维积分:
        ∫_{R^d} f(x) * exp(-|x|^2) dx

    由于标准 Hermite 权重已包含 exp(-x^2)，积分公式为:
        I ≈ Σ_i w_i * f(x_i)

    Parameters
    ----------
    f : callable
        被积函数，接受 shape (d,) 的 ndarray。
    dim_num : int
        维度数。
    level_max : int
        稀疏网格最大层级。

    Returns
    -------
    result : float
        积分近似值。
    """
    indices, comb_weights = sparse_grid_hermite_index(dim_num, level_max)
    result = 0.0
    for idx_tuple, cw in zip(indices, comb_weights):
        point = np.zeros(dim_num, dtype=float)
        w_prod = 1.0
        for d in range(dim_num):
            order = level_to_order_open(0)
            # 推断 order 从索引大小
            if idx_tuple[d] >= 1:
                # 反向推断层级
                for lev in range(1, level_max + 1):
                    ord_lev = level_to_order_open(lev)
                    if idx_tuple[d] < ord_lev:
                        order = ord_lev
                        break
                else:
                    order = level_to_order_open(level_max)
            else:
                order = level_to_order_open(0)
            x_1d, w_1d = hermite_rule(order)
            point[d] = x_1d[idx_tuple[d]]
            w_prod *= w_1d[idx_tuple[d]]
        result += cw * w_prod * f(point)
    return float(result)


def integrate_energy_deposition_along_ray(s_vals, intensity_vals, ne_vals, Te_val, omega0, Z=1):
    """
    沿射线计算逆轫致吸收导致的能量沉积。

    功率沉积密度:
        dP/ds = -κ_ib * I(s)
    其中 κ_ib = (ν_ei / c) * (ω_p^2 / ω_0^2) * (1 / η)

    总沉积能量（沿射线）:
        E_dep = ∫ κ_ib(s) * I(s) ds

    Parameters
    ----------
    s_vals : ndarray
        光程节点 [m]。
    intensity_vals : ndarray
        各节点上的激光强度 [W/m^2]。
    ne_vals : ndarray
        各节点上的电子密度 [m^{-3}]。
    Te_val : float
        电子温度 [K]（假设为常数沿射线）。
    omega0 : float
        激光角频率 [rad/s]。
    Z : int, optional
        离子电荷数。

    Returns
    -------
    energy_dep : float
        单位截面积上的沉积能量 [J/m^2]。
    """
    from physics_constants import plasma_frequency, electron_ion_collision_frequency, C_LIGHT

    s_vals = np.asarray(s_vals, dtype=float)
    intensity_vals = np.asarray(intensity_vals, dtype=float)
    ne_vals = np.asarray(ne_vals, dtype=float)

    if len(s_vals) < 2 or len(intensity_vals) != len(s_vals) or len(ne_vals) != len(s_vals):
        raise ValueError("输入数组长度不一致或不足。")

    kappa_vals = np.zeros_like(ne_vals)
    for i, ne in enumerate(ne_vals):
        if ne <= 0:
            kappa_vals[i] = 0.0
            continue
        nu_ei = electron_ion_collision_frequency(ne, Te_val, Z)
        omega_p = plasma_frequency(ne)
        ratio = (omega_p / omega0) ** 2
        ratio = np.clip(ratio, 0.0, 1.0)
        eta = np.sqrt(1.0 - ratio)
        eta_safe = max(eta, 1e-6)
        kappa_vals[i] = (nu_ei / C_LIGHT) * ratio * (1.0 / eta_safe)
        if not np.isfinite(kappa_vals[i]):
            kappa_vals[i] = 0.0

    # 梯形法则积分
    integrand = kappa_vals * intensity_vals
    energy_dep = np.trapezoid(integrand, s_vals)
    return float(energy_dep)
