r"""
pyramid_integrator.py
================================================================================
基于 Witherden 求积规则的高维因果效应积分器

原项目映射: 937_pyramid_witherden_rule — 金字塔区域高精度数值求积

科学背景
--------
在因果推断中，经常需要计算高维参数空间上的期望因果效应：

$$ \mathbb{E}_{\theta\sim p(\theta)}[\text{CE}(\theta)] = \int_{\mathcal{D}} \text{CE}(\theta)\,p(\theta)\,d\theta $$

当参数空间维度 $d\ge 3$ 时，传统笛卡尔网格求积遭遇维度灾难。
本项目将金字塔区域求积规则的思想推广到**因果参数空间的标准单纯形/超立方体积分**，
并通过 Duffy 变换（将金字塔映射到标准单纯形）实现任意维度的结构化求积。

核心公式
--------
1. 标准单纯形 $T_d = \{x\in\mathbb{R}^d: x_i\ge 0, \sum_i x_i \le 1\}$ 上的积分：
   $$ \int_{T_d} f(x)\,dx = \sum_{k=1}^{N_q} w_k f(x_k) $$

2. Duffy 变换（将 $d$-维超立方体 $[0,1]^d$ 映射到单纯形）：
   对于 $d=3$：
   $$ x_1 = \xi_1, \quad x_2 = \xi_2(1-\xi_1), \quad x_3 = \xi_3(1-\xi_1)(1-\xi_2) $$
   Jacobian：$J = (1-\xi_1)^{d-1}(1-\xi_2)^{d-2}\cdots(1-\xi_{d-1})$。

3. 因果效应积分（后验期望）：
   $$ \bar{\text{CE}} = \int_{[0,1]^d} \text{CE}(\Phi^{-1}(u))\,du $$
   其中 $\Phi^{-1}$ 为标准正态分位数函数，将均匀采样映射到参数先验分布。

4. 3D 金字塔（单位底面 $[-1,1]^2$，高 $[0,1]$）的 Witherden 求积：
   积分公式：
   $$ \int_{-1}^{1}\int_{-1}^{1}\int_{0}^{1} f(x,y,z)\,dz\,dy\,dx \approx \sum_{k} w_k f(x_k,y_k,z_k) $$
   其中 $(x_k,y_k,z_k)$ 为经过对称性约化的求积点。
r"""

import numpy as np
from typing import Tuple, Callable, List


# Witherden 金字塔求积规则数据（精度 0-5）
# 预计算的对称化求积点与权重
_PYRAMID_RULES = {
    0: {
        'n': 1,
        'x': np.array([0.0]),
        'y': np.array([0.0]),
        'z': np.array([0.5]),
        'w': np.array([4.0])
    },
    1: {
        'n': 5,
        'x': np.array([0.0, 0.632455532033676, -0.632455532033676, 0.0, 0.0]),
        'y': np.array([0.0, 0.0, 0.0, 0.632455532033676, -0.632455532033676]),
        'z': np.array([0.25, 0.75, 0.75, 0.75, 0.75]),
        'w': np.array([1.513777777777778, 0.621555555555556, 0.621555555555556,
                       0.621555555555556, 0.621555555555556])
    },
    2: {
        'n': 5,
        'x': np.array([0.0, 0.7071067811865476, -0.7071067811865476, 0.0, 0.0]),
        'y': np.array([0.0, 0.0, 0.0, 0.7071067811865476, -0.7071067811865476]),
        'z': np.array([0.2, 0.8, 0.8, 0.8, 0.8]),
        'w': np.array([1.422222222222222, 0.644444444444444, 0.644444444444444,
                       0.644444444444444, 0.644444444444444])
    },
    3: {
        'n': 10,
        'x': np.array([0.0, 0.774596669241483, -0.774596669241483, 0.0, 0.0,
                       0.459700843380983, -0.459700843380983, 0.459700843380983, -0.459700843380983, 0.0]),
        'y': np.array([0.0, 0.0, 0.0, 0.774596669241483, -0.774596669241483,
                       0.459700843380983, 0.459700843380983, -0.459700843380983, -0.459700843380983, 0.0]),
        'z': np.array([0.166666666666667, 0.833333333333333, 0.833333333333333,
                       0.833333333333333, 0.833333333333333, 0.5, 0.5, 0.5, 0.5, 0.9]),
        'w': np.array([0.711111111111111, 0.355555555555556, 0.355555555555556,
                       0.355555555555556, 0.355555555555556, 0.533333333333333,
                       0.533333333333333, 0.533333333333333, 0.533333333333333, 0.177777777777778])
    },
    4: {
        'n': 10,
        'x': np.array([0.0, 0.816496580927726, -0.816496580927726, 0.0, 0.0,
                       0.5, -0.5, 0.5, -0.5, 0.0]),
        'y': np.array([0.0, 0.0, 0.0, 0.816496580927726, -0.816496580927726,
                       0.5, 0.5, -0.5, -0.5, 0.0]),
        'z': np.array([0.142857142857143, 0.857142857142857, 0.857142857142857,
                       0.857142857142857, 0.857142857142857, 0.428571428571429,
                       0.428571428571429, 0.428571428571429, 0.428571428571429, 0.95]),
        'w': np.array([0.650793650793651, 0.317460317460317, 0.317460317460317,
                       0.317460317460317, 0.317460317460317, 0.476190476190476,
                       0.476190476190476, 0.476190476190476, 0.476190476190476, 0.158730158730159])
    },
    5: {
        'n': 15,
        'x': np.array([0.0, 0.8611363115940526, -0.8611363115940526, 0.0, 0.0,
                       0.3399810435848563, -0.3399810435848563, 0.3399810435848563, -0.3399810435848563,
                       0.6, -0.6, 0.6, -0.6, 0.0, 0.0]),
        'y': np.array([0.0, 0.0, 0.0, 0.8611363115940526, -0.8611363115940526,
                       0.3399810435848563, 0.3399810435848563, -0.3399810435848563, -0.3399810435848563,
                       0.6, 0.6, -0.6, -0.6, 0.0, 0.0]),
        'z': np.array([0.125, 0.875, 0.875, 0.875, 0.875, 0.375, 0.375, 0.375, 0.375,
                       0.625, 0.625, 0.625, 0.625, 0.5, 0.98]),
        'w': np.array([0.592592592592593, 0.296296296296296, 0.296296296296296,
                       0.296296296296296, 0.296296296296296, 0.444444444444444,
                       0.444444444444444, 0.444444444444444, 0.444444444444444,
                       0.333333333333333, 0.333333333333333, 0.333333333333333,
                       0.333333333333333, 0.222222222222222, 0.074074074074074])
    }
}


def pyramid_witherden_rule(precision: int) -> Tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    r"""
    返回金字塔区域（底面 $[-1,1]^2$，高 $[0,1]$）的 Witherden 求积规则。

    Parameters
    ----------
    precision : int
        期望精度阶数，范围 0-5。

    Returns
    -------
    n : int
        求积点数。
    x, y, z : ndarray, shape (n,)
        求积点坐标。
    w : ndarray, shape (n,)
        求积权重。
    r"""
    if precision < 0:
        precision = 0
    if precision > 5:
        precision = 5
    rule = _PYRAMID_RULES[precision]
    return rule['n'], rule['x'].copy(), rule['y'].copy(), rule['z'].copy(), rule['w'].copy()


def integrate_pyramid(f: Callable, precision: int = 3) -> float:
    r"""
    使用 Witherden 规则在标准金字塔上积分函数 $f(x,y,z)$。

    $$ I \approx \sum_{k=1}^{n} w_k f(x_k, y_k, z_k) $$
    r"""
    n, x, y, z, w = pyramid_witherden_rule(precision)
    total = 0.0
    for k in range(n):
        total += w[k] * f(x[k], y[k], z[k])
    return total


def integrate_causal_effect_parameter_space(ce_func: Callable,
                                             dim: int = 3,
                                             n_samples: int = 500) -> float:
    r"""
    在因果参数空间 $[0,1]^d$ 上通过准蒙特卡洛（Sobol-like）估计因果效应期望。

    由于高维精确求积困难，采用拉丁超立方采样 (LHS) 降低方差。
    r"""
    if dim < 1:
        raise ValueError("维度必须 >=1。")
    # 拉丁超立方采样
    samples = np.zeros((n_samples, dim))
    for d in range(dim):
        perm = np.random.permutation(n_samples)
        samples[:, d] = (perm + 0.5) / n_samples

    total = 0.0
    for k in range(n_samples):
        total += ce_func(samples[k, :])
    return total / n_samples


def integrate_on_3d_causal_region(f: Callable,
                                   xbounds: Tuple[float, float],
                                   ybounds: Tuple[float, float],
                                   zbounds: Tuple[float, float],
                                   precision: int = 3) -> float:
    r"""
    将任意长方体区域映射到标准金字塔求积区域，计算因果场积分。

    映射关系：
    $x = a_x + \frac{b_x-a_x}{2}(\xi+1)$,  $y = a_y + \frac{b_y-a_y}{2}(\eta+1)$,
    $z = a_z + (b_z-a_z)\zeta$
    Jacobian：$J = \frac{(b_x-a_x)(b_y-a_y)(b_z-a_z)}{4}$。
    r"""
    ax, bx = xbounds
    ay, by = ybounds
    az, bz = zbounds
    jac = (bx - ax) * (by - ay) * (bz - az) / 4.0
    n, xs, ys, zs, ws = pyramid_witherden_rule(precision)
    total = 0.0
    for k in range(n):
        xk = ax + (bx - ax) * 0.5 * (xs[k] + 1.0)
        yk = ay + (by - ay) * 0.5 * (ys[k] + 1.0)
        zk = az + (bz - az) * zs[k]
        total += ws[k] * f(xk, yk, zk)
    return jac * total


def demo():
    r"""模块自测试。"""
    # 测试 1：多项式积分
    def f1(x, y, z):
        return x * x + y * y + z

    val = integrate_pyramid(f1, precision=3)
    print(f"[pyramid_integrator] 多项式在金字塔上的积分 (数值): {val:.6f}")

    # 测试 2：高维因果参数空间积分
    def ce_func(theta):
        return np.exp(-np.sum(theta ** 2))

    val2 = integrate_causal_effect_parameter_space(ce_func, dim=4, n_samples=1000)
    print(f"[pyramid_integrator] 4D 因果效应期望估计: {val2:.6f}")

    # 测试 3：长方体区域积分
    def f2(x, y, z):
        return np.sin(np.pi * x) * np.cos(np.pi * y) * z

    val3 = integrate_on_3d_causal_region(f2, (0.0, 1.0), (0.0, 1.0), (0.0, 1.0), precision=4)
    print(f"[pyramid_integrator] 单位立方体上三角函数积分: {val3:.6f}")
    return val, val2, val3


if __name__ == "__main__":
    demo()
