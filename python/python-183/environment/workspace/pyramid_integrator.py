
import numpy as np
from typing import Tuple, Callable, List




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
    if precision < 0:
        precision = 0
    if precision > 5:
        precision = 5
    rule = _PYRAMID_RULES[precision]
    return rule['n'], rule['x'].copy(), rule['y'].copy(), rule['z'].copy(), rule['w'].copy()


def integrate_pyramid(f: Callable, precision: int = 3) -> float:
    n, x, y, z, w = pyramid_witherden_rule(precision)
    total = 0.0
    for k in range(n):
        total += w[k] * f(x[k], y[k], z[k])
    return total


def integrate_causal_effect_parameter_space(ce_func: Callable,
                                             dim: int = 3,
                                             n_samples: int = 500) -> float:
    if dim < 1:
        raise ValueError("维度必须 >=1。")

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

    def f1(x, y, z):
        return x * x + y * y + z

    val = integrate_pyramid(f1, precision=3)
    print(f"[pyramid_integrator] 多项式在金字塔上的积分 (数值): {val:.6f}")


    def ce_func(theta):
        return np.exp(-np.sum(theta ** 2))

    val2 = integrate_causal_effect_parameter_space(ce_func, dim=4, n_samples=1000)
    print(f"[pyramid_integrator] 4D 因果效应期望估计: {val2:.6f}")


    def f2(x, y, z):
        return np.sin(np.pi * x) * np.cos(np.pi * y) * z

    val3 = integrate_on_3d_causal_region(f2, (0.0, 1.0), (0.0, 1.0), (0.0, 1.0), precision=4)
    print(f"[pyramid_integrator] 单位立方体上三角函数积分: {val3:.6f}")
    return val, val2, val3


if __name__ == "__main__":
    demo()
