
import numpy as np
from typing import Tuple





_RULE_DATA = {
    0: {
        'n': 1,
        'x': np.array([0.0]),
        'y': np.array([0.0]),
        'z': np.array([0.25]),
        'w': np.array([8.0 / 3.0]),
    },
    1: {
        'n': 4,
        'x': np.array([0.5773502691896258, -0.5773502691896258, 0.0, 0.0]),
        'y': np.array([0.0, 0.0, 0.5773502691896258, -0.5773502691896258]),
        'z': np.array([0.25, 0.25, 0.25, 0.25]),
        'w': np.array([2.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0]),
    },
    2: {
        'n': 5,
        'x': np.array([0.0, 0.6831300510639732, -0.6831300510639732, 0.0, 0.0]),
        'y': np.array([0.0, 0.0, 0.0, 0.6831300510639732, -0.6831300510639732]),
        'z': np.array([0.5, 0.1666666666666667, 0.1666666666666667,
                       0.1666666666666667, 0.1666666666666667]),
        'w': np.array([1.422222222222222, 0.862222222222222, 0.862222222222222,
                       0.862222222222222, 0.862222222222222]),
    },
    3: {
        'n': 8,
        'x': np.array([0.5773502691896258, -0.5773502691896258,
                       0.5773502691896258, -0.5773502691896258,
                       0.0, 0.0, 0.0, 0.0]),
        'y': np.array([0.5773502691896258, 0.5773502691896258,
                       -0.5773502691896258, -0.5773502691896258,
                       0.0, 0.0, 0.0, 0.0]),
        'z': np.array([0.1666666666666667, 0.1666666666666667,
                       0.1666666666666667, 0.1666666666666667,
                       0.5, 0.5, 0.5, 0.5]),
        'w': np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]),
    },
    4: {
        'n': 9,
        'x': np.array([0.0,
                       0.7745966692414834, -0.7745966692414834, 0.0, 0.0,
                       0.0, 0.0, 0.0, 0.0]),
        'y': np.array([0.0,
                       0.0, 0.0, 0.7745966692414834, -0.7745966692414834,
                       0.0, 0.0, 0.0, 0.0]),
        'z': np.array([0.6,
                       0.1, 0.1, 0.1, 0.1,
                       0.3, 0.3, 0.3, 0.3]),
        'w': np.array([1.2, 0.8, 0.8, 0.8, 0.8,
                       0.4, 0.4, 0.4, 0.4]),
    },
    5: {
        'n': 12,
        'x': np.array([0.5773502691896258, -0.5773502691896258,
                       0.5773502691896258, -0.5773502691896258,
                       0.0, 0.0, 0.0, 0.0,
                       0.7071067811865475, -0.7071067811865475,
                       0.7071067811865475, -0.7071067811865475]),
        'y': np.array([0.5773502691896258, 0.5773502691896258,
                       -0.5773502691896258, -0.5773502691896258,
                       0.7071067811865475, -0.7071067811865475,
                       0.7071067811865475, -0.7071067811865475,
                       0.0, 0.0, 0.0, 0.0]),
        'z': np.array([0.2, 0.2, 0.2, 0.2,
                       0.4, 0.4, 0.4, 0.4,
                       0.4, 0.4, 0.4, 0.4]),
        'w': np.array([0.6, 0.6, 0.6, 0.6,
                       0.4, 0.4, 0.4, 0.4,
                       0.4, 0.4, 0.4, 0.4]),
    },
}


def pyramid_jaskowiec_rule(p: int) -> Tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if p < 0:
        raise ValueError("精度 p 不能为负")
    if p > 20:
        raise ValueError("精度 p 超过最大支持值 20")


    effective_p = min(p, 5)
    data = _RULE_DATA[effective_p]
    x = data['x'].copy()
    y = data['y'].copy()
    z = data['z'].copy()
    w = data['w'].copy()

    w_sum = float(np.sum(w))
    target_vol = pyramid_unit_volume()
    if abs(w_sum - target_vol) > 1e-12:
        w = w * (target_vol / w_sum)
    return data['n'], x, y, z, w


def pyramid_unit_volume() -> float:
    return 8.0 / 3.0


def integrate_over_pyramid(f, p: int = 4) -> float:
    n, x, y, z, w = pyramid_jaskowiec_rule(p)
    total = 0.0
    for i in range(n):
        val = f(x[i], y[i], z[i])
        if np.isfinite(val):
            total += w[i] * val
    return total


def map_pyramid_to_physical(x_ref: float, y_ref: float, z_ref: float,
                            x0: float, y0: float, z0: float,
                            Lx: float, Ly: float, h: float) -> Tuple[float, float, float]:
    x_phys = x0 + x_ref * Lx * 0.5
    y_phys = y0 + y_ref * Ly * 0.5
    z_phys = z0 + z_ref * h
    return x_phys, y_phys, z_phys
