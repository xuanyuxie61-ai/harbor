
import numpy as np
from typing import Tuple, Optional


def lagrange_basis(t_data: np.ndarray, j: int, t: float) -> float:
    n = len(t_data)
    result = 1.0
    for m in range(n):
        if m != j:
            denom = t_data[j] - t_data[m]
            if abs(denom) < 1e-15:
                return 0.0
            result *= (t - t_data[m]) / denom
    return result


def lagrange_interpolate(t_data: np.ndarray, p_data: np.ndarray, t_eval: np.ndarray) -> np.ndarray:
    if len(t_data) != len(p_data):
        raise ValueError("节点与值维度不匹配")
    if len(t_data) < 2:
        raise ValueError("至少需要 2 个节点")

    result = np.zeros_like(t_eval, dtype=float)
    for j in range(len(t_data)):
        lj = np.array([lagrange_basis(t_data, j, t) for t in t_eval])
        result += p_data[j] * lj
    return result


def piecewise_linear_interpolate(t_data: np.ndarray, p_data: np.ndarray, t_eval: np.ndarray) -> np.ndarray:
    if len(t_data) != len(p_data):
        raise ValueError("节点与值维度不匹配")
    t_data = np.asarray(t_data)
    p_data = np.asarray(p_data)
    t_eval = np.asarray(t_eval)

    result = np.zeros_like(t_eval, dtype=float)
    for idx, t in enumerate(t_eval):
        if t <= t_data[0]:
            result[idx] = p_data[0]
        elif t >= t_data[-1]:
            result[idx] = p_data[-1]
        else:

            i = np.searchsorted(t_data, t) - 1
            i = max(0, min(i, len(t_data) - 2))
            dt = t_data[i + 1] - t_data[i]
            if abs(dt) < 1e-15:
                result[idx] = p_data[i]
            else:
                w = (t - t_data[i]) / dt
                result[idx] = p_data[i] * (1.0 - w) + p_data[i + 1] * w
    return result


def clenshaw_curtis_nodes(n: int, a: float = -1.0, b: float = 1.0) -> np.ndarray:
    if n < 2:
        return np.array([(a + b) / 2.0])
    i = np.arange(n)
    x = np.cos((n - 1 - i) * np.pi / (n - 1))

    return 0.5 * (b - a) * x + 0.5 * (b + a)


def fejer1_nodes(n: int, a: float = -1.0, b: float = 1.0) -> np.ndarray:
    i = np.arange(n)
    x = np.cos((2.0 * i + 1.0) * np.pi / (2.0 * n))
    return 0.5 * (b - a) * x + 0.5 * (b + a)


def fejer2_nodes(n: int, a: float = -1.0, b: float = 1.0) -> np.ndarray:
    i = np.arange(n)
    x = np.cos(i * np.pi / (n - 1)) if n > 1 else np.array([0.0])
    if n == 1:
        return np.array([(a + b) / 2.0])
    return 0.5 * (b - a) * x + 0.5 * (b + a)


def arc_length_parameterize(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if points.ndim != 2:
        raise ValueError("points 必须为二维数组 (n_points x dim)")
    n = points.shape[0]
    s = np.zeros(n, dtype=float)
    for i in range(1, n):
        s[i] = s[i - 1] + np.linalg.norm(points[i] - points[i - 1])

    if s[-1] > 1e-15:
        t = s / s[-1]
    else:
        t = np.linspace(0.0, 1.0, n)
    return t, s


def interpolate_credit_curve(
    maturities: np.ndarray,
    values: np.ndarray,
    eval_maturities: np.ndarray,
    method: str = "linear"
) -> np.ndarray:

    idx = np.argsort(maturities)
    maturities = maturities[idx]
    values = values[idx]


    if np.any(maturities < 0):
        raise ValueError("期限不能为负")
    if np.any(eval_maturities < 0):
        raise ValueError("求值期限不能为负")

    if method.lower() == "linear":
        return piecewise_linear_interpolate(maturities, values, eval_maturities)
    elif method.lower() == "lagrange":

        if len(maturities) > 10:
            raise ValueError("Lagrange 插值节点数超过 10，建议使用分段线性")
        return lagrange_interpolate(maturities, values, eval_maturities)
    else:
        raise ValueError(f"不支持的插值方法: {method}")


def test_interpolation():
    t_data = np.array([0.0, 1.0, 2.0, 3.0, 5.0])
    p_data = np.array([0.01, 0.015, 0.022, 0.028, 0.035])
    t_eval = np.array([0.5, 1.5, 4.0])

    p_lin = piecewise_linear_interpolate(t_data, p_data, t_eval)
    assert np.all(p_lin >= np.min(p_data)) and np.all(p_lin <= np.max(p_data)), "线性插值越界"

    p_lag = lagrange_interpolate(t_data, p_data, t_eval)

    p_lag_nodes = lagrange_interpolate(t_data, p_data, t_data)
    assert np.allclose(p_lag_nodes, p_data), "Lagrange 插值节点恢复失败"


    pts = np.random.randn(10, 3)
    t_param, s = arc_length_parameterize(pts)
    assert len(t_param) == 10 and np.isclose(t_param[-1], 1.0), "弧长参数化错误"

    print("interpolation_surfaces test passed.")


if __name__ == "__main__":
    test_interpolation()
