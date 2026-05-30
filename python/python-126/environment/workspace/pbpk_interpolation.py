
import numpy as np
from typing import Callable, Tuple





def runge_function(x: float) -> float:
    return 1.0 / (1.0 + x * x)


def bernstein_example(x: float) -> float:
    return abs(x)


def step_function(x: float, x0: float = 0.0) -> float:
    return 1.0 if x >= x0 else 0.0


def oscillatory_function(x: float) -> float:
    if x < 0.0 or x > 1.0:
        return 0.0
    val = np.sqrt(max(x * (1.0 - x), 0.0)) * np.sin(2.1 * np.pi / (x + 0.05))
    return val


def piecewise_composite(x: float) -> float:
    if x < 0.0:
        return 0.0
    elif x <= 5.0:
        return max(np.sin(x) + np.sin(x * x), 0.0)
    else:
        return max(1.0 - abs(x - 5.0) / 5.0, 0.0)






def lagrange_interpolate(x_nodes: np.ndarray, y_nodes: np.ndarray,
                          x_eval: np.ndarray) -> np.ndarray:
    if len(x_nodes) != len(y_nodes):
        raise ValueError("x_nodes and y_nodes must have same length")
    n = len(x_nodes)
    result = np.zeros_like(x_eval, dtype=float)
    for i in range(n):
        Li = np.ones_like(x_eval, dtype=float)
        for j in range(n):
            if i != j:
                diff = x_nodes[i] - x_nodes[j]
                if abs(diff) < 1e-15:
                    raise ValueError("Duplicate nodes")
                Li *= (x_eval - x_nodes[j]) / diff
        result += y_nodes[i] * Li
    return result


def chebyshev_nodes(n: int, a: float = -1.0, b: float = 1.0) -> np.ndarray:
    if n < 1:
        raise ValueError("n must be positive")
    k = np.arange(n)
    x = 0.5 * (a + b) + 0.5 * (b - a) * np.cos((2.0 * k + 1.0) * np.pi / (2.0 * n))
    return x






def piecewise_linear_interpolate(x_nodes: np.ndarray, y_nodes: np.ndarray,
                                  x_eval: np.ndarray) -> np.ndarray:
    if len(x_nodes) != len(y_nodes):
        raise ValueError("x_nodes and y_nodes must have same length")
    if len(x_nodes) < 2:
        raise ValueError("Need at least 2 nodes")
    sorted_idx = np.argsort(x_nodes)
    x_s = x_nodes[sorted_idx]
    y_s = y_nodes[sorted_idx]
    result = np.empty_like(x_eval, dtype=float)
    for i, x in enumerate(x_eval):
        if x <= x_s[0]:
            result[i] = y_s[0]
        elif x >= x_s[-1]:
            result[i] = y_s[-1]
        else:
            idx = np.searchsorted(x_s, x) - 1
            idx = max(0, min(idx, len(x_s) - 2))
            dx = x_s[idx + 1] - x_s[idx]
            if abs(dx) < 1e-15:
                result[i] = y_s[idx]
            else:
                t = (x - x_s[idx]) / dx
                result[i] = y_s[idx] * (1.0 - t) + y_s[idx + 1] * t
    return result






def pd_effect_interpolate(concentration: float,
                           C_nodes: np.ndarray,
                           E_nodes: np.ndarray,
                           method: str = "linear") -> float:
    if concentration < 0.0:
        raise ValueError("Concentration must be non-negative")
    if len(C_nodes) != len(E_nodes):
        raise ValueError("Node arrays must have same length")
    C = np.asarray(C_nodes, dtype=float)
    E = np.asarray(E_nodes, dtype=float)
    if method == "linear":
        val = piecewise_linear_interpolate(C, E, np.array([concentration]))
        return float(val[0])
    elif method == "lagrange":
        if len(C) > 10:
            raise ValueError("Lagrange interpolation unstable for >10 nodes")
        val = lagrange_interpolate(C, E, np.array([concentration]))
        return float(val[0])
    else:
        raise ValueError("method must be 'linear' or 'lagrange'")


def build_pd_curve_from_hill(C50: float, Emax: float, n_hill: float,
                              n_points: int = 50) -> Tuple[np.ndarray, np.ndarray]:
    if C50 <= 0 or Emax < 0 or n_hill <= 0:
        raise ValueError("Invalid Hill parameters")

    C_nodes = np.logspace(-3, 3, n_points) * C50
    E_nodes = Emax * (C_nodes ** n_hill) / (C50 ** n_hill + C_nodes ** n_hill)
    return C_nodes, E_nodes


def pharmacodynamic_response(C_plasma: float, C50: float, Emax: float,
                              n_hill: float, baseline: float = 0.0) -> float:
    if C_plasma < 0 or C50 <= 0 or Emax < 0 or n_hill <= 0:
        raise ValueError("Invalid PD parameters")
    effect = baseline + Emax * (C_plasma ** n_hill) / (C50 ** n_hill + C_plasma ** n_hill)
    return effect






if __name__ == "__main__":
    x_test = np.linspace(-1, 1, 100)
    print(f"Runge at 0.5: {runge_function(0.5):.6f}")
    print(f"Bernstein at -0.3: {bernstein_example(-0.3):.6f}")
    print(f"Oscillatory at 0.1: {oscillatory_function(0.1):.6f}")

    nodes = chebyshev_nodes(10, -1, 1)
    vals = np.array([runge_function(xi) for xi in nodes])
    x_fine = np.linspace(-1, 1, 200)
    y_interp = lagrange_interpolate(nodes, vals, x_fine)
    y_exact = np.array([runge_function(xi) for xi in x_fine])
    print(f"Chebyshev Lagrange max error: {np.max(np.abs(y_interp - y_exact)):.4e}")

    C_nodes, E_nodes = build_pd_curve_from_hill(1.0, 100.0, 2.0)
    eff = pd_effect_interpolate(1.5, C_nodes, E_nodes, method="linear")
    print(f"PD effect at 1.5xC50: {eff:.2f}")
    resp = pharmacodynamic_response(2.0, 1.0, 100.0, 2.0, 10.0)
    print(f"Pharmacodynamic response: {resp:.2f}")
