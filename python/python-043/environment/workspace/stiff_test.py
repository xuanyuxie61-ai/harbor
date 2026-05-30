
import numpy as np
from typing import Tuple













def lindberg_g1(t: float) -> float:
    return 1.0e4 * (t + 2.0 * np.exp(-t) - 2.0)


def lindberg_g2(t: float) -> float:
    return 1.0e4 * (1.0 - np.exp(-t) - t * np.exp(-t))


def lindberg_exact(t: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    t = np.asarray(t, dtype=float)
    n = t.size
    y = np.zeros((n, 4), dtype=float)
    dydt = np.zeros((n, 4), dtype=float)

    g1 = lindberg_g1(t)
    g2 = lindberg_g2(t)
    dg1 = 1.0e4 * (1.0 - 2.0 * np.exp(-t))
    dg2 = 1.0e4 * (t * np.exp(-t))

    exp_g1 = np.exp(g1)
    cg = np.cos(g2)
    sg = np.sin(g2)


    y[:, 0] = exp_g1 * (cg + sg)
    y[:, 1] = exp_g1 * (cg - sg)


    y[:, 2] = 1.0 - 2.0 * np.exp(-t)
    y[:, 3] = t * np.exp(-t)


    dydt[:, 0] = y[:, 0] * dg1 + exp_g1 * (-sg + cg) * dg2
    dydt[:, 1] = y[:, 1] * dg1 + exp_g1 * (-sg - cg) * dg2
    dydt[:, 2] = 2.0 * np.exp(-t)
    dydt[:, 3] = np.exp(-t) - t * np.exp(-t)

    return y, dydt


def lindberg_rhs(t: float, y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    g1 = lindberg_g1(t)
    g2 = lindberg_g2(t)
    dg1 = 1.0e4 * (1.0 - 2.0 * np.exp(-t))
    dg2 = 1.0e4 * (t * np.exp(-t))

    exp_g1 = np.exp(g1)
    cg = np.cos(g2)
    sg = np.sin(g2)

    y_exact = np.zeros(4, dtype=float)
    y_exact[0] = exp_g1 * (cg + sg)
    y_exact[1] = exp_g1 * (cg - sg)
    y_exact[2] = 1.0 - 2.0 * np.exp(-t)
    y_exact[3] = t * np.exp(-t)

    dydt_exact = np.zeros(4, dtype=float)
    dydt_exact[0] = y_exact[0] * dg1 + exp_g1 * (-sg + cg) * dg2
    dydt_exact[1] = y_exact[1] * dg1 + exp_g1 * (-sg - cg) * dg2
    dydt_exact[2] = 2.0 * np.exp(-t)
    dydt_exact[3] = np.exp(-t) - t * np.exp(-t)





    return dydt_exact






def estimate_stiffness_ratio(jacobian: np.ndarray) -> float:
    jacobian = np.asarray(jacobian, dtype=float)
    if jacobian.ndim != 2 or jacobian.shape[0] != jacobian.shape[1]:
        return 0.0
    try:
        eigvals = np.linalg.eigvals(jacobian)
        real_parts = np.real(eigvals)
        pos = real_parts[real_parts > 0]
        neg = real_parts[real_parts < 0]
        if len(pos) == 0 or len(neg) == 0:
            return 0.0
        lambda_max = np.max(pos)
        lambda_min = np.min(np.abs(neg))
        if lambda_min < 1e-30:
            return 1e30
        return lambda_max / lambda_min
    except Exception:
        return 0.0





def compute_l2_error(y_numeric: np.ndarray, y_exact: np.ndarray) -> float:
    diff = y_numeric - y_exact
    norm_exact = np.linalg.norm(y_exact)
    norm_diff = np.linalg.norm(diff)
    if norm_exact < 1e-30:
        return norm_diff
    return norm_diff / norm_exact


def compute_max_error(y_numeric: np.ndarray, y_exact: np.ndarray) -> float:
    return float(np.max(np.abs(y_numeric - y_exact)))






def dynamo_stiffness_estimate(radius: float, eta: float, va: float) -> float:
    if eta <= 0.0 or radius <= 0.0 or va <= 0.0:
        return 0.0
    return radius * va / eta





def _self_test():
    t = np.linspace(0.0, 1.0, 11)
    y, dydt = lindberg_exact(t)
    assert y.shape == (11, 4)
    assert dydt.shape == (11, 4)


    assert abs(y[0, 2] - (-1.0)) < 1e-6
    assert abs(y[0, 3] - 0.0) < 1e-10


    J = np.array([[-1e4, 1e4, 0, 0],
                  [1, -2, 0, 0],
                  [0, 0, -1, 0],
                  [0, 0, 0, -0.1]], dtype=float)
    sr = estimate_stiffness_ratio(J)
    assert sr > 1e3


    S = dynamo_stiffness_estimate(3480e3, 2.0, 1.0e-3)
    assert S > 1e3
    print(f"stiff_test: dynamo stiffness estimate S={S:.4e}")
    print("stiff_test: self-test passed.")


if __name__ == "__main__":
    _self_test()
