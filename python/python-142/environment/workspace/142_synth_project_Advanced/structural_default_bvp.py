
import numpy as np
from typing import Tuple, Optional


def solve_ill_bvp_fd(
    epsilon: float,
    n_nodes: int = 200,
    x_left: float = -1.0,
    x_right: float = 1.0,
    bc_left: float = 2.0,
    bc_right: float = 1.0
) -> Tuple[np.ndarray, np.ndarray]:
    if epsilon <= 0:
        raise ValueError("epsilon 必须为正")






    uniform = np.linspace(0.0, 1.0, n_nodes)

    xi = 0.5 * (1.0 + np.tanh(3.0 * (uniform - 0.5)) / np.tanh(1.5))
    x = x_left + (x_right - x_left) * xi


    n_total = n_nodes
    a = np.zeros(n_total - 1, dtype=float)
    b = np.zeros(n_total, dtype=float)
    c = np.zeros(n_total - 1, dtype=float)
    d = np.zeros(n_total, dtype=float)

    for i in range(1, n_total - 1):
        h_m = x[i] - x[i - 1]
        h_p = x[i + 1] - x[i]
        h_avg = 0.5 * (h_m + h_p)


        pe = abs(x[i]) * h_avg / (2.0 * epsilon)

        if pe > 2.0:

            if x[i] > 0:

                a[i - 1] = epsilon / (h_m * h_avg)
                b[i] = -epsilon / (h_m * h_avg) - x[i] / h_avg + 1.0
                c[i] = 0.0
            else:

                a[i - 1] = 0.0
                b[i] = -epsilon / (h_p * h_avg) + x[i] / h_avg + 1.0
                c[i] = epsilon / (h_p * h_avg)
        else:

            a[i - 1] = epsilon / (h_m * h_avg) + x[i] / (h_m + h_p)
            b[i] = -epsilon * (1.0 / (h_m * h_avg) + 1.0 / (h_p * h_avg)) + 1.0
            c[i] = epsilon / (h_p * h_avg) - x[i] / (h_m + h_p)


    b[0] = 1.0
    d[0] = bc_left
    c[0] = 0.0

    b[-1] = 1.0
    d[-1] = bc_right
    a[-1] = 0.0


    from utils import tridiagonal_solve
    y = tridiagonal_solve(a, b, c, d)
    return x, y


def structural_default_probability_density(
    asset_values: np.ndarray,
    mu: float = 0.05,
    sigma: float = 0.2,
    v_min: float = 10.0,
    v_max: float = 200.0,
    default_barrier: float = 30.0
) -> np.ndarray:

    x_eval = 2.0 * (asset_values - v_min) / (v_max - v_min) - 1.0

    epsilon = max(sigma**2 / (2.0 * max(abs(mu), 1e-8)), 1e-6)




    bc_left = 2.0
    bc_right = 1.0

    x_grid, y_grid = solve_ill_bvp_fd(
        epsilon=epsilon,
        n_nodes=200,
        x_left=-1.0,
        x_right=1.0,
        bc_left=bc_left,
        bc_right=bc_right
    )


    from interpolation_surfaces import piecewise_linear_interpolate
    pdf = piecewise_linear_interpolate(x_grid, y_grid, x_eval)


    pdf = np.where(asset_values < default_barrier, 0.0, pdf)

    pdf = np.maximum(pdf, 0.0)


    integral = np.trapezoid(pdf, asset_values)
    if integral > 1e-15:
        pdf = pdf / integral

    return pdf


def default_probability_from_structural(
    v0: float,
    mu: float,
    sigma: float,
    default_barrier: float,
    maturity: float
) -> float:
    from utils import normal_cdf
    if v0 <= 0 or default_barrier <= 0 or maturity <= 0:
        return 1.0
    d = (np.log(v0 / default_barrier) + (mu - 0.5 * sigma**2) * maturity) / (sigma * np.sqrt(maturity))
    return normal_cdf(-d)


def test_structural_bvp():
    x, y = solve_ill_bvp_fd(epsilon=0.01, n_nodes=100)
    assert len(x) == 100, "节点数错误"
    assert np.isclose(y[0], 2.0, atol=1e-6), "左边界条件不满足"
    assert np.isclose(y[-1], 1.0, atol=1e-6), "右边界条件不满足"


    v_grid = np.linspace(20, 150, 100)
    pdf = structural_default_probability_density(v_grid, mu=0.05, sigma=0.2, default_barrier=30.0)
    assert np.all(pdf >= 0), "PDF 存在负值"
    integral = np.trapezoid(pdf, v_grid)
    assert abs(integral - 1.0) < 0.1, f"PDF 积分不归一: {integral}"


    pd_analytic = default_probability_from_structural(100.0, 0.05, 0.2, 30.0, 5.0)
    assert 0.0 <= pd_analytic <= 1.0, "解析 PD 越界"

    print(f"structural_default_bvp test passed. epsilon=0.01, PD_analytic={pd_analytic:.6f}")


if __name__ == "__main__":
    test_structural_bvp()
