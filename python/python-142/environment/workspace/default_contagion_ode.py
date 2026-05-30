
import numpy as np
from typing import Tuple, Callable, Optional


def contagion_rhs(t: float, y: np.ndarray, params: dict) -> np.ndarray:
    u, v, w = y
    eta1 = params.get("eta1", 0.01)
    eta2 = params.get("eta2", 0.01)
    q = params.get("q", 0.01)
    f = params.get("f", 1.0)

    du = (q * v - u * v + u * (1.0 - u)) / eta1
    dv = (-q * v - u * v + f * w) / eta2
    dw = u - w

    return np.array([du, dv, dw], dtype=float)


def theta_method_solve(
    f: Callable[[float, np.ndarray], np.ndarray],
    tspan: Tuple[float, float],
    y0: np.ndarray,
    n_steps: int,
    theta: float = 0.5,
    max_newton_iter: int = 20,
    newton_tol: float = 1e-10
) -> Tuple[np.ndarray, np.ndarray]:
    t0, tf = tspan
    h = (tf - t0) / n_steps
    n_vars = len(y0)
    t = np.linspace(t0, tf, n_steps + 1)
    y = np.zeros((n_steps + 1, n_vars), dtype=float)
    y[0, :] = y0

    for n in range(n_steps):
        yn = y[n, :]
        tn = t[n]
        tnp1 = t[n + 1]
        fn = f(tn, yn)

        if theta >= 0.9999999:

            y[n + 1, :] = yn + h * fn
        else:

            y_pred = yn + h * fn

            y_curr = y_pred.copy()
            for _ in range(max_newton_iter):
                fnp1 = f(tnp1, y_curr)
                residual = y_curr - yn - h * (theta * fn + (1.0 - theta) * fnp1)


                y_next = y_curr - 0.5 * residual
                if np.linalg.norm(y_next - y_curr) < newton_tol:
                    y_curr = y_next
                    break
                y_curr = y_next
            y[n + 1, :] = y_curr


        y[n + 1, 0] = np.clip(y[n + 1, 0], 0.0, 1.0)
        y[n + 1, 1] = np.clip(y[n + 1, 1], 0.0, 1.0)
        y[n + 1, 2] = np.clip(y[n + 1, 2], 0.0, 1.0)

    return t, y


def simulate_default_contagion(
    initial_default_rate: float = 0.05,
    initial_pressure: float = 0.1,
    initial_buffer: float = 0.5,
    t_max: float = 5.0,
    n_steps: int = 500,
    theta: float = 0.5,
    params: Optional[dict] = None
) -> Tuple[np.ndarray, np.ndarray]:
    if params is None:
        params = {
            "eta1": 0.02,
            "eta2": 0.05,
            "q": 0.02,
            "f": 0.8
        }

    y0 = np.array([initial_default_rate, initial_pressure, initial_buffer], dtype=float)
    f = lambda t, y: contagion_rhs(t, y, params)
    t, y = theta_method_solve(f, (0.0, t_max), y0, n_steps, theta)
    return t, y


def network_cascade_intensity(
    adjacency: np.ndarray,
    local_intensities: np.ndarray,
    coupling_strength: float = 0.1
) -> np.ndarray:
    adj = np.asarray(adjacency, dtype=float)
    local = np.asarray(local_intensities, dtype=float)
    network_effect = adj @ local
    result = local + coupling_strength * network_effect

    return np.clip(result, 0.0, 1.0)


def test_default_contagion():
    t, y = simulate_default_contagion(
        initial_default_rate=0.05,
        initial_pressure=0.1,
        initial_buffer=0.5,
        t_max=2.0,
        n_steps=200,
        theta=0.5
    )
    assert len(t) == 201, "时间步数错误"
    assert np.all(y[:, 0] >= 0) and np.all(y[:, 0] <= 1), "违约强度越界"
    assert np.all(y[:, 1] >= 0) and np.all(y[:, 1] <= 1), "传染压力越界"
    assert np.all(y[:, 2] >= 0) and np.all(y[:, 2] <= 1), "缓冲水平越界"


    adj = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    local = np.array([0.1, 0.2, 0.15])
    net = network_cascade_intensity(adj, local, 0.1)
    assert np.all(net >= local), "网络效应应为增强"
    print(f"default_contagion_ode test passed. final u={y[-1,0]:.4f}, v={y[-1,1]:.4f}, w={y[-1,2]:.4f}")


if __name__ == "__main__":
    test_default_contagion()
