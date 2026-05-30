
import numpy as np
from typing import Callable, Tuple, Optional


def rk4_integrate(f: Callable, t_span: Tuple[float, float],
                  y0: np.ndarray, n_steps: int) -> Tuple[np.ndarray, np.ndarray]:
    if n_steps < 1:
        raise ValueError("n_steps 必须至少为 1。")
    t0, tf = t_span
    dt = (tf - t0) / n_steps
    m = len(y0)
    t = np.zeros(n_steps + 1)
    y = np.zeros((n_steps + 1, m))
    t[0] = t0
    y[0, :] = y0

    for i in range(n_steps):
        ti = t[i]
        yi = y[i, :]
        k1 = dt * np.array(f(ti, yi))
        k2 = dt * np.array(f(ti + 0.5 * dt, yi + 0.5 * k1))
        k3 = dt * np.array(f(ti + 0.5 * dt, yi + 0.5 * k2))
        k4 = dt * np.array(f(ti + dt, yi + k3))
        y[i + 1, :] = yi + (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
        t[i + 1] = ti + dt
    return t, y


def ball_unit_sample_nd(dim: int) -> np.ndarray:
    if dim < 1:
        raise ValueError("维度必须至少为 1。")
    z = np.random.randn(dim)
    z = z / np.linalg.norm(z)
    u = np.random.rand()
    r = u ** (1.0 / dim)
    return r * z


def causal_ode_system(t: float, y: np.ndarray,
                       A: np.ndarray, B: np.ndarray,
                       u_func: Callable, bilinear: bool = False,
                       C: Optional[np.ndarray] = None) -> np.ndarray:
    u = np.array(u_func(t))
    dydt = A @ y + B @ u
    if bilinear and C is not None:
        dydt = dydt + C @ (y * u)
    return dydt


def simulate_intervention_diffusion(A: np.ndarray,
                                     B: np.ndarray,
                                     y0: np.ndarray,
                                     t_span: Tuple[float, float],
                                     n_steps: int = 200,
                                     intervention_time: float = 0.5,
                                     intervention_idx: Optional[int] = None,
                                     intervention_magnitude: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    p = A.shape[0]
    if intervention_idx is None:
        intervention_idx = 0
    if not (0 <= intervention_idx < p):
        raise ValueError("intervention_idx 超出范围。")

    sigma = (t_span[1] - t_span[0]) / n_steps * 3.0

    def u_func(t):
        u = np.zeros(p)
        u[intervention_idx] = intervention_magnitude * np.exp(-0.5 * ((t - intervention_time) / sigma) ** 2)
        return u

    def f(t, y):
        return causal_ode_system(t, y, A, B, u_func)

    t, y = rk4_integrate(f, t_span, y0, n_steps)
    return t, y


def monte_carlo_causal_distance(A: np.ndarray,
                                 B: np.ndarray,
                                 y0: np.ndarray,
                                 t_span: Tuple[float, float],
                                 n_steps: int = 100,
                                 n_samples: int = 200) -> Tuple[float, float]:
    p = len(y0)


    def u_zero(t):
        return np.zeros(p)

    def f_base(t, y):
        return causal_ode_system(t, y, A, B, u_zero)

    _, y_base = rk4_integrate(f_base, t_span, y0, n_steps)
    y_final_base = y_base[-1, :]

    distances = np.zeros(n_samples)
    for k in range(n_samples):
        delta = ball_unit_sample_nd(p) * 0.1
        y0_pert = y0 + delta
        _, y_pert = rk4_integrate(f_base, t_span, y0_pert, n_steps)
        distances[k] = np.linalg.norm(y_pert[-1, :] - y_final_base)

    mu = float(np.mean(distances))
    var = float(np.var(distances, ddof=1)) if n_samples > 1 else 0.0
    return mu, var


def demo():
    np.random.seed(13)
    p = 5

    A = np.zeros((p, p))
    for i in range(p):
        for j in range(i):
            A[i, j] = 0.2 * np.random.randn()
    np.fill_diagonal(A, -0.5)
    B = np.eye(p) * 0.8
    y0 = np.zeros(p)

    t, y = simulate_intervention_diffusion(A, B, y0, (0.0, 2.0), n_steps=200,
                                            intervention_time=0.5,
                                            intervention_idx=1,
                                            intervention_magnitude=2.0)
    print(f"[causal_ode_dynamics] 干预扩散模拟完成: t_end={t[-1]:.3f}, max_state={np.max(np.abs(y)):.4f}")

    mu, var = monte_carlo_causal_distance(A, B, y0, (0.0, 1.0), n_steps=100, n_samples=100)
    print(f"[causal_ode_dynamics] MC 因果距离: mean={mu:.4f}, var={var:.6f}")
    return t, y


if __name__ == "__main__":
    demo()
