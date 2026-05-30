
import numpy as np


def coupled_market_dynamics(y: np.ndarray, t: float,
                             k1: float, K2: np.ndarray,
                             gamma: float, m: np.ndarray) -> np.ndarray:
    n = len(m)
    if len(y) != 2 * n:
        raise ValueError("coupled_market_dynamics: 状态向量维度必须是 2n。")
    u = y[0::2]
    v = y[1::2]
    dydt = np.zeros_like(y)











    raise NotImplementedError("Hole 3: 耦合市场动力学核心方程待实现")


def trapezoidal_sde_solver(f, g, tspan: tuple, y0: np.ndarray,
                            n_steps: int, rng: np.random.Generator = None) -> tuple:
    if rng is None:
        rng = np.random.default_rng()
    t0, tf = tspan
    dt = (tf - t0) / n_steps
    m = len(y0)
    t = np.linspace(t0, tf, n_steps + 1)
    y = np.zeros((n_steps + 1, m))
    y[0, :] = y0

    for i in range(n_steps):
        ti = t[i]
        yi = y[i, :]
        fi = f(ti, yi)
        gi = g(ti, yi)
        dW = np.sqrt(dt) * rng.standard_normal(m)
        explicit_part = yi + dt * fi + gi * dW


        yn = explicit_part.copy()
        for _ in range(20):
            fn = f(ti + dt, yn)
            yn_new = yi + 0.5 * dt * (fi + fn) + gi * dW
            if np.linalg.norm(yn_new - yn) < 1e-12:
                break
            yn = yn_new
        y[i + 1, :] = yn
    return t, y


def simulate_contagion(n_assets: int, T: float = 1.0, dt: float = 0.01,
                        k1: float = 2.0, gamma: float = 0.5,
                        sigma_noise: float = 0.3,
                        rng: np.random.Generator = None) -> tuple:
    if rng is None:
        rng = np.random.default_rng()
    n_steps = int(np.round(T / dt))
    m = np.ones(n_assets)

    K2 = np.zeros((n_assets, n_assets))
    for i in range(1, n_assets):
        K2[0, i] = 1.0
        K2[i, 0] = 1.0

    K2 += 0.2 * rng.random((n_assets, n_assets))
    K2 = 0.5 * (K2 + K2.T)
    np.fill_diagonal(K2, 0.0)

    y0 = np.zeros(2 * n_assets)


    def f(t, y):
        return coupled_market_dynamics(y, t, k1, K2, gamma, m)

    def g(t, y):
        noise = np.zeros(2 * n_assets)
        noise[1::2] = sigma_noise
        return noise

    t, y = trapezoidal_sde_solver(f, g, (0.0, T), y0, n_steps, rng)


    shock_step = n_steps // 2
    y[shock_step:, 0] += 0.5 * np.exp(-k1 * (t[shock_step:] - t[shock_step]))

    u = y[:, 0::2]
    max_deviation = np.max(np.abs(u), axis=0)
    return t, y, max_deviation


def trapezoidal_ode_solver(f, tspan: tuple, y0: np.ndarray,
                            n_steps: int) -> tuple:
    t0, tf = tspan
    dt = (tf - t0) / n_steps
    m = len(y0)
    t = np.linspace(t0, tf, n_steps + 1)
    y = np.zeros((n_steps + 1, m))
    y[0, :] = y0
    for i in range(n_steps):
        ti = t[i]
        yi = y[i, :]
        fi = f(ti, yi)
        yn = yi + dt * fi
        for _ in range(20):
            fn = f(ti + dt, yn)
            yn_new = yi + 0.5 * dt * (fi + fn)
            if np.linalg.norm(yn_new - yn) < 1e-12:
                break
            yn = yn_new
        y[i + 1, :] = yn
    return t, y
