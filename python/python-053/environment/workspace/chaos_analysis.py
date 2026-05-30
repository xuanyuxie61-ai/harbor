
import numpy as np
from typing import Tuple, List, Optional


def lyapunov_exponent_1d(f: callable, df: callable, x0: float,
                         n_iter: int = 10000, n_transient: int = 1000) -> float:
    x = float(x0)

    for _ in range(n_transient):
        x = f(x)
        if not np.isfinite(x):
            x = x0

    lyap_sum = 0.0
    count = 0
    for _ in range(n_iter):
        x = f(x)
        dfx = df(x)
        if abs(dfx) < 1e-15:
            dfx = 1e-15
        if np.isfinite(dfx) and dfx != 0.0:
            lyap_sum += np.log(abs(dfx))
            count += 1
        if not np.isfinite(x):
            x = x0

    if count == 0:
        return 0.0
    return lyap_sum / count


def correlation_dimension(trajectory: np.ndarray,
                          r_min: float = 1e-3,
                          r_max: float = 1.0,
                          n_r: int = 50) -> float:
    trajectory = np.atleast_2d(trajectory).T
    if trajectory.ndim != 2:
        raise ValueError("trajectory must be 2D array")

    N = trajectory.shape[0]
    if N < 100:
        return 0.0


    max_samples = 2000
    if N > max_samples:
        idx = np.random.choice(N, max_samples, replace=False)
        traj = trajectory[idx]
        N = max_samples
    else:
        traj = trajectory

    r_vals = np.logspace(np.log10(r_min), np.log10(r_max), n_r)
    c_vals = np.zeros(n_r)

    for i_r, r in enumerate(r_vals):
        count = 0
        for i in range(N):
            dists = np.linalg.norm(traj[i + 1:] - traj[i], axis=1)
            count += np.sum(dists < r)
        c_vals[i_r] = 2.0 * count / (N * (N - 1))


    valid = (c_vals > 1e-4) & (c_vals < 0.5)
    if np.sum(valid) < 5:
        valid = c_vals > 0

    if np.sum(valid) < 3:
        return 0.0

    log_r = np.log(r_vals[valid])
    log_c = np.log(c_vals[valid])


    A = np.vstack([log_r, np.ones_like(log_r)]).T
    slope, _ = np.linalg.lstsq(A, log_c, rcond=None)[0]
    return float(slope)


def levy_dragon_ifs(n_iter: int = 10000) -> np.ndarray:
    A0 = np.array([[0.5, 0.5], [-0.5, 0.5]])
    A1 = np.array([[0.5, -0.5], [0.5, 0.5]])
    b0 = np.array([0.5, 0.5])
    b1 = np.array([-0.5, 0.5])

    x = np.random.rand(2)
    points = np.zeros((n_iter, 2))

    for i in range(n_iter):
        if np.random.rand() < 0.5:
            x = A0 @ x + b0
        else:
            x = A1 @ x + b1
        points[i] = x

    return points


def cross_chaos_ifs(n_iter: int = 10000) -> np.ndarray:
    A = np.array([[1.0 / 3.0, 0.0],
                  [0.0, 1.0 / 3.0]])
    b = np.array([
        [1.0 / 3.0, 0.0],
        [0.0, 1.0 / 3.0],
        [1.0 / 3.0, 1.0 / 3.0],
        [2.0 / 3.0, 1.0 / 3.0],
        [1.0 / 3.0, 2.0 / 3.0]
    ]).T

    x = np.random.rand(2)
    points = np.zeros((n_iter, 2))

    for i in range(n_iter):
        j = np.random.randint(0, 5)
        x = A @ x + b[:, j]
        points[i] = x

    return points


def enso_poincare_map(h_n: float,
                      r: float = 0.25,
                      alpha: float = 0.5,
                      R: float = 1.0,
                      epsilon: float = 0.3,
                      gamma: float = 0.4) -> float:
    dt = 0.01
    mu = 1.0 + dt * (gamma - R * alpha / r)
    K = (r / (R * alpha)) * (gamma - R * alpha / r) / epsilon if epsilon > 0 else 1e10

    if K <= 0:
        return h_n * np.exp(-dt * r)

    h_next = mu * h_n * (1.0 - h_n / K)
    return h_next


def enso_lyapunov_exponent(r: float = 0.25,
                           alpha: float = 0.5,
                           R: float = 1.0,
                           epsilon: float = 0.3,
                           gamma: float = 0.4,
                           n_iter: int = 5000) -> float:
    dt = 0.01
    mu = 1.0 + dt * (gamma - R * alpha / r)
    K = (r / (R * alpha)) * (gamma - R * alpha / r) / epsilon if epsilon > 0 else 1e10

    def f(h):
        if K <= 0:
            return h * np.exp(-dt * r)
        return mu * h * (1.0 - h / K)

    def df(h):
        if K <= 0:
            return np.exp(-dt * r)
        return mu * (1.0 - 2.0 * h / K)

    return lyapunov_exponent_1d(f, df, 0.1, n_iter=n_iter)


def bifurcation_diagram(param_name: str,
                        param_range: np.ndarray,
                        r: float = 0.25,
                        alpha: float = 0.5,
                        R: float = 1.0,
                        epsilon: float = 0.3,
                        gamma: float = 0.4) -> Tuple[np.ndarray, List[np.ndarray]]:
    attractors = []
    params = []

    for p in param_range:
        kw = {"r": r, "alpha": alpha, "R": R, "epsilon": epsilon, "gamma": gamma}
        kw[param_name] = p

        dt = 0.01
        mu = 1.0 + dt * (kw["gamma"] - kw["R"] * kw["alpha"] / kw["r"])
        K = (kw["r"] / (kw["R"] * kw["alpha"])) * (kw["gamma"] - kw["R"] * kw["alpha"] / kw["r"]) / kw["epsilon"] \
            if kw["epsilon"] > 0 else 1e10

        if K <= 0:
            attractors.append(np.array([0.0]))
            params.append(p)
            continue


        h = 0.1
        for _ in range(2000):
            h = mu * h * (1.0 - h / K)


        points = []
        for _ in range(100):
            h = mu * h * (1.0 - h / K)
            points.append(h)

        attractors.append(np.array(points))
        params.append(p)

    return np.array(params), attractors
