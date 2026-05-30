
import numpy as np
from scipy.linalg import cholesky


def simulate_returns_mc(mu: np.ndarray, sigma: np.ndarray, corr: np.ndarray,
                        T: int = 252, n_paths: int = 5000,
                        rng: np.random.Generator = None) -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng()
    n = len(mu)
    if sigma.shape != (n,) or corr.shape != (n, n):
        raise ValueError("simulate_returns_mc: 参数维度不匹配。")

    cov = np.outer(sigma, sigma) * corr

    try:
        L = cholesky(cov, lower=True)
    except np.linalg.LinAlgError:

        eigvals, eigvecs = np.linalg.eigh(cov)
        eigvals = np.maximum(eigvals, 1e-8)
        cov = eigvecs @ np.diag(eigvals) @ eigvecs.T
        L = cholesky(cov, lower=True)

    dt = 1.0 / 252.0
    Z = rng.standard_normal((n_paths, T, n))
    returns = np.zeros((n_paths, T, n))
    for t in range(T):
        shocks = Z[:, t, :] @ L.T
        returns[:, t, :] = mu * dt + shocks * np.sqrt(dt)
    return returns


def bootstrap_risk_analysis(returns: np.ndarray, n_bootstrap: int = 2000,
                            alpha: float = 0.05,
                            rng: np.random.Generator = None) -> dict:
    if rng is None:
        rng = np.random.default_rng()
    T, n = returns.shape
    if T < 30:
        raise ValueError("bootstrap_risk_analysis: 样本量至少为30。")

    mean_boot = np.zeros((n_bootstrap, n))
    var_boot = np.zeros(n_bootstrap)
    cvar_boot = np.zeros(n_bootstrap)

    for b in range(n_bootstrap):
        idx = rng.integers(0, T, size=T)
        sample = returns[idx, :]
        mean_boot[b, :] = np.mean(sample, axis=0)

        port_ret = np.mean(sample, axis=1)
        var_boot[b] = np.percentile(port_ret, alpha * 100)
        cvar_boot[b] = np.mean(port_ret[port_ret <= var_boot[b]])

    def ci(arr):
        return (float(np.percentile(arr, alpha / 2 * 100)),
                float(np.percentile(arr, (1 - alpha / 2) * 100)))

    return {
        "mean_estimate": np.mean(mean_boot, axis=0).tolist(),
        "mean_ci": [ci(mean_boot[:, i]) for i in range(n)],
        "VaR_mean": float(np.mean(var_boot)),
        "VaR_ci": ci(var_boot),
        "CVaR_mean": float(np.mean(cvar_boot)),
        "CVaR_ci": ci(cvar_boot),
        "n_bootstrap": n_bootstrap,
    }


def tournament_risk_simulation(strengths: np.ndarray, n_games: int = 10000,
                                rng: np.random.Generator = None) -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng()
    strengths = np.asarray(strengths, dtype=float)
    if np.any(strengths <= 0):
        raise ValueError("tournament_risk_simulation: 强度必须为正数。")
    n = len(strengths)
    stats = np.zeros(n, dtype=int)
    probs = strengths / np.sum(strengths)
    for _ in range(n_games):
        winner = rng.choice(n, p=probs)
        stats[winner] += 1
    return stats / n_games


def high_dim_sphere_sampling(n_samples: int, dim: int,
                              rng: np.random.Generator = None) -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng()
    Z = rng.standard_normal((n_samples, dim))
    norms = np.linalg.norm(Z, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    return Z / norms
