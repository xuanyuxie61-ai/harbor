
import numpy as np
from scipy.optimize import minimize
from typing import Tuple, Optional






def pade_approximant(z_points: np.ndarray, g_points: np.ndarray, z_eval: np.ndarray) -> np.ndarray:
    n = len(z_points)
    if n != len(g_points):
        raise ValueError("z_points 与 g_points 长度不一致")
    if n < 2:
        raise ValueError("至少需要 2 个点")

    N = n // 2
    if 2 * N + 1 > n:
        N = (n - 1) // 2



    m = 2 * N + 1
    A = np.zeros((m, m), dtype=np.complex128)
    b = np.zeros(m, dtype=np.complex128)
    for i in range(m):
        zi = z_points[i]
        gi = g_points[i]

        for k in range(N + 1):
            A[i, k] = zi ** k

        for k in range(1, N + 1):
            A[i, N + k] = -gi * zi ** k
        b[i] = gi

    try:
        x = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    p = x[:N + 1]
    q = np.concatenate([[1.0], x[N + 1:]])

    g_eval = np.zeros(len(z_eval), dtype=np.complex128)
    for k, z in enumerate(z_eval):
        num = np.polyval(p[::-1], z)
        den = np.polyval(q[::-1], z)
        if abs(den) < 1e-14:
            den = 1e-14
        g_eval[k] = num / den
    g_eval = np.where(np.isfinite(g_eval), g_eval, 0.0)
    return g_eval


def pade_spectral_function(omega_n: np.ndarray, g_iw: np.ndarray,
                           omega_real: np.ndarray, eta: float = 0.05) -> np.ndarray:

    z_full = np.concatenate([-omega_n[::-1], omega_n])
    g_full = np.concatenate([g_iw.conj()[::-1], g_iw])
    z_eval = omega_real + 1j * eta
    g_real = pade_approximant(z_full, g_full, z_eval)
    A = -g_real.imag / np.pi
    A = np.where(A > 0, A, 0.0)
    return A






def maxent_spectral_function(omega_n: np.ndarray, g_iw: np.ndarray,
                             omega_real: np.ndarray, default_model: Optional[np.ndarray] = None,
                             alpha: float = 1.0) -> np.ndarray:
    n_omega = len(omega_real)
    domega = omega_real[1] - omega_real[0] if n_omega > 1 else 1.0
    if default_model is None:
        default_model = np.ones(n_omega) / (n_omega * domega)
    default_model = np.abs(default_model)
    default_model = default_model / np.trapezoid(default_model, omega_real)
    

    K = np.zeros((len(omega_n), n_omega), dtype=np.complex128)
    for n, wn in enumerate(omega_n):
        for m, w in enumerate(omega_real):
            K[n, m] = 1.0 / (1j * wn - w)
    
    def objective(A):
        A = np.abs(A)

        g_model = K @ A * domega
        chi2 = 0.5 * np.sum(np.abs(g_iw - g_model) ** 2)

        ratio = A / (default_model + 1e-14)
        ratio = np.where(ratio > 1e-14, ratio, 1e-14)
        S = np.sum(A * np.log(ratio)) * domega
        return chi2 - alpha * S
    

    from scipy.optimize import minimize
    A0 = default_model.copy()
    bounds = [(0.0, None) for _ in range(n_omega)]

    def eq_con(A):
        return np.trapezoid(A, omega_real) - 1.0
    
    cons = {"type": "eq", "fun": eq_con}
    result = minimize(objective, A0, method="SLSQP", bounds=bounds, constraints=cons,
                      options={"maxiter": 500, "ftol": 1e-8})
    A_opt = np.abs(result.x)

    norm = np.trapezoid(A_opt, omega_real)
    if norm > 0:
        A_opt /= norm
    return A_opt






def spectral_moments(A: np.ndarray, omega: np.ndarray, max_moment: int = 4) -> dict:
    if max_moment < 0:
        raise ValueError("max_moment >= 0")
    moments = {}
    domega = omega[1] - omega[0] if len(omega) > 1 else 1.0
    for n in range(max_moment + 1):
        M = np.trapezoid(omega ** n * A, omega)
        moments[f"M_{n}"] = float(M)
    return moments


def self_energy_from_greens_function(omega: np.ndarray, g: np.ndarray, epsilon_k: float) -> np.ndarray:
    g = np.where(np.abs(g) > 1e-14, g, 1e-14)
    return omega - epsilon_k - 1.0 / g


def kramers_kronig_relation(imag_part: np.ndarray, omega: np.ndarray) -> np.ndarray:
    n = len(omega)
    real_part = np.zeros(n)
    domega = omega[1] - omega[0]
    for i in range(n):
        integrand = imag_part / (omega - omega[i])

        mask = np.abs(omega - omega[i]) > 1e-10
        if np.any(mask):
            real_part[i] = np.trapezoid(integrand[mask], omega[mask]) / np.pi
    return real_part


if __name__ == "__main__":

    omega_n = np.array([1.0, 3.0, 5.0, 7.0, 9.0]) * np.pi
    g_iw = 1.0 / (1j * omega_n + 0.5)
    omega_real = np.linspace(-5, 5, 100)
    A = pade_spectral_function(omega_n, g_iw, omega_real, eta=0.1)
    print(f"Spectral sum rule: {np.trapezoid(A, omega_real):.6f} (expect ~1.0)")
