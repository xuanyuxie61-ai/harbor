
import numpy as np
from typing import Tuple, List


def spherical_to_cartesian(theta: np.ndarray, phi: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    st = np.sin(theta)
    x = st * np.cos(phi)
    y = st * np.sin(phi)
    z = np.cos(theta)
    return x, y, z


def cartesian_to_spherical(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.zeros_like(r)
    phi = np.zeros_like(r)
    mask = r > 1e-15
    theta[mask] = np.arccos(np.clip(z[mask] / r[mask], -1.0, 1.0))
    phi[mask] = np.arctan2(y[mask], x[mask])
    phi = np.where(phi < 0, phi + 2 * np.pi, phi)
    return theta, phi


def gauss_legendre_angles(n_polar: int, n_azimuth: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if n_polar <= 0 or n_azimuth <= 0:
        raise ValueError("节点数必须为正")

    mu, w_mu = np.polynomial.legendre.leggauss(n_polar)
    phi = np.linspace(0, 2 * np.pi, n_azimuth, endpoint=False)
    w_phi = np.full(n_azimuth, 2 * np.pi / n_azimuth)
    return mu, w_mu, phi, w_phi


def integrate_sphere_function(f_values: np.ndarray, w_mu: np.ndarray, w_phi: np.ndarray) -> float:
    if f_values.shape[0] != w_mu.shape[0] or f_values.shape[1] != w_phi.shape[0]:
        raise ValueError("函数值维度与权重不匹配")
    return float(np.dot(w_mu, np.dot(f_values, w_phi)))


def henyey_greenstein_phase_function(cos_scatter: np.ndarray, g: float) -> np.ndarray:
    cos_scatter = np.asarray(cos_scatter, dtype=np.float64)
    if abs(g) >= 1.0:
        raise ValueError(f"Henyey-Greenstein参数 g 必须在 (-1, 1) 内，得到 g={g}")

    denom = 1.0 + g**2 - 2.0 * g * cos_scatter
    denom = np.maximum(denom, 1e-15)
    p = (1.0 - g**2) / (4.0 * np.pi * denom**1.5)
    return p


def compute_scatter_angles(mu_in: np.ndarray, phi_in: np.ndarray,
                           mu_out: np.ndarray, phi_out: np.ndarray) -> np.ndarray:
    dphi = phi_in - phi_out
    term1 = np.sqrt(np.maximum(1.0 - mu_in**2, 0.0)) * np.sqrt(np.maximum(1.0 - mu_out**2, 0.0)) * np.cos(dphi)
    term2 = mu_in * mu_out
    return np.clip(term1 + term2, -1.0, 1.0)


def delta_eddington_approximation(tau: np.ndarray, omega: np.ndarray, g: float,
                                  mu0: float) -> Tuple[np.ndarray, np.ndarray]:
    tau = np.asarray(tau, dtype=np.float64)
    omega = np.asarray(omega, dtype=np.float64)

    if np.any(tau < 0):
        raise ValueError("光学厚度不能为负")
    if np.any((omega < 0) | (omega > 1)):
        raise ValueError("单次散射反照率必须在 [0, 1] 内")
    if abs(g) >= 1.0:
        raise ValueError("不对称参数 g 必须在 (-1, 1) 内")
    if mu0 <= 0 or mu0 > 1:
        raise ValueError("入射角余弦 mu0 必须在 (0, 1] 内")

    f = g**3
    tau_star = tau * (1.0 - omega * f)
    omega_star = np.where(1.0 - omega * f > 1e-15,
                          (1.0 - f) * omega / (1.0 - omega * f),
                          0.0)
    g_star = np.where(1.0 - f > 1e-15, (g - f) / (1.0 - f), 0.0)

    k = np.sqrt(3.0 * (1.0 - omega_star) * (1.0 - omega_star * g_star))
    k = np.maximum(k, 1e-15)

    gamma3 = np.sqrt(1.0 - omega_star) / 3.0

    r_inf = (1.0 + 2.0 * k / 3.0) / np.maximum(1.0 - 2.0 * k / 3.0, 1e-15)
    r_0 = (1.0 + 2.0 * gamma3) / np.maximum(1.0 - 2.0 * gamma3, 1e-15)

    exp_term = np.exp(-2.0 * k * tau_star)
    denom = (r_inf + r_0)**2 - (r_inf - r_0)**2 * exp_term
    denom = np.maximum(denom, 1e-15)

    R = (r_inf - r_0) * (r_inf + r_0) * (1.0 - exp_term) / denom

    direct_trans = np.exp(-tau_star / mu0)
    T = direct_trans + (1.0 - R) * (1.0 - direct_trans)

    R = np.clip(R, 0.0, 1.0)
    T = np.clip(T, 0.0, 1.0)
    return R, T
