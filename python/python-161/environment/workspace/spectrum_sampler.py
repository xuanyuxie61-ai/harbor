
import numpy as np
from typing import Tuple


PLANCK_CONSTANT: float = 6.62607015e-34
SPEED_OF_LIGHT: float = 2.99792458e8
ELEMENTARY_CHARGE: float = 1.602176634e-19


def build_am15_spectrum(
    lambda_min: float = 280.0,
    lambda_max: float = 1200.0,
    n_bins: int = 64,
    theta_max_deg: float = 15.0,
    n_theta: int = 16,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if lambda_min >= lambda_max or lambda_min <= 0:
        raise ValueError("波长范围必须满足 0 < lambda_min < lambda_max")
    if n_bins <= 0 or n_theta <= 0:
        raise ValueError("离散格点数必须为正整数")

    lambdas = np.linspace(lambda_min, lambda_max, n_bins)
    thetas = np.linspace(0.0, theta_max_deg, n_theta)



    def am15_irradiance(lam: np.ndarray) -> np.ndarray:
        lam_m = lam * 1e-9

        T_sun = 5778.0
        h, c, k_B = PLANCK_CONSTANT, SPEED_OF_LIGHT, 1.380649e-23

        term1 = (2.0 * h * c ** 2) / (lam_m ** 5)
        term2 = 1.0 / (np.exp((h * c) / (lam_m * k_B * T_sun)) - 1.0)
        bb = term1 * term2

        bb_peak = bb.max() if bb.max() > 0 else 1.0
        am15 = 1000.0 * bb / bb_peak

        am15 *= (1.0 - 0.15 * np.exp(-((lam - 940.0) / 60.0) ** 2))
        am15 *= (1.0 - 0.08 * np.exp(-((lam - 760.0) / 40.0) ** 2))
        return np.maximum(am15, 0.0)

    irr = am15_irradiance(lambdas)


    theta_rad = np.deg2rad(thetas)
    cos_factor = np.cos(theta_rad) * (1.0 + 0.05 * np.cos(2.0 * theta_rad))


    pdf = np.outer(irr, cos_factor)


    total = pdf.sum()
    if total <= 0.0 or not np.isfinite(total):
        pdf = np.ones_like(pdf) / (n_bins * n_theta)
    else:
        pdf /= total


    cdf_flat = np.cumsum(pdf.ravel(order='C'))
    cdf = cdf_flat.reshape(pdf.shape, order='C')

    if cdf.size > 0:
        cdf.flat[-1] = 1.0

    return lambdas, thetas, pdf, cdf


def discrete_cdf_to_xy(
    n1: int,
    n2: int,
    cdf: np.ndarray,
    n_samples: int,
    u: np.ndarray,
) -> np.ndarray:
    if cdf.shape != (n1, n2):
        raise ValueError(f"CDF 形状 {cdf.shape} 与 ({n1},{n2}) 不符")
    if u.size != n_samples:
        raise ValueError("随机数数组长度与 n_samples 不符")
    if n_samples <= 0:
        return np.zeros((2, 0))


    u = np.clip(u, 0.0, 1.0)

    xy = np.zeros((2, n_samples))
    cdf_flat = cdf.ravel(order='C')


    for k in range(n_samples):
        idx = np.searchsorted(cdf_flat, u[k], side='left')
        idx = min(idx, n1 * n2 - 1)
        i = idx // n2
        j = idx % n2


        r = np.random.rand(2)
        xy[0, k] = (i + r[0]) / n1
        xy[1, k] = (j + r[1]) / n2

    return xy


def sample_photons(
    n_photons: int = 10000,
    lambda_min: float = 280.0,
    lambda_max: float = 1200.0,
    theta_max_deg: float = 15.0,
) -> Tuple[np.ndarray, np.ndarray]:
    if n_photons <= 0:
        return np.array([]), np.array([])

    n_bins, n_theta = 64, 16
    lambdas, thetas, pdf, cdf = build_am15_spectrum(
        lambda_min, lambda_max, n_bins, theta_max_deg, n_theta
    )

    u = np.random.rand(n_photons)
    xy = discrete_cdf_to_xy(n_bins, n_theta, cdf, n_photons, u)


    lambdas_sample = lambda_min + xy[0, :] * (lambda_max - lambda_min)
    thetas_sample = 0.0 + xy[1, :] * theta_max_deg

    return lambdas_sample, thetas_sample


def photon_energy_ev(lambda_nm: np.ndarray) -> np.ndarray:
    lambda_nm = np.asarray(lambda_nm)
    with np.errstate(divide='ignore', invalid='ignore'):
        energy = (PLANCK_CONSTANT * SPEED_OF_LIGHT) / (ELEMENTARY_CHARGE * lambda_nm * 1e-9)
    energy = np.where(lambda_nm > 0, energy, 0.0)
    return energy


if __name__ == "__main__":

    lams, thetas = sample_photons(1000)
    print(f"采样光子数: {len(lams)}, λ 范围 [{lams.min():.1f}, {lams.max():.1f}] nm")
    print(f"θ 范围 [{thetas.min():.2f}, {thetas.max():.2f}] deg")
    E = photon_energy_ev(lams)
    print(f"光子能量范围 [{E.min():.3f}, {E.max():.3f}] eV")
