
import numpy as np
from typing import Union, List
from autodiff_core import DualScalar, HyperDualScalar
from autodiff_core import dual_exp, dual_sqrt, dual_min, dual_abs
from autodiff_core import hdual_exp, hdual_sqrt


def lennard_jones_potential(r: float, epsilon: float = 1.0,
                            sigma: float = 1.0) -> float:
    if r <= 0:
        raise ValueError("LJ potential requires r > 0")
    sr = sigma / r
    sr6 = sr ** 6
    sr12 = sr6 * sr6
    return 4.0 * epsilon * (sr12 - sr6)


def lennard_jones_force(r: float, epsilon: float = 1.0,
                        sigma: float = 1.0) -> float:


    raise NotImplementedError("Hole_1: 请补全 lennard_jones_force 的解析导数实现")


def lennard_jones_dual(r: DualScalar, epsilon: float = 1.0,
                       sigma: float = 1.0) -> DualScalar:
    r_eff = dual_min(r, DualScalar(0.8 * sigma, 0.0))


    if r.val < 0.8 * sigma:
        r_eff = DualScalar(0.8 * sigma, 0.0)
    else:
        r_eff = r

    sr = DualScalar(sigma, 0.0) / r_eff
    sr6 = sr ** 6
    sr12 = sr6 * sr6
    return DualScalar(4.0 * epsilon, 0.0) * (sr12 - sr6)


def lennard_jones_hyperdual(r: HyperDualScalar, epsilon: float = 1.0,
                            sigma: float = 1.0) -> HyperDualScalar:
    if r.f0 < 0.8 * sigma:
        r_eff = HyperDualScalar(0.8 * sigma, 0.0, 0.0, 0.0)
    else:
        r_eff = r
    sr = HyperDualScalar(sigma, 0.0, 0.0, 0.0) / r_eff
    sr6 = sr ** 6
    sr12 = sr6 * sr6
    return HyperDualScalar(4.0 * epsilon, 0.0, 0.0, 0.0) * (sr12 - sr6)


def gaussian_potential_2d(x: float, y: float,
                          xmu: float = 0.0, ymu: float = 0.0,
                          xsigma: float = 1.0, ysigma: float = 1.0,
                          A: float = 1.0,
                          corr_matrix: np.ndarray = None) -> float:
    vx = (x - xmu) / xsigma
    vy = (y - ymu) / ysigma
    v = np.array([vx, vy])
    if corr_matrix is None:
        C = np.eye(2)
    else:
        C = np.asarray(corr_matrix)

        eigvals, eigvecs = np.linalg.eigh(C)
        eigvals = np.maximum(eigvals, 1e-10)
        C = eigvecs @ np.diag(eigvals) @ eigvecs.T
    quad = float(v @ C @ v)
    return A * np.exp(-0.5 * quad)


def gaussian_potential_dual(x: DualScalar, y: DualScalar,
                            xmu: float = 0.0, ymu: float = 0.0,
                            xsigma: float = 1.0, ysigma: float = 1.0,
                            A: float = 1.0,
                            corr_matrix: np.ndarray = None) -> DualScalar:
    vx = (x - DualScalar(xmu, 0.0)) / DualScalar(xsigma, 0.0)
    vy = (y - DualScalar(ymu, 0.0)) / DualScalar(ysigma, 0.0)
    if corr_matrix is None:
        quad = vx * vx + vy * vy
    else:
        C = np.asarray(corr_matrix)

        quad = (DualScalar(C[0, 0], 0.0) * vx * vx
                + DualScalar(2.0 * C[0, 1], 0.0) * vx * vy
                + DualScalar(C[1, 1], 0.0) * vy * vy)
    return DualScalar(A, 0.0) * dual_exp(DualScalar(-0.5, 0.0) * quad)


def total_potential_lj(positions: np.ndarray,
                       epsilon: float = 1.0,
                       sigma: float = 1.0,
                       rcut: float = 2.5,
                       box_size: float = None) -> float:
    n = positions.shape[0]
    v_total = 0.0
    delta = 0.3 * sigma
    rcut_inner = rcut - delta
    rcut_sq = rcut * rcut

    for i in range(n):
        for j in range(i + 1, n):
            rij = positions[i] - positions[j]
            if box_size is not None:
                rij -= box_size * np.round(rij / box_size)
            r_sq = float(np.dot(rij, rij))
            if r_sq >= rcut_sq:
                continue
            r = np.sqrt(r_sq)
            if r < 1e-12:
                continue

            if r < rcut_inner:
                s = 1.0
            else:

                dr = rcut - r
                s = dr * dr * (2.0 * delta + r - rcut) / (delta ** 3)
            v = lennard_jones_potential(r, epsilon, sigma)
            v_total += v * s
    return v_total


def total_forces_lj(positions: np.ndarray,
                    epsilon: float = 1.0,
                    sigma: float = 1.0,
                    rcut: float = 2.5,
                    box_size: float = None) -> np.ndarray:
    n, d = positions.shape
    forces = np.zeros_like(positions)
    delta = 0.3 * sigma
    rcut_inner = rcut - delta
    rcut_sq = rcut * rcut

    for i in range(n):
        for j in range(i + 1, n):
            rij = positions[i] - positions[j]
            if box_size is not None:
                rij -= box_size * np.round(rij / box_size)
            r_sq = float(np.dot(rij, rij))
            if r_sq >= rcut_sq or r_sq < 1e-24:
                continue
            r = np.sqrt(r_sq)
            if r < rcut_inner:
                s = 1.0
                ds = 0.0
            else:
                dr = rcut - r
                s = dr * dr * (2.0 * delta + r - rcut) / (delta ** 3)
                ds = -2.0 * dr * (2.0 * delta + r - rcut) / (delta ** 3) \
                     + dr * dr / (delta ** 3)
            f_mag = lennard_jones_force(r, epsilon, sigma)


            factor = (f_mag * s + lennard_jones_potential(r, epsilon, sigma) * ds) / r
            f_vec = factor * rij
            forces[i] += f_vec
            forces[j] -= f_vec
    return forces


def total_potential_with_gaussian(positions: np.ndarray,
                                   epsilon: float = 1.0,
                                   sigma: float = 1.0,
                                   rcut: float = 2.5,
                                   gaussian_centers: np.ndarray = None,
                                   gaussian_params: List[dict] = None) -> float:
    v = total_potential_lj(positions, epsilon, sigma, rcut)
    if gaussian_centers is not None and gaussian_params is not None:
        for i in range(positions.shape[0]):
            for k, center in enumerate(gaussian_centers):
                params = gaussian_params[k % len(gaussian_params)]
                dx = positions[i, 0] - center[0]
                dy = positions[i, 1] - center[1] if positions.shape[1] > 1 else 0.0
                v += gaussian_potential_2d(
                    dx, dy,
                    xmu=0.0, ymu=0.0,
                    xsigma=params.get('xsigma', 1.0),
                    ysigma=params.get('ysigma', 1.0),
                    A=params.get('A', 0.1),
                    corr_matrix=params.get('corr', None)
                )
    return v


def virial_stress_lj(positions: np.ndarray,
                     epsilon: float = 1.0,
                     sigma: float = 1.0,
                     rcut: float = 2.5,
                     volume: float = 1.0,
                     box_size: float = None) -> np.ndarray:
    n, d = positions.shape
    stress = np.zeros((d, d))
    delta = 0.3 * sigma
    rcut_inner = rcut - delta
    rcut_sq = rcut * rcut

    for i in range(n):
        for j in range(i + 1, n):
            rij = positions[i] - positions[j]
            r_sq = float(np.dot(rij, rij))
            if r_sq >= rcut_sq or r_sq < 1e-24:
                continue
            r = np.sqrt(r_sq)
            if r < rcut_inner:
                s = 1.0
            else:
                dr = rcut - r
                s = dr * dr * (2.0 * delta + r - rcut) / (delta ** 3)
            f_mag = lennard_jones_force(r, epsilon, sigma)
            factor = f_mag * s / r
            for alpha in range(d):
                for beta in range(d):
                    stress[alpha, beta] += rij[alpha] * factor * rij[beta]
    return stress / volume
