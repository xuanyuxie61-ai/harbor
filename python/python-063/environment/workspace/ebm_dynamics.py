
import numpy as np


SIGMA = 5.670374419e-8
Q_SOLAR = 1361.0 / 4.0
C_OCEAN = 2.5e8
C_LAND = 5.0e6


def ice_albedo_feedback(T, T_ice=263.15, alpha_ocean=0.08, alpha_ice=0.65, k=0.5):
    T = np.asarray(T, dtype=np.float64)
    return alpha_ocean + (alpha_ice - alpha_ocean) / (1.0 + np.exp(k * (T - T_ice)))


def outgoing_longwave_radiation(T, epsilon=0.62):
    T_safe = np.maximum(np.asarray(T, dtype=np.float64), 100.0)
    return epsilon * SIGMA * T_safe**4


def solar_insolation(lat, t=None, orbital_variation=False):
    sin_lat = np.sin(np.asarray(lat, dtype=np.float64))
    P2 = 0.5 * (3.0 * sin_lat**2 - 1.0)
    S = 1.0 - 0.482 * P2

    if orbital_variation and t is not None:
        ecc = 0.0167 + 0.011 * np.cos(2.0 * np.pi * t / 100000.0)
        precession = np.cos(2.0 * np.pi * t / 23000.0 + np.pi / 6.0)
        obliquity = 0.4091 + 0.025 * np.cos(2.0 * np.pi * t / 41000.0)
        modulation = 1.0 + ecc * precession * sin_lat * np.cos(obliquity)
        S = S * modulation

    return np.maximum(S, 0.1)


def spherical_laplacian(T, vertices, faces, dual_areas):
    n_nodes = len(vertices)
    T = np.asarray(T, dtype=np.float64)
    neighbor_sum = np.zeros(n_nodes, dtype=np.float64)
    neighbor_cnt = np.zeros(n_nodes, dtype=int)

    for tri in faces:
        i, j, k = tri
        neighbor_sum[i] += T[j] + T[k]
        neighbor_sum[j] += T[i] + T[k]
        neighbor_sum[k] += T[i] + T[j]
        neighbor_cnt[i] += 2
        neighbor_cnt[j] += 2
        neighbor_cnt[k] += 2

    neighbor_cnt = np.maximum(neighbor_cnt, 1)
    neighbor_avg = neighbor_sum / neighbor_cnt
    lap = (neighbor_avg - T) / (0.1 + dual_areas)
    return lap


def compute_heat_capacity(lat):
    return C_OCEAN * 0.5 + (C_OCEAN - C_LAND) * 0.5 * np.cos(np.asarray(lat))**2


def ebm_rhs(T, vertices, faces, areas, dual_areas, t,
            D_diff=0.55, epsilon=0.6, volcanic_forcing=0.0, solar_forcing=0.0):












    pass


def implicit_trapezoidal_step(T_n, dt, rhs_func, max_iter=15, tol=1e-8):
    T_n = np.asarray(T_n, dtype=np.float64)
    f_n = rhs_func(T_n)
    z = T_n + dt * f_n

    for _ in range(max_iter):
        f_z = rhs_func(z)
        z_new = T_n + 0.5 * dt * (f_n + f_z)
        diff = np.max(np.abs(z_new - z))
        z = z_new
        if diff < tol:
            break
    return z
