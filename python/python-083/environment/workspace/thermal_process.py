
import numpy as np
from typing import Tuple, Optional






def thermal_advection_lax_1d(T0: np.ndarray, z: np.ndarray, dt: float,
                              n_steps: int, v: float, alpha: float,
                              source: Optional[np.ndarray] = None,
                              bc_type: str = "dirichlet") -> np.ndarray:
    nz = len(z)
    dz = z[1] - z[0]
    if dz <= 0:
        raise ValueError("Spatial grid must be strictly increasing.")


    cfl_adv = abs(v) * dt / dz
    cfl_diff = alpha * dt / (dz * dz)
    if cfl_adv > 1.0 or cfl_diff > 0.5:

        dt_max1 = dz / (abs(v) + 1e-10)
        dt_max2 = 0.5 * dz * dz / (alpha + 1e-10)
        dt = min(dt, 0.9 * min(dt_max1, dt_max2))
        cfl_adv = abs(v) * dt / dz
        cfl_diff = alpha * dt / (dz * dz)

    T = T0.copy().astype(np.float64)
    T_new = np.zeros_like(T)

    for _ in range(n_steps):

        for i in range(1, nz - 1):
            adv = -cfl_adv * 0.5 * (T[i+1] - T[i-1])
            diff = cfl_diff * (T[i+1] - 2.0*T[i] + T[i-1])
            lax_avg = 0.5 * (T[i-1] + T[i+1])
            T_new[i] = lax_avg + adv + diff
            if source is not None:
                T_new[i] += dt * source[i]


        if bc_type == "dirichlet":
            T_new[0] = T[0]
            T_new[-1] = T[-1]
        elif bc_type == "neumann":
            T_new[0] = T_new[1]
            T_new[-1] = T_new[-2]
        elif bc_type == "periodic":

            i = 0
            adv = -cfl_adv * 0.5 * (T[1] - T[-2])
            diff = cfl_diff * (T[1] - 2.0*T[i] + T[-2])
            lax_avg = 0.5 * (T[-2] + T[1])
            T_new[i] = lax_avg + adv + diff
            i = nz - 1
            T_new[i] = T_new[0]
        else:
            raise ValueError(f"Unknown bc_type: {bc_type}")

        T, T_new = T_new, T

    return T


def gaussian_laser_source(z: np.ndarray, z0: float, power: float,
                          spot_size: float, absorptivity: float = 0.3) -> np.ndarray:
    Q = absorptivity * power / (np.sqrt(np.pi) * spot_size) * \
        np.exp(-((z - z0)**2) / (spot_size**2))
    return Q






def simulate_layer_deposition_thermal(n_layers: int, layer_thickness: float,
                                       scan_speed: float, laser_power: float,
                                       thermal_diffusivity: float,
                                       dt_per_layer: int = 200) -> dict:

    z_max = n_layers * layer_thickness * 3.0
    nz = 101
    z = np.linspace(0, z_max, nz)
    dz = z[1] - z[0]


    T_room = 300.0
    T_preheat = 400.0
    T = np.full(nz, T_preheat, dtype=np.float64)

    peak_temps = []
    cooling_rates = []

    for layer in range(n_layers):
        z_surface = layer * layer_thickness

        spot = layer_thickness * 2.0
        source = gaussian_laser_source(z, z_surface, laser_power, spot)



        source = source * 2.5e3

        dt = 0.5 * min(dz / (abs(scan_speed / 1000.0) + 1e-6),
                        0.5 * dz * dz / (thermal_diffusivity + 1e-10))
        T_before = T.copy()
        T = thermal_advection_lax_1d(T, z, dt, dt_per_layer, scan_speed,
                                      thermal_diffusivity, source=source,
                                      bc_type="neumann")
        peak_temps.append(np.max(T))

        cooling = (np.max(T_before) - np.max(T)) / (dt * dt_per_layer)
        cooling_rates.append(cooling)

    return {
        "peak_temps": np.array(peak_temps),
        "cooling_rates": np.array(cooling_rates),
        "final_profile": T,
        "depth_grid": z,
    }






def ethier_steinman_solution(X: np.ndarray, Y: np.ndarray, Z: np.ndarray,
                              T: float, a: float = np.pi/4.0, d: float = np.pi/2.0,
                              nu: float = 0.01) -> Tuple[np.ndarray, np.ndarray,
                                                          np.ndarray, np.ndarray]:
    exp_nu = np.exp(-nu * d * d * T)
    exp_2nu = np.exp(-2.0 * nu * d * d * T)

    U = -a * (np.exp(a * X) * np.sin(a * Y + d * Z) +
              np.exp(a * Z) * np.cos(a * X + d * Y)) * exp_nu
    V = -a * (np.exp(a * Y) * np.sin(a * Z + d * X) +
              np.exp(a * X) * np.cos(a * Y + d * Z)) * exp_nu
    W = -a * (np.exp(a * Z) * np.sin(a * X + d * Y) +
              np.exp(a * Y) * np.cos(a * Z + d * X)) * exp_nu

    P = -0.5 * a * a * (
        np.exp(2.0 * a * X) + np.exp(2.0 * a * Y) + np.exp(2.0 * a * Z) +
        2.0 * np.sin(a * X + d * Y) * np.cos(a * Z + d * X) * np.exp(a * (Y + Z)) +
        2.0 * np.sin(a * Y + d * Z) * np.cos(a * X + d * Y) * np.exp(a * (Z + X)) +
        2.0 * np.sin(a * Z + d * X) * np.cos(a * Y + d * Z) * np.exp(a * (X + Y))
    ) * exp_2nu

    return U, V, W, P


def verify_ns_residual_3d(X: np.ndarray, Y: np.ndarray, Z: np.ndarray,
                           T: float, a: float = np.pi/4.0, d: float = np.pi/2.0,
                           nu: float = 0.01) -> Tuple[np.ndarray, np.ndarray,
                                                       np.ndarray, np.ndarray]:
    U, V, W, P = ethier_steinman_solution(X, Y, Z, T, a, d, nu)

    dx = X[1, 0, 0] - X[0, 0, 0] if X.ndim >= 3 else 0.1
    dy = Y[0, 1, 0] - Y[0, 0, 0] if Y.ndim >= 3 else 0.1
    dz = Z[0, 0, 1] - Z[0, 0, 0] if Z.ndim >= 3 else 0.1


    def grad_x(F):
        dF = np.zeros_like(F)
        dF[1:-1, :, :] = (F[2:, :, :] - F[:-2, :, :]) / (2.0 * dx)
        return dF

    def grad_y(F):
        dF = np.zeros_like(F)
        dF[:, 1:-1, :] = (F[:, 2:, :] - F[:, :-2, :]) / (2.0 * dy)
        return dF

    def grad_z(F):
        dF = np.zeros_like(F)
        dF[:, :, 1:-1] = (F[:, :, 2:] - F[:, :, :-2]) / (2.0 * dz)
        return dF

    def laplacian(F):
        return grad_x(grad_x(F)) + grad_y(grad_y(F)) + grad_z(grad_z(F))

    rho = 1.0
    dUdt = np.zeros_like(U)
    conv_x = U * grad_x(U) + V * grad_y(U) + W * grad_z(U)
    R_u = dUdt + conv_x + grad_x(P) / rho - nu * laplacian(U)

    conv_y = U * grad_x(V) + V * grad_y(V) + W * grad_z(V)
    R_v = dUdt + conv_y + grad_y(P) / rho - nu * laplacian(V)

    conv_z = U * grad_x(W) + V * grad_y(W) + W * grad_z(W)
    R_w = dUdt + conv_z + grad_z(P) / rho - nu * laplacian(W)

    R_cont = grad_x(U) + grad_y(V) + grad_z(W)

    return R_u, R_v, R_w, R_cont






def estimate_melt_pool_size(laser_power: float, scan_speed: float,
                             absorptivity: float, thermal_diffusivity: float,
                             melting_temp: float, ambient_temp: float) -> dict:

    rho = 4430.0
    cp = 580.0
    k = thermal_diffusivity * rho * cp

    delta_T = melting_temp - ambient_temp
    if delta_T < 1.0:
        delta_T = 1.0


    v_ms = scan_speed / 1000.0


    L_char = 1e-4
    Pe = v_ms * L_char / thermal_diffusivity



    width = (2.0 * absorptivity * laser_power) / (np.pi * np.e * k * delta_T * max(v_ms, 1e-6))

    width = max(width, 1e-6)
    depth = width / (2.0 + min(Pe, 100.0))
    length = width * (1.0 + 0.5 * min(Pe, 100.0))

    return {
        "width_m": width,
        "depth_m": depth,
        "length_m": length,
        "peclet": Pe,
        "thermal_diffusivity": thermal_diffusivity,
    }






def thermal_strain_gradient(T_profile: np.ndarray, z: np.ndarray,
                             thermal_expansion: float = 9e-6) -> np.ndarray:
    T_ref = T_profile[-1]
    eps_th = thermal_expansion * (T_profile - T_ref)

    d_eps_dz = np.gradient(eps_th, z)
    return d_eps_dz
