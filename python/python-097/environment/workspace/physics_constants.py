
import numpy as np


MU_0 = 4.0 * np.pi * 1e-7
EPSILON_0 = 8.854187817e-12
C_0 = 1.0 / np.sqrt(MU_0 * EPSILON_0)
ETA_0 = np.sqrt(MU_0 / EPSILON_0)


def curl_electric_to_magnetic(E, dx, dy, dz):
    Ex, Ey, Ez = E






    raise NotImplementedError("Hole 1: 请实现curl_electric_to_magnetic的旋度离散计算")


def curl_magnetic_to_electric(H, dx, dy, dz):
    Hx, Hy, Hz = H


    dHz_dy = np.zeros_like(Hx)
    dHy_dz = np.zeros_like(Hx)
    dHz_dy[:, :-1, :] = (Hz[:, 1:, :] - Hz[:, :-1, :]) / dy
    dHy_dz[:, :, :-1] = (Hy[:, :, 1:] - Hy[:, :, :-1]) / dz
    curl_x = dHz_dy - dHy_dz


    dHx_dz = np.zeros_like(Hy)
    dHz_dx = np.zeros_like(Hy)
    dHx_dz[:, :, :-1] = (Hx[:, :, 1:] - Hx[:, :, :-1]) / dz
    dHz_dx[:-1, :, :] = (Hz[1:, :, :] - Hz[:-1, :, :]) / dx
    curl_y = dHx_dz - dHz_dx


    dHy_dx = np.zeros_like(Hz)
    dHx_dy = np.zeros_like(Hz)
    dHy_dx[:-1, :, :] = (Hy[1:, :, :] - Hy[:-1, :, :]) / dx
    dHx_dy[:, :-1, :] = (Hx[:, 1:, :] - Hx[:, :-1, :]) / dy
    curl_z = dHy_dx - dHx_dy

    return curl_x, curl_y, curl_z


def electromagnetic_energy_density(E, H, epsilon, mu):
    Ex, Ey, Ez = E
    Hx, Hy, Hz = H
    E_magnitude_sq = Ex**2 + Ey**2 + Ez**2
    H_magnitude_sq = Hx**2 + Hy**2 + Hz**2
    return 0.5 * (epsilon * E_magnitude_sq + mu * H_magnitude_sq)


def poynting_vector(E, H):
    Ex, Ey, Ez = E
    Hx, Hy, Hz = H
    Sx = Ey * Hz - Ez * Hy
    Sy = Ez * Hx - Ex * Hz
    Sz = Ex * Hy - Ey * Hx
    return Sx, Sy, Sz


def quality_factor(omega, W_stored, P_loss, eps=1e-30):
    if P_loss < eps:
        return float('inf')
    return omega * W_stored / P_loss


def cfl_condition_3d(dx, dy, dz, c_max):
    return 1.0 / (c_max * np.sqrt(1.0/dx**2 + 1.0/dy**2 + 1.0/dz**2))


def wavenumber_frequency_relation(omega, epsilon, mu):
    return omega * np.sqrt(epsilon * mu)
