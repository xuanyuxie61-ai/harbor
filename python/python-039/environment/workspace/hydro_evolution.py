
import numpy as np
from typing import Tuple, Optional, Callable


class HydroEvolution:

    def __init__(self, eta_s_over_s: float = 0.08,
                 cs2: float = 1.0 / 3.0,
                 g_star: float = 47.5,
                 tau0: float = 0.6,
                 tau_f: float = 10.0,
                 dtau: float = 0.05):
        self.eta_s_over_s = eta_s_over_s
        self.cs2 = cs2
        self.g_star = g_star
        self.tau0 = tau0
        self.tau_f = tau_f
        self.dtau = dtau

    def equation_of_state(self, epsilon: np.ndarray) -> np.ndarray:
        return self.cs2 * epsilon

    def energy_to_temperature(self, epsilon: np.ndarray) -> np.ndarray:
        epsilon = np.asarray(epsilon)
        epsilon = np.where(epsilon < 1e-15, 1e-15, epsilon)

        pass

    def temperature_to_energy(self, T: np.ndarray) -> np.ndarray:
        T = np.asarray(T)
        hbarc = 0.1973269804
        return (np.pi ** 2 * self.g_star / 30.0) * (T ** 4) / (hbarc ** 3)

    def specific_entropy(self, epsilon: np.ndarray) -> np.ndarray:
        T = self.energy_to_temperature(epsilon)
        P = self.equation_of_state(epsilon)
        s = (epsilon + P) / T
        return s

    def bjorken_1d(self, epsilon0: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        tau_grid = np.arange(self.tau0, self.tau_f, self.dtau)
        n_tau = len(tau_grid)

        epsilon = np.zeros(n_tau)
        T = np.zeros(n_tau)

        epsilon[0] = epsilon0
        T[0] = self.energy_to_temperature(epsilon0)

        for i in range(1, n_tau):
            tau = tau_grid[i]
            tau_prev = tau_grid[i - 1]


            eps_ideal = epsilon0 * (self.tau0 / tau) ** (1.0 + self.cs2)


            T0 = self.energy_to_temperature(epsilon0)
            visc_corr = 1.0 + (4.0 / 3.0) * self.eta_s_over_s * (
                1.0 - self.tau0 / tau
            ) / (self.tau0 * T0)
            visc_corr = np.clip(visc_corr, 0.5, 5.0)

            epsilon[i] = eps_ideal * visc_corr
            T[i] = self.energy_to_temperature(epsilon[i])

        return tau_grid, epsilon, T

    def evolve_2d(self, x_grid: np.ndarray, y_grid: np.ndarray,
                  epsilon_init: np.ndarray,
                  nx: int = 40, ny: int = 40) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        tau_grid = np.arange(self.tau0, self.tau_f, self.dtau)
        nt = len(tau_grid)

        epsilon_history = np.zeros((nt, nx, ny))
        T_history = np.zeros((nt, nx, ny))
        entropy_history = np.zeros((nt, nx, ny))

        epsilon_history[0] = epsilon_init
        T_history[0] = self.energy_to_temperature(epsilon_init)
        entropy_history[0] = self.specific_entropy(epsilon_init)

        dx = x_grid[1] - x_grid[0] if len(x_grid) > 1 else 1.0
        dy = y_grid[1] - y_grid[0] if len(y_grid) > 1 else 1.0

        for it in range(1, nt):
            tau = tau_grid[it]
            eps_prev = epsilon_history[it - 1].copy()


            eps_pad = np.pad(eps_prev, ((1, 1), (1, 1)), mode='edge')


            laplacian = (
                (eps_pad[2:, 1:-1] - 2 * eps_pad[1:-1, 1:-1] + eps_pad[:-2, 1:-1]) / dx ** 2 +
                (eps_pad[1:-1, 2:] - 2 * eps_pad[1:-1, 1:-1] + eps_pad[1:-1, :-2]) / dy ** 2
            )


            T_prev = T_history[it - 1]
            s_prev = entropy_history[it - 1]
            D = np.zeros_like(T_prev)
            mask = (T_prev > 1e-6) & (s_prev > 1e-15) & (tau > 1e-6)
            D[mask] = self.eta_s_over_s / (tau * T_prev[mask] * s_prev[mask])
            D = np.clip(D, 0.0, 2.0)



            damping = -(1.0 + self.cs2) * eps_prev / tau
            diffusion = D * laplacian

            eps_new = eps_prev + self.dtau * (damping + diffusion)
            eps_new = np.clip(eps_new, 1e-6, 1e6)

            epsilon_history[it] = eps_new
            T_history[it] = self.energy_to_temperature(eps_new)
            entropy_history[it] = self.specific_entropy(eps_new)

        return tau_grid, epsilon_history, T_history, entropy_history

    def freezeout_surface(self, tau_grid: np.ndarray,
                          T_history: np.ndarray,
                          T_freezeout: float = 0.154) -> np.ndarray:
        nt, nx, ny = T_history.shape
        tau_fo = np.full((nx, ny), -1.0)

        for i in range(nx):
            for j in range(ny):
                T_line = T_history[:, i, j]

                below = np.where(T_line < T_freezeout)[0]
                if len(below) > 0:
                    idx = below[0]
                    if idx == 0:
                        tau_fo[i, j] = tau_grid[0]
                    else:

                        t1, t2 = tau_grid[idx - 1], tau_grid[idx]
                        T1, T2 = T_line[idx - 1], T_line[idx]
                        if abs(T2 - T1) > 1e-15:
                            tau_fo[i, j] = t1 + (T_freezeout - T1) * (t2 - t1) / (T2 - T1)
                        else:
                            tau_fo[i, j] = t1
        return tau_fo

    def flow_velocity(self, epsilon_history: np.ndarray,
                      x_grid: np.ndarray, y_grid: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        nt, nx, ny = epsilon_history.shape
        ux = np.zeros_like(epsilon_history)
        uy = np.zeros_like(epsilon_history)

        dx = x_grid[1] - x_grid[0] if len(x_grid) > 1 else 1.0
        dy = y_grid[1] - y_grid[0] if len(y_grid) > 1 else 1.0

        for it in range(nt):
            eps = epsilon_history[it]

            eps_pad_x = np.pad(eps, ((0, 0), (1, 1)), mode='edge')
            eps_pad_y = np.pad(eps, ((1, 1), (0, 0)), mode='edge')

            d_eps_dx = (eps_pad_x[:, 2:] - eps_pad_x[:, :-2]) / (2.0 * dx)
            d_eps_dy = (eps_pad_y[2:, :] - eps_pad_y[:-2, :]) / (2.0 * dy)

            ux[it] = -self.cs2 * d_eps_dx * 0.1
            uy[it] = -self.cs2 * d_eps_dy * 0.1

        return ux, uy

    def entropy_production(self, tau_grid: np.ndarray,
                           epsilon_history: np.ndarray,
                           ux: np.ndarray, uy: np.ndarray) -> np.ndarray:
        nt, nx, ny = epsilon_history.shape
        S_production = np.zeros(nt)

        dx = 1.0 if nx <= 1 else 1.0
        dy = 1.0 if ny <= 1 else 1.0

        for it in range(1, nt):

            ux_pad = np.pad(ux[it], ((0, 0), (1, 1)), mode='edge')
            uy_pad = np.pad(uy[it], ((1, 1), (0, 0)), mode='edge')
            div_u = ((ux_pad[:, 2:] - ux_pad[:, :-2]) / (2 * dx) +
                     (uy_pad[2:, :] - uy_pad[:-2, :]) / (2 * dy))

            T = self.energy_to_temperature(epsilon_history[it])
            T = np.where(T < 1e-6, 1e-6, T)


            local_rate = self.eta_s_over_s * (div_u ** 2) / T
            S_production[it] = S_production[it - 1] + self.dtau * np.sum(local_rate) * dx * dy

        return S_production
