
import numpy as np
from typing import Tuple, Callable, Optional
from landau_free_energy import (
    MultiferroicMaterialParams,
    variational_derivative_P,
    variational_derivative_M,
    landau_free_energy_density,
    thermal_fluctuation_correction
)
from reaction_diffusion_solver import ReactionDiffusionFTCS, coupled_reaction_diffusion_step
from adaptive_ode_integrator import AdaptiveMidpointIntegrator, tdgl_rhs
from monte_carlo_sampler import (
    MetropolisMCSampler,
    batch_monte_carlo_statistics,
    compute_correlation_function
)
from domain_optimizer import optimize_domain_configuration


class MultiferroicSimulator:

    def __init__(self, nx: int = 64, ny: int = 64,
                 Lx: float = 1.0, Ly: float = 1.0,
                 temperature: float = 300.0,
                 gamma_P: float = 0.1, gamma_M: float = 0.05):
        self.nx = nx
        self.ny = ny
        self.Lx = Lx
        self.Ly = Ly
        self.dx = Lx / (nx - 1)
        self.dy = Ly / (ny - 1)
        self.params = MultiferroicMaterialParams(temperature)
        self.params.validate()
        self.gamma_P = gamma_P
        self.gamma_M = gamma_M



        D_P = max(self.params.g11, self.params.g12, self.params.g44) * 1e10
        D_M = max(self.params.A11, self.params.A12) * 1e10
        self.solver_P = ReactionDiffusionFTCS(nx, ny, Lx, Ly, D=D_P)
        self.solver_M = ReactionDiffusionFTCS(nx, ny, Lx, Ly, D=D_M)


        self.P = self._initialize_polarization()
        self.M = self._initialize_magnetization()

    def _initialize_polarization(self) -> np.ndarray:
        P = np.zeros((self.ny, self.nx))
        Ps = 0.3
        center_x = self.nx // 2
        P[:, :center_x] = Ps
        P[:, center_x:] = -Ps

        P += np.random.default_rng(123).normal(0, 0.01, (self.ny, self.nx))
        return P

    def _initialize_magnetization(self) -> np.ndarray:
        M = np.zeros((self.ny, self.nx))
        Ms = 1e5
        x = np.linspace(0, self.Lx, self.nx)
        y = np.linspace(0, self.Ly, self.ny)
        X, Y = np.meshgrid(x, y)

        M = Ms * np.sin(2.0 * np.pi * X / self.Lx) * np.cos(2.0 * np.pi * Y / self.Ly)
        M += np.random.default_rng(456).normal(0, Ms * 0.01, (self.ny, self.nx))
        return M

    def compute_laplacian_tensor(self, field: np.ndarray) -> np.ndarray:
        lap = np.zeros((2, 2, self.ny, self.nx), dtype=float)
        f = field


        lap[0, 0, 1:-1, 1:-1] = (f[1:-1, 2:] - 2.0 * f[1:-1, 1:-1] + f[1:-1, :-2]) / self.dx ** 2

        lap[1, 1, 1:-1, 1:-1] = (f[2:, 1:-1] - 2.0 * f[1:-1, 1:-1] + f[:-2, 1:-1]) / self.dy ** 2


        lap[0, 0, :, 0] = lap[0, 0, :, 1]
        lap[0, 0, :, -1] = lap[0, 0, :, -2]
        lap[0, 0, 0, :] = lap[0, 0, 1, :]
        lap[0, 0, -1, :] = lap[0, 0, -2, :]

        lap[1, 1, 0, :] = lap[1, 1, 1, :]
        lap[1, 1, -1, :] = lap[1, 1, -2, :]
        lap[1, 1, :, 0] = lap[1, 1, :, 1]
        lap[1, 1, :, -1] = lap[1, 1, :, -2]

        return lap

    def reaction_P(self, P: np.ndarray, M: np.ndarray) -> np.ndarray:


        pass

    def reaction_M(self, P: np.ndarray, M: np.ndarray) -> np.ndarray:


        pass

    def step_ftcs(self) -> Tuple[np.ndarray, np.ndarray]:
        self.P, self.M = coupled_reaction_diffusion_step(
            self.P, self.M,
            self.solver_P, self.solver_M,
            self.reaction_P, self.reaction_M
        )

        self.P = np.clip(self.P, -1.0, 1.0)
        self.M = np.clip(self.M, -5e5, 5e5)
        return self.P, self.M

    def compute_total_free_energy(self) -> float:
        E = 0.0
        for i in range(self.ny):
            for j in range(self.nx):
                P_loc = np.array([self.P[i, j], 0.0])
                M_loc = np.array([0.0, self.M[i, j]])

                dPdx = np.array([(self.P[i, min(j+1, self.nx-1)] - self.P[i, j]) / self.dx, 0.0])
                dPdy = np.array([0.0, (self.P[min(i+1, self.ny-1), j] - self.P[i, j]) / self.dy])
                dMdx = np.array([0.0, (self.M[i, min(j+1, self.nx-1)] - self.M[i, j]) / self.dx])
                dMdy = np.array([0.0, (self.M[min(i+1, self.ny-1), j] - self.M[i, j]) / self.dy])
                f = landau_free_energy_density(P_loc, M_loc, dPdx, dPdy, dMdx, dMdy, self.params)
                E += f * self.dx * self.dy
        return E

    def compute_magnetoelectric_coefficient(self) -> float:
        V = self.Lx * self.Ly
        product = np.mean(self.P * self.M) * V
        P_mean = np.mean(np.abs(self.P))
        M_mean = np.mean(np.abs(self.M))
        if P_mean < 1e-20 or M_mean < 1e-20:
            return 0.0
        alpha = product / (P_mean * M_mean * V)
        return float(alpha)

    def run_ftcs_simulation(self, nsteps: int = 200) -> dict:
        energy_history = []
        alpha_history = []
        dw_position = []

        for step in range(nsteps):
            self.step_ftcs()
            if step % 10 == 0:
                E = self.compute_total_free_energy()
                alpha = self.compute_magnetoelectric_coefficient()
                energy_history.append(E)
                alpha_history.append(alpha)

                zero_crossings = []
                for i in range(self.ny):
                    for j in range(self.nx - 1):
                        if self.P[i, j] * self.P[i, j + 1] < 0:
                            zero_crossings.append(j)
                if zero_crossings:
                    dw_position.append(np.mean(zero_crossings) * self.dx)
                else:
                    dw_position.append(self.Lx / 2.0)

        return {
            'P_final': self.P.copy(),
            'M_final': self.M.copy(),
            'energy_history': np.array(energy_history),
            'alpha_me_history': np.array(alpha_history),
            'domain_wall_position': np.array(dw_position),
        }

    def run_monte_carlo_thermalization(self, n_steps: int = 500,
                                        amplitude: float = 0.05) -> dict:
        def energy_func(state_flat: np.ndarray) -> float:
            half = len(state_flat) // 2
            P_tmp = state_flat[:half].reshape((self.ny, self.nx))
            M_tmp = state_flat[half:].reshape((self.ny, self.nx))
            self.P = P_tmp
            self.M = M_tmp
            return self.compute_total_free_energy()

        state0 = np.concatenate([self.P.flatten(), self.M.flatten()])
        sampler = MetropolisMCSampler(self.params.T)
        state_final, energies, obs = sampler.sample(
            state0, energy_func, n_steps=n_steps, amplitude=amplitude, burn_in=100
        )

        half = len(state_final) // 2
        self.P = state_final[:half].reshape((self.ny, self.nx))
        self.M = state_final[half:].reshape((self.ny, self.nx))

        return {
            'energies': np.array(energies),
            'observables': np.array(obs),
            'acceptance_rate': len(energies) / max(n_steps, 1),
        }

    def compute_correlation_length(self) -> Tuple[float, float]:
        C_P = compute_correlation_function(self.P, max_r=min(self.nx, self.ny) // 4)
        C_M = compute_correlation_function(self.M, max_r=min(self.nx, self.ny) // 4)


        xi_P = self._find_decay_length(C_P)
        xi_M = self._find_decay_length(C_M)
        return xi_P, xi_M

    def _find_decay_length(self, C: np.ndarray) -> float:
        threshold = 1.0 / np.e
        for r in range(1, len(C)):
            if C[r] < threshold:

                if C[r-1] > C[r]:
                    frac = (C[r-1] - threshold) / (C[r-1] - C[r] + 1e-20)
                    return (r - 1 + frac) * self.dx
        return len(C) * self.dx
