
import numpy as np
from typing import Callable, Optional, Tuple
from spatial_operators import SpatialDiscretization1D
from stochastic_rk import (
    StochasticIntegrator,
    sde_euler_maruyama_step,
    sde_srk_platen_step,
    sde_milstein_step,
    stiff_sde_semiimplicit_step,
    adaptive_rk12_sde_step,
)
from wiener_process import QWienerProcess
from numerical_utils import apply_dirichlet_bc, apply_neumann_bc_rhs


class SPDESolver1D:

    def __init__(self,
                 spatial: SpatialDiscretization1D,
                 wiener: QWienerProcess,
                 integrator: StochasticIntegrator,
                 sigma_noise: float = 0.1,
                 dirichlet_bc: Optional[Tuple[np.ndarray, np.ndarray]] = None,
                 neumann_bc: Optional[Tuple[np.ndarray, np.ndarray]] = None):
        self.spatial = spatial
        self.wiener = wiener
        self.integrator = integrator
        self.sigma_noise = sigma_noise
        self.dirichlet_bc = dirichlet_bc
        self.neumann_bc = neumann_bc


        self.M, self.K = spatial.assemble_fem_matrices()


        self.A_implicit = -np.diag(1.0 / self.M) @ self.K

    def _drift(self, u: np.ndarray) -> np.ndarray:
        return self.spatial.full_rhs_deterministic(u, scheme="auto")

    def _drift_split_nonlinear(self, u: np.ndarray) -> np.ndarray:
        return -self.spatial.advection_operator(u, scheme="auto") + self.spatial.reaction_operator(u)

    def _diffusion(self, u: np.ndarray) -> np.ndarray:
        K = self.spatial.K
        u_clip = np.clip(u, 0.0, None)
        return self.sigma_noise * u_clip * (1.0 - u_clip / K)

    def _diffusion_jacobian_diag(self, u: np.ndarray) -> np.ndarray:

        pass

    def _apply_bc_to_rhs(self, b: np.ndarray, t: float) -> np.ndarray:
        if self.neumann_bc is not None:
            nodes, flux = self.neumann_bc
            b = apply_neumann_bc_rhs(b, self.spatial.dx.mean(), nodes, flux)
        return b

    def solve(self,
              u0: np.ndarray,
              t_span: Tuple[float, float],
              store_every: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        t0, tf = t_span
        if tf <= t0:
            raise ValueError("tf must be > t0")

        t = t0
        u = u0.copy()
        nx = self.spatial.nx


        est_steps = int((tf - t0) / self.integrator.dt) + 10
        t_list = [t]
        u_list = [u.copy()]

        step_count = 0
        while t < tf - 1e-14:
            dt = min(self.integrator.dt, tf - t)
            if dt <= 1e-14:
                break

            dW = self.wiener.increment(dt)


            if self.integrator.method == "semiimplicit":

                y_new = stiff_sde_semiimplicit_step(
                    u, self.A_implicit, self._drift_split_nonlinear,
                    self._diffusion, dt, dW
                )
            elif self.integrator.method == "milstein":

                pass
            elif self.integrator.method == "adaptive_rk12":
                y_new, h_new, accepted = adaptive_rk12_sde_step(
                    u, self._drift, self._diffusion, dt, dW, self.integrator.tol
                )
                self.integrator.dt = h_new
                if not accepted:
                    continue
            elif self.integrator.method == "srk_platen":
                y_new = sde_srk_platen_step(
                    u, self._drift, self._diffusion, dt, dW
                )
            else:
                y_new = sde_euler_maruyama_step(
                    u, self._drift, self._diffusion, dt, dW
                )


            y_new = np.clip(y_new, 0.0, self.spatial.K * 1.5)


            if self.dirichlet_bc is not None:
                bc_nodes, bc_vals = self.dirichlet_bc
                y_new[bc_nodes] = bc_vals

            u = y_new
            t += dt
            step_count += 1

            if step_count % store_every == 0:
                t_list.append(t)
                u_list.append(u.copy())

        if t_list[-1] < tf:
            t_list.append(t)
            u_list.append(u.copy())

        return np.array(t_list, dtype=np.float64), np.array(u_list, dtype=np.float64)

    def compute_energy(self, u: np.ndarray) -> float:
        return float(np.sum(u ** 2 * self.M))

    def compute_total_mass(self, u: np.ndarray) -> float:
        return float(np.sum(u * self.M))


