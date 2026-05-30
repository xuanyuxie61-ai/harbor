
import numpy as np
from typing import Callable, Tuple


class ManufacturedVerification:

    def __init__(self, kappa: float = 1.0):
        self.kappa = kappa

    def heat_exact(self, x: np.ndarray, t: float, omega: float = np.pi, lam: float = 1.0) -> np.ndarray:
        return np.sin(omega * x) * np.exp(-lam * t)

    def heat_forcing(self, x: np.ndarray, t: float, omega: float = np.pi, lam: float = 1.0) -> np.ndarray:
        return (-lam + self.kappa * omega ** 2) * np.sin(omega * x) * np.exp(-lam * t)

    def verify_heat_fem(self, x_grid: np.ndarray, dt: float, n_steps: int) -> dict:
        from pde_spatiotemporal_model import PDE1DHeatExplicit
        solver = PDE1DHeatExplicit(kappa=self.kappa)
        omega = np.pi / (x_grid[-1] - x_grid[0])
        lam = 1.0

        u0 = self.heat_exact(x_grid, 0.0, omega, lam)

        def source(x, t, u):
            return self.heat_forcing(x, t, omega, lam)

        u_num = solver.solve(x_grid, u0, dt, n_steps, source=source)
        t_final = dt * n_steps
        u_exact = self.heat_exact(x_grid, t_final, omega, lam)


        h = np.diff(x_grid)
        h_avg = np.mean(h)
        err = u_num - u_exact
        l2_err = np.sqrt(np.mean(err ** 2))
        linf_err = np.max(np.abs(err))

        return {
            "l2_error": float(l2_err),
            "linf_error": float(linf_err),
            "relative_l2": float(l2_err / (np.sqrt(np.mean(u_exact ** 2)) + 1e-15)),
            "grid_size": len(x_grid),
            "dt": dt,
            "final_time": t_final
        }

    def convergence_study(self, n_grids: list, dt: float, n_steps: int) -> dict:
        results = []
        for n in n_grids:
            x_grid = np.linspace(0.0, 1.0, n)
            res = self.verify_heat_fem(x_grid, dt, n_steps)
            results.append(res)


        orders = []
        for i in range(1, len(results)):
            h_ratio = results[i - 1]["grid_size"] / results[i]["grid_size"]
            err_ratio = results[i - 1]["l2_error"] / (results[i]["l2_error"] + 1e-15)
            if err_ratio > 0:
                p = np.log(err_ratio) / np.log(h_ratio)
                orders.append(p)

        return {
            "results": results,
            "estimated_spatial_order": float(np.mean(orders)) if orders else 0.0
        }

    def reaction_diffusion_exact(self, x: np.ndarray, t: float) -> np.ndarray:
        return np.exp(-t) * np.sin(np.pi * x) + 0.5

    def reaction_diffusion_forcing(self, x: np.ndarray, t: float, D: float = 0.1) -> np.ndarray:
        u = self.reaction_diffusion_exact(x, t)
        ut = -np.exp(-t) * np.sin(np.pi * x)
        uxx = -(np.pi ** 2) * np.exp(-t) * np.sin(np.pi * x)
        return ut - D * uxx

    def verify_reaction_diffusion(self, x_grid: np.ndarray, dt: float, n_steps: int, D: float = 0.1) -> dict:
        from pde_spatiotemporal_model import ReactionDiffusion1D
        solver = ReactionDiffusion1D(D=D, rho=0.0, K=1.0, mu=0.0, c_s=1.0)
        u0 = self.reaction_diffusion_exact(x_grid, 0.0)


        original_reaction = solver.reaction
        solver.reaction = lambda u: self.reaction_diffusion_forcing(x_grid, 0.0, D) * np.ones_like(u)

        u_num = solver.solve(x_grid, u0, dt, n_steps, scheme="heun")
        t_final = dt * n_steps
        u_exact = self.reaction_diffusion_exact(x_grid, t_final)

        err = u_num - u_exact
        l2_err = np.sqrt(np.mean(err ** 2))

        return {
            "l2_error": float(l2_err),
            "relative_l2": float(l2_err / (np.sqrt(np.mean(u_exact ** 2)) + 1e-15)),
            "verified": l2_err < 0.1
        }
