
import numpy as np
from typing import List, Callable, Optional
from collections import deque


class TravelTimeTree:

    def __init__(self, x0: float, t0: float, v_func: Callable[[float], float],
                 D: float = 0.0, dt: float = 1.0):
        self.x0 = float(x0)
        self.t0 = float(t0)
        self.v_func = v_func
        self.D = float(D)
        self.dt = float(dt)

    def forward_step(self, x: float) -> float:
        return x + self.v_func(x) * self.dt

    def backward_step(self, x: float, n_branches: int = 1,
                      rng: Optional[np.random.Generator] = None) -> list[float]:
        if rng is None:
            rng = np.random.default_rng()
        v = self.v_func(x)
        x_base = x - v * self.dt
        if self.D <= 0.0 or n_branches <= 1:
            return [x_base]

        branches = []
        std = np.sqrt(2.0 * self.D * self.dt)
        for _ in range(n_branches):
            xi = rng.normal(0.0, std)
            branches.append(x_base + xi)
        return branches

    def build_backward_tree(self, max_levels: int = 10,
                            n_branches: int = 2) -> dict:
        rng = np.random.default_rng(42)
        levels = [[self.x0]]
        times = [self.t0]

        for level in range(1, max_levels + 1):
            current_level = []
            for x_parent in levels[-1]:
                children = self.backward_step(x_parent, n_branches=n_branches, rng=rng)
                current_level.extend(children)
            levels.append(current_level)
            times.append(self.t0 + level * self.dt)

        return {"levels": levels, "times": times}

    def compute_travel_time_distribution(self, x_source: float,
                                         n_particles: int = 10000,
                                         max_steps: int = 500,
                                         x_bounds: tuple = (-1e9, 1e9)) -> np.ndarray:
        if n_particles < 1:
            raise ValueError("粒子数必须 ≥ 1")
        rng = np.random.default_rng(123)
        arrival_times = np.full(n_particles, -1.0)

        for p in range(n_particles):
            x = float(x_source)
            for step in range(1, max_steps + 1):
                v = self.v_func(x)
                std = np.sqrt(2.0 * self.D * self.dt) if self.D > 0 else 0.0
                x = x + v * self.dt + rng.normal(0.0, std)
                t = step * self.dt


                tol = max(abs(v) * self.dt * 2.0, 1e-3)
                if abs(x - self.x0) < tol:
                    arrival_times[p] = t
                    break
                if x < x_bounds[0] or x > x_bounds[1]:
                    break

        return arrival_times


def discrete_dynamical_stability_map(v_func: Callable[[float], float],
                                     x_grid: np.ndarray,
                                     dt: float = 1.0,
                                     n_iter: int = 100) -> np.ndarray:
    if len(x_grid) == 0:
        raise ValueError("网格不能为空")
    lyap = np.zeros_like(x_grid)
    h_deriv = 1e-6

    for i, x0 in enumerate(x_grid):
        x = x0
        lam_sum = 0.0
        for _ in range(n_iter):

            vp = v_func(x + h_deriv)
            vm = v_func(x - h_deriv)
            dv_dx = (vp - vm) / (2.0 * h_deriv)
            jac = abs(1.0 + dv_dx * dt)
            if jac > 1e-15:
                lam_sum += np.log(jac)
            x = x + v_func(x) * dt
            if not np.isfinite(x):
                lam_sum = np.nan
                break
        lyap[i] = lam_sum / n_iter if np.isfinite(lam_sum) else np.nan

    return lyap


if __name__ == "__main__":

    tree = TravelTimeTree(x0=50.0, t0=0.0, v_func=lambda x: 0.5, D=0.1, dt=2.0)
    btree = tree.build_backward_tree(max_levels=5, n_branches=2)
    assert len(btree["levels"]) == 6

    ttd = tree.compute_travel_time_distribution(x_source=0.0, n_particles=500, max_steps=200)
    reached = ttd[ttd > 0]
    assert len(reached) > 0

    x_grid = np.linspace(0, 100, 50)
    lyap = discrete_dynamical_stability_map(lambda x: 0.5, x_grid)
    assert np.allclose(lyap, np.log(1.0), atol=0.1)
    print("travel_time_tree: 自测试通过")
