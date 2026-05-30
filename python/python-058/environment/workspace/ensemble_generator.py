
import numpy as np
from typing import List, Tuple


def hypercube_grid(dim: int, ns: List[int], bounds: List[Tuple[float, float]],
                   centering: List[int] = None) -> np.ndarray:
    if centering is None:
        centering = [0] * dim


    grids_1d = []
    for d in range(dim):
        a, b = bounds[d]
        n = max(2, ns[d])
        ctype = centering[d]

        if ctype == 0:
            x = np.linspace(a, b, n)
        elif ctype == 1:
            if n == 1:
                x = np.array([(a+b)/2.0])
            else:
                dx = (b - a) / (n + 1.0)
                x = a + dx * np.arange(1, n + 1)
        elif ctype == 2:
            dx = (b - a) / n
            x = a + dx * np.arange(n)
        elif ctype == 3:
            dx = (b - a) / n
            x = a + dx * np.arange(1, n + 1)
        elif ctype == 4:
            dx = (b - a) / n
            x = a + dx * (np.arange(n) + 0.5)
        else:
            x = np.linspace(a, b, n)
        grids_1d.append(x)


    N = int(np.prod(ns))
    grid = np.zeros((N, dim))


    idx = 0
    def recurse(d, current):
        nonlocal idx
        if d == dim:
            grid[idx, :] = current
            idx += 1
            return
        for val in grids_1d[d]:
            recurse(d + 1, current + [val])

    recurse(0, [])
    return grid


def nearest_interp_1d(xd: np.ndarray, yd: np.ndarray, xi: np.ndarray) -> np.ndarray:
    xd = np.asarray(xd)
    yd = np.asarray(yd)
    xi = np.asarray(xi)
    result = np.zeros_like(xi)

    for i, x in enumerate(xi.flat):

        dist = np.abs(xd - x)
        idx = int(np.argmin(dist))
        result.flat[i] = yd[idx]
    return result


class EnsembleParameterSampler:

    def __init__(self, param_names: List[str], param_bounds: List[Tuple[float, float]],
                 samples_per_dim: int = 3):
        self.param_names = param_names
        self.param_bounds = param_bounds
        self.dim = len(param_names)
        self.samples_per_dim = samples_per_dim
        self.grid = hypercube_grid(self.dim, [samples_per_dim] * self.dim,
                                   param_bounds, centering=[1] * self.dim)
        self.n_ensemble = len(self.grid)

    def get_member_params(self, member_idx: int) -> dict:
        if member_idx < 0 or member_idx >= self.n_ensemble:
            raise IndexError("Member index out of range")
        return {name: float(self.grid[member_idx, d])
                for d, name in enumerate(self.param_names)}


class SoundingProfileInterpolator:

    def __init__(self, pressure_levels: np.ndarray, temperature: np.ndarray,
                 dewpoint: np.ndarray, wind_u: np.ndarray, wind_v: np.ndarray):
        self.p_src = np.asarray(pressure_levels)
        self.T_src = np.asarray(temperature)
        self.Td_src = np.asarray(dewpoint)
        self.u_src = np.asarray(wind_u)
        self.v_src = np.asarray(wind_v)

        if len(self.p_src) > 1 and self.p_src[0] < self.p_src[1]:
            self.p_src = self.p_src[::-1]
            self.T_src = self.T_src[::-1]
            self.Td_src = self.Td_src[::-1]
            self.u_src = self.u_src[::-1]
            self.v_src = self.v_src[::-1]

    def interpolate(self, p_target: np.ndarray) -> Tuple[np.ndarray, np.ndarray,
                                                           np.ndarray, np.ndarray, np.ndarray]:
        T = nearest_interp_1d(self.p_src, self.T_src, p_target)
        Td = nearest_interp_1d(self.p_src, self.Td_src, p_target)
        u = nearest_interp_1d(self.p_src, self.u_src, p_target)
        v = nearest_interp_1d(self.p_src, self.v_src, p_target)
        return p_target, T, Td, u, v
