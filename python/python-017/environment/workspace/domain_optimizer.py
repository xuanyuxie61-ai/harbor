
import numpy as np
from typing import Callable, Tuple, Optional


def best_nearby(delta: np.ndarray, point: np.ndarray, prevbest: float,
                nvars: int, f: Callable[[np.ndarray], float],
                funevals: int) -> Tuple[float, np.ndarray, int]:
    z = point.copy()
    minf = prevbest

    for i in range(nvars):

        z[i] = point[i] + delta[i]
        ftmp = f(z)
        funevals += 1
        if ftmp < minf:
            minf = ftmp
        else:

            z[i] = point[i] - delta[i]
            ftmp = f(z)
            funevals += 1
            if ftmp < minf:
                minf = ftmp
            else:
                z[i] = point[i]

    return minf, z, funevals


def hooke_jeeves(nvars: int, startpt: np.ndarray, rho: float,
                 eps: float, itermax: int,
                 f: Callable[[np.ndarray], float]) -> Tuple[int, np.ndarray]:
    if not (0 < rho < 1):
        raise ValueError("rho 必须在 (0, 1) 之间")

    newx = startpt.copy()
    xbefore = startpt.copy()
    delta = np.zeros(nvars)
    for i in range(nvars):
        if startpt[i] == 0.0:
            delta[i] = rho
        else:
            delta[i] = rho * abs(startpt[i])

    funevals = 0
    steplength = rho
    iters = 0
    fbefore = f(newx)
    funevals += 1
    newf = fbefore

    while iters < itermax and eps < steplength:
        iters += 1
        newx = xbefore.copy()
        newf, newx, funevals = best_nearby(delta, newx, fbefore, nvars, f, funevals)

        keep = True
        while newf < fbefore and keep:

            for i in range(nvars):
                tmp = xbefore[i]
                xbefore[i] = newx[i]
                if newx[i] <= tmp:
                    delta[i] = -abs(delta[i])
                else:
                    delta[i] = abs(delta[i])
                newx[i] = newx[i] + newx[i] - tmp

            fbefore = newf
            newf, newx, funevals = best_nearby(delta, newx, fbefore, nvars, f, funevals)

            if fbefore <= newf:
                break

            keep = False
            for i in range(nvars):
                if 0.5 * abs(delta[i]) < abs(newx[i] - xbefore[i]):
                    keep = True
                    break

        if eps <= steplength and fbefore <= newf:
            steplength *= rho
            delta *= rho

    endpt = xbefore.copy()
    return iters, endpt


def tsp_descent_style_domain_optimization(
    initial_state: np.ndarray,
    energy_func: Callable[[np.ndarray], float],
    n_variations: int = 500,
    step_size: float = 0.05
) -> Tuple[np.ndarray, float]:
    n = len(initial_state)
    state = initial_state.copy()
    best_energy = energy_func(state)

    rng = np.random.default_rng(seed=42)

    for _ in range(n_variations):

        idx = rng.integers(0, n)
        perturbation = state.copy()
        width = max(1, n // 20)
        start = max(0, idx - width)
        end = min(n, idx + width)
        perturbation[start:end] += rng.normal(0, step_size, end - start)

        e_new = energy_func(perturbation)
        if e_new < best_energy:
            state = perturbation
            best_energy = e_new


        idx1, idx2 = sorted(rng.integers(0, n, 2))
        if idx2 - idx1 < 2:
            continue
        perturbation = state.copy()
        perturbation[idx1:idx2] = -perturbation[idx1:idx2]

        e_new = energy_func(perturbation)
        if e_new < best_energy:
            state = perturbation
            best_energy = e_new

    return state, best_energy


def optimize_domain_configuration(
    P0: np.ndarray, M0: np.ndarray,
    total_energy_func: Callable[[np.ndarray, np.ndarray], float],
    max_iter: int = 100
) -> Tuple[np.ndarray, np.ndarray, float]:
    shape_P = P0.shape
    shape_M = M0.shape
    P_flat = P0.flatten().copy()
    M_flat = M0.flatten().copy()


    nP = len(P_flat)
    nM = len(M_flat)

    def joint_energy(z: np.ndarray) -> float:
        P = z[:nP].reshape(shape_P)
        M = z[nM:].reshape(shape_M)
        e = total_energy_func(P, M)
        if not np.isfinite(e):
            return 1e20
        return e

    z0 = np.concatenate([P_flat, M_flat])


    _, z_opt = hooke_jeeves(len(z0), z0, rho=0.7, eps=1e-5,
                            itermax=max(20, max_iter // 2),
                            f=joint_energy)


    z_opt, E_min = tsp_descent_style_domain_optimization(
        z_opt, joint_energy, n_variations=max_iter * 2, step_size=0.02
    )

    P_opt = z_opt[:nP].reshape(shape_P)
    M_opt = z_opt[nM:].reshape(shape_M)
    return P_opt, M_opt, E_min
