
import numpy as np


def best_nearby(delta: np.ndarray, x: np.ndarray, fbefore: float,
                nvars: int, f, funevals: int) -> tuple:
    newx = np.copy(x)
    newf = fbefore
    for i in range(nvars):

        z = np.copy(newx)
        z[i] += delta[i]
        fz = f(z)
        funevals += 1
        if fz < newf:
            newf = fz
            newx = z
        else:

            z = np.copy(newx)
            z[i] -= delta[i]
            fz = f(z)
            funevals += 1
            if fz < newf:
                newf = fz
                newx = z
    return newf, newx, funevals


def hooke_jeeves(nvars: int, startpt: np.ndarray, rho: float,
                 eps: float, itermax: int, f) -> tuple:
    startpt = np.asarray(startpt, dtype=float)
    if startpt.shape[0] != nvars:
        raise ValueError("startpt 维度与 nvars 不匹配。")
    if not (0.0 < rho < 1.0):
        raise ValueError("rho 必须在 (0,1) 区间内。")

    newx = np.copy(startpt)
    xbefore = np.copy(startpt)
    delta = np.zeros(nvars)
    for i in range(nvars):
        if startpt[i] == 0.0:
            delta[i] = rho
        else:
            delta[i] = rho * abs(startpt[i])

    funevals = 0
    steplength = float(np.max(np.abs(delta)))
    iters = 0
    fbefore = f(newx)
    funevals += 1
    newf = fbefore

    while iters < itermax and eps < steplength:
        iters += 1
        newx = np.copy(xbefore)
        newf, newx, funevals = best_nearby(delta, newx, fbefore,
                                           nvars, f, funevals)

        keep = 1
        while newf < fbefore and keep == 1:
            for i in range(nvars):
                if newx[i] <= xbefore[i]:
                    delta[i] = -abs(delta[i])
                else:
                    delta[i] = abs(delta[i])
                tmp = xbefore[i]
                xbefore[i] = newx[i]
                newx[i] = newx[i] + newx[i] - tmp

            fbefore = newf
            newf, newx, funevals = best_nearby(delta, newx, fbefore,
                                               nvars, f, funevals)

            if fbefore <= newf:
                break

            keep = 0
            for i in range(nvars):
                if 0.5 * abs(delta[i]) < abs(newx[i] - xbefore[i]):
                    keep = 1
                    break

        if eps <= steplength and fbefore <= newf:
            steplength *= rho
            delta *= rho

    endpt = np.copy(xbefore)
    return iters, endpt


def optimize_gan_hyperparams(initial_params: np.ndarray,
                             loss_evaluator, rho: float = 0.85,
                             eps: float = 1e-4, itermax: int = 30) -> tuple:
    nvars = len(initial_params)
    history = []

    def wrapped_f(x):
        val = loss_evaluator(x)
        history.append(float(val))
        return float(val)

    iters, best = hooke_jeeves(nvars, initial_params, rho, eps, itermax, wrapped_f)
    return best, history
