
import numpy as np


def fixed_point_iteration(g_func, x0, tol=1e-10, max_iter=200,
                          bounds=None, verbose=False):
    x = np.asarray(x0, dtype=float).copy()
    n = x.size
    residual_history = []

    for it in range(1, max_iter + 1):
        x_new = g_func(x)
        if bounds is not None:
            low, upp = bounds
            x_new = np.clip(x_new, low, upp)
        diff = np.linalg.norm(x_new - x, ord=np.inf)
        residual_history.append(diff)
        x = x_new

        if diff < tol:
            if verbose:
                print(f"[FixedPoint] Converged in {it} iterations, diff={diff:.3e}")
            return x, diff, it, True


        if it > 10 and len(residual_history) >= 4:
            if residual_history[-1] > residual_history[-2] > residual_history[-3] > residual_history[-4]:
                if verbose:
                    print(f"[FixedPoint] Divergence detected at iter {it}")
                return x, diff, it, False

    if verbose:
        print(f"[FixedPoint] Max iter reached, diff={diff:.3e}")
    return x, diff, max_iter, False


def newton_solver(f_func, j_func, x0, tol=1e-10, max_iter=100,
                  lambda_init=1.0, bounds=None, verbose=False):
    x = np.asarray(x0, dtype=float).copy()
    n = x.size
    fx = f_func(x)
    fx_norm0 = np.linalg.norm(fx, ord=np.inf)
    fx_norm = fx_norm0
    big = 100.0 * fx_norm0
    small = 1e-12

    for it in range(1, max_iter + 1):
        J = j_func(x)

        for i in range(n):
            if abs(J[i, i]) < small:
                J[i, i] += 1e-10

        try:
            dx = np.linalg.solve(J, -fx)
        except np.linalg.LinAlgError:

            dx = np.linalg.lstsq(J, -fx, rcond=None)[0]


        lam = lambda_init
        for _ in range(10):
            x_trial = x + lam * dx
            if bounds is not None:
                low, upp = bounds
                x_trial = np.clip(x_trial, low, upp)
            fx_trial = f_func(x_trial)
            fx_trial_norm = np.linalg.norm(fx_trial, ord=np.inf)
            if fx_trial_norm < fx_norm:
                x = x_trial
                fx = fx_trial
                fx_norm = fx_trial_norm
                break
            lam *= 0.5
        else:

            x = x + 0.1 * dx
            if bounds is not None:
                low, upp = bounds
                x = np.clip(x, low, upp)
            fx = f_func(x)
            fx_norm = np.linalg.norm(fx, ord=np.inf)

        if big < fx_norm:
            if verbose:
                print(f"[Newton] Divergence: |f| grew too large at iter {it}")
            return x, fx_norm, it, False

        if fx_norm < tol:
            if verbose:
                print(f"[Newton] Converged in {it} iterations, |f|={fx_norm:.3e}")
            return x, fx_norm, it, True

    if verbose:
        print(f"[Newton] Max iter reached, |f|={fx_norm:.3e}")
    return x, fx_norm, max_iter, False


def reactor_algebraic_residual(state, params):




    raise NotImplementedError("Hole 1: 请实现 reactor_algebraic_residual 的残差公式")


def reactor_jacobian(state, params):
    eps = 1e-8
    n = len(state)
    J = np.zeros((n, n))
    f0 = reactor_algebraic_residual(state, params)
    for j in range(n):
        state_pert = state.copy()
        state_pert[j] += eps
        f_pert = reactor_algebraic_residual(state_pert, params)
        J[:, j] = (f_pert - f0) / eps
    return J
