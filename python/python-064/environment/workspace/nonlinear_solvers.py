
import numpy as np


def newton_maehly(coeffs, max_iter=100, tol=1e-12):
    c = np.array(coeffs, dtype=complex)
    d = len(c) - 1
    if d <= 0:
        return np.array([])


    radius = 1.0 + np.max(np.abs(c[:-1] / (c[-1] + 1e-20)))
    radius = min(radius, 100.0)


    theta = np.linspace(0, 2.0 * np.pi, d, endpoint=False)
    roots = radius * np.exp(1j * theta)

    for iteration in range(max_iter):
        roots_old = roots.copy()

        for i in range(d):
            pz, dpz = poly_and_derivative(c, roots[i])
            s = 0.0
            for j in range(d):
                if j != i:
                    diff = roots[i] - roots[j]
                    if abs(diff) > 1e-15:
                        s += 1.0 / diff
            denom = dpz - pz * s
            if abs(denom) < 1e-15:
                denom = 1e-15
            roots[i] = roots[i] - pz / denom


        max_change = np.max(np.abs(roots - roots_old))
        max_poly = np.max(np.abs([poly_and_derivative(c, z)[0] for z in roots]))

        if max_change < tol and max_poly < tol * 10:
            return roots

    return roots


def poly_and_derivative(c, z):
    c = np.array(c, dtype=complex)
    d = len(c) - 1
    if d < 0:
        return 0.0, 0.0

    p = c[d]
    dp = 0.0
    for k in range(d - 1, -1, -1):
        dp = dp * z + p
        p = p * z + c[k]

    return p, dp


def roots_rc(n, x, fx, q):
    x = np.asarray(x, dtype=float).flatten()
    fx = np.asarray(fx, dtype=float).flatten()
    q = np.array(q, dtype=float)

    ferr = np.sum(np.abs(fx))


    if abs(q[2 * n, 0]) < 1e-15:

        for i in range(n):
            for j in range(n + 1):
                q[i, j] = 0.0
                q[i + n, j] = 0.0
            q[i, i] = 100.0
            q[i + n, i] = 1.0

        q[2 * n, 0:n] = 1.0e30
        q[2 * n + 1, 0:n] = n

        for i in range(n):
            q[i + n, n] = x[i]

        q[0:n, n] = fx
        q[2 * n, n] = ferr
        q[2 * n + 1, n] = 0.0


        xnew = x.copy()
        xnew[0] += 0.01
        return xnew, ferr, q, False


    jsus = 0
    for i in range(1, n + 1):
        if 2 * n <= q[2 * n + 1, i]:
            q[2 * n, i] = 1.0e30
        if q[2 * n + 1, jsus] < (n + 3) / 2:
            jsus = i
        if (n + 3) / 2 <= q[2 * n + 1, i] and q[2 * n, jsus] < q[2 * n, i]:
            jsus = i

    for i in range(n):
        q[i + n, jsus] = x[i]
        q[i, jsus] = fx[i]

    q[2 * n, jsus] = ferr
    q[2 * n + 1, jsus] = 0.0


    jsma = 0
    for j in range(n + 1):
        if q[2 * n, j] < q[2 * n, jsma]:
            jsma = j


    if jsma != n:
        for i in range(2 * n + 2):
            q[i, jsma], q[i, n] = q[i, n], q[i, jsma]


    xnew = x.copy()
    if ferr < 1e-8:
        return xnew, ferr, q, True


    for i in range(n):

        xnew[i] = x[i] * 0.9 + 0.1 * q[i + n, n]

    return xnew, ferr, q, False


def continuation_solver(fun, x0, lambda0, lambda_target, dlambda=0.01,
                         max_steps=1000, tol=1e-6):
    x = np.array(x0, dtype=float)
    lam = lambda0
    lambdas = [lam]
    solutions = [x.copy()]

    direction = 1.0 if lambda_target > lambda0 else -1.0
    step = 0

    while direction * (lam - lambda_target) < 0 and step < max_steps:
        step += 1

        f_val = fun(x, lam)
        n = len(x)


        dlam_test = dlambda * 0.01
        f_plus = fun(x, lam + dlam_test)
        dx_dlam = (x - x)

        try:

            dx_dlam_approx = np.zeros(n)
            for i in range(n):
                x_pert = x.copy()
                h_pert = max(abs(x[i]) * 1e-6, 1e-8)
                x_pert[i] += h_pert
                f_pert = fun(x_pert, lam)
                df_dx = (f_pert - f_val) / h_pert
                dx_dlam_approx[i] = -np.sum(df_dx * (f_plus - f_val)) / (dlam_test * max(np.sum(df_dx ** 2), 1e-20))
        except Exception:
            dx_dlam_approx = np.zeros(n)


        x_pred = x + dx_dlam_approx * dlambda
        lam_pred = lam + direction * dlambda


        x_corr = x_pred.copy()
        for newton_iter in range(20):
            f_corr = fun(x_corr, lam_pred)
            if np.linalg.norm(f_corr) < tol:
                break


            J = np.zeros((n, n))
            for i in range(n):
                h_pert = max(abs(x_corr[i]) * 1e-6, 1e-8)
                x_pert = x_corr.copy()
                x_pert[i] += h_pert
                J[:, i] = (fun(x_pert, lam_pred) - f_corr) / h_pert

            try:
                delta = np.linalg.solve(J, -f_corr)
                x_corr = x_corr + delta
                x_corr = np.clip(x_corr, -1e6, 1e6)
            except np.linalg.LinAlgError:

                dlambda *= 0.5
                break
        else:

            dlambda *= 0.5
            if abs(dlambda) < 1e-8:
                break
            continue


        x = x_corr
        lam = lam_pred
        lambdas.append(lam)
        solutions.append(x.copy())


        dlambda = min(abs(dlambda) * 1.1, abs(lambda_target - lambda0) * 0.1)
        dlambda *= direction

    return lambdas, solutions


def find_equilibrium_temperature_ebm(insolation, albedo_func, olr_func,
                                      T_guess=280.0, tol=1e-4, max_iter=100):





    raise NotImplementedError("Hole_3: Equilibrium temperature solver is not implemented.")


def stability_eigenvalues(J):
    n = J.shape[0]
    if n <= 2:

        eigenvalues = np.linalg.eigvals(J)
    else:


        coeffs = np.poly(J)

        eigenvalues = newton_maehly(coeffs)

    max_real = np.max(np.real(eigenvalues))
    if max_real < -1e-6:
        stability = 'stable'
    elif max_real > 1e-6:
        stability = 'unstable'
    else:

        if np.max(np.abs(np.imag(eigenvalues))) > 1e-6:
            stability = 'oscillatory'
        else:
            stability = 'marginally_stable'

    return eigenvalues, stability
