
import numpy as np






def givapp(c, s, vin, k):
    vout = np.asarray(vin, dtype=float).copy()
    for i in range(k):
        w1 = c[i] * vout[i] - s[i] * vout[i + 1]
        w2 = s[i] * vout[i] + c[i] * vout[i + 1]
        vout[i] = w1
        vout[i + 1] = w2
    return vout


def gmres_solve(x0, b, atv, params):
    b = np.asarray(b, dtype=float)
    x0 = np.asarray(x0, dtype=float)
    n = len(b)
    errtol = params[0]
    kmax = int(params[1])
    reorth = 1
    if len(params) >= 3:
        reorth = int(params[2])

    x = x0.copy()
    if np.linalg.norm(x) > 1e-14:
        r = b - atv(x)
    else:
        r = b.copy()
    rho = np.linalg.norm(r)
    g_vec = rho * np.eye(kmax + 1, 1)[:, 0]
    errtol_abs = errtol * np.linalg.norm(b)
    error = [rho]
    total_iters = 0
    if rho < errtol_abs:
        return x, error, total_iters

    h = np.zeros((kmax, kmax), dtype=float)
    v = np.zeros((n, kmax), dtype=float)
    c = np.zeros(kmax + 1, dtype=float)
    s = np.zeros(kmax + 1, dtype=float)
    v[:, 0] = r / rho
    beta = rho
    k = 0

    while rho > errtol_abs and k < kmax:
        k += 1
        v[:, k - 1] = np.where(np.abs(v[:, k - 1]) < 1e-15, 0.0, v[:, k - 1])
        av = atv(v[:, k - 1])
        v[:, k] = av
        normav = np.linalg.norm(v[:, k])


        for j in range(k):
            h[j, k - 1] = np.dot(v[:, j], v[:, k])
            v[:, k] = v[:, k] - h[j, k - 1] * v[:, j]
        h[k, k - 1] = np.linalg.norm(v[:, k])
        normav2 = h[k, k - 1]


        if (reorth == 1 and normav + 0.001 * normav2 == normav) or reorth == 3:
            for j in range(k):
                hr = np.dot(v[:, j], v[:, k])
                h[j, k - 1] += hr
                v[:, k] = v[:, k] - hr * v[:, j]
            h[k, k - 1] = np.linalg.norm(v[:, k])

        if h[k, k - 1] > 1e-14:
            v[:, k] = v[:, k] / h[k, k - 1]


        if k > 1:
            h[0:k, k - 1] = givapp(c[0:k - 1], s[0:k - 1], h[0:k, k - 1], k - 1)

        nu = np.linalg.norm(h[k - 1:k + 1, k - 1])
        if nu > 1e-14:
            c[k - 1] = h[k - 1, k - 1] / nu
            s[k - 1] = -h[k, k - 1] / nu
            h[k - 1, k - 1] = c[k - 1] * h[k - 1, k - 1] - s[k - 1] * h[k, k - 1]
            h[k, k - 1] = 0.0
            g_vec[k - 1:k + 1] = givapp(c[k - 1], s[k - 1], g_vec[k - 1:k + 1], 1)

        rho = abs(g_vec[k])
        error.append(rho)
        total_iters = k

    y = np.linalg.solve(h[0:k, 0:k], g_vec[0:k])
    x = x0 + np.dot(v[:, 0:k], y)
    return x, error, total_iters






def cg_solve(x0, b, atv, params):
    x = np.asarray(x0, dtype=float).copy()
    b = np.asarray(b, dtype=float)
    errtol = params[0] * np.linalg.norm(b)
    kmax = int(params[1])

    r = b - atv(x)
    k = 1
    rho = np.dot(r, r)
    zeta = np.linalg.norm(r)
    total_iters = 1
    err = [zeta]
    p = r.copy()

    while zeta > errtol and k < kmax + 1:
        w = atv(p)
        alpha = rho / (np.dot(p, w) + 1e-14)
        x = x + alpha * p
        r = r - alpha * w
        rho_new = np.dot(r, r)
        zeta = np.linalg.norm(r)
        total_iters += 1
        err.append(zeta)
        beta = rho_new / (rho + 1e-14)
        p = r + beta * p
        rho = rho_new
        k += 1

    return x, err, total_iters






def diffjac(x, f, f0, eps=1e-7):
    n = len(x)
    J = np.zeros((n, n), dtype=float)
    for j in range(n):
        xj = x.copy()
        h = eps * max(abs(xj[j]), 1.0)
        if h < eps:
            h = eps
        xj[j] = xj[j] + h
        fj = f(xj)
        J[:, j] = (fj - f0) / h

    l, u = np.linalg.qr(J)
    return l, u


def nsol_solve(x, f, tol, parms=None):
    x = np.asarray(x, dtype=float).copy()
    atol = tol[0]
    rtol = tol[1]
    maxit = 40
    isham = 1000
    rsham = 0.5
    if parms is not None:
        maxit = int(parms[0])
        isham = int(parms[1])
        rsham = float(parms[2])

    ierr = 0
    it_hist = []
    n = len(x)
    f0 = f(x)
    fnrm = np.linalg.norm(f0, np.inf)
    it_hist.append(fnrm)
    fnrmo = 1.0
    itc = 0
    stop_tol = atol + rtol * fnrm
    itsham = isham
    outstat = []

    while stop_tol < fnrm and itc < maxit:
        ratio = fnrm / fnrmo if fnrmo > 0 else fnrm
        outstat.append([itc, fnrm, ratio])
        fnrmo = fnrm
        itc += 1

        if itc == 1 or ratio > rsham or itsham == 0:
            itsham = isham
            l, u = diffjac(x, f, f0)
        itsham -= 1


        step = -np.linalg.solve(l @ u, f0)
        xold = x.copy()
        x = x + step
        f0 = f(x)
        fnrm = np.linalg.norm(f0, np.inf)
        it_hist.append(fnrm)
        ratio = fnrm / fnrmo if fnrmo > 0 else fnrm
        outstat.append([itc, fnrm, ratio])

        if ratio >= 1.0:
            ierr = 1
            return xold, it_hist, ierr

    sol = x.copy()
    if stop_tol < fnrm:
        ierr = 1
    return sol, it_hist, ierr






def build_forward_model_oct(z_scan, layer_boundaries, k_min, k_max, n_gl=32):
    from spectral_integration import integrate_depth_resolved_signal
    from oct_physics import source_spectrum_gaussian
    from scipy.special import erf

    boundaries = np.asarray(layer_boundaries, dtype=float)
    n_layers = len(boundaries) - 1

    def forward_func(params):
        params = np.asarray(params, dtype=float)
        if len(params) != 3 * n_layers:
            raise ValueError(f"Expected {3*n_layers} parameters, got {len(params)}")

        def reflectivity_func(k, z):

            layer = 0
            for i in range(n_layers):
                if boundaries[i] <= z < boundaries[i + 1]:
                    layer = i
                    break
            idx = 3 * layer
            mu_a = params[idx]
            mu_s = params[idx + 1]
            g = params[idx + 2]

            mu_t = mu_a + mu_s

            refl = mu_s * np.exp(-2.0 * mu_t * z)
            return refl

        A = integrate_depth_resolved_signal(reflectivity_func, z_scan, k_min, k_max, n_gl)
        return A

    return forward_func


def inverse_problem_oct(z_scan, A_measured, layer_boundaries, param_guess,
                        k_min, k_max, max_iter=20):
    forward_func = build_forward_model_oct(z_scan, layer_boundaries, k_min, k_max)
    n_params = len(param_guess)
    n_data = len(z_scan)

    def objective(p):
        return forward_func(p) - A_measured

    def jacobian_atv(p, vec):
        eps = 1e-6
        return (objective(p + eps * vec) - objective(p - eps * vec)) / (2.0 * eps)

    params = np.asarray(param_guess, dtype=float).copy()
    residuals = []
    lam = 0.01

    for it in range(max_iter):
        res = objective(params)
        rnorm = np.linalg.norm(res)
        residuals.append(rnorm)
        if rnorm < 1e-4:
            break



        def normal_atv(vec):

            jv = jacobian_atv(params, vec)

            jt_jv = np.zeros(n_params)
            for j in range(n_params):
                ej = np.zeros(n_params)
                ej[j] = 1.0
                jt_jv[j] = np.dot(jacobian_atv(params, ej), jv)
            return jt_jv + lam * vec

        rhs = np.zeros(n_params)
        for j in range(n_params):
            ej = np.zeros(n_params)
            ej[j] = 1.0
            rhs[j] = -np.dot(jacobian_atv(params, ej), res)

        delta, _, _ = cg_solve(np.zeros(n_params), rhs, normal_atv, [1e-3, min(n_params, 50)])
        params_new = params + delta
        rnew = np.linalg.norm(objective(params_new))
        if rnew < rnorm:
            params = params_new
            lam *= 0.7
        else:
            lam *= 2.0
            if lam > 1e6:
                break


    for i in range(n_params // 3):
        params[3 * i] = max(params[3 * i], 1e-6)
        params[3 * i + 1] = max(params[3 * i + 1], 1e-6)
        params[3 * i + 2] = np.clip(params[3 * i + 2], -0.99, 0.99)

    return params, residuals
