"""
nonlinear_inverse_solver.py

Inverse problem solvers for quantitative OCT tissue characterization.
Reconstructs tissue optical properties (mu_a, mu_s) from simulated A-scan
measurements using Newton-GMRES and CG iterative methods.

Adapted from kelley (Tim Kelley, SIAM 2004):
- nsol: hybrid Newton-Shamanskii-Chord solver
- gmres: Saad-Schultz GMRES with Givens rotations
- cg: conjugate gradient

Scientific formulation:
  Given measured A-scan A_meas(z), find optical parameters p such that
      F(p) = A_sim(p) - A_meas = 0
  where A_sim(p) is the forward model (DG diffusion + spectral integration).
"""

import numpy as np


# ---------------------------------------------------------------------------
# GMRES with Givens rotations (from gmres)
# ---------------------------------------------------------------------------

def givapp(c, s, vin, k):
    """
    Apply k Givens rotations to vector vin.

    Parameters
    ----------
    c, s : array_like
        Cosine and sine of rotations.
    vin : ndarray
        Input vector.
    k : int
        Number of rotations to apply.

    Returns
    -------
    vout : ndarray
    """
    vout = np.asarray(vin, dtype=float).copy()
    for i in range(k):
        w1 = c[i] * vout[i] - s[i] * vout[i + 1]
        w2 = s[i] * vout[i] + c[i] * vout[i + 1]
        vout[i] = w1
        vout[i + 1] = w2
    return vout


def gmres_solve(x0, b, atv, params):
    """
    GMRES solver for A x = b with matrix-free operator atv(x) = A @ x.

    Parameters
    ----------
    x0 : ndarray
        Initial guess.
    b : ndarray
        Right-hand side.
    atv : callable
        Function atv(x) returning A @ x.
    params : array_like
        [errtol, kmax, reorth] where reorth:
        1 = Brown/Hindmarsh condition (default)
        2 = never
        3 = always

    Returns
    -------
    x : ndarray
        Solution.
    error : list
        Residual norms.
    total_iters : int
    """
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

        # Modified Gram-Schmidt
        for j in range(k):
            h[j, k - 1] = np.dot(v[:, j], v[:, k])
            v[:, k] = v[:, k] - h[j, k - 1] * v[:, j]
        h[k, k - 1] = np.linalg.norm(v[:, k])
        normav2 = h[k, k - 1]

        # Reorthogonalize
        if (reorth == 1 and normav + 0.001 * normav2 == normav) or reorth == 3:
            for j in range(k):
                hr = np.dot(v[:, j], v[:, k])
                h[j, k - 1] += hr
                v[:, k] = v[:, k] - hr * v[:, j]
            h[k, k - 1] = np.linalg.norm(v[:, k])

        if h[k, k - 1] > 1e-14:
            v[:, k] = v[:, k] / h[k, k - 1]

        # Givens rotations
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


# ---------------------------------------------------------------------------
# Conjugate Gradient (from cg)
# ---------------------------------------------------------------------------

def cg_solve(x0, b, atv, params):
    """
    Conjugate gradient solver for symmetric positive definite A x = b.

    Parameters
    ----------
    x0 : ndarray
        Initial guess.
    b : ndarray
        RHS.
    atv : callable
        atv(x) = A @ x.
    params : array_like
        [errtol, kmax].

    Returns
    -------
    x : ndarray
    err : list
    total_iters : int
    """
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


# ---------------------------------------------------------------------------
# Newton solver with finite-difference Jacobian (from nsol)
# ---------------------------------------------------------------------------

def diffjac(x, f, f0, eps=1e-7):
    """
    Finite-difference Jacobian approximation.

    Returns L, U such that J ~ L @ U (via LU decomposition).

    Parameters
    ----------
    x : ndarray
    f : callable
    f0 : ndarray
        f(x).
    eps : float

    Returns
    -------
    l, u : ndarray
        LU factors.
    """
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
    # LU decomposition
    l, u = np.linalg.qr(J)  # Use QR for stability
    return l, u


def nsol_solve(x, f, tol, parms=None):
    """
    Hybrid Newton-Shamanskii-Chord solver for nonlinear systems F(x)=0.

    Parameters
    ----------
    x : ndarray
        Initial iterate.
    f : callable
        f(x) -> ndarray.
    tol : array_like
        [atol, rtol].
    parms : array_like, optional
        [maxit, isham, rsham].

    Returns
    -------
    sol : ndarray
        Solution.
    it_hist : list
        Residual norms.
    ierr : int
        0 = success, 1 = failure.
    """
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

        # Solve J step = -f0 using QR factors
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


# ---------------------------------------------------------------------------
# OCT inverse problem: reconstruct optical properties from A-scan
# ---------------------------------------------------------------------------

def build_forward_model_oct(z_scan, layer_boundaries, k_min, k_max, n_gl=32):
    """
    Build a forward model function that computes A-scan from optical parameters.

    Parameters
    ----------
    z_scan : ndarray
        Depth sampling points.
    layer_boundaries : array_like
        Layer z boundaries.
    k_min, k_max : float
        Spectral range.
    n_gl : int
        Quadrature order.

    Returns
    -------
    forward_func : callable
        forward_func(params) -> A_scan.
    """
    from spectral_integration import integrate_depth_resolved_signal
    from oct_physics import source_spectrum_gaussian
    from scipy.special import erf

    boundaries = np.asarray(layer_boundaries, dtype=float)
    n_layers = len(boundaries) - 1

    def forward_func(params):
        """
        params = [mu_a_0, mu_s_0, g_0, mu_a_1, mu_s_1, g_1, ...]
        """
        params = np.asarray(params, dtype=float)
        if len(params) != 3 * n_layers:
            raise ValueError(f"Expected {3*n_layers} parameters, got {len(params)}")

        def reflectivity_func(k, z):
            # Find layer
            layer = 0
            for i in range(n_layers):
                if boundaries[i] <= z < boundaries[i + 1]:
                    layer = i
                    break
            idx = 3 * layer
            mu_a = params[idx]
            mu_s = params[idx + 1]
            g = params[idx + 2]
            # Simple reflectivity model: backscattering ~ mu_s * exp(-2 mu_t z)
            mu_t = mu_a + mu_s
            # Use error function for smooth transition at boundaries
            refl = mu_s * np.exp(-2.0 * mu_t * z)
            return refl

        A = integrate_depth_resolved_signal(reflectivity_func, z_scan, k_min, k_max, n_gl)
        return A

    return forward_func


def inverse_problem_oct(z_scan, A_measured, layer_boundaries, param_guess,
                        k_min, k_max, max_iter=20):
    """
    Solve inverse problem: reconstruct [mu_a, mu_s, g] per layer from A-scan.

    Uses Levenberg-Marquardt with GMRES for linear subproblems.

    Parameters
    ----------
    z_scan : ndarray
        Depth coordinates.
    A_measured : ndarray
        Measured A-scan.
    layer_boundaries : array_like
    param_guess : ndarray
        Initial parameter guess.
    k_min, k_max : float
        Spectral range.
    max_iter : int

    Returns
    -------
    params : ndarray
        Reconstructed parameters.
    residuals : list
        Residual history.
    """
    forward_func = build_forward_model_oct(z_scan, layer_boundaries, k_min, k_max)
    n_params = len(param_guess)
    n_data = len(z_scan)

    def objective(p):
        return forward_func(p) - A_measured

    def jacobian_atv(p, vec):
        """Matrix-free Jacobian-vector product J @ vec."""
        eps = 1e-6
        return (objective(p + eps * vec) - objective(p - eps * vec)) / (2.0 * eps)

    params = np.asarray(param_guess, dtype=float).copy()
    residuals = []
    lam = 0.01  # Levenberg-Marquardt damping

    for it in range(max_iter):
        res = objective(params)
        rnorm = np.linalg.norm(res)
        residuals.append(rnorm)
        if rnorm < 1e-4:
            break

        # Normal equations: (J^T J + lam I) delta = -J^T res
        # Use CG on normal equations
        def normal_atv(vec):
            # Approximate J^T J vec + lam vec
            jv = jacobian_atv(params, vec)
            # Approximate J^T by finite differences on adjoint
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

    # Clamp parameters to physical ranges
    for i in range(n_params // 3):
        params[3 * i] = max(params[3 * i], 1e-6)      # mu_a > 0
        params[3 * i + 1] = max(params[3 * i + 1], 1e-6)  # mu_s > 0
        params[3 * i + 2] = np.clip(params[3 * i + 2], -0.99, 0.99)  # g

    return params, residuals
