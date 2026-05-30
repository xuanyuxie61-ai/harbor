
import numpy as np






def jacobi_p(x, alpha, beta, n):
    x = np.asarray(x, dtype=float)
    if n == 0:
        return np.ones_like(x)
    if n == 1:
        return 0.5 * (alpha - beta) + 0.5 * (alpha + beta + 2.0) * x


    pl = np.ones_like(x)
    p = 0.5 * (alpha - beta) + 0.5 * (alpha + beta + 2.0) * x
    for k in range(1, n):
        a1 = 2.0 * (k + 1.0) * (k + alpha + beta + 1.0) * (2.0 * k + alpha + beta)
        a2 = (2.0 * k + alpha + beta + 1.0) * (alpha * alpha - beta * beta)
        a3 = (2.0 * k + alpha + beta) * (2.0 * k + alpha + beta + 1.0) * (2.0 * k + alpha + beta + 2.0)
        a4 = 2.0 * (k + alpha) * (k + beta) * (2.0 * k + alpha + beta + 2.0)
        denom = a1
        if abs(denom) < 1e-14:
            denom = 1e-14
        p_new = ((a2 + a3 * x) * p - a4 * pl) / denom
        pl = p.copy()
        p = p_new
    return p


def grad_jacobi_p(x, alpha, beta, n):
    x = np.asarray(x, dtype=float)
    if n == 0:
        return np.zeros_like(x)
    return 0.5 * (alpha + beta + n + 1.0) * jacobi_p(x, alpha + 1.0, beta + 1.0, n - 1)






def jacobi_gl_nodes(alpha, beta, n):
    if n == 0:
        return np.array([-1.0, 1.0])
    if n == 1:
        return np.array([-1.0, 0.0, 1.0])



    r = -np.cos(np.pi * np.arange(n + 1) / n)

    for i in range(1, n):
        x0 = r[i]
        for _ in range(50):
            p = jacobi_p(x0, alpha + 1.0, beta + 1.0, n - 1)
            dp = grad_jacobi_p(x0, alpha + 1.0, beta + 1.0, n - 1)
            if abs(dp) < 1e-14:
                break
            dx = p / dp
            x0 = x0 - dx
            if abs(dx) < 1e-14:
                break
        r[i] = x0
    r[0] = -1.0
    r[-1] = 1.0
    return r






def vandermonde_1d(n, r):
    r = np.asarray(r, dtype=float)
    Np = len(r)
    V = np.zeros((Np, n + 1), dtype=float)
    for j in range(n + 1):
        V[:, j] = jacobi_p(r, 0.0, 0.0, j)
    return V


def d_matrix_1d(n, r, V):
    r = np.asarray(r, dtype=float)
    Np = len(r)
    Vx = np.zeros((Np, n + 1), dtype=float)
    for j in range(n + 1):
        Vx[:, j] = grad_jacobi_p(r, 0.0, 0.0, j)
    D = np.linalg.solve(V.T, Vx.T).T
    return D






def dg_diffusion_solve_1d(z_min, z_max, n_elements, poly_order,
                          diffusivity_func, absorption_func, source_func,
                          robin_left=(0.5, 0.0), robin_right=(0.5, 0.0)):
    if n_elements < 1 or poly_order < 1:
        raise ValueError("n_elements and poly_order must be >= 1.")


    r = jacobi_gl_nodes(0.0, 0.0, poly_order)
    Np = len(r)
    V = vandermonde_1d(poly_order, r)
    Dr = d_matrix_1d(poly_order, r, V)




    invV = np.linalg.inv(V)
    M = np.dot(invV, invV.T)


    va = np.linspace(z_min, z_max, n_elements + 1)[:-1]
    vb = np.linspace(z_min, z_max, n_elements + 1)[1:]

    total_dof = n_elements * Np
    A = np.zeros((total_dof, total_dof), dtype=float)
    b_vec = np.zeros(total_dof, dtype=float)
    z_global = np.zeros(total_dof, dtype=float)

    for k in range(n_elements):
        h = vb[k] - va[k]

        x_local = va[k] + 0.5 * (r + 1.0) * h

        J = h / 2.0

        D_local = Dr / J
        M_local = M * J


        D_vals = np.array([diffusivity_func(xi) for xi in x_local])
        mu_vals = np.array([absorption_func(xi) for xi in x_local])
        S_vals = np.array([source_func(xi) for xi in x_local])



        K_local = np.zeros((Np, Np), dtype=float)
        for q in range(Np):
            wq = M_local[q, q]
            for i in range(Np):
                for j in range(Np):
                    K_local[i, j] += D_vals[q] * D_local[q, i] * D_local[q, j] * wq
                    if i == j:
                        K_local[i, j] += mu_vals[q] * wq

        f_local = S_vals * np.diag(M_local)


        dof_start = k * Np
        A[dof_start:dof_start + Np, dof_start:dof_start + Np] += K_local
        b_vec[dof_start:dof_start + Np] += f_local
        z_global[dof_start:dof_start + Np] = x_local


    for k in range(n_elements - 1):
        dof_k_end = (k + 1) * Np - 1
        dof_kp1_start = (k + 1) * Np

        dz = z_global[dof_kp1_start] - z_global[dof_k_end]
        if dz < 1e-14:
            dz = 1e-14
        penalty = 1.0 / dz
        A[dof_k_end, dof_k_end] += penalty
        A[dof_k_end, dof_kp1_start] -= penalty
        A[dof_kp1_start, dof_k_end] -= penalty
        A[dof_kp1_start, dof_kp1_start] += penalty



    aL, gL = robin_left
    D0 = diffusivity_func(z_global[0])
    A[0, 0] += aL
    b_vec[0] += gL


    aR, gR = robin_right
    Dn = diffusivity_func(z_global[-1])
    A[-1, -1] += aR
    b_vec[-1] += gR


    phi = np.linalg.solve(A, b_vec)
    return z_global, phi






def solve_tissue_diffusion_dg(layer_boundaries, layer_optical_props,
                              source_profile='uniform', poly_order=4, n_elements_per_layer=4):
    boundaries = np.asarray(layer_boundaries, dtype=float)
    n_layers = len(boundaries) - 1

    def D_func(z):


        raise NotImplementedError("Hole 2: D_func in solve_tissue_diffusion_dg needs to be implemented.")

    def mu_a_func(z):
        for i in range(n_layers):
            if boundaries[i] <= z <= boundaries[i + 1]:
                return layer_optical_props[i]['mu_a']
        return 0.0

    def S_func(z):
        if source_profile == 'uniform':
            return 1.0 if boundaries[0] <= z <= boundaries[-1] else 0.0
        elif source_profile == 'gaussian':
            z0 = (boundaries[0] + boundaries[-1]) / 2.0
            sigma = (boundaries[-1] - boundaries[0]) / 4.0
            return np.exp(-0.5 * ((z - z0) / sigma) ** 2)
        else:
            return 1.0

    n_elements = n_layers * n_elements_per_layer
    z, phi = dg_diffusion_solve_1d(
        boundaries[0], boundaries[-1], n_elements, poly_order,
        D_func, mu_a_func, S_func,
        robin_left=(0.5, 1.0), robin_right=(0.5, 0.0)
    )
    return z, phi
