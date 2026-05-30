
import numpy as np
from typing import Callable, Tuple


def legendre_basis_2d() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    xi = np.array([-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0)])
    w = np.array([1.0, 1.0])



    nodes = np.array([-1.0, 0.0, 1.0])
    phi = np.zeros((3, 2))
    dphi = np.zeros((3, 2))
    for k in range(3):
        for gp in range(2):

            x = xi[gp]
            L = 1.0
            dL = 0.0
            for m in range(3):
                if m != k:
                    L *= (x - nodes[m]) / (nodes[k] - nodes[m])

            dx = 1e-8
            Lp = 1.0
            Lm = 1.0
            for m in range(3):
                if m != k:
                    Lp *= ((x + dx) - nodes[m]) / (nodes[k] - nodes[m])
                    Lm *= ((x - dx) - nodes[m]) / (nodes[k] - nodes[m])
            dL = (Lp - Lm) / (2.0 * dx)
            phi[k, gp] = L
            dphi[k, gp] = dL
    return phi, dphi, w


def assemble_dg_matrices(nel: int, K: float = 1.0,
                         penal: float = 10.0,
                         ss: float = -1.0) -> Tuple[np.ndarray, np.ndarray]:
    if nel < 1:
        raise ValueError("nel 必须至少为 1。")
    locdim = 3
    ndof = nel * locdim
    h = 1.0 / nel

    phi, dphi, wg = legendre_basis_2d()

    M = np.zeros((ndof, ndof))
    A = np.zeros((ndof, ndof))


    Mloc = np.zeros((locdim, locdim))
    for ii in range(locdim):
        for jj in range(locdim):
            s = 0.0
            for gp in range(2):
                s += wg[gp] * phi[ii, gp] * phi[jj, gp]
            Mloc[ii, jj] = s * (h / 2.0)


    Aloc_diff = np.zeros((locdim, locdim))
    for ii in range(locdim):
        for jj in range(locdim):
            s = 0.0
            for gp in range(2):
                s += wg[gp] * dphi[ii, gp] * dphi[jj, gp]
            Aloc_diff[ii, jj] = K * s * (2.0 / h)



    scale = 0.1
    Bmat = scale * np.array([
        [penal, 1.0 - penal, -2.0 + penal],
        [-ss - penal, -1.0 + ss - penal, 2.0 - ss - penal],
        [2.0 * ss + penal, 1.0 - 2.0 * ss - penal, -2.0 + 2.0 * ss + penal]
    ])
    Cmat = scale * np.array([
        [penal, -1.0 + penal, -2.0 + penal],
        [ss + penal, -1.0 + ss + penal, -2.0 + ss + penal],
        [2.0 * ss + penal, -1.0 + 2.0 * ss + penal, -2.0 + 2.0 * ss + penal]
    ])
    Dmat = scale * np.array([
        [-penal, -1.0 + penal, 2.0 - penal],
        [-ss - penal, -1.0 + ss + penal, 2.0 - ss - penal],
        [-2.0 * ss - penal, -1.0 + 2.0 * ss + penal, 2.0 - 2.0 * ss - penal]
    ])
    Emat = scale * np.array([
        [-penal, 1.0 - penal, 2.0 - penal],
        [ss + penal, -1.0 + ss + penal, -2.0 + ss + penal],
        [-2.0 * ss - penal, 1.0 - 2.0 * ss - penal, 2.0 - 2.0 * ss - penal]
    ])
    F0mat = scale * np.array([
        [penal, 2.0 - penal, -4.0 + penal],
        [-2.0 * ss - penal, -2.0 + 2.0 * ss + penal, 4.0 - 2.0 * ss - penal],
        [4.0 * ss + penal, 2.0 - 4.0 * ss - penal, -4.0 + 4.0 * ss + penal]
    ])
    FNmat = scale * np.array([
        [penal, -2.0 + penal, -4.0 + penal],
        [2.0 * ss + penal, -2.0 + 2.0 * ss + penal, -4.0 + 2.0 * ss + penal],
        [4.0 * ss + penal, -2.0 + 4.0 * ss + penal, -4.0 + 4.0 * ss + penal]
    ])


    for i in range(nel):
        base = i * locdim
        for ii in range(locdim):
            for jj in range(locdim):

                M[base + ii, base + jj] += Mloc[ii, jj]
                A[base + ii, base + jj] += Aloc_diff[ii, jj]


        if i == 0:
            for ii in range(locdim):
                for jj in range(locdim):
                    A[base + ii, base + jj] += (F0mat[ii, jj] + Cmat[ii, jj])
                    if nel > 1:
                        A[base + ii, base + locdim + jj] += Dmat[ii, jj]
        elif i == nel - 1:
            for ii in range(locdim):
                for jj in range(locdim):
                    A[base + ii, base + jj] += (FNmat[ii, jj] + Bmat[ii, jj])
                    A[base + ii, base - locdim + jj] += Emat[ii, jj]
        else:
            for ii in range(locdim):
                for jj in range(locdim):
                    A[base + ii, base + jj] += (Bmat[ii, jj] + Cmat[ii, jj])
                    A[base + ii, base - locdim + jj] += Emat[ii, jj]
                    A[base + ii, base + locdim + jj] += Dmat[ii, jj]

    return M, A


def solve_causal_diffusion_dg(nel: int = 8,
                               nsteps: int = 50,
                               dt: float = 0.001,
                               K: float = 1.0,
                               source_func: Callable = None,
                               u0_func: Callable = None) -> Tuple[np.ndarray, np.ndarray]:
    if dt <= 0.0:
        raise ValueError("时间步长 dt 必须为正。")
    if nsteps < 0:
        raise ValueError("时间步数必须非负。")

    M, A = assemble_dg_matrices(nel, K=K, penal=1.0, ss=-1.0)
    ndof = M.shape[0]




    h = 1.0 / nel
    locdim = 3
    b_dirichlet = np.zeros(ndof)


    b_dirichlet[0] = 0.0
    b_dirichlet[2] = 0.0

    b_dirichlet[ndof - 3] = 0.0
    b_dirichlet[ndof - 1] = 0.0


    if u0_func is None:
        u = np.zeros(ndof)
    else:
        u = np.zeros(ndof)
        for i in range(nel):
            base = i * locdim
            xmid = (i + 0.5) * h
            u[base] = u0_func(xmid)

    if source_func is None:

        def source_func(x, t):
            return 1.0 * np.exp(-((x - 0.3) ** 2) / 0.02) * np.exp(-t * 2.0) + \
                   0.5 * np.exp(-((x - 0.7) ** 2) / 0.02) * np.exp(-t * 2.0)

    t = 0.0
    t_history = [0.0]
    u_history = [u.copy()]


    LHS = M + dt * A

    LHS = LHS + 1e-8 * np.eye(ndof)

    for _ in range(nsteps):
        F = np.zeros(ndof)
        for i in range(nel):
            base = i * locdim
            xmid = (i + 0.5) * h
            fval = source_func(xmid, t)
            F[base] = fval

        rhs = M @ u + dt * (F + b_dirichlet)
        u = np.linalg.solve(LHS, rhs)
        t += dt
        t_history.append(t)
        u_history.append(u.copy())

    return np.array(t_history), np.array(u_history)


def demo():
    t_hist, u_hist = solve_causal_diffusion_dg(nel=8, nsteps=100, dt=0.001, K=1.0)
    print(f"[dg_causal_solver] DG 扩散求解完成: t_final={t_hist[-1]:.4f}, "
          f"u_max={np.max(np.abs(u_hist[-1])):.6e}")
    return t_hist, u_hist


if __name__ == "__main__":
    demo()
