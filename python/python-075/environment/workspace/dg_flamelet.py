
import numpy as np


def jacobi_polynomial(x, alpha, beta, N):
    x = np.atleast_1d(x)
    if N == 0:
        return np.ones_like(x)


    from math import gamma as math_gamma
    gamma0 = (2.0**(alpha + beta + 1.0) / (alpha + beta + 1.0)
              * math_gamma(alpha + 1.0) * math_gamma(beta + 1.0)
              / math_gamma(alpha + beta + 1.0))


    P = np.zeros((N + 1, len(x)))
    P[0, :] = 1.0 / np.sqrt(gamma0)

    if N >= 1:
        gamma1 = (alpha + 1.0) * (beta + 1.0) / (alpha + beta + 3.0) * gamma0
        P[1, :] = ((alpha + beta + 2.0) * x / 2.0 + (alpha - beta) / 2.0) / np.sqrt(gamma1)

    aold = 2.0 / (2.0 + alpha + beta) * np.sqrt((alpha + 1.0) * (beta + 1.0) / (alpha + beta + 3.0))

    for i in range(1, N):
        h1 = 2.0 * i + alpha + beta
        anew = (2.0 / (h1 + 2.0)
                * np.sqrt((i + 1.0) * (i + 1.0 + alpha + beta)
                          * (i + 1.0 + alpha) * (i + 1.0 + beta)
                          / ((h1 + 1.0) * (h1 + 3.0))))
        bnew = -(alpha**2 - beta**2) / (h1 * (h1 + 2.0))
        P[i + 1, :] = (1.0 / anew) * (-aold * P[i - 1, :] + (x - bnew) * P[i, :])
        aold = anew

    return P[N, :]


def vandermonde_1d(N, r):
    r = np.atleast_1d(r)
    V = np.zeros((len(r), N + 1))
    for j in range(N + 1):
        V[:, j] = jacobi_polynomial(r, 0.0, 0.0, j)
    return V


def jacobi_gauss_lobatto(alpha, beta, N):
    if N == 0:
        return np.array([-1.0, 1.0]), np.array([1.0, 1.0])
    if N == 1:
        return np.array([-1.0, 0.0, 1.0]), np.array([1.0/3.0, 4.0/3.0, 1.0/3.0])


    from numpy.linalg import eigvalsh

    n = N - 1
    diag = np.zeros(n)
    offdiag = np.zeros(n - 1)
    for i in range(n):
        diag[i] = (beta**2 - alpha**2) / ((2.0 * i + alpha + beta + 2.0)
                                          * (2.0 * i + alpha + beta + 4.0))
    for i in range(n - 1):
        num = 4.0 * (i + 1.0) * (i + 1.0 + alpha + beta + 1.0) \
              * (i + 1.0 + alpha) * (i + 1.0 + beta)
        den = ((2.0 * i + alpha + beta + 2.0)**2
               * (2.0 * i + alpha + beta + 3.0)
               * (2.0 * i + alpha + beta + 1.0))
        offdiag[i] = np.sqrt(num / den)

    J = np.diag(diag) + np.diag(offdiag, 1) + np.diag(offdiag, -1)
    x_int = eigvalsh(J)
    x = np.concatenate([[-1.0], np.sort(x_int), [1.0]])


    V = vandermonde_1d(N, x)
    w = np.zeros(N + 1)


    rhs = np.zeros(N + 1)
    rhs[0] = np.sqrt(2.0)
    w = np.linalg.solve(V.T, rhs)
    w = w**2
    return x, w


class DGFlameletSolver:

    def __init__(self, N, K, xmin, xmax, D, bc_type='neumann'):
        self.N = N
        self.K = K
        self.xmin = xmin
        self.xmax = xmax
        self.D = D
        self.bc_type = bc_type


        self.r, self.w = jacobi_gauss_lobatto(0.0, 0.0, N)
        self.Np = N + 1


        self.V = vandermonde_1d(N, self.r)
        self.Vinv = np.linalg.inv(self.V)



        dV = np.zeros((self.Np, self.Np))
        for j in range(self.Np):

            if j == 0:
                dV[:, j] = 0.0
            else:

                dV[:, j] = jacobi_polynomial(self.r, 1.0, 1.0, j - 1) * np.sqrt(j * (j + 1.0))
        self.Dr = np.dot(dV, self.Vinv)


        self.M = np.diag(self.w)
        self.Minv = np.diag(1.0 / self.w)


        self.dx_elem = (xmax - xmin) / K


        self.x = np.zeros((self.Np, K))
        for k in range(K):
            self.x[:, k] = xmin + (k + 0.5) * self.dx_elem + 0.5 * self.dx_elem * self.r


        self._build_maps()



        Emat = np.zeros((self.Np, 2))
        V_face = vandermonde_1d(N, np.array([-1.0, 1.0]))
        Emat = np.dot(self.Vinv.T, V_face.T)
        self.LIFT = np.dot(self.V, np.dot(self.V.T, Emat))


        self.rk4a = np.array([0.0,
                              -567301805773.0 / 1357537059087.0,
                              -2404267990393.0 / 2016746695238.0,
                              -3550918686646.0 / 2091501179385.0,
                              -1275806237668.0 / 842570457699.0])
        self.rk4b = np.array([1432997174477.0 / 9575080441755.0,
                              5161836677717.0 / 13612068292357.0,
                              1720146321549.0 / 2090206949498.0,
                              3134564353537.0 / 4481467310338.0,
                              2277821191437.0 / 14882151754819.0])
        self.rk4c = np.array([0.0,
                              1432997174477.0 / 9575080441755.0,
                              2526269341429.0 / 6820363962896.0,
                              2006345519317.0 / 3224310063776.0,
                              2802321613138.0 / 2924317926251.0])

    def _build_maps(self):
        self.vmapM = np.zeros((2, self.K), dtype=int)
        self.vmapP = np.zeros((2, self.K), dtype=int)
        self.mapI = 0
        self.mapO = 1

        for k in range(self.K):
            self.vmapM[0, k] = k * self.Np
            self.vmapM[1, k] = (k + 1) * self.Np - 1

        for k in range(self.K):
            if k == 0:
                self.vmapP[0, k] = self.vmapM[1, self.K - 1]
            else:
                self.vmapP[0, k] = self.vmapM[1, k - 1]

            if k == self.K - 1:
                self.vmapP[1, k] = self.vmapM[0, 0]
            else:
                self.vmapP[1, k] = self.vmapM[0, k + 1]

        self.vmapM = self.vmapM.flatten()
        self.vmapP = self.vmapP.flatten()

    def compute_rhs(self, u, source_func=None, t=0.0):

        u_local = u.reshape((self.Np, self.K), order='F')


        ux = np.dot(self.Dr, u_local)


        ux = (2.0 / self.dx_elem) * ux


        du = np.zeros(2 * self.K)
        du[:] = (u[self.vmapM] - u[self.vmapP]) / 2.0


        if self.bc_type == 'neumann':
            du[self.mapI] = 0.0
            du[self.mapO] = 0.0
        elif self.bc_type == 'dirichlet':
            uin = -u[self.vmapI]
            du[self.mapI] = (u[self.vmapI] - uin) / 2.0
            uout = -u[self.vmapO]
            du[self.mapO] = (u[self.vmapO] - uout) / 2.0


        q = self.D * ux - np.dot(self.LIFT, du.reshape((2, self.K), order='F'))


        qx = np.dot(self.Dr, q)
        qx = (2.0 / self.dx_elem) * qx


        dq = np.zeros(2 * self.K)
        dq[:] = (q.flatten('F')[self.vmapM] - q.flatten('F')[self.vmapP]) / 2.0

        if self.bc_type == 'neumann':
            dq[self.mapI] = 0.0
            dq[self.mapO] = 0.0
        elif self.bc_type == 'dirichlet':
            qin = q.flatten('F')[self.vmapI]
            dq[self.mapI] = (q.flatten('F')[self.vmapI] - qin) / 2.0
            qout = q.flatten('F')[self.vmapO]
            dq[self.mapO] = (q.flatten('F')[self.vmapO] - qout) / 2.0


        rhs = self.D * qx - np.dot(self.LIFT, dq.reshape((2, self.K), order='F'))
        rhs = rhs.flatten('F')


        if source_func is not None:
            x_flat = self.x.flatten('F')
            src = source_func(x_flat, t)
            rhs = rhs + src

        return rhs

    def step(self, u, dt, source_func=None, t=0.0):
        resu = np.zeros_like(u)
        time = t
        for intrk in range(5):
            timelocal = time + self.rk4c[intrk] * dt
            rhsu = self.compute_rhs(u, source_func, timelocal)

            rhsu = np.clip(rhsu, -1e6, 1e6)
            resu = self.rk4a[intrk] * resu + dt * rhsu
            resu = np.clip(resu, -1e6, 1e6)
            u = u + self.rk4b[intrk] * resu
            u = np.clip(u, -1e3, 1e3)
        return u

    def solve_steady_flamelet(self, source_func, max_iter=1000, tol=1e-8):
        u = np.zeros(self.Np * self.K)
        dt = 0.25 * (self.dx_elem / self.Np) ** 2 / max(self.D, 1e-12)
        for _ in range(max_iter):
            u_new = self.step(u, dt, source_func)
            if np.linalg.norm(u_new - u) < tol:
                return u_new
            u = u_new
        return u
