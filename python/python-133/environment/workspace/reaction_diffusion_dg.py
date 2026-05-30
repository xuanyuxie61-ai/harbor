
import numpy as np
import math
from typing import Tuple, Callable, Optional


def jacobi_polynomial(x: np.ndarray, alpha: float, beta: float, N: int) -> np.ndarray:
    x = np.asarray(x).flatten()
    PL = np.zeros((N + 1, x.size))

    gamma0 = (2.0 ** (alpha + beta + 1.0) / (alpha + beta + 1.0)
              * math.gamma(alpha + 1.0) * math.gamma(beta + 1.0)
              / math.gamma(alpha + beta + 1.0))
    PL[0, :] = 1.0 / np.sqrt(gamma0)

    if N == 0:
        return PL[0, :]

    gamma1 = (alpha + 1.0) * (beta + 1.0) / (alpha + beta + 3.0) * gamma0
    PL[1, :] = ((alpha + beta + 2.0) * x / 2.0 + (alpha - beta) / 2.0) / np.sqrt(gamma1)

    if N == 1:
        return PL[1, :]

    aold = 2.0 / (2.0 + alpha + beta) * np.sqrt(
        (alpha + 1.0) * (beta + 1.0) / (alpha + beta + 3.0))

    for i in range(1, N):
        h1 = 2.0 * i + alpha + beta
        anew = (2.0 / (h1 + 2.0)
                * np.sqrt((i + 1.0) * (i + 1.0 + alpha + beta)
                          * (i + 1.0 + alpha) * (i + 1.0 + beta)
                          / (h1 + 1.0) / (h1 + 3.0)))
        bnew = -(alpha ** 2 - beta ** 2) / h1 / (h1 + 2.0)
        PL[i + 1, :] = (1.0 / anew) * (
            -aold * PL[i - 1, :] + (x - bnew) * PL[i, :])
        aold = anew

    return PL[N, :]


def vandermonde_1d(N: int, r: np.ndarray) -> np.ndarray:
    r = np.asarray(r).flatten()
    V = np.zeros((r.size, N + 1))
    for j in range(N + 1):
        V[:, j] = jacobi_polynomial(r, 0.0, 0.0, j)
    return V


def jacobi_gauss_lobatto(alpha: float, beta: float, N: int) -> np.ndarray:
    if N == 1:
        return np.array([-1.0, 1.0])

    xint, _ = jacobi_gauss_quadrature(alpha + 1.0, beta + 1.0, N - 2)
    x = np.concatenate(([-1.0], xint, [1.0]))
    return x


def jacobi_gauss_quadrature(alpha: float, beta: float, N: int) -> Tuple[np.ndarray, np.ndarray]:
    if N == 0:
        return np.array([-(alpha - beta) / (alpha + beta + 2.0)]), np.array([2.0])


    h1 = 2.0 * np.arange(N + 1) + alpha + beta
    J = np.diag(-0.5 * (alpha ** 2 - beta ** 2) / (h1 + 2.0) / h1)


    buf = np.sqrt(
        (np.arange(1, N + 1) + alpha) * (np.arange(1, N + 1) + beta)
        * (np.arange(1, N + 1) + alpha + beta)
        * np.arange(1, N + 1)
        / (h1[1:] + 1.0) / (h1[1:] - 1.0)
    ) / (h1[1:] / 2.0)

    if N >= 1 and buf.size > 0:
        n_sub = min(buf.size, J.shape[0] - 1)
        if n_sub > 0:
            J += np.diag(buf[:n_sub], k=1) + np.diag(buf[:n_sub], k=-1)

    eigvals, eigvecs = np.linalg.eigh(J)
    x = eigvals
    w = (eigvecs[0, :] ** 2) * (2.0 ** (alpha + beta + 1.0))
    w *= math.gamma(alpha + 1.0) * math.gamma(beta + 1.0)
    w /= math.gamma(alpha + beta + 1.0)


    w = np.maximum(w, 1.0e-16)
    return x, w


def d_matrix_1d(N: int, r: np.ndarray, V: Optional[np.ndarray] = None) -> np.ndarray:
    if V is None:
        V = vandermonde_1d(N, r)
    Vr = np.zeros_like(V)
    for j in range(N + 1):

        if j > 0:
            Vr[:, j] = jacobi_polynomial(r, 1.0, 1.0, j - 1) * np.sqrt(j * (j + 1.0))
    D = Vr @ np.linalg.inv(V)
    return D


class DG1DReactionDiffusion:

    def __init__(self,
                 N: int = 4,
                 K: int = 10,
                 x_left: float = 0.0,
                 x_right: float = 1.0,
                 v: float = 0.01,
                 D_diff: float = 1.0e-4,
                 ):
        self.N = N
        self.K = K
        self.x_left = x_left
        self.x_right = x_right
        self.v = v
        self.D_diff = D_diff


        self.r = jacobi_gauss_lobatto(0.0, 0.0, N)
        self.V = vandermonde_1d(N, self.r)
        self.V_inv = np.linalg.inv(self.V)
        self.Dr = d_matrix_1d(N, self.r, self.V)


        self.M = self.V_inv.T @ self.V_inv


        self._build_mesh()
        self._build_operators()

    def _build_mesh(self) -> None:
        Np = self.N + 1
        K = self.K

        vx = np.linspace(self.x_left, self.x_right, K + 1)
        self.vx = vx


        x = np.zeros((Np, K))
        for k in range(K):
            x[:, k] = 0.5 * (1.0 - self.r) * vx[k] + 0.5 * (1.0 + self.r) * vx[k + 1]
        self.x = x


        self.J = 0.5 * (vx[1:] - vx[:-1])
        self.rx = 1.0 / self.J

    def _build_operators(self) -> None:
        Np = self.N + 1
        K = self.K
        Nfaces = 2


        self.vmapM = np.zeros((Nfaces, K), dtype=int)
        self.vmapP = np.zeros((Nfaces, K), dtype=int)
        self.nx = np.zeros((Nfaces, K))

        for k in range(K):
            self.vmapM[0, k] = k * Np
            self.vmapM[1, k] = (k + 1) * Np - 1
            self.vmapP[0, k] = (k - 1) * Np + Np - 1 if k > 0 else k * Np
            self.vmapP[1, k] = (k + 1) * Np if k < K - 1 else (k + 1) * Np - 1
            self.nx[0, k] = -1.0
            self.nx[1, k] = 1.0


        self.mapI = 0
        self.mapO = Nfaces * K - 1
        self.vmapI = 0
        self.vmapO = Np * K - 1


        Emat = np.zeros((Np, Nfaces))
        Emat[0, 0] = 1.0
        Emat[-1, 1] = 1.0
        self.LIFT = self.V @ (self.V.T @ Emat)


        self.Fscale = np.ones((Nfaces, K)) / self.J[np.newaxis, :]

    def _compute_flux_diffusion(self, u: np.ndarray, q: np.ndarray,
                                 flux_type: str = 'u') -> np.ndarray:
        Np = self.N + 1
        K = self.K
        Nfaces = 2
        du = np.zeros((Nfaces, K))


        u_faces = np.zeros((Nfaces, K))
        for k in range(K):
            u_faces[0, k] = u[0, k]
            u_faces[1, k] = u[-1, k]


        for k in range(K):
            for f in range(Nfaces):
                neighbor_k = k - 1 if f == 0 else k + 1
                if 0 <= neighbor_k < K:
                    neighbor_f = 1 if f == 0 else 0
                    uP = u_faces[neighbor_f, neighbor_k]
                else:

                    uP = -u_faces[f, k]

                du[f, k] = 0.5 * (u_faces[f, k] - uP)

        return du

    def rhs(self, u: np.ndarray,
            source_func: Callable[[np.ndarray, float], np.ndarray],
            time: float) -> np.ndarray:
        Np = self.N + 1
        K = self.K


        q_ref = self.Dr @ u
        q = np.zeros_like(q_ref)
        for k in range(K):
            q[:, k] = q_ref[:, k] * self.rx[k]


        du = self._compute_flux_diffusion(u, q)


        Ldu = np.zeros_like(u)
        for k in range(K):
            for i in range(Np):
                Ldu[i, k] = (self.LIFT[i, 0] * du[0, k] * self.Fscale[0, k]
                             + self.LIFT[i, 1] * du[1, k] * self.Fscale[1, k])


        q_corrected = np.zeros_like(q)
        for k in range(K):
            q_corrected[:, k] = q[:, k] - Ldu[:, k]


        dq_ref = self.Dr @ q_corrected
        dq = np.zeros_like(dq_ref)
        for k in range(K):
            dq[:, k] = dq_ref[:, k] * self.rx[k]


        dq_flux = self._compute_flux_diffusion(q_corrected, np.zeros_like(q_corrected))
        Ldq = np.zeros_like(u)
        for k in range(K):
            for i in range(Np):
                Ldq[i, k] = (self.LIFT[i, 0] * dq_flux[0, k] * self.Fscale[0, k]
                             + self.LIFT[i, 1] * dq_flux[1, k] * self.Fscale[1, k])


        u_faces = np.array([u[0, :], u[-1, :]])
        du_conv = np.zeros((2, K))
        for k in range(K):
            for f in range(2):
                neighbor_k = k - 1 if f == 0 else k + 1
                if 0 <= neighbor_k < K:
                    neighbor_f = 1 if f == 0 else 0
                    uP = u_faces[neighbor_f, neighbor_k]
                else:
                    uP = 0.0

                nx = self.nx[f, k]
                v_n = self.v * nx
                du_conv[f, k] = 0.5 * (self.v * u_faces[f, k] + self.v * uP) * nx
                du_conv[f, k] += 0.5 * abs(v_n) * (u_faces[f, k] - uP)

        Lconv = np.zeros_like(u)
        for k in range(K):
            for i in range(Np):
                Lconv[i, k] = (self.LIFT[i, 0] * du_conv[0, k] * self.Fscale[0, k]
                               + self.LIFT[i, 1] * du_conv[1, k] * self.Fscale[1, k])


        conv_ref = self.v * (self.Dr @ u)
        conv = np.zeros_like(conv_ref)
        for k in range(K):
            conv[:, k] = conv_ref[:, k] * self.rx[k]


        R = source_func(self.x, time)
        R = np.reshape(R, (Np, K))

        rhsu = (-conv
                + self.D_diff * (dq - Ldq)
                - Lconv
                + R)


        rhsu = np.clip(rhsu, -1.0e6, 1.0e6)
        return rhsu

    def solve(self,
              u0: np.ndarray,
              final_time: float,
              source_func: Callable[[np.ndarray, float], np.ndarray],
              dt_factor: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
        Np = self.N + 1
        K = self.K
        u = u0.reshape((Np, K)).copy()


        x_global = np.zeros(Np * K)
        for k in range(K):
            x_global[k * Np:(k + 1) * Np] = self.x[:, k]

        x_global = np.sort(np.unique(np.round(x_global, 12)))
        n_global = x_global.size


        u_global = np.zeros(n_global)
        count = np.zeros(n_global)
        for k in range(K):
            for i in range(Np):
                idx = np.searchsorted(x_global, self.x[i, k])
                idx = min(idx, n_global - 1)
                u_global[idx] += u[i, k]
                count[idx] += 1
        count = np.maximum(count, 1)
        u_global /= count


        dx_min = np.min(np.diff(x_global))
        dt = dt_factor * dx_min ** 2 / max(self.D_diff, 1.0e-12)
        if abs(self.v) > 1.0e-12:
            dt = min(dt, dt_factor * dx_min / abs(self.v))
        n_steps = max(1, int(np.ceil(final_time / dt)))
        dt = final_time / n_steps

        def fd_rhs(u_g, t):
            rhs = np.zeros_like(u_g)
            dx = np.diff(x_global)

            for i in range(1, n_global - 1):
                dx_avg = 0.5 * (dx[i - 1] + dx[i])
                d2u = (u_g[i + 1] - u_g[i]) / dx[i] - (u_g[i] - u_g[i - 1]) / dx[i - 1]
                d2u /= dx_avg
                du_dx = (u_g[i + 1] - u_g[i - 1]) / (dx[i - 1] + dx[i])
                rhs[i] = self.D_diff * d2u - self.v * du_dx

            R = source_func(x_global, t)
            rhs += R

            rhs[0] = 0.0
            rhs[-1] = 0.0
            return rhs


        time = 0.0
        for _ in range(n_steps):
            k1 = fd_rhs(u_global, time)
            k2 = fd_rhs(u_global + 0.5 * dt * k1, time + 0.5 * dt)
            k3 = fd_rhs(u_global + 0.5 * dt * k2, time + 0.5 * dt)
            k4 = fd_rhs(u_global + dt * k3, time + dt)
            u_global += (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            u_global = np.maximum(u_global, 0.0)
            time += dt


        u_out = np.zeros((Np, K))
        for k in range(K):
            for i in range(Np):
                idx = np.searchsorted(x_global, self.x[i, k])
                idx = min(idx, n_global - 1)
                u_out[i, k] = u_global[idx]

        return self.x, u_out
