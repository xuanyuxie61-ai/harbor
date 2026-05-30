
import numpy as np


def compute_etdrk4_coefficients(L: np.ndarray, dt: float, M: int = 16) -> dict:
    N = len(L)
    E = np.exp(dt * L)
    E2 = np.exp(dt * L / 2.0)


    Q = np.zeros(N, dtype=complex)
    f1 = np.zeros(N, dtype=complex)
    f2 = np.zeros(N, dtype=complex)
    f3 = np.zeros(N, dtype=complex)

    for i in range(N):
        if abs(L[i]) < 1e-10:

            Q[i] = dt ** 2 / 2.0 - dt ** 3 * L[i] / 6.0
            f1[i] = dt ** 3 / 3.0 - dt ** 4 * L[i] / 8.0
            f2[i] = dt ** 3 / 6.0 - dt ** 4 * L[i] / 24.0
            f3[i] = dt ** 3 / 6.0 - dt ** 4 * L[i] / 12.0
        else:

            r = np.exp(1j * np.pi * (np.arange(M) + 0.5) / M)
            z = dt * L[i] + r
            Q[i] = dt * np.mean((np.exp(z / 2.0) - 1.0) / z)
            f1[i] = dt * np.mean((-4.0 - z + np.exp(z) * (4.0 - 3.0 * z + z ** 2)) / z ** 3)
            f2[i] = dt * np.mean((2.0 + z + np.exp(z) * (-2.0 + z)) / z ** 3)
            f3[i] = dt * np.mean((-4.0 - 3.0 * z - z ** 2 + np.exp(z) * (4.0 - z)) / z ** 3)

    return {
        'E': E,
        'E2': E2,
        'Q': Q,
        'f1': f1,
        'f2': f2,
        'f3': f3,
    }


class ETDRK4Solver:

    def __init__(
        self,
        nx: int,
        ny: int,
        Lx: float,
        Ly: float,
        D: list[float],
        vx: float = 0.0,
        vy: float = 0.0,
        dt: float = 0.01,
        n_fields: int = 6
    ):
        self.nx = nx
        self.ny = ny
        self.Lx = Lx
        self.Ly = Ly
        self.dt = dt
        self.n_fields = n_fields
        self.D = D
        self.vx = vx
        self.vy = vy


        self.kx = 2.0 * np.pi * np.fft.fftfreq(nx, d=Lx / nx)
        self.ky = 2.0 * np.pi * np.fft.fftfreq(ny, d=Ly / ny)
        self.KX, self.KY = np.meshgrid(self.kx, self.ky, indexing='ij')


        self.L = []
        self.coeffs = []
        for i in range(n_fields):
            Li = -D[i] * (self.KX ** 2 + self.KY ** 2) - 1j * (vx * self.KX + vy * self.KY)
            self.L.append(Li)
            self.coeffs.append(compute_etdrk4_coefficients(Li.ravel(), dt))

    def step(self, u: np.ndarray, nonlinear_func) -> np.ndarray:
        n_fields = self.n_fields
        nx, ny = self.nx, self.ny


        v = np.zeros((n_fields, nx, ny), dtype=complex)
        for i in range(n_fields):
            v[i] = np.fft.fft2(u[i])


        N_phys = nonlinear_func(u)
        Nv = np.zeros((n_fields, nx, ny), dtype=complex)
        for i in range(n_fields):
            Nv[i] = np.fft.fft2(N_phys[i])


        a = np.zeros((n_fields, nx, ny), dtype=complex)
        b = np.zeros((n_fields, nx, ny), dtype=complex)
        c = np.zeros((n_fields, nx, ny), dtype=complex)
        v_new = np.zeros((n_fields, nx, ny), dtype=complex)

        for i in range(n_fields):
            Li = self.L[i].ravel()
            cdict = self.coeffs[i]
            E = cdict['E'].reshape(nx, ny)
            E2 = cdict['E2'].reshape(nx, ny)
            Q = cdict['Q'].reshape(nx, ny)
            f1 = cdict['f1'].reshape(nx, ny)
            f2 = cdict['f2'].reshape(nx, ny)
            f3 = cdict['f3'].reshape(nx, ny)

            vi = v[i]
            Nvi = Nv[i]

            a[i] = E2 * vi + Q * Nvi
            Na = np.fft.fft2(nonlinear_func(np.fft.ifft2(a).real)[i])
            b[i] = E2 * vi + Q * Na
            Nb = np.fft.fft2(nonlinear_func(np.fft.ifft2(b).real)[i])
            c[i] = E2 * a[i] + Q * (2.0 * Nb - Nvi)
            Nc = np.fft.fft2(nonlinear_func(np.fft.ifft2(c).real)[i])

            v_new[i] = E * vi + Nvi * f1 + 2.0 * (Na + Nb) * f2 + Nc * f3


        u_new = np.zeros((n_fields, nx, ny))
        for i in range(n_fields):
            u_new[i] = np.fft.ifft2(v_new[i]).real

        return u_new

    def solve(self, u0: np.ndarray, nonlinear_func, n_steps: int) -> np.ndarray:
        u = u0.copy()
        for _ in range(n_steps):
            u = self.step(u, nonlinear_func)
        return u

    def solve_with_history(self, u0: np.ndarray, nonlinear_func, n_steps: int, save_every: int = 1) -> np.ndarray:
        u = u0.copy()
        history = [u.copy()]
        for step in range(n_steps):
            u = self.step(u, nonlinear_func)
            if (step + 1) % save_every == 0:
                history.append(u.copy())
        return np.array(history)
