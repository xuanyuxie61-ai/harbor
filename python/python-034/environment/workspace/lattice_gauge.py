
import numpy as np


class Lattice:

    def __init__(self, nx: int = 4, ny: int = 4, nz: int = 4, nt: int = 8):
        self.dims = np.array([nx, ny, nz, nt], dtype=int)
        self.nd = 4
        self.vol = nx * ny * nz * nt
        self.shape = (nx, ny, nz, nt)

    def site_index(self, x: np.ndarray) -> int:
        x = np.mod(x, self.dims)
        return int(x[0] + self.dims[0] * (x[1] + self.dims[1] * (x[2] + self.dims[2] * x[3])))

    def index_to_site(self, idx: int) -> np.ndarray:
        x = np.zeros(4, dtype=int)
        x[0] = idx % self.dims[0]
        idx //= self.dims[0]
        x[1] = idx % self.dims[1]
        idx //= self.dims[1]
        x[2] = idx % self.dims[2]
        idx //= self.dims[2]
        x[3] = idx
        return x

    def neighbor(self, x: np.ndarray, mu: int, sign: int = 1) -> np.ndarray:
        y = x.copy()
        y[mu] = (y[mu] + sign) % self.dims[mu]
        return y


def su2_random() -> np.ndarray:
    v = np.random.randn(4)
    v /= np.linalg.norm(v)
    return np.array([[v[0] + 1j * v[3], v[1] + 1j * v[2]],
                     [-v[1] + 1j * v[2], v[0] - 1j * v[3]]])


def su2_identity() -> np.ndarray:
    return np.eye(2, dtype=complex)


def su2_dagger(u: np.ndarray) -> np.ndarray:
    return u.conj().T


def su2_trace(u: np.ndarray) -> complex:
    return np.trace(u)


def su2_stereographic_project(u: np.ndarray) -> np.ndarray:
    a0 = u[0, 0].real
    a1 = u[0, 1].real
    a2 = u[0, 1].imag
    a3 = u[0, 0].imag
    eps = 1e-12
    denom = 1.0 + a0
    if abs(denom) < eps:
        denom = eps * np.sign(denom + eps)
    q = np.array([a1, a2, a3]) / denom
    return q


def su2_stereographic_inverse(q: np.ndarray) -> np.ndarray:
    norm_sq = np.dot(q, q)
    denom = 1.0 + norm_sq
    a0 = (1.0 - norm_sq) / denom
    a1 = 2.0 * q[0] / denom
    a2 = 2.0 * q[1] / denom
    a3 = 2.0 * q[2] / denom
    return np.array([[a0 + 1j * a3, a1 + 1j * a2],
                     [-a1 + 1j * a2, a0 - 1j * a3]], dtype=complex)


class GaugeConfig:

    def __init__(self, lattice: Lattice):
        self.lat = lattice
        self.U = np.zeros((4, *lattice.shape, 2, 2), dtype=complex)
        for mu in range(4):
            for idx in range(lattice.vol):
                x = lattice.index_to_site(idx)
                self.U[(mu, *x)] = su2_identity()

    def randomize(self):
        for mu in range(4):
            for idx in range(self.lat.vol):
                x = self.lat.index_to_site(idx)
                self.U[(mu, *x)] = su2_random()

    def get_link(self, mu: int, x: np.ndarray) -> np.ndarray:
        x = np.mod(x, self.lat.dims)
        return self.U[(mu, *x)].copy()

    def set_link(self, mu: int, x: np.ndarray, u: np.ndarray):
        x = np.mod(x, self.lat.dims)
        self.U[(mu, *x)] = u.copy()

    def _link_array(self, mu: int) -> np.ndarray:
        return self.U[mu]

    def plaquette(self, x: np.ndarray, mu: int, nu: int) -> np.ndarray:
        u1 = self.get_link(mu, x)
        u2 = self.get_link(nu, self.lat.neighbor(x, mu, 1))
        u3 = su2_dagger(self.get_link(mu, self.lat.neighbor(x, nu, 1)))
        u4 = su2_dagger(self.get_link(nu, x))
        return u1 @ u2 @ u3 @ u4

    def average_plaquette(self) -> float:
        s = 0.0
        count = 0
        nx, ny, nz, nt = self.lat.shape

        for mu in range(4):
            for nu in range(mu + 1, 4):
                for ix in range(nx):
                    for iy in range(ny):
                        for iz in range(nz):
                            for it in range(nt):
                                x = np.array([ix, iy, iz, it])
                                p = self.plaquette(x, mu, nu)
                                s += su2_trace(p).real
                                count += 1
        return s / count if count > 0 else 0.0

    def wilson_action(self, beta: float = 2.4) -> float:
        avg = self.average_plaquette()
        return beta * (1.0 - 0.5 * avg) * self.lat.vol * 6.0


def ifs_thermalize_gauge(gauge: GaugeConfig, n_iter: int = 50):
    lat = gauge.lat
    for _ in range(n_iter):
        for idx in range(lat.vol):
            x = lat.index_to_site(idx)
            for mu in range(4):
                u = gauge.get_link(mu, x)
                if np.random.rand() < 0.5:
                    v = su2_random()
                else:
                    phi = np.random.randn(3) * 0.1
                    v = su2_stereographic_inverse(phi)
                gauge.set_link(mu, x, v @ u)
