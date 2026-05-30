
import numpy as np


class Poisson2DSolver:

    def __init__(self, nx, ny, dh, boundary_box=None):
        self.nx = nx
        self.ny = ny
        self.dh = dh
        self.nn = nx * ny
        self.boundary_box = boundary_box


        self.A = self._build_stencil()

    def _build_stencil(self):
        nx, ny, dh = self.nx, self.ny, self.dh
        nn = nx * ny
        A = np.zeros((nn, nn), dtype=float)
        h2 = dh * dh


        for j in range(1, ny - 1):
            for i in range(1, nx - 1):
                u = j * nx + i
                A[u, u] = -4.0 / h2
                A[u, u - 1] = 1.0 / h2
                A[u, u + 1] = 1.0 / h2
                A[u, u - nx] = 1.0 / h2
                A[u, u + nx] = 1.0 / h2


        j = 0
        for i in range(nx):
            u = j * nx + i
            A[u, u] = -1.0 / dh
            A[u, u + nx] = 1.0 / dh


        j = ny - 1
        for i in range(nx):
            u = j * nx + i
            A[u, u - nx] = 1.0 / dh
            A[u, u] = -1.0 / dh


        i = nx - 1
        for j in range(ny):
            u = j * nx + i
            A[u, :] = 0.0
            A[u, u - 1] = 1.0 / dh
            A[u, u] = -1.0 / dh


        i = 0
        for j in range(ny):
            u = j * nx + i
            A[u, :] = 0.0
            A[u, u] = 1.0


        if self.boundary_box is not None:
            bx1, bx2 = self.boundary_box[0]
            by1, by2 = self.boundary_box[1]
            for j in range(by1, by2 + 1):
                for i in range(bx1, bx2 + 1):
                    u = j * nx + i
                    A[u, :] = 0.0
                    A[u, u] = 1.0

        return A

    def solve_gs(self, phi_init, den, n0, phi0, te, phi_p, eps0, qe, max_iter=2000, tol=0.1):
        nx, ny = self.nx, self.ny
        nn = self.nn
        phi = phi_init.copy().ravel()
        den_vec = den.copy().ravel()

        for it in range(1, max_iter + 1):


            phi_clipped = np.clip(phi, phi0 - 10.0 * te, phi0 + 10.0 * te)
            b = den_vec - n0 * np.exp((phi_clipped - phi0) / te)
            b = -b * qe / eps0


            b[0:nx] = 0.0
            b[nn - nx:nn] = 0.0
            b[nx - 1:nn:nx] = 0.0
            b[0:nn:nx] = phi0

            if self.boundary_box is not None:
                bx1, bx2 = self.boundary_box[0]
                by1, by2 = self.boundary_box[1]
                for j in range(by1, by2 + 1):
                    b[bx1 + j * nx:bx2 + 1 + j * nx] = phi_p


            for i in range(nn):
                phi[i] = (b[i] - np.dot(self.A[i, 0:i], phi[0:i])
                          - np.dot(self.A[i, i + 1:nn], phi[i + 1:nn])) / self.A[i, i]


            if it % 10 == 0:
                res = np.linalg.norm(b - self.A @ phi)
                if res <= tol:
                    break

        return phi.reshape((nx, ny))

    def compute_electric_field(self, phi):
        nx, ny, dh = self.nx, self.ny, self.dh
        efx = np.zeros((nx, ny), dtype=float)
        efy = np.zeros((nx, ny), dtype=float)


        efx[1:nx - 1, :] = (phi[0:nx - 2, :] - phi[2:nx, :]) / (2.0 * dh)
        efy[:, 1:ny - 1] = (phi[:, 0:ny - 2] - phi[:, 2:ny]) / (2.0 * dh)


        efx[0, :] = (phi[0, :] - phi[1, :]) / dh
        efx[nx - 1, :] = (phi[nx - 2, :] - phi[nx - 1, :]) / dh
        efy[:, 0] = (phi[:, 0] - phi[:, 1]) / dh
        efy[:, ny - 1] = (phi[:, ny - 2] - phi[:, ny - 1]) / dh

        return efx, efy


def pic_charge_density(nx, ny, dh, part_x, part_v, np_part, spwt, mp_q):
    chg = np.zeros((nx, ny), dtype=float)

    for p in range(np_part):
        fi = 1.0 + part_x[p, 0] / dh
        i = int(np.floor(fi))
        hx = fi - i

        fj = 1.0 + part_x[p, 1] / dh
        j = int(np.floor(fj))
        hy = fj - j


        if i < 0 or i >= nx - 1 or j < 0 or j >= ny - 1:
            continue

        chg[i, j] += (1.0 - hx) * (1.0 - hy)
        chg[i + 1, j] += hx * (1.0 - hy)
        chg[i, j + 1] += (1.0 - hx) * hy
        chg[i + 1, j + 1] += hx * hy


    den = spwt * mp_q * chg / (dh * dh)


    den[0, :] *= 2.0
    den[nx - 1, :] *= 2.0
    den[:, 0] *= 2.0
    den[:, ny - 1] *= 2.0


    den = den + 10000.0

    return den


def poisson_2d_exact_solution(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    denom = (3.0 + x) ** 2 + (1.0 + y) ** 2
    u = 2.0 * (1.0 + y) / denom
    ux = (-2.0 * x - 6.0) * (2.0 * y + 2.0) / (denom ** 2)
    uy = (-2.0 * y - 2.0) * (2.0 * y + 2.0) / (denom ** 2) + 2.0 / denom
    uxx = 4.0 * (y + 1.0) * (4.0 * (x + 3.0) ** 2 / denom - 1.0) / (denom ** 2)
    uxy = 4.0 * (x + 3.0) * (4.0 * (y + 1.0) ** 2 / denom - 1.0) / (denom ** 2)
    uyy = 4.0 * (y + 1.0) * (4.0 * (y + 1.0) ** 2 / denom - 3.0) / (denom ** 2)

    return u, ux, uy, uxx, uxy, uyy


def electrostatic_stabilization_energy(phi, rho, dx, dy):
    return 0.5 * np.sum(rho * phi) * dx * dy
