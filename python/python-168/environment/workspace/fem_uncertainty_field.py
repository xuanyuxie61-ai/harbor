
import numpy as np


class FEMUncertaintyField:

    def __init__(self, nx=20, ny=20):
        self.nx = max(int(nx), 2)
        self.ny = max(int(ny), 2)

    def solve_uncertainty_field(self, domain, a_func, c_func, f_func):
        (xmin, xmax), (ymin, ymax) = domain
        x = np.linspace(xmin, xmax, self.nx)
        y = np.linspace(ymin, ymax, self.ny)

        mn = self.nx * self.ny
        A_mat = np.zeros((mn, mn), dtype=np.float64)
        b_vec = np.zeros(mn, dtype=np.float64)


        quad_xi = np.array([-0.7745966692414834, 0.0, 0.7745966692414834])
        quad_w = np.array([0.5555555555555556, 0.8888888888888889, 0.5555555555555556])

        for ex in range(self.nx - 1):
            xw = x[ex]
            xe = x[ex + 1]
            hx = xe - xw

            for ey in range(self.ny - 1):
                ys = y[ey]
                yn = y[ey + 1]
                hy = yn - ys


                sw = ey * self.nx + ex
                se = ey * self.nx + ex + 1
                nw = (ey + 1) * self.nx + ex
                ne = (ey + 1) * self.nx + ex + 1
                nodes = [sw, se, nw, ne]


                for qx in range(3):
                    xi = quad_xi[qx]
                    xq = 0.5 * ((1.0 - xi) * xw + (1.0 + xi) * xe)
                    wx = quad_w[qx] * hx * 0.5

                    for qy in range(3):
                        eta = quad_xi[qy]
                        yq = 0.5 * ((1.0 - eta) * ys + (1.0 + eta) * yn)
                        wy = quad_w[qy] * hy * 0.5
                        wq = wx * wy






                        N = np.array([
                            0.25 * (1.0 - xi) * (1.0 - eta),
                            0.25 * (1.0 + xi) * (1.0 - eta),
                            0.25 * (1.0 - xi) * (1.0 + eta),
                            0.25 * (1.0 + xi) * (1.0 + eta)
                        ], dtype=np.float64)


                        dN_dxi = np.array([-0.25 * (1.0 - eta), 0.25 * (1.0 - eta),
                                           -0.25 * (1.0 + eta), 0.25 * (1.0 + eta)])
                        dN_deta = np.array([-0.25 * (1.0 - xi), -0.25 * (1.0 + xi),
                                            0.25 * (1.0 - xi), 0.25 * (1.0 + xi)])


                        J = np.array([
                            [dN_dxi @ np.array([xw, xe, xw, xe]),
                             dN_dxi @ np.array([ys, ys, yn, yn])],
                            [dN_deta @ np.array([xw, xe, xw, xe]),
                             dN_deta @ np.array([ys, ys, yn, yn])]
                        ])
                        detJ = J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0]
                        if abs(detJ) < 1e-14:
                            continue


                        invJ = np.array([[J[1, 1], -J[0, 1]], [-J[1, 0], J[0, 0]]]) / detJ
                        gradN = np.zeros((4, 2))
                        for i in range(4):
                            gradN[i] = invJ @ np.array([dN_dxi[i], dN_deta[i]])

                        aq = a_func(xq, yq)
                        cq = c_func(xq, yq)
                        fq = f_func(xq, yq)


                        for i in range(4):
                            for j in range(4):
                                A_mat[nodes[i], nodes[j]] += wq * (
                                    aq * np.dot(gradN[i], gradN[j]) +
                                    cq * N[i] * N[j]
                                )
                            b_vec[nodes[i]] += wq * fq * N[i]


        for iy in range(self.ny):
            for ix in range(self.nx):
                k = iy * self.nx + ix
                if ix == 0 or ix == self.nx - 1 or iy == 0 or iy == self.ny - 1:
                    A_mat[k, :] = 0.0
                    A_mat[:, k] = 0.0
                    A_mat[k, k] = 1.0
                    b_vec[k] = 0.0


        try:
            u = np.linalg.solve(A_mat, b_vec)
        except np.linalg.LinAlgError:
            u = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]

        u_grid = u.reshape(self.ny, self.nx).T
        return u_grid, x, y

    @staticmethod
    def sample_field_at_points(u_grid, x_grid, y_grid, query_points):
        query_points = np.asarray(query_points, dtype=np.float64)
        values = np.zeros(query_points.shape[0], dtype=np.float64)

        nx = len(x_grid)
        ny = len(y_grid)
        dx = x_grid[1] - x_grid[0] if nx > 1 else 1.0
        dy = y_grid[1] - y_grid[0] if ny > 1 else 1.0

        for idx, (px, py) in enumerate(query_points):

            ix = int((px - x_grid[0]) / dx)
            iy = int((py - y_grid[0]) / dy)
            ix = max(0, min(ix, nx - 2))
            iy = max(0, min(iy, ny - 2))


            xi = 2.0 * (px - x_grid[ix]) / dx - 1.0
            eta = 2.0 * (py - y_grid[iy]) / dy - 1.0
            xi = np.clip(xi, -1.0, 1.0)
            eta = np.clip(eta, -1.0, 1.0)


            N1 = 0.25 * (1.0 - xi) * (1.0 - eta)
            N2 = 0.25 * (1.0 + xi) * (1.0 - eta)
            N3 = 0.25 * (1.0 - xi) * (1.0 + eta)
            N4 = 0.25 * (1.0 + xi) * (1.0 + eta)

            values[idx] = (N1 * u_grid[ix, iy] +
                           N2 * u_grid[ix + 1, iy] +
                           N3 * u_grid[ix, iy + 1] +
                           N4 * u_grid[ix + 1, iy + 1])

        return values


class AnnularCovarianceEstimator:

    def __init__(self, nr=8, nt=32):
        self.nr = max(int(nr), 1)
        self.nt = max(int(nt), 4)

    @staticmethod
    def legendre_gauss_nodes(n):
        try:
            xi, w = np.polynomial.legendre.leggauss(n)
        except Exception:

            xi = np.linspace(-1, 1, n, endpoint=False) + 1.0 / n
            w = np.full(n, 2.0 / n)
        return xi, w

    def integrate_annular_covariance(self, center, r1, r2, covariance_func):
        cx, cy = center
        r1 = float(r1)
        r2 = float(r2)
        if r2 <= r1:
            r2 = r1 + 1e-6

        area = np.pi * (r2 * r2 - r1 * r1)


        ra, rw = self.legendre_gauss_nodes(self.nr)


        r_sq_nodes = 0.5 * ((r2 ** 2 - r1 ** 2) * ra + (r2 ** 2 + r1 ** 2))
        r_nodes = np.sqrt(r_sq_nodes)

        w_r = rw * 0.5 * (r2 ** 2 - r1 ** 2)

        w_r = w_r / (r2 ** 2 - r1 ** 2)

        tw = 1.0 / self.nt


        sample_val = covariance_func(cx + r_nodes[0] * np.cos(0.0),
                                      cy + r_nodes[0] * np.sin(0.0))
        if np.isscalar(sample_val):
            integral = 0.0
            is_scalar = True
        else:
            integral = np.zeros((2, 2), dtype=np.float64)
            is_scalar = False

        for i in range(self.nt):
            theta = 2.0 * np.pi * i / self.nt
            c_t = np.cos(theta)
            s_t = np.sin(theta)
            for j in range(self.nr):
                r_val = r_nodes[j]
                px = cx + r_val * c_t
                py = cy + r_val * s_t
                val = covariance_func(px, py)
                weight = area * tw * w_r[j]
                if is_scalar:
                    integral += weight * val
                else:
                    integral += weight * np.asarray(val, dtype=np.float64)

        return integral
