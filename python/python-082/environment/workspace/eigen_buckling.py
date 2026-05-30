
import numpy as np
from utils import validate_matrix_nonsingular


def generate_orthogonal_matrix(n):
    A = np.random.randn(n, n)
    Q, R = np.linalg.qr(A)

    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1.0
    return Q


def generate_symmetric_eigenproblem(n, lambda_mean=1.0, lambda_dev=0.1):
    lambda_vals = lambda_mean + lambda_dev * np.random.randn(n)
    lambda_vals = np.sort(lambda_vals)[::-1]
    Q = generate_orthogonal_matrix(n)
    A = Q @ np.diag(lambda_vals) @ Q.T
    return A, Q, lambda_vals


def generate_nonsymmetric_eigenproblem(n, lambda_mean=1.0, lambda_dev=0.1):
    T = np.triu(np.random.randn(n, n))
    np.fill_diagonal(T, lambda_mean + lambda_dev * np.random.randn(n))
    Q = generate_orthogonal_matrix(n)
    A = Q.T @ T @ Q
    return A, Q, T


class BucklingAnalysis:

    def __init__(self, D_matrix, plate_length, plate_width, nx=20, ny=20):
        self.D = np.asarray(D_matrix)
        self.L = float(plate_length)
        self.W = float(plate_width)
        self.nx = nx
        self.ny = ny
        self.dx = self.L / (nx - 1)
        self.dy = self.W / (ny - 1)

    def build_bending_stiffness_matrix(self):
        n = self.nx * self.ny
        K_b = np.zeros((n, n))

        D11, D12, D66 = self.D[0, 0], self.D[0, 1], self.D[2, 2]
        D22 = self.D[1, 1]

        coeff_x = D11 / self.dx ** 4
        coeff_y = D22 / self.dy ** 4
        coeff_xy = 2.0 * (D12 + 2.0 * D66) / (self.dx ** 2 * self.dy ** 2)

        for j in range(self.ny):
            for i in range(self.nx):
                idx = j * self.nx + i

                K_b[idx, idx] = 6.0 * coeff_x + 6.0 * coeff_y + 4.0 * coeff_xy


                if i > 0:
                    K_b[idx, idx - 1] += -4.0 * coeff_x
                if i < self.nx - 1:
                    K_b[idx, idx + 1] += -4.0 * coeff_x
                if i > 1:
                    K_b[idx, idx - 2] += coeff_x
                if i < self.nx - 2:
                    K_b[idx, idx + 2] += coeff_x


                if j > 0:
                    K_b[idx, idx - self.nx] += -4.0 * coeff_y
                if j < self.ny - 1:
                    K_b[idx, idx + self.nx] += -4.0 * coeff_y
                if j > 1:
                    K_b[idx, idx - 2 * self.nx] += coeff_y
                if j < self.ny - 2:
                    K_b[idx, idx + 2 * self.nx] += coeff_y


                for di, dj in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < self.nx and 0 <= nj < self.ny:
                        K_b[idx, nj * self.nx + ni] += coeff_xy


        for i in range(self.nx):
            K_b[i, :] = 0.0
            K_b[i, i] = 1.0
            K_b[(self.ny - 1) * self.nx + i, :] = 0.0
            K_b[(self.ny - 1) * self.nx + i, (self.ny - 1) * self.nx + i] = 1.0
        for j in range(self.ny):
            K_b[j * self.nx, :] = 0.0
            K_b[j * self.nx, j * self.nx] = 1.0
            K_b[j * self.nx + self.nx - 1, :] = 0.0
            K_b[j * self.nx + self.nx - 1, j * self.nx + self.nx - 1] = 1.0

        return K_b

    def build_geometric_stiffness_matrix(self, N_x, N_y, N_xy):
        n = self.nx * self.ny
        K_g = np.zeros((n, n))

        coeff_xx = N_x / self.dx ** 2
        coeff_yy = N_y / self.dy ** 2
        coeff_xy = N_xy / (4.0 * self.dx * self.dy)

        for j in range(self.ny):
            for i in range(self.nx):
                idx = j * self.nx + i
                K_g[idx, idx] = -2.0 * (coeff_xx + coeff_yy)

                if i > 0:
                    K_g[idx, idx - 1] += coeff_xx
                if i < self.nx - 1:
                    K_g[idx, idx + 1] += coeff_xx
                if j > 0:
                    K_g[idx, idx - self.nx] += coeff_yy
                if j < self.ny - 1:
                    K_g[idx, idx + self.nx] += coeff_yy

                for di, dj in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < self.nx and 0 <= nj < self.ny:
                        K_g[idx, nj * self.nx + ni] += coeff_xy


        for i in range(self.nx):
            K_g[i, :] = 0.0
            K_g[(self.ny - 1) * self.nx + i, :] = 0.0
        for j in range(self.ny):
            K_g[j * self.nx, :] = 0.0
            K_g[j * self.nx + self.nx - 1, :] = 0.0

        return K_g

    def solve_buckling_loads(self, N_x=1.0, N_y=0.0, N_xy=0.0, n_modes=5):
        K_b = self.build_bending_stiffness_matrix()
        K_g = self.build_geometric_stiffness_matrix(N_x, N_y, N_xy)


        try:

            K_g_inv = np.linalg.inv(K_g + 1e-8 * np.eye(K_g.shape[0]))
            M = K_g_inv @ K_b
            eigvals, eigvecs = np.linalg.eig(M)

            buckling_vals = np.real(1.0 / eigvals)
        except np.linalg.LinAlgError:
            K_b_inv = np.linalg.pinv(K_b)
            eigvals, eigvecs = np.linalg.eig(K_b_inv @ K_g)
            buckling_vals = np.real(eigvals)


        valid = buckling_vals > 1e-6
        if not np.any(valid):
            return np.array([np.inf]), np.zeros((K_b.shape[0], 1))

        sorted_idx = np.argsort(buckling_vals[valid])
        lambdas = buckling_vals[valid][sorted_idx][:n_modes]
        modes = eigvecs[:, valid][:, sorted_idx][:, :n_modes]

        return lambdas, modes

    def compute_critical_buckling_load(self, N_x=1.0):
        lambdas, _ = self.solve_buckling_loads(N_x=N_x, n_modes=1)
        if len(lambdas) > 0 and lambdas[0] < np.inf:
            return lambdas[0]
        return np.inf


class VibrationAnalysis:

    def __init__(self, D_matrix=None, rho=1600.0, thickness=1.0, plate_length=100.0, plate_width=100.0, nx=12, ny=12):
        self.D = np.asarray(D_matrix) if D_matrix is not None else np.eye(3)
        self.rho = float(rho)
        self.h = float(thickness)
        self.L = float(plate_length)
        self.W = float(plate_width)
        self.nx = nx
        self.ny = ny

    def solve_natural_frequencies(self, D_matrix, rho, thickness, plate_length, plate_width, nx=20, ny=20):
        buckling = BucklingAnalysis(D_matrix, plate_length, plate_width, nx, ny)
        K = buckling.build_bending_stiffness_matrix()
        n = K.shape[0]


        dx = plate_length / (nx - 1)
        dy = plate_width / (ny - 1)
        M = np.eye(n) * rho * thickness * dx * dy

        try:
            K_inv = np.linalg.inv(K)
            eigvals, eigvecs = np.linalg.eig(K_inv @ M)
        except np.linalg.LinAlgError:
            K_inv = np.linalg.pinv(K)
            eigvals, eigvecs = np.linalg.eig(K_inv @ M)


        omega_sq = np.real(1.0 / eigvals)
        valid = omega_sq > 1e-6
        if not np.any(valid):
            return np.array([]), np.zeros((n, 0))

        omega_sq_valid = omega_sq[valid]
        sorted_idx = np.argsort(omega_sq_valid)
        omegas = np.sqrt(omega_sq_valid[sorted_idx])
        modes = eigvecs[:, valid][:, sorted_idx]

        return omegas, modes
