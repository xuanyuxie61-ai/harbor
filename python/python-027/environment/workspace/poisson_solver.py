# -*- coding: utf-8 -*-

import numpy as np
from parameters import get_parameters


class PoissonSolver:

    def __init__(self, params=None):
        if params is None:
            params = get_parameters()
        self.params = params
        self.epsilon_0 = 8.854187817e-12
        self.e_charge = 1.602176634e-19
        self.n0 = params.get('n_0')
        self.Te = params.get('T_e')
        self.nx = params.get('nx')
        self.tol = params.get('tol')
        self.max_iter = params.get('max_iter')

    def build_laplacian_1d(self, nx, dx):
        if nx < 3:
            raise ValueError("nx 必须至少为 3")
        if dx <= 0:
            raise ValueError("dx 必须为正")

        L = np.zeros((nx, nx))
        coeff = 1.0 / (dx * dx)

        for i in range(nx):
            L[i, i] = 2.0 * coeff
            if i > 0:
                L[i, i-1] = -1.0 * coeff
            if i < nx - 1:
                L[i, i+1] = -1.0 * coeff

        return L

    def build_laplacian_2d(self, nx, ny, dx, dy):
        if nx < 3 or ny < 3:
            raise ValueError("nx 和 ny 必须至少为 3")
        if dx <= 0 or dy <= 0:
            raise ValueError("dx 和 dy 必须为正")


        Tx = np.diag(2.0 * np.ones(nx)) + np.diag(-1.0 * np.ones(nx-1), 1) + np.diag(-1.0 * np.ones(nx-1), -1)
        Ty = np.diag(2.0 * np.ones(ny)) + np.diag(-1.0 * np.ones(ny-1), 1) + np.diag(-1.0 * np.ones(ny-1), -1)

        Ix = np.eye(nx)
        Iy = np.eye(ny)

        L = (1.0/dx**2) * np.kron(Iy, Tx) + (1.0/dy**2) * np.kron(Ty, Ix)

        return L

    def jacobi_solve(self, A, b, x0=None, max_iter=None, tol=None, omega=1.0):
        n = A.shape[0]
        if A.shape[0] != A.shape[1]:
            raise ValueError("A 必须是方阵")
        if len(b) != n:
            raise ValueError("b 维度必须与 A 匹配")

        if max_iter is None:
            max_iter = self.max_iter
        if tol is None:
            tol = self.tol
        if x0 is None:
            x0 = np.zeros(n)


        diag_A = np.diag(A)
        if np.any(np.abs(diag_A) < 1.0e-30):
            raise ValueError("A 的对角线元素包含零或接近零的值")

        x = x0.copy()
        res_history = []

        for it in range(max_iter):

            residual = b - A.dot(x)
            x = x + omega * residual / diag_A


            res_norm = np.linalg.norm(residual) / np.linalg.norm(b)
            res_history.append(res_norm)

            if res_norm < tol:
                return x, np.array(res_history), it + 1


        print(f"  [警告] Jacobi迭代未在 {max_iter} 步内收敛，最终残差 = {res_history[-1]:.3e}")
        return x, np.array(res_history), max_iter

    def gauss_seidel_solve(self, A, b, x0=None, max_iter=None, tol=None, omega=1.0):
        n = A.shape[0]
        if max_iter is None:
            max_iter = self.max_iter
        if tol is None:
            tol = self.tol
        if x0 is None:
            x0 = np.zeros(n)

        x = x0.copy()
        res_history = []

        for it in range(max_iter):
            x_old = x.copy()
            for i in range(n):
                sigma = np.dot(A[i, :i], x[:i]) + np.dot(A[i, i+1:], x[i+1:])
                x[i] = x_old[i] + omega * (b[i] - sigma - A[i, i]*x_old[i]) / A[i, i]

            residual = b - A.dot(x)
            res_norm = np.linalg.norm(residual) / (np.linalg.norm(b) + 1.0e-30)
            res_history.append(res_norm)

            if res_norm < tol:
                return x, np.array(res_history), it + 1

        print(f"  [警告] GS迭代未在 {max_iter} 步内收敛，最终残差 = {res_history[-1]:.3e}")
        return x, np.array(res_history), max_iter

    def solve_1d_sheath_poisson(self, n_i_profile, x_arr):
        nx = len(x_arr)
        if len(n_i_profile) != nx:
            raise ValueError("n_i_profile 长度必须与 x_arr 匹配")

        dx = x_arr[1] - x_arr[0]
        if dx <= 0:
            raise ValueError("x_arr 必须是单调递增的")



        n_int = nx - 2
        L_int = np.zeros((n_int, n_int))
        coeff = 1.0 / (dx * dx)

        for i in range(n_int):
            L_int[i, i] = 2.0 * coeff
            if i > 0:
                L_int[i, i-1] = -1.0 * coeff
            if i < n_int - 1:
                L_int[i, i+1] = -1.0 * coeff

        phi = np.zeros(nx)
        n_e = np.full(nx, self.n0)
        rho = np.zeros(nx)


        kB_Te = self.Te
        e_eps = self.e_charge / self.epsilon_0




        raise NotImplementedError("Hole_2: 请实现 Picard 迭代核心循环")


        return phi, n_e, rho

    def solve_2d_laplace(self, nx, ny, dx, dy, boundary_func):
        L = self.build_laplacian_2d(nx, ny, dx, dy)
        N = nx * ny
        b = np.zeros(N)
        phi_vec = np.zeros(N)


        known = np.zeros(N, dtype=bool)
        for j in range(ny):
            for i in range(nx):
                idx = j * nx + i
                val = boundary_func(i, j)
                if val is not None:
                    known[idx] = True
                    phi_vec[idx] = val

                    L[idx, :] = 0.0
                    L[idx, idx] = 1.0
                    b[idx] = val


        if N <= 2500:
            phi_vec = np.linalg.solve(L, b)
        else:
            phi_vec, _, _ = self.jacobi_solve(L, b, x0=phi_vec, max_iter=20000, tol=1.0e-8, omega=0.9)

        phi = phi_vec.reshape((ny, nx))
        return phi


def demo_poisson():
    solver = PoissonSolver()


    nx = 64
    x = np.linspace(0.0, 0.005, nx)
    n_i = 1.0e19 * (0.5 + 0.5 * np.exp(-x / 1.0e-3))

    phi, n_e, rho = solver.solve_1d_sheath_poisson(n_i, x)

    print("一维鞘层泊松方程求解:")
    print(f"  壁面电势 phi_w     = {phi[-1]:.3f} V")
    print(f"  壁面电荷密度 rho_w = {rho[-1]:.3e} C/m^3")
    print(f"  壁面电子密度 n_e,w = {n_e[-1]:.3e} m^-3")

    return phi, n_e, rho


if __name__ == "__main__":
    demo_poisson()
