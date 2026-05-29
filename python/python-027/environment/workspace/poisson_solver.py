# -*- coding: utf-8 -*-
"""
poisson_solver.py
鞘层泊松方程有限差分求解模块
基于种子项目 269_delsq (五点差分Laplacian) 和 603_jacobi (Jacobi迭代) 重构

本模块求解一维/二维泊松方程:
    nabla^2 phi = -rho/epsilon_0
其中 rho = e*(n_i - n_e) 为电荷密度，n_e = n_0*exp(e*phi/(k_B*T_e)) 为Boltzmann电子。
"""

import numpy as np
from parameters import get_parameters


class PoissonSolver:
    """
    泊松方程数值求解器
    
    一维模型（鞘层）:
        d^2 phi / dx^2 = -(e/epsilon_0) * (n_i(x) - n_0 * exp(e*phi/(k_B*T_e)))
    
    二维模型（靶板表面附近）:
        d^2 phi / dx^2 + d^2 phi / dy^2 = -(e/epsilon_0) * rho(x,y)
    
    使用五点差分离散 Laplacian (delsq)，配合 Jacobi 迭代求解。
    """

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
        """
        构造一维负Laplacian矩阵（五点差分的一维退化形式）
        
        基于 delsq.m 的核心思想:
            -d^2/dx^2 的离散形式:  (1/dx^2) * tridiag(-1, 2, -1)
        
        Returns:
            L: (nx, nx) 稀疏矩阵格式（此处用稠密数组演示结构）
        """
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
        """
        构造二维五点差分负Laplacian矩阵
        
        基于 delsq.m 的核心思想，对 nx x ny 网格:
            -Delta_h = (1/dx^2) * (4*u_{i,j} - u_{i-1,j} - u_{i+1,j} - u_{i,j-1} - u_{i,j+1})
        
        使用 Kronecker 积构造块三对角矩阵。
        
        Returns:
            L: (nx*ny, nx*ny) 矩阵
        """
        if nx < 3 or ny < 3:
            raise ValueError("nx 和 ny 必须至少为 3")
        if dx <= 0 or dy <= 0:
            raise ValueError("dx 和 dy 必须为正")

        # 一维三对角矩阵
        Tx = np.diag(2.0 * np.ones(nx)) + np.diag(-1.0 * np.ones(nx-1), 1) + np.diag(-1.0 * np.ones(nx-1), -1)
        Ty = np.diag(2.0 * np.ones(ny)) + np.diag(-1.0 * np.ones(ny-1), 1) + np.diag(-1.0 * np.ones(ny-1), -1)

        Ix = np.eye(nx)
        Iy = np.eye(ny)

        L = (1.0/dx**2) * np.kron(Iy, Tx) + (1.0/dy**2) * np.kron(Ty, Ix)

        return L

    def jacobi_solve(self, A, b, x0=None, max_iter=None, tol=None, omega=1.0):
        """
        Jacobi迭代求解 Ax = b
        
        基于 jacobi1.m / jacobi2.m / jacobi3.m 的核心算法:
            x^{new} = x + omega * (b - A*x) ./ diag(A)
        
        Parameters:
            A:        系数矩阵
            b:        右端项
            x0:       初始猜测
            max_iter: 最大迭代次数
            tol:      收敛容差（基于残差范数）
            omega:    松弛因子 (0 < omega <= 1)
        
        Returns:
            x:        解向量
            res_history: 残差历史
            iters:    实际迭代次数
        """
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

        # 提取对角线
        diag_A = np.diag(A)
        if np.any(np.abs(diag_A) < 1.0e-30):
            raise ValueError("A 的对角线元素包含零或接近零的值")

        x = x0.copy()
        res_history = []

        for it in range(max_iter):
            # Jacobi 更新（jacobi3 风格）
            residual = b - A.dot(x)
            x = x + omega * residual / diag_A

            # 计算残差范数
            res_norm = np.linalg.norm(residual) / np.linalg.norm(b)
            res_history.append(res_norm)

            if res_norm < tol:
                return x, np.array(res_history), it + 1

        # 若未收敛，给出警告但仍返回结果
        print(f"  [警告] Jacobi迭代未在 {max_iter} 步内收敛，最终残差 = {res_history[-1]:.3e}")
        return x, np.array(res_history), max_iter

    def gauss_seidel_solve(self, A, b, x0=None, max_iter=None, tol=None, omega=1.0):
        """
        Gauss-Seidel/SOR 迭代求解（作为Jacobi的加速对比）
        
        迭代格式:
            x_i^{new} = x_i + omega * (b_i - sum_{j<i} A_{ij}*x_j^{new} - sum_{j>=i} A_{ij}*x_j) / A_{ii}
        """
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
        """
        求解一维鞘层泊松方程
        
        方程:
            d^2 phi / dx^2 = -(e/epsilon_0) * (n_i - n_e)
            n_e = n_0 * exp(e*phi/(k_B*T_e))
        
        边界条件:
            phi(0) = 0    (鞘层边缘，准中性)
            phi(L) = phi_w  (壁面电势，由鞘层条件确定)
        
        由于电子项的非线性，采用 Picard 迭代:
            d^2 phi^{k+1} / dx^2 = -(e/epsilon_0) * (n_i - n_0*exp(e*phi^k/(k_B*T_e)))
        
        Parameters:
            n_i_profile: 离子密度剖面 [m^-3]
            x_arr:       空间网格 [m]
        
        Returns:
            phi: 电势剖面 [V]
            n_e: 电子密度剖面 [m^-3]
            rho: 电荷密度剖面 [C/m^3]
        """
        nx = len(x_arr)
        if len(n_i_profile) != nx:
            raise ValueError("n_i_profile 长度必须与 x_arr 匹配")

        dx = x_arr[1] - x_arr[0]
        if dx <= 0:
            raise ValueError("x_arr 必须是单调递增的")

        # 构造 Laplacian（内部点）
        # 对 Dirichlet 边界，内部 nx-2 个点
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

        # Picard 迭代处理非线性
        kB_Te = self.Te  # eV -> J/e = eV (因为 e 在两边约掉)
        e_eps = self.e_charge / self.epsilon_0

        # TODO: 实现 Picard 迭代求解非线性泊松方程的核心循环
        # 提示: 需迭代计算电子密度 n_e = n_0*exp(e*phi/(k_B*T_e))、电荷密度 rho、
        #       右端项 b_int，并施加壁面电势边界条件，最后求解线性系统
        raise NotImplementedError("Hole_2: 请实现 Picard 迭代核心循环")


        return phi, n_e, rho

    def solve_2d_laplace(self, nx, ny, dx, dy, boundary_func):
        """
        求解二维Laplace方程（用于靶板表面电势分布）
        
        方程:
            nabla^2 phi = 0
        
        boundary_func(i, j) -> 给定边界上的 phi 值，内部点返回 None
        """
        L = self.build_laplacian_2d(nx, ny, dx, dy)
        N = nx * ny
        b = np.zeros(N)
        phi_vec = np.zeros(N)

        # 标记已知边界点
        known = np.zeros(N, dtype=bool)
        for j in range(ny):
            for i in range(nx):
                idx = j * nx + i
                val = boundary_func(i, j)
                if val is not None:
                    known[idx] = True
                    phi_vec[idx] = val
                    # 修改矩阵：已知点对应方程 phi = val
                    L[idx, :] = 0.0
                    L[idx, idx] = 1.0
                    b[idx] = val

        # 求解（小规模2D问题用直接求解器）
        if N <= 2500:
            phi_vec = np.linalg.solve(L, b)
        else:
            phi_vec, _, _ = self.jacobi_solve(L, b, x0=phi_vec, max_iter=20000, tol=1.0e-8, omega=0.9)

        phi = phi_vec.reshape((ny, nx))
        return phi


def demo_poisson():
    """演示泊松方程求解"""
    solver = PoissonSolver()

    # 1D测试
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
