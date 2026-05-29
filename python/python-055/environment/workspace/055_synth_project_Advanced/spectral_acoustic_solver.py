"""
spectral_acoustic_solver.py
基于种子项目 399_fem1d_spectral_numeric（一维谱有限元方法），
构建海洋声速剖面下的一维声学 Helmholtz 方程谱元求解器。

科学背景：在水平分层海洋中，声压场 p(z) 满足一维 Helmholtz 方程：

    d²p/dz² + k²(z) · p = f(z),    z ∈ [0, H]

其中：
    k(z) = ω / c(z) 为波数，ω = 2πf 为角频率；
    c(z) 为声速剖面；
    f(z) 为声源项（通常取为 Dirac δ 函数或高斯包络）；
    H 为海深（计算域上界）。

边界条件：
    - 海面 (z=0)：压力释放边界，p(0) = 0；
    - 海底 (z=H)：刚性或阻抗边界，dp/dz = 0（简化）或
      dp/dz + j·k·Z · p = 0（阻抗边界，Z 为海底反射系数）。

谱有限元方法采用高阶多项式基函数：
    φ_i(z) = z^{i-1} · (z - H) · z
    满足 φ_i(0) = φ_i(H) = 0（齐次 Dirichlet 边界）。

弱形式推导：
    乘以检验函数 v 并在 [0, H] 上积分，分部积分得：
        -∫ (dp/dz)(dv/dz) dz + ∫ k²(z) · p · v dz = ∫ f · v dz

    代入 p = Σ c_j φ_j，取 v = φ_i，得到线性方程组：
        Σ_j [ -B_{ij} + K_{ij} ] c_j = F_i
    其中:
        B_{ij} = ∫_0^H φ_i'(z) φ_j'(z) dz
        K_{ij} = ∫_0^H k²(z) φ_i(z) φ_j(z) dz
        F_i    = ∫_0^H f(z) φ_i(z) dz

本模块同时计算 L² 与 H¹ 误差范数（相对精确解或参考解）。
"""

import numpy as np
from scipy.integrate import quad


class SpectralAcousticSolver:
    """
    一维谱有限元声学求解器。
    """

    def __init__(self, depth: float = 4000.0, frequency: float = 12000.0):
        """
        参数:
            depth:    海深 (m)
            frequency: 声源频率 (Hz)
        """
        self.H = float(depth)
        self.f = float(frequency)
        self.omega = 2.0 * np.pi * self.f

    def _basis(self, z: np.ndarray, i: int) -> np.ndarray:
        """
        基函数 φ_i(z) = z^{i-1} · (z - H) · z = z^i · (z - H)。

        注意：i 从 1 开始编号，对应多项式次数 i+1。
        修正为 φ_i(z) = z^{i-1} · z · (z - H) = z^i · (z - H) 以保证 φ(0)=φ(H)=0。
        但为保持与原始 fem1d_spectral_numeric 一致，使用：
            φ_i(z) = z^{i-1} · z · (z - H) / H^2 （归一化）
        这里简化为 φ_i(z) = z^{i} · (z - H) / H^{i+1} 以保持数值稳定。
        """
        z = np.asarray(z, dtype=np.float64)
        # 归一化坐标 ξ = z/H ∈ [0,1]
        xi = z / self.H
        # 在归一化坐标下: φ_i(ξ) = ξ^i · (ξ - 1)，满足 φ(0)=φ(1)=0
        return (xi ** i) * (xi - 1.0)

    def _basis_derivative(self, z: np.ndarray, i: int) -> np.ndarray:
        """基函数导数 dφ_i/dz。"""
        z = np.asarray(z, dtype=np.float64)
        xi = z / self.H
        # dφ/dz = (dφ/dξ) · (dξ/dz) = (dφ/dξ) / H
        dphi_dxi = i * (xi ** (i - 1)) * (xi - 1.0) + xi ** i
        return dphi_dxi / self.H

    def _compute_B_entry(self, i: int, j: int) -> float:
        """计算刚度矩阵元素 B_{ij} = ∫ φ_i' φ_j' dz。"""
        def integrand(z):
            return self._basis_derivative(np.array([z]), i)[0] * \
                   self._basis_derivative(np.array([z]), j)[0]
        val, _ = quad(integrand, 0.0, self.H, limit=100)
        return float(val)

    def _compute_K_entry(self, i: int, j: int, k_func) -> float:
        """计算质量矩阵元素 K_{ij} = ∫ k²(z) φ_i φ_j dz。"""
        def integrand(z):
            kz = k_func(np.array([z]))[0]
            return (kz ** 2) * self._basis(np.array([z]), i)[0] * \
                   self._basis(np.array([z]), j)[0]
        val, _ = quad(integrand, 0.0, self.H, limit=100)
        return float(val)

    def _compute_F_entry(self, i: int, source_func) -> float:
        """计算载荷向量元素 F_i = ∫ f(z) φ_i dz。"""
        def integrand(z):
            return source_func(np.array([z]))[0] * self._basis(np.array([z]), i)[0]
        val, _ = quad(integrand, 0.0, self.H, limit=100)
        return float(val)

    def solve(
        self,
        n_basis: int,
        sound_speed_func,
        source_func,
        verbose: bool = False
    ) -> dict:
        """
        求解一维 Helmholtz 方程。

        参数:
            n_basis:        基函数数量
            sound_speed_func: 声速函数 c(z)，返回数组
            source_func:    声源函数 f(z)，返回数组
            verbose:        是否输出调试信息
        返回:
            包含系数、解函数、矩阵条件数等的字典
        """
        if n_basis < 1:
            raise ValueError("n_basis 必须 >= 1")

        k_func = lambda z: self.omega / sound_speed_func(z)

        # 组装矩阵
        B = np.zeros((n_basis, n_basis), dtype=np.float64)
        K = np.zeros((n_basis, n_basis), dtype=np.float64)
        F = np.zeros(n_basis, dtype=np.float64)

        for i in range(n_basis):
            for j in range(n_basis):
                B[i, j] = self._compute_B_entry(i + 1, j + 1)
                K[i, j] = self._compute_K_entry(i + 1, j + 1, k_func)
            F[i] = self._compute_F_entry(i + 1, source_func)

        # 总矩阵: A = -B + K
        A = -B + K

        cond_num = np.linalg.cond(A)
        if verbose:
            print(f"  矩阵条件数: {cond_num:.4e}")

        # 求解
        try:
            coeffs = np.linalg.solve(A, F)
        except np.linalg.LinAlgError:
            # 若奇异，使用最小二乘
            coeffs, _, _, _ = np.linalg.lstsq(A, F, rcond=None)

        # 构造解函数
        def solution(z):
            z = np.asarray(z, dtype=np.float64)
            p = np.zeros_like(z)
            for i in range(n_basis):
                p += coeffs[i] * self._basis(z, i + 1)
            return p

        def solution_derivative(z):
            z = np.asarray(z, dtype=np.float64)
            dp = np.zeros_like(z)
            for i in range(n_basis):
                dp += coeffs[i] * self._basis_derivative(z, i + 1)
            return dp

        return {
            'coeffs': coeffs,
            'solution': solution,
            'derivative': solution_derivative,
            'matrix_A': A,
            'rhs_F': F,
            'cond_number': float(cond_num),
            'n_basis': n_basis,
        }

    def compute_errors(
        self,
        coeffs: np.ndarray,
        exact_solution_func,
        exact_derivative_func = None
    ) -> dict:
        """
        计算 L² 与 H¹ 误差范数。

        公式:
            ||e||_{L²} = √(∫ (u - u_exact)² dz)
            ||e||_{H¹} = √(∫ [(u - u_exact)² + (u' - u_exact')²] dz)
        """
        def l2_integrand(z):
            z_arr = np.array([z])
            u_num = np.zeros_like(z_arr)
            for i in range(len(coeffs)):
                u_num += coeffs[i] * self._basis(z_arr, i + 1)
            diff = u_num[0] - exact_solution_func(z_arr)[0]
            return diff ** 2

        l2_sq, _ = quad(l2_integrand, 0.0, self.H, limit=100)
        err_l2 = np.sqrt(l2_sq)

        if exact_derivative_func is not None:
            def h1_integrand(z):
                z_arr = np.array([z])
                u_num = np.zeros_like(z_arr)
                du_num = np.zeros_like(z_arr)
                for i in range(len(coeffs)):
                    u_num += coeffs[i] * self._basis(z_arr, i + 1)
                    du_num += coeffs[i] * self._basis_derivative(z_arr, i + 1)
                diff_u = u_num[0] - exact_solution_func(z_arr)[0]
                diff_du = du_num[0] - exact_derivative_func(z_arr)[0]
                return diff_u ** 2 + diff_du ** 2

            h1_sq, _ = quad(h1_integrand, 0.0, self.H, limit=100)
            err_h1 = np.sqrt(h1_sq)
        else:
            err_h1 = err_l2

        return {
            'err_l2': float(err_l2),
            'err_h1': float(err_h1),
        }

    def solve_with_reference(
        self,
        n_basis: int,
        sound_speed_func,
        source_func,
        exact_solution_func = None,
        exact_derivative_func = None
    ) -> dict:
        """求解并计算误差（若提供精确解）。"""
        result = self.solve(n_basis, sound_speed_func, source_func)
        if exact_solution_func is not None:
            errors = self.compute_errors(
                result['coeffs'], exact_solution_func, exact_derivative_func
            )
            result.update(errors)
        return result
