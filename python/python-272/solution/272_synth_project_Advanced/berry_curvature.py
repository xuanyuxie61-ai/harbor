"""
Berry Curvature 计算模块
实现多种数值方法计算 Berry curvature 和 Chern 数

核心物理：
Berry curvature Ω_n(k) = -2·Im Σ_{m≠n} <u_nk|v_x|u_mk><u_mk|v_y|u_nk> / (E_m - E_n)²

Fukui-Hatsugai-Suzuki 方法：
通过离散 U(1) 联络计算 Chern 数，保证严格的整数化
"""

import numpy as np
from typing import Tuple, List, Dict
from weyl_hamiltonian import WeylHamiltonian
from brillouin_mesh import BrillouinMesh


class BerryCurvatureCalculator:
    """
    Berry curvature 计算引擎

    支持方法：
    1. Kubo 公式 (连续极限)
    2. Fukui-Hatsugai-Suzuki (离散规范化)
    3. Wilson loop 方法
    4. 高阶有限差分近似
    """

    def __init__(self, hamiltonian: WeylHamiltonian, mesh: BrillouinMesh):
        """
        初始化计算引擎

        Parameters:
        -----------
        hamiltonian : WeylHamiltonian
            Hamiltonian 构造器
        mesh : BrillouinMesh
            Brillouin 区网格
        """
        self.ham = hamiltonian
        self.mesh = mesh
        self.gauge_cache = {}  # 规范场缓存

    def solve_eigenproblem(self, k: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        求解 Hamiltonian 本征问题

        H(k)|u_n(k)> = E_n(k)|u_n(k)>

        Parameters:
        -----------
        k : np.ndarray
            k 点坐标 (分数坐标)

        Returns:
        --------
        Tuple[np.ndarray, np.ndarray]
            eigenvalues: 能量本征值
            eigenvectors: 本征态 (列向量)
        """
        # 将分数坐标转为笛卡尔坐标
        k_cart = k[0] * self.mesh.b1 + k[1] * self.mesh.b2 + k[2] * self.mesh.b3
        H = self.ham.hamiltonian_matrix(k_cart)
        eigenvalues, eigenvectors = np.linalg.eigh(H)
        return eigenvalues, eigenvectors

    def berry_curvature_kubo(self, k: np.ndarray, band_idx: int = 0,
                             eta: float = 1e-4) -> np.ndarray:
        """
        使用 Kubo 公式计算 Berry curvature

        Ω_n^{xy}(k) = -2·Im Σ_{m≠n} <u_n|v_x|u_m><u_m|v_y|u_n> / (E_m - E_n)²

        Parameters:
        -----------
        k : np.ndarray
            k 点坐标
        band_idx : int
            能带索引
        eta : float
            正则化参数

        Returns:
        --------
        np.ndarray
            Berry curvature 的三个分量 [Ω_xy, Ω_yz, Ω_zx]
        """
        eigenvalues, eigenvectors = self.solve_eigenproblem(k)

        # 速度矩阵
        v_x = self.ham.velocity_operator(k, 'x')
        v_y = self.ham.velocity_operator(k, 'y')
        v_z = self.ham.velocity_operator(k, 'z')

        n_bands = len(eigenvalues)
        omega_xy = 0.0 + 0.0j
        omega_yz = 0.0 + 0.0j
        omega_zx = 0.0 + 0.0j

        u_n = eigenvectors[:, band_idx]
        E_n = eigenvalues[band_idx]

        for m in range(n_bands):
            if m == band_idx:
                continue

            u_m = eigenvectors[:, m]
            E_m = eigenvalues[m]
            dE = E_m - E_n

            # 避免简并点发散
            if np.abs(dE) < eta:
                dE = eta * np.sign(dE + 1e-10)

            # 矩阵元 <u_n|v_i|u_m>
            vx_nm = np.conj(u_n) @ v_x @ u_m
            vy_nm = np.conj(u_n) @ v_y @ u_m
            vz_nm = np.conj(u_n) @ v_z @ u_m

            # Kubo 公式各项
            denominator = dE**2 + eta**2

            omega_xy += np.conj(vx_nm) * vy_nm / denominator
            omega_yz += np.conj(vy_nm) * vz_nm / denominator
            omega_zx += np.conj(vz_nm) * vx_nm / denominator

        omega_xy = -2 * np.imag(omega_xy)
        omega_yz = -2 * np.imag(omega_yz)
        omega_zx = -2 * np.imag(omega_zx)

        return np.array([omega_xy, omega_yz, omega_zx])

    def berry_connection(self, k: np.ndarray, dk: np.ndarray,
                         band_idx: int = 0) -> complex:
        """
        计算 Berry connection A_n(k) = i<u_n|∇_k|u_n>

        使用有限差分近似：
        A_n(k)·dk ≈ -Im ln <u_n(k)|u_n(k+dk)>

        Parameters:
        -----------
        k : np.ndarray
            当前 k 点
        dk : np.ndarray
            k 空间位移
        band_idx : int
            能带索引

        Returns:
        --------
        complex
            Berry connection 的近似值
        """
        _, u_k = self.solve_eigenproblem(k)
        _, u_k_dk = self.solve_eigenproblem(k + dk)

        # 重叠 <u_n(k)|u_n(k+dk)>
        overlap = np.conj(u_k[:, band_idx]) @ u_k_dk[:, band_idx]

        # A·dk ≈ -Im ln(overlap)
        A_dk = -np.imag(np.log(overlap + 1e-15))
        return A_dk

    def fhs_chern_number(self, plane: str = 'kz_const',
                          kz_value: float = 0.0,
                          band_idx: int = 0,
                          n_grid: int = 20) -> Tuple[int, np.ndarray]:
        """
        Fukui-Hatsugai-Suzuki 方法计算 Chern 数

        步骤：
        1. 在 2D BZ 上建立网格
        2. 计算每个 link 上的 U(1) 联络
           U_1(k) = <u(k)|u(k+b1)> / |<u(k)|u(k+b1)>|
        3. 计算每个 plaquette 上的场强
           F_12 = ln(U_1·U_2·U_1^{-1}·U_2^{-1}) / i
        4. Chern 数 C = (1/2π) Σ F_12

        Parameters:
        -----------
        plane : str
            切片平面
        kz_value : float
            固定坐标
        band_idx : int
            能带索引
        n_grid : int
            网格密度

        Returns:
        --------
        Tuple[int, np.ndarray]
            chern_number: Chern 数 (整数)
            flux_density: 每个 plaquette 的 Berry flux
        """
        # 生成网格
        vertices, triangles = self.mesh.generate_triangular_mesh_2d(
            plane, kz_value, n_grid, n_grid
        )

        # 计算 U(1) 联络
        u_fields = {}
        for i, v in enumerate(vertices):
            _, u = self.solve_eigenproblem(v)
            u_fields[i] = u[:, band_idx]

        # 计算 plaquette flux
        n1 = n_grid
        n2 = n_grid
        flux = np.zeros((n1, n2))

        for i in range(n1):
            for j in range(n2):
                # 四个顶点
                idx_00 = i * (n2 + 1) + j
                idx_10 = (i + 1) * (n2 + 1) + j
                idx_01 = i * (n2 + 1) + (j + 1)
                idx_11 = (i + 1) * (n2 + 1) + (j + 1)

                # 周期性边界
                if i == n1 - 1:
                    idx_10 = j
                    idx_11 = j + 1
                if j == n2 - 1:
                    idx_01 = i * (n2 + 1)
                    idx_11 = (i + 1) * (n2 + 1) if i < n1 - 1 else 0

                # U(1) links
                u1 = self._compute_link(u_fields[idx_00], u_fields[idx_10])
                u2 = self._compute_link(u_fields[idx_10], u_fields[idx_11])
                u3 = self._compute_link(u_fields[idx_01], u_fields[idx_11])
                u4 = self._compute_link(u_fields[idx_00], u_fields[idx_01])

                # plaquette = u1 * u2 * u3^* * u4^*
                plaquette = u1 * u2 * np.conj(u3) * np.conj(u4)

                # F_12 = Im ln(plaquette)
                flux[i, j] = np.imag(np.log(plaquette + 1e-15))

        # Chern 数 = (1/2π) Σ F
        chern_number = int(np.round(np.sum(flux) / (2 * np.pi)))
        return chern_number, flux

    def _compute_link(self, u_i: np.ndarray, u_j: np.ndarray) -> complex:
        """
        计算 U(1) link 变量
        U(k_i → k_j) = <u(k_i)|u(k_j)> / |<u(k_i)|u(k_j)>|
        """
        overlap = np.conj(u_i) @ u_j
        norm = np.abs(overlap)
        if norm < 1e-10:
            return 1.0 + 0.0j
        return overlap / norm

    def berry_curvature_high_order_fd(self, k: np.ndarray, band_idx: int = 0,
                                       order: int = 4, dk: float = 0.01) -> np.ndarray:
        """
        高阶有限差分计算 Berry curvature

        使用 O(dk^4) 精度的中心差分：
        ∂u/∂k_x ≈ [-u(k+2dk) + 8u(k+dk) - 8u(k-dk) + u(k-2dk)] / (12dk)

        Parameters:
        -----------
        k : np.ndarray
            k 点
        band_idx : int
            能带索引
        order : int
            差分阶数 (2 或 4)
        dk : float
            步长

        Returns:
        --------
        np.ndarray
            Berry curvature [Ω_xy, Ω_yz, Ω_zx]
        """
        if order == 2:
            return self._berry_curvature_fd2(k, band_idx, dk)
        elif order == 4:
            return self._berry_curvature_fd4(k, band_idx, dk)
        elif order == 6:
            # 使用四阶方法，但更小步长作为近似
            return self._berry_curvature_fd4(k, band_idx, dk/2)
        elif order == 8:
            return self._berry_curvature_fd4(k, band_idx, dk/3)
        else:
            raise ValueError(f"Unsupported order: {order}")

    def _berry_curvature_fd2(self, k: np.ndarray, band_idx: int,
                              dk: float) -> np.ndarray:
        """二阶有限差分 Berry curvature"""
        # 计算各方向导数
        du_dkx = self._gradient_fd2(k, 0, dk, band_idx)
        du_dky = self._gradient_fd2(k, 1, dk, band_idx)
        du_dkz = self._gradient_fd2(k, 2, dk, band_idx)

        # Ω_xy = -2 Im <∂u/∂kx | ∂u/∂ky>
        omega_xy = -2 * np.imag(np.conj(du_dkx) @ du_dky)
        omega_yz = -2 * np.imag(np.conj(du_dky) @ du_dkz)
        omega_zx = -2 * np.imag(np.conj(du_dkz) @ du_dkx)

        return np.array([omega_xy, omega_yz, omega_zx])

    def _berry_curvature_fd4(self, k: np.ndarray, band_idx: int,
                              dk: float) -> np.ndarray:
        """四阶有限差分 Berry curvature"""
        du_dkx = self._gradient_fd4(k, 0, dk, band_idx)
        du_dky = self._gradient_fd4(k, 1, dk, band_idx)
        du_dkz = self._gradient_fd4(k, 2, dk, band_idx)

        omega_xy = -2 * np.imag(np.conj(du_dkx) @ du_dky)
        omega_yz = -2 * np.imag(np.conj(du_dky) @ du_dkz)
        omega_zx = -2 * np.imag(np.conj(du_dkz) @ du_dkx)

        return np.array([omega_xy, omega_yz, omega_zx])

    def _gradient_fd2(self, k: np.ndarray, direction: int,
                       dk: float, band_idx: int) -> np.ndarray:
        """
        二阶中心差分梯度
        ∂u/∂k_i ≈ [u(k+dk·e_i) - u(k-dk·e_i)] / (2dk)
        """
        dk_vec = np.zeros(3)
        dk_vec[direction] = dk

        _, u_plus = self.solve_eigenproblem(k + dk_vec)
        _, u_minus = self.solve_eigenproblem(k - dk_vec)

        # 规范固定
        phase = np.conj(u_plus[:, band_idx]) @ u_minus[:, band_idx]
        phase = phase / (np.abs(phase) + 1e-15)
        u_minus = u_minus * phase

        return (u_plus[:, band_idx] - u_minus[:, band_idx]) / (2 * dk)

    def _gradient_fd4(self, k: np.ndarray, direction: int,
                       dk: float, band_idx: int) -> np.ndarray:
        """
        四阶中心差分梯度
        ∂u/∂k_i ≈ [-u(k+2dk) + 8u(k+dk) - 8u(k-dk) + u(k-2dk)] / (12dk)
        """
        dk_vec = np.zeros(3)
        dk_vec[direction] = dk

        _, u_p2 = self.solve_eigenproblem(k + 2*dk_vec)
        _, u_p1 = self.solve_eigenproblem(k + dk_vec)
        _, u_m1 = self.solve_eigenproblem(k - dk_vec)
        _, u_m2 = self.solve_eigenproblem(k - 2*dk_vec)

        # 规范对齐到 u_p1
        ref = u_p1[:, band_idx]
        for u in [u_p2, u_m1, u_m2]:
            phase = np.conj(ref) @ u[:, band_idx]
            phase = phase / (np.abs(phase) + 1e-15)
            u[:] = u * phase

        grad = (-u_p2[:, band_idx] + 8*u_p1[:, band_idx]
                - 8*u_m1[:, band_idx] + u_m2[:, band_idx]) / (12 * dk)
        return grad

    def wilson_loop(self, k_path: np.ndarray, band_idx: int = 0) -> np.ndarray:
        """
        Wilson loop 方法计算 Berry phase

        W = Π_n U(k_n → k_{n+1})
        θ = -Im ln(det W)

        Parameters:
        -----------
        k_path : np.ndarray
            k 点路径
        band_idx : int
            能带索引

        Returns:
        --------
        np.ndarray
            Wilson loop 相位
        """
        n_points = len(k_path)
        W = np.eye(1, dtype=complex)

        for i in range(n_points - 1):
            _, u1 = self.solve_eigenproblem(k_path[i])
            _, u2 = self.solve_eigenproblem(k_path[i + 1])

            overlap = np.conj(u1[:, band_idx]) @ u2[:, band_idx]
            link = overlap / (np.abs(overlap) + 1e-15)
            W = W * link

        # 闭合回路
        _, u1 = self.solve_eigenproblem(k_path[-1])
        _, u2 = self.solve_eigenproblem(k_path[0])
        overlap = np.conj(u1[:, band_idx]) @ u2[:, band_idx]
        link = overlap / (np.abs(overlap) + 1e-15)
        W = W * link

        berry_phase = -np.imag(np.log(W[0, 0] + 1e-15))
        return np.array([berry_phase])

    def compute_berry_curvature_field(self, k_points: np.ndarray,
                                       band_idx: int = 0,
                                       method: str = 'fd4',
                                       dk: float = 0.01) -> np.ndarray:
        """
        在网格点上批量计算 Berry curvature

        Parameters:
        -----------
        k_points : np.ndarray
            k 点网格
        band_idx : int
            能带索引
        method : str
            方法 ('kubo', 'fd2', 'fd4')
        dk : float
            有限差分步长

        Returns:
        --------
        np.ndarray
            Berry curvature 场 (N, 3)
        """
        n_points = len(k_points)
        omega_field = np.zeros((n_points, 3))

        for i, k in enumerate(k_points):
            if method == 'kubo':
                omega_field[i] = self.berry_curvature_kubo(k, band_idx)
            elif method == 'fd2':
                omega_field[i] = self.berry_curvature_high_order_fd(k, band_idx, 2, dk)
            elif method == 'fd4':
                omega_field[i] = self.berry_curvature_high_order_fd(k, band_idx, 4, dk)
            else:
                raise ValueError(f"Unknown method: {method}")

        return omega_field
