"""
eigen_buckling.py
受损复合材料层合板的屈曲与自由振动特征值分析。
原项目映射：
  - 1206_test_eigen 的对称/非对称矩阵特征结构生成与正交变换
  - r8nsymm_gen, r8symm_gen 的Schur分解思想
科学背景：
  经典层合板屈曲方程：
    D_11 ∂^4w/∂x^4 + 2(D_12 + 2D_66) ∂^4w/∂x^2∂y^2 + D_22 ∂^4w/∂y^4
    + N_x ∂^2w/∂x^2 + 2N_xy ∂^2w/∂x∂y + N_y ∂^2w/∂y^2 = 0
  简化为特征值问题：
    (K_buckling - λ K_geo) φ = 0
  其中 λ = N_cr / N_ref 为屈曲载荷系数。
  自由振动方程：
    (K - ω^2 M) φ = 0
  对于损伤层合板，D_ij 替换为退化后的 D̃_ij(d_f, d_m, d_s)。
  特征值问题求解：A x = λ x，使用QR算法或幂法。
"""

import numpy as np
from utils import validate_matrix_nonsingular


def generate_orthogonal_matrix(n):
    """
    生成随机正交矩阵 Q（从 r8mat_orth_uniform 迁移）。
    使用QR分解：对随机矩阵 A 做 QR = A，Q 即为正交矩阵。
    """
    A = np.random.randn(n, n)
    Q, R = np.linalg.qr(A)
    # 保证行列式为+1（真旋转）
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1.0
    return Q


def generate_symmetric_eigenproblem(n, lambda_mean=1.0, lambda_dev=0.1):
    """
    生成具有指定特征结构的对称矩阵（从 r8symm_gen 迁移）。
    A = Q * diag(λ) * Q^T
    用于模拟层合板刚度矩阵的特征结构。
    """
    lambda_vals = lambda_mean + lambda_dev * np.random.randn(n)
    lambda_vals = np.sort(lambda_vals)[::-1]  # 降序
    Q = generate_orthogonal_matrix(n)
    A = Q @ np.diag(lambda_vals) @ Q.T
    return A, Q, lambda_vals


def generate_nonsymmetric_eigenproblem(n, lambda_mean=1.0, lambda_dev=0.1):
    """
    生成具有指定特征结构的非对称矩阵（从 r8nsymm_gen 迁移）。
    A = Q^T * T * Q，其中 T 为上三角Schur矩阵。
    用于模拟含阻尼或损伤耦合的动力学系统。
    """
    T = np.triu(np.random.randn(n, n))
    np.fill_diagonal(T, lambda_mean + lambda_dev * np.random.randn(n))
    Q = generate_orthogonal_matrix(n)
    A = Q.T @ T @ Q
    return A, Q, T


class BucklingAnalysis:
    """层合板屈曲特征值分析。"""

    def __init__(self, D_matrix, plate_length, plate_width, nx=20, ny=20):
        """
        D_matrix: 3x3 弯曲刚度矩阵 D_ij
        plate_length, plate_width: 板尺寸
        nx, ny: 离散网格数
        """
        self.D = np.asarray(D_matrix)
        self.L = float(plate_length)
        self.W = float(plate_width)
        self.nx = nx
        self.ny = ny
        self.dx = self.L / (nx - 1)
        self.dy = self.W / (ny - 1)

    def build_bending_stiffness_matrix(self):
        """
        构建弯曲刚度矩阵 K_b（基于有限差分离散）。
        双调和算子 ∇^4 w 的5点差分近似：
          ∂^4w/∂x^4 ≈ (w_{i-2,j} - 4w_{i-1,j} + 6w_{i,j} - 4w_{i+1,j} + w_{i+2,j}) / dx^4
        """
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
                # 中心点
                K_b[idx, idx] = 6.0 * coeff_x + 6.0 * coeff_y + 4.0 * coeff_xy

                # x方向邻居
                if i > 0:
                    K_b[idx, idx - 1] += -4.0 * coeff_x
                if i < self.nx - 1:
                    K_b[idx, idx + 1] += -4.0 * coeff_x
                if i > 1:
                    K_b[idx, idx - 2] += coeff_x
                if i < self.nx - 2:
                    K_b[idx, idx + 2] += coeff_x

                # y方向邻居
                if j > 0:
                    K_b[idx, idx - self.nx] += -4.0 * coeff_y
                if j < self.ny - 1:
                    K_b[idx, idx + self.nx] += -4.0 * coeff_y
                if j > 1:
                    K_b[idx, idx - 2 * self.nx] += coeff_y
                if j < self.ny - 2:
                    K_b[idx, idx + 2 * self.nx] += coeff_y

                # 混合项 (x,y)
                for di, dj in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < self.nx and 0 <= nj < self.ny:
                        K_b[idx, nj * self.nx + ni] += coeff_xy

        # 简支边界条件处理
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
        """
        构建几何刚度矩阵 K_g（膜力引起的等效刚度）。
        """
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

        # 边界条件
        for i in range(self.nx):
            K_g[i, :] = 0.0
            K_g[(self.ny - 1) * self.nx + i, :] = 0.0
        for j in range(self.ny):
            K_g[j * self.nx, :] = 0.0
            K_g[j * self.nx + self.nx - 1, :] = 0.0

        return K_g

    def solve_buckling_loads(self, N_x=1.0, N_y=0.0, N_xy=0.0, n_modes=5):
        """
        求解屈曲特征值问题：(K_b - λ K_g) φ = 0。
        返回前 n_modes 个屈曲载荷系数 λ 和模态 φ。
        """
        K_b = self.build_bending_stiffness_matrix()
        K_g = self.build_geometric_stiffness_matrix(N_x, N_y, N_xy)

        # 广义特征值问题：使用特征值分解
        try:
            # K_b u = λ K_g u  =>  K_g^{-1} K_b u = (1/λ) u
            K_g_inv = np.linalg.inv(K_g + 1e-8 * np.eye(K_g.shape[0]))
            M = K_g_inv @ K_b
            eigvals, eigvecs = np.linalg.eig(M)
            # λ_buckling = 1 / eigvals
            buckling_vals = np.real(1.0 / eigvals)
        except np.linalg.LinAlgError:
            K_b_inv = np.linalg.pinv(K_b)
            eigvals, eigvecs = np.linalg.eig(K_b_inv @ K_g)
            buckling_vals = np.real(eigvals)

        # 取正值并排序
        valid = buckling_vals > 1e-6
        if not np.any(valid):
            return np.array([np.inf]), np.zeros((K_b.shape[0], 1))

        sorted_idx = np.argsort(buckling_vals[valid])
        lambdas = buckling_vals[valid][sorted_idx][:n_modes]
        modes = eigvecs[:, valid][:, sorted_idx][:, :n_modes]

        return lambdas, modes

    def compute_critical_buckling_load(self, N_x=1.0):
        """计算临界屈曲载荷。"""
        lambdas, _ = self.solve_buckling_loads(N_x=N_x, n_modes=1)
        if len(lambdas) > 0 and lambdas[0] < np.inf:
            return lambdas[0]
        return np.inf


class VibrationAnalysis:
    """自由振动特征值分析。"""

    def __init__(self, D_matrix=None, rho=1600.0, thickness=1.0, plate_length=100.0, plate_width=100.0, nx=12, ny=12):
        self.D = np.asarray(D_matrix) if D_matrix is not None else np.eye(3)
        self.rho = float(rho)
        self.h = float(thickness)
        self.L = float(plate_length)
        self.W = float(plate_width)
        self.nx = nx
        self.ny = ny

    def solve_natural_frequencies(self, D_matrix, rho, thickness, plate_length, plate_width, nx=20, ny=20):
        """
        求解自由振动频率：(K - ω^2 M) φ = 0。
        返回自然频率 (rad/s) 和振型。
        """
        buckling = BucklingAnalysis(D_matrix, plate_length, plate_width, nx, ny)
        K = buckling.build_bending_stiffness_matrix()
        n = K.shape[0]

        # 一致质量矩阵（简化 lumped）
        dx = plate_length / (nx - 1)
        dy = plate_width / (ny - 1)
        M = np.eye(n) * rho * thickness * dx * dy

        try:
            K_inv = np.linalg.inv(K)
            eigvals, eigvecs = np.linalg.eig(K_inv @ M)
        except np.linalg.LinAlgError:
            K_inv = np.linalg.pinv(K)
            eigvals, eigvecs = np.linalg.eig(K_inv @ M)

        # ω^2 = 1/λ，取正实部
        omega_sq = np.real(1.0 / eigvals)
        valid = omega_sq > 1e-6
        if not np.any(valid):
            return np.array([]), np.zeros((n, 0))

        omega_sq_valid = omega_sq[valid]
        sorted_idx = np.argsort(omega_sq_valid)
        omegas = np.sqrt(omega_sq_valid[sorted_idx])
        modes = eigvecs[:, valid][:, sorted_idx]

        return omegas, modes
