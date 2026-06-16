"""
qgp_operators.py — 高阶有限差分算子: WENO5 重构与 Lax-Friedrichs 分裂
=========================================================================

融合种子项目: 350_fd_predator_prey (有限差分法)

本模块实现求解相对论流体力学方程所需的高阶空间差分算子.
核心方法借鉴有限差分思想 (如 350_fd_predator_prey 中的 Euler 格式),
但大幅升级到五阶 WENO (Weighted Essentially Non-Oscillatory) 精度.

WENO5 重构 (Jiang & Shu 1996):
------------------------------

对于半离散形式 du/dt = -[F(u)]_x, 在网格点 x_i 处:

1. 特征分解: 将守恒量变换到局部特征变量
   w = L * u, 其中 L 为左特征向量矩阵

2. Lax-Friedrichs 通量分裂:
   F^{+}(u) = 0.5 * (F(u) + alpha * u)
   F^{-}(u) = 0.5 * (F(u) - alpha * u)
   其中 alpha = max |eigenvalues of dF/du| (全局 Lax-Friedrichs)

3. WENO5 重构:
   对 F^+ 和 F^- 分别在 i+1/2 界面进行五阶重构

   三个子模板:
   q_0 = (2*f_{i-2} - 7*f_{i-1} + 11*f_i) / 6
   q_1 = (-f_{i-1} + 5*f_i + 2*f_{i+1}) / 6
   q_2 = (2*f_i + 5*f_{i+1} - f_{i+2}) / 6

   光滑度指标:
   beta_0 = (13/12)*(f_{i-2} - 2*f_{i-1} + f_i)^2 + (1/4)*(f_{i-2} - 4*f_{i-1} + 3*f_i)^2
   beta_1 = (13/12)*(f_{i-1} - 2*f_i + f_{i+1})^2 + (1/4)*(f_{i-1} - f_{i+1})^2
   beta_2 = (13/12)*(f_i - 2*f_{i+1} + f_{i+2})^2 + (1/4)*(3*f_i - 4*f_{i+1} + f_{i+2})^2

   非线性权重:
   alpha_k = d_k / (epsilon + beta_k)^p
   omega_k = alpha_k / sum(alpha_j)

   最终重构:
   f_{i+1/2}^{+} = omega_0 * q_0 + omega_1 * q_1 + omega_2 * q_2

数值耗散:
    WENO5 的数值耗散正比于 dx^5 * |u^{(5)}|
    在光滑区域: 五阶精度
    在间断附近: 自动降为一阶 (ENO 选择), 避免 Gibbs 振荡

人工粘滞修正:
    F_artificial = -mu * nabla^2 u
    其中 mu ~ O(dx^2) 为人工粘滞系数
"""

import numpy as np
from qgp_config import NumericalParams
from qgp_grid import QGPGrid


class WENO5Operator:
    """
    WENO5 有限差分算子

    用于计算守恒律 du/dt + dF/dx + dG/dy = S 中的空间导数项.

    通量分裂采用全局 Lax-Friedrichs (Rusanov):
        F^+ = 0.5*(F + alpha*u),  F^- = 0.5*(F - alpha*u)
        alpha = max |lambda_max|  (全局最大特征速度)

    Attributes:
        grid: QGP 计算网格
        epsilon: WENO 小量 (防止除零)
        mu_artificial: 人工粘滞系数
    """

    def __init__(self, grid: QGPGrid):
        """
        初始化 WENO5 算子

        Args:
            grid: 计算网格对象
        """
        self.grid = grid
        self.nx = grid.nx
        self.ny = grid.ny
        self.ng = grid.ng
        self.dx = grid.dx
        self.dy = grid.dy
        self.epsilon = NumericalParams.WENO_EPSILON
        self.mu_artificial = NumericalParams.ARTIFICIAL_VISCOSITY

        # WENO5 理想线性权重 (三阶子模板)
        self.d_plus = np.array([1.0/10.0, 6.0/10.0, 3.0/10.0])
        self.d_minus = np.array([3.0/10.0, 6.0/10.0, 1.0/10.0])

    def weno5_reconstruct(self, f: np.ndarray, direction: str) -> np.ndarray:
        """
        WENO5 重构: 从 f_i 计算 f_{i+1/2}

        对一维数组 f[...], 在相邻网格点中间进行五阶 WENO 重构.

        数学公式:
            f_{i+1/2} = sum_{k=0}^{2} omega_k * q_k

        其中:
            q_0 = (2*f_{i-2} - 7*f_{i-1} + 11*f_i) / 6
            q_1 = (-f_{i-1} + 5*f_i + 2*f_{i+1}) / 6
            q_2 = (2*f_i + 5*f_{i+1} - f_{i+2}) / 6

        Args:
            f: 一维场数据 (长度 >= 5)
            direction: 'x' 或 'y', 决定差分方向

        Returns:
            f_{i+1/2} 重构值 (长度 = N-1)
        """
        eps = self.epsilon
        n = len(f)

        # 三个子模板的候选值
        q0 = (2.0 * f[:-4] - 7.0 * f[1:-3] + 11.0 * f[2:-2]) / 6.0
        q1 = (-f[1:-3] + 5.0 * f[2:-2] + 2.0 * f[3:-1]) / 6.0
        q2 = (2.0 * f[2:-2] + 5.0 * f[3:-1] - f[4:]) / 6.0

        # 光滑度指标
        beta0 = ((13.0/12.0) * (f[:-4] - 2.0*f[1:-3] + f[2:-2])**2 +
                 0.25 * (f[:-4] - 4.0*f[1:-3] + 3.0*f[2:-2])**2)
        beta1 = ((13.0/12.0) * (f[1:-3] - 2.0*f[2:-2] + f[3:-1])**2 +
                 0.25 * (f[1:-3] - f[3:-1])**2)
        beta2 = ((13.0/12.0) * (f[2:-2] - 2.0*f[3:-1] + f[4:])**2 +
                 0.25 * (3.0*f[2:-2] - 4.0*f[3:-1] + f[4:])**2)

        # 非线性权重
        alpha0 = self.d_plus[0] / (eps + beta0)**2
        alpha1 = self.d_plus[1] / (eps + beta1)**2
        alpha2 = self.d_plus[2] / (eps + beta2)**2
        alpha_sum = alpha0 + alpha1 + alpha2 + 1.0e-30

        omega0 = alpha0 / alpha_sum
        omega1 = alpha1 / alpha_sum
        omega2 = alpha2 / alpha_sum

        return omega0 * q0 + omega1 * q1 + omega2 * q2

    def weno5_reconstruct_left(self, f: np.ndarray) -> np.ndarray:
        """
        WENO5 左重构: f_{i+1/2}^{-} (使用右偏模板)

        用于负通量 F^- 的重构 (从右侧插值).
        与 weno5_reconstruct 对称但使用 d_minus 权重.
        """
        eps = self.epsilon
        # 镜像: 反转数组, 使用正重构, 再反转结果
        f_rev = f[::-1]
        n = len(f)

        q0 = (2.0 * f_rev[:-4] - 7.0 * f_rev[1:-3] + 11.0 * f_rev[2:-2]) / 6.0
        q1 = (-f_rev[1:-3] + 5.0 * f_rev[2:-2] + 2.0 * f_rev[3:-1]) / 6.0
        q2 = (2.0 * f_rev[2:-2] + 5.0 * f_rev[3:-1] - f_rev[4:]) / 6.0

        beta0 = ((13.0/12.0) * (f_rev[:-4] - 2.0*f_rev[1:-3] + f_rev[2:-2])**2 +
                 0.25 * (f_rev[:-4] - 4.0*f_rev[1:-3] + 3.0*f_rev[2:-2])**2)
        beta1 = ((13.0/12.0) * (f_rev[1:-3] - 2.0*f_rev[2:-2] + f_rev[3:-1])**2 +
                 0.25 * (f_rev[1:-3] - f_rev[3:-1])**2)
        beta2 = ((13.0/12.0) * (f_rev[2:-2] - 2.0*f_rev[3:-1] + f_rev[4:])**2 +
                 0.25 * (3.0*f_rev[2:-2] - 4.0*f_rev[3:-1] + f_rev[4:])**2)

        alpha0 = self.d_minus[0] / (eps + beta0)**2
        alpha1 = self.d_minus[1] / (eps + beta1)**2
        alpha2 = self.d_minus[2] / (eps + beta2)**2
        alpha_sum = alpha0 + alpha1 + alpha2 + 1.0e-30

        omega0 = alpha0 / alpha_sum
        omega1 = alpha1 / alpha_sum
        omega2 = alpha2 / alpha_sum

        result = omega0 * q0 + omega1 * q1 + omega2 * q2
        return result[::-1]

    def lf_flux_split(self, flux: np.ndarray, u: np.ndarray,
                       alpha: float) -> tuple:
        """
        Lax-Friedrichs 通量分裂

        F^+ = 0.5 * (F + alpha * u)  (右行波)
        F^- = 0.5 * (F - alpha * u)  (左行波)

        其中 alpha >= max|eigenvalue(dF/du)| 保证分裂的正确性.

        物理: 将通量分解为只向右传播和只向左传播的两部分,
        每部分都可以用单侧模板安全重构.

        Args:
            flux: 物理通量 F(u)
            u: 守恒变量
            alpha: 全局最大特征速度

        Returns:
            (F_plus, F_minus)
        """
        f_plus = 0.5 * (flux + alpha * u)
        f_minus = 0.5 * (flux - alpha * u)
        return f_plus, f_minus

    def spatial_derivative_x(self, field: np.ndarray,
                              alpha: float) -> np.ndarray:
        """
        计算 x 方向的空间导数 dF/dx (WENO5 + LF 分裂)

        半离散格式:
            (dF/dx)_i = (hat{F}_{i+1/2} - hat{F}_{i-1/2}) / dx

        其中 hat{F}_{i+1/2} = F^+_{i+1/2} + F^-_{i+1/2}
        (分别从左侧和右侧 WENO 重构)

        Args:
            field: 守恒变量场 (ny+2ng, nx+2ng)
            alpha: 最大特征速度

        Returns:
            dF/dx 在内部区域 (ny, nx)
        """
        ng = self.ng
        ny, nx = self.ny, self.nx
        result = np.zeros((ny, nx))

        for j in range(ny):
            # 提取 j 行 (含鬼单元), 需要 5 个点用于 WENO5
            row = field[ng + j, :]  # 长度 nx + 2*ng

            # 正向通量重构 (F^+)
            f_plus_recon = self.weno5_reconstruct(row, 'x')
            # 负向通量重构 (F^-)
            f_minus_recon = self.weno5_reconstruct_left(row)

            # 数值通量 at i+1/2
            f_hat = f_plus_recon + f_minus_recon

            # 导数: (hat{F}_{i+1/2} - hat{F}_{i-1/2}) / dx
            # f_hat 的长度 = nx + 2*ng - 4
            # 我们需要 f_hat[i+1/2] - f_hat[i-1/2] for i in [ng, ng+nx-1]
            idx_start = ng - 2  # 因为 WENO 消耗了 4 个点
            if idx_start >= 0 and idx_start + nx <= len(f_hat):
                result[j, :] = (f_hat[idx_start + 1:idx_start + nx + 1] -
                                f_hat[idx_start:idx_start + nx]) / self.dx
            else:
                # 回退到中心差分 (边界保护)
                result[j, :] = (row[ng + 1:ng + nx + 1] -
                                row[ng - 1:ng + nx - 1]) / (2.0 * self.dx)

        return result

    def spatial_derivative_y(self, field: np.ndarray,
                              alpha: float) -> np.ndarray:
        """
        计算 y 方向的空间导数 dG/dy

        与 spatial_derivative_x 相同但在 y 方向操作.

        Args:
            field: 守恒变量场
            alpha: 最大特征速度

        Returns:
            dG/dy 在内部区域
        """
        ng = self.ng
        ny, nx = self.ny, self.nx
        result = np.zeros((ny, nx))

        for i in range(nx):
            col = field[:, ng + i]  # y 方向列
            f_plus_recon = self.weno5_reconstruct(col, 'y')
            f_minus_recon = self.weno5_reconstruct_left(col)
            f_hat = f_plus_recon + f_minus_recon

            idx_start = ng - 2
            if idx_start >= 0 and idx_start + ny <= len(f_hat):
                result[:, i] = (f_hat[idx_start + 1:idx_start + ny + 1] -
                                f_hat[idx_start:idx_start + ny]) / self.dy
            else:
                result[:, i] = (col[ng + 1:ng + ny + 1] -
                                col[ng - 1:ng + ny - 1]) / (2.0 * self.dy)

        return result

    def artificial_viscosity(self, field: np.ndarray) -> np.ndarray:
        """
        人工粘滞项 (熵修正)

        F_visc = mu * nabla^2 u

        用于在强间断处提供额外耗散, 防止非物理振荡.
        系数 mu ~ O(dx^2) 保证不破坏整体精度.

        使用标准二阶中心差分 Laplacian.

        Args:
            field: 场 (含鬼单元)

        Returns:
            人工粘滞修正 (内部区域)
        """
        ng = self.ng
        ny, nx = self.ny, self.nx
        f = self.grid.apply_boundary_conditions(field)

        # 二阶 Laplacian
        lap_x = (f[ng:ng+ny, ng+1:ng+nx+1] - 2.0*f[ng:ng+ny, ng:ng+nx] +
                 f[ng:ng+ny, ng-1:ng+nx-1]) / self.dx**2
        lap_y = (f[ng+1:ng+ny+1, ng:ng+nx] - 2.0*f[ng:ng+ny, ng:ng+nx] +
                 f[ng-1:ng+ny-1, ng:ng+nx]) / self.dy**2

        return self.mu_artificial * (lap_x + lap_y)

    def compute_rhs(self, u: np.ndarray, flux_func,
                     source_func=None, alpha: float = 1.0) -> np.ndarray:
        """
        计算空间离散化的右端项

        du/dt = -dF/dx - dG/dy + S + mu*nabla^2(u)

        其中:
            F, G 为 x, y 方向的物理通量
            S 为源项 (Bjorken 膨胀几何项)
            mu*nabla^2(u) 为人工粘滞

        通量分裂:
            F = flux_func(u), alpha 为最大特征速度

        Args:
            u: 守恒变量 (含鬼单元)
            flux_func: 通量函数, 返回 (F_x, F_y)
            source_func: 源项函数 (可选)
            alpha: 最大特征速度

        Returns:
            右端项 (内部区域 ny, nx)
        """
        # 施加边界条件
        u = self.grid.apply_boundary_conditions(u)

        # 计算物理通量
        F_x, F_y = flux_func(u)

        # WENO5 空间导数
        dFdx = self.spatial_derivative_x(F_x, alpha)
        dGdy = self.spatial_derivative_y(F_y, alpha)

        # 人工粘滞
        visc = self.artificial_viscosity(u)

        # 右端项
        rhs = -(dFdx + dGdy) + visc

        # 源项
        if source_func is not None:
            rhs += source_func(u)

        return rhs
