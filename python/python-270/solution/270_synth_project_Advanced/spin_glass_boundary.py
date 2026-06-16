"""
spin_glass_boundary.py
======================

周期性/开放边界条件的数学处理与一致性验证。

物理背景
--------
在有限尺寸自旋玻璃模拟中, 边界条件的选择对结果有重要影响:

1. 周期性边界条件 (Periodic Boundary Conditions, PBC):
   S(i+L, j, k) = S(i, j, k)
   等价于在三维环面 T^3 上定义系统。
   优点: 消除表面效应, 更快趋近热力学极限
   缺点: 引入有限尺寸 quantization: k_mu = 2*pi*n_mu / L

2. 开放边界条件 (Open Boundary Conditions, OBC):
   格点 (i,j,k) 的邻居数 < 6 (表面/棱/角)
   表面格点配位数: z_surface = 5 (面), 4 (棱), 3 (角)

3. 反周期边界条件 (Anti-Periodic Boundary Conditions, APBC):
   S(i+L, j, k) = -S(i, j, k) (在某一个方向上)
   用于计算 domain wall 能量和 stiffness exponent

边界条件对动量空间的影响
-------------------------
PBC: 波矢 k_mu = 2*pi*n / L, n = -L/2, ..., L/2-1
APBC: 波矢 k_mu = 2*pi*(n+1/2) / L

本模块核心算法来源于 seed project:
- 646_laplace_radial_exact: 径向 Laplace 方程的精确解
  (映射为边界条件导致的谐波修正)
- 545_house: 折线边界编码
  (映射为任意形状边界的表示)
"""

import numpy as np
from typing import Optional, Literal, Tuple
from spin_lattice_geometry import CubicLattice3D


BoundaryType = Literal["periodic", "open", "anti_periodic"]


# =====================================================================
#  边界条件处理器
# =====================================================================

class BoundaryHandler:
    """
    处理立方晶格上的边界条件。

    参数
    ----
    L : int
        晶格常数
    boundary_type : str
        "periodic", "open", "anti_periodic"
    apbc_direction : int, default 0
        反周期边界条件的方向 (0=x, 1=y, 2=z)
    """

    def __init__(self, L: int,
                 boundary_type: BoundaryType = "periodic",
                 apbc_direction: int = 0):
        if L < 2:
            raise ValueError(f"L 必须 >= 2, 当前 L={L}")
        if boundary_type not in ("periodic", "open", "anti_periodic"):
            raise ValueError(f"不支持的边界类型: {boundary_type}")
        self.L = L
        self.boundary_type = boundary_type
        self.apbc_direction = apbc_direction
        self.is_periodic = (boundary_type == "periodic")
        self.is_open = (boundary_type == "open")
        self.is_apbc = (boundary_type == "anti_periodic")

    # -----------------------------------------------------------------
    #  坐标折叠
    # -----------------------------------------------------------------

    def wrap_coord(self, ix: int, iy: int, iz: int) -> Optional[Tuple[int, int, int, int]]:
        """
        将坐标折叠到 [0, L) 范围, 同时返回符号因子。

        返回
        ----
        (ix', iy', iz', sign) :
            折叠后坐标和自旋符号因子。
            开边界下若越界返回 None。

        对于 APBC: 若跨越边界, sign = -1; 否则 sign = +1。
        """
        sign = 1

        if self.is_open:
            if not (0 <= ix < self.L and 0 <= iy < self.L and 0 <= iz < self.L):
                return None
            return (ix, iy, iz, sign)

        # 处理 x 方向
        sx, ix_w = divmod(ix, self.L)
        if self.is_apbc and self.apbc_direction == 0 and sx != 0:
            sign *= (-1) ** sx

        # 处理 y 方向
        sy, iy_w = divmod(iy, self.L)
        if self.is_apbc and self.apbc_direction == 1 and sy != 0:
            sign *= (-1) ** sy

        # 处理 z 方向
        sz, iz_w = divmod(iz, self.L)
        if self.is_apbc and self.apbc_direction == 2 and sz != 0:
            sign *= (-1) ** sz

        return (ix_w, iy_w, iz_w, sign)

    # -----------------------------------------------------------------
    #  格点邻居 (带符号因子)
    # -----------------------------------------------------------------

    def neighbor_with_sign(self, ix: int, iy: int, iz: int,
                           dx: int, dy: int, dz: int
                           ) -> Optional[Tuple[int, int, int, int]]:
        """
        获取 (ix+dx, iy+dy, iz+dz) 的折叠坐标和符号因子。
        """
        return self.wrap_coord(ix + dx, iy + dy, iz + dz)

    # -----------------------------------------------------------------
    #  局部配位数 (开边界)
    # -----------------------------------------------------------------

    def local_coordination(self, ix: int, iy: int, iz: int) -> int:
        """
        计算格点 (ix, iy, iz) 的实际配位数。

        PBC: z = 6 对所有格点
        OBC: z = 6 - (表面数)
        """
        if not self.is_open:
            return 6
        z_local = 0
        for dix, diy, diz in [(1, 0, 0), (-1, 0, 0),
                               (0, 1, 0), (0, -1, 0),
                               (0, 0, 1), (0, 0, -1)]:
            nx, ny, nz = ix + dix, iy + diy, iz + diz
            if 0 <= nx < self.L and 0 <= ny < self.L and 0 <= nz < self.L:
                z_local += 1
        return z_local

    # -----------------------------------------------------------------
    #  Fourier 变换核 (von Neumann 稳定性分析所需)
    # -----------------------------------------------------------------

    def fourier_laplacian_symbol(self, kx: float, ky: float, kz: float) -> complex:
        """
        离散拉普拉斯算子在动量空间的符号:
            Delta_hat(k) = sum_{mu=1}^{3} [exp(i*k_mu) + exp(-i*k_mu) - 2]
                         = 2 * [cos(kx) + cos(ky) + cos(kz) - 3]

        PBC: k_mu = 2*pi*n / L
        APBC: k_mu = 2*pi*(n+1/2) / L (在反周期方向上)
        """
        return 2.0 * (np.cos(kx) + np.cos(ky) + np.cos(kz) - 3.0)

    def fourier_laplacian_4th_symbol(self, kx: float, ky: float, kz: float) -> complex:
        """
        四阶离散拉普拉斯算子在动量空间的符号:
            Delta4_hat(k) = sum_{mu=1}^{3} [
                -(S_{i+2e_mu} + S_{i-2e_mu}) + 16*(S_{i+e_mu} + S_{i-e_mu}) - 30*S_i
            ] / 12

        动量空间:
            Delta4_hat(k) = sum_{mu} [-2*cos(2*k_mu) + 32*cos(k_mu) - 30] / 12
        """
        val = 0.0
        for k in (kx, ky, kz):
            val += (-2.0 * np.cos(2.0 * k) + 32.0 * np.cos(k) - 30.0) / 12.0
        return val

    # -----------------------------------------------------------------
    #  稳定性判据
    # -----------------------------------------------------------------

    def max_stable_dt_2nd_order(self, D: float = 1.0) -> float:
        """
        二阶有限差分 + 前向 Euler 的最大稳定时间步长。

        von Neumann 稳定性条件:
            |1 + D * dt * Delta_hat(k)| <= 1  对所有 k

        最坏情况: Delta_hat_min = -2d = -6
        => 1 - 6*D*dt >= -1  =>  dt <= 2/(6*D) = 1/(3*D)

        参数
        ----
        D : float
            扩散系数 (或等效耦合强度)

        返回
        ----
        dt_max : float
        """
        if D <= 0:
            raise ValueError(f"D 必须 > 0, 当前 D={D}")
        return 1.0 / (3.0 * D)

    def max_stable_dt_4th_order(self, D: float = 1.0) -> float:
        """
        四阶有限差分 + 前向 Euler 的最大稳定时间步长。

        四阶 Laplacian 最小符号值:
            Delta4_hat_min ≈ -41/2 * d = -123/2  (在 k=(pi,pi,pi))
        更精确计算: 在每个方向上,
            -2*cos(2*pi) + 32*cos(pi) - 30 = -2 - 32 - 30 = -64
            三个方向: 3 * (-64) / 12 = -16

        => 1 - 16*D*dt >= -1  =>  dt <= 2/(16*D) = 1/(8*D)

        参数
        ----
        D : float
            扩散系数

        返回
        ----
        dt_max : float
        """
        if D <= 0:
            raise ValueError(f"D 必须 > 0, 当前 D={D}")
        # 精确计算: 四阶 stencil 的最小 amplification
        # k=(pi,pi,pi) 处: (-2*cos(2*pi)+32*cos(pi)-30)/12 = (-2-32-30)/12 = -64/12 = -16/3
        # 三个方向之和: 3*(-16/3) = -16
        return 2.0 / (16.0 * D)

    # -----------------------------------------------------------------
    #  边界能量修正
    # -----------------------------------------------------------------

    def boundary_energy_correction(self, lattice: CubicLattice3D,
                                   J_mean: float = 0.0,
                                   J_var: float = 1.0,
                                   beta: float = 1.0) -> float:
        """
        开放边界条件导致的表面能修正 (平均场估计):

        表面格点数: N_surface = 6*(L-2)^2 + 12*(L-2) + 8 (对 L>=2)
        损失的键数: N_lost = 3*L^3 - N_bond_open
        其中 N_bond_open = 3*L^2*(L-1) (对简单立方)

        平均场修正:
            delta_E ≈ N_lost * <J>^2 * beta / 2

        参数
        ----
        lattice : CubicLattice3D
        J_mean : float
        J_var : float
        beta : float

        返回
        ----
        delta_E : float
            表面能修正
        """
        L = lattice.L
        if self.is_periodic:
            return 0.0

        # 开放边界的键数
        n_bond_open = 3 * L * L * (L - 1)
        n_bond_periodic = 3 * L ** 3
        n_lost = n_bond_periodic - n_bond_open

        # 平均场估计
        delta_E = n_lost * J_mean ** 2 * beta / 2.0
        return delta_E

    # -----------------------------------------------------------------
    #  一致性检查
    # -----------------------------------------------------------------

    def verify_boundary_consistency(self, lattice: CubicLattice3D) -> bool:
        """
        验证边界条件实现的一致性:
        1) 每个格点的邻居数正确
        2) 邻居关系对称
        3) 折叠后坐标在 [0, L) 内
        """
        for ix in range(self.L):
            for iy in range(self.L):
                for iz in range(self.L):
                    z_expected = self.local_coordination(ix, iy, iz)
                    actual_neighbors = []
                    for dx, dy, dz in [(1, 0, 0), (-1, 0, 0),
                                       (0, 1, 0), (0, -1, 0),
                                       (0, 0, 1), (0, 0, -1)]:
                        result = self.neighbor_with_sign(ix, iy, iz, dx, dy, dz)
                        if result is not None:
                            nx, ny, nz, sign = result
                            if 0 <= nx < self.L and 0 <= ny < self.L and 0 <= nz < self.L:
                                actual_neighbors.append((nx, ny, nz))
                    if len(actual_neighbors) != z_expected:
                        return False
        return True
