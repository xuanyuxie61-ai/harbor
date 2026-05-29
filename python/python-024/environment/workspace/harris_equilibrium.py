r"""
harris_equilibrium.py
===================
基于 Harris 电流片模型的太阳耀斑磁重联平衡态构造，
并在四边形有限元网格上进行双线性插值与物理场重构。

核心物理模型
------------
Harris 电流片是磁重联研究中最经典的平衡态解析解，描述
太阳日冕中相反方向磁场之间的过渡层。

磁场分布（一维电流片，沿 y 方向变化）：
    B_x(y) = B_0 * tanh(y / lambda)
    B_y = 0
    B_z(y) = B_{g}                     (引导场)

压强分布（由力学平衡 nabla p = J x B 导出）：
    p(y) = p_0 + (B_0^2 / (2*mu_0)) * sech^2(y / lambda)

电流密度（由安培定律 J = (1/mu_0) nabla x B）：
    J_z(y) = (B_0 / (mu_0 * lambda)) * sech^2(y / lambda)

等离子体密度（理想气体状态方程 p = n*k_B*T，假设等温）：
    rho(y) = rho_0 * sech^2(y / lambda) + rho_{\infty}

四边形网格双线性形函数：
    N_1(xi, eta) = (1-xi)(1-eta)/4
    N_2(xi, eta) = (1+xi)(1-eta)/4
    N_3(xi, eta) = (1+xi)(1+eta)/4
    N_4(xi, eta) = (1-xi)(1+eta)/4

其中 (xi, eta) in [-1, 1]^2 为参考坐标。

融入原项目:
- 956_quadrilateral_surface_display: 四边形网格节点插值与双线性形函数
- 503_hand_mesh2d: 2D 网格拓扑结构生成思想
"""

import numpy as np
from typing import Tuple, Optional

# 物理常数（SI单位）
MU_0 = 4.0 * np.pi * 1e-7          # 真空磁导率 [H/m]
K_B = 1.380649e-23                 # 玻尔兹曼常数 [J/K]
M_P = 1.6726219e-27                # 质子质量 [kg]


class HarrisEquilibrium:
    """
    Harris 电流片平衡态构造器。
    """

    def __init__(self,
                 B0: float = 1.0e-2,          # 背景磁场 [T]
                 lambda_cs: float = 5.0e4,    # 电流片半厚度 [m]
                 B_guide: float = 2.0e-3,     # 引导场 [T]
                 p0: float = 2.0e-3,          # 背景压强 [Pa]
                 T_plasma: float = 2.0e6,     # 等离子体温度 [K]
                 rho_inf: float = 1.0e-12,    # 背景密度 [kg/m^3]
                 y_max: float = 3.0e5):       # 计算域半宽 [m]
        """
        初始化 Harris 电流片参数，并进行边界检查。
        """
        if B0 <= 0.0:
            raise ValueError("B0 必须为正")
        if lambda_cs <= 0.0:
            raise ValueError("lambda_cs 必须为正")
        if T_plasma <= 0.0:
            raise ValueError("T_plasma 必须为正")
        if y_max <= 0.0:
            raise ValueError("y_max 必须为正")

        self.B0 = B0
        self.lambda_cs = lambda_cs
        self.B_guide = B_guide
        self.p0 = p0
        self.T_plasma = T_plasma
        self.rho_inf = rho_inf
        self.y_max = y_max

        # 由状态方程计算特征密度
        self.n0 = p0 / (K_B * T_plasma)          # 背景数密度 [m^-3]
        self.rho0 = self.n0 * M_P                # 背景质量密度 [kg/m^3]

    def B_field(self, y: np.ndarray) -> np.ndarray:
        """
        计算磁场 B = (B_x, B_y, B_z)。
        公式: B_x(y) = B_0 * tanh(y / lambda)
        """
        y = np.asarray(y, dtype=float)
        if np.any(np.abs(y) > 1e10):
            raise ValueError("y 值过大，可能导致数值溢出")

        Bx = self.B0 * np.tanh(y / self.lambda_cs)
        By = np.zeros_like(y)
        Bz = np.full_like(y, self.B_guide)
        return np.stack([Bx, By, Bz], axis=-1)

    def pressure(self, y: np.ndarray) -> np.ndarray:
        """
        计算热压强。
        公式: p(y) = p_0 + (B_0^2 / (2*mu_0)) * sech^2(y / lambda)
        """
        y = np.asarray(y, dtype=float)
        sech2 = 1.0 / np.cosh(y / self.lambda_cs) ** 2
        return self.p0 + (self.B0 ** 2 / (2.0 * MU_0)) * sech2

    def current_density(self, y: np.ndarray) -> np.ndarray:
        """
        计算电流密度 J = (1/mu_0) nabla x B。
        公式: J_z(y) = (B_0 / (mu_0 * lambda)) * sech^2(y / lambda)
        """
        y = np.asarray(y, dtype=float)
        sech2 = 1.0 / np.cosh(y / self.lambda_cs) ** 2
        Jz = (self.B0 / (MU_0 * self.lambda_cs)) * sech2
        Jx = np.zeros_like(y)
        Jy = np.zeros_like(y)
        return np.stack([Jx, Jy, Jz], axis=-1)

    def mass_density(self, y: np.ndarray) -> np.ndarray:
        """
        计算质量密度（假设等温）。
        公式: rho(y) = rho_inf + rho_0 * sech^2(y / lambda)
        """
        y = np.asarray(y, dtype=float)
        sech2 = 1.0 / np.cosh(y / self.lambda_cs) ** 2
        return self.rho_inf + self.rho0 * sech2

    def alfven_speed(self, y: np.ndarray) -> np.ndarray:
        """
        计算阿尔芬速度。
        公式: v_A = |B| / sqrt(mu_0 * rho)
        """
        # HOLE 1: 请实现阿尔芬速度的计算公式
        # 提示: v_A = |B| / sqrt(mu_0 * rho)，注意处理 rho 接近零的情况
        raise NotImplementedError("Hole 1: 请实现阿尔芬速度公式 v_A = |B| / sqrt(mu_0 * rho)")

    def plasma_beta(self, y: np.ndarray) -> np.ndarray:
        """
        计算等离子体 beta = 2*mu_0*p / |B|^2。
        """
        p = self.pressure(y)
        B = self.B_field(y)
        B2 = np.sum(B ** 2, axis=-1)
        B2_safe = np.where(B2 < 1e-30, 1e-30, B2)
        return 2.0 * MU_0 * p / B2_safe

    def generate_quadrilateral_mesh(self, nx: int = 32, ny: int = 64) -> Tuple[np.ndarray, np.ndarray]:
        """
        在 [0, L_x] x [-y_max, y_max] 上生成结构化四边形网格。
        返回: nodes (nnodes, 2), elements (nelems, 4)
        """
        if nx < 2 or ny < 2:
            raise ValueError("nx 和 ny 至少为 2")

        Lx = 2.0 * self.y_max  # 取 x 方向宽度与 y 方向相同
        x = np.linspace(0.0, Lx, nx)
        # y 方向在电流片附近加密
        y_uniform = np.linspace(-1.0, 1.0, ny)
        # 使用 tanh 映射在电流片附近加密
        y = self.y_max * np.tanh(1.5 * y_uniform) / np.tanh(1.5)

        X, Y = np.meshgrid(x, y, indexing='ij')
        nodes = np.column_stack([X.ravel(), Y.ravel()])

        nelems = (nx - 1) * (ny - 1)
        elements = np.zeros((nelems, 4), dtype=int)
        idx = 0
        for i in range(nx - 1):
            for j in range(ny - 1):
                n1 = i * ny + j
                n2 = (i + 1) * ny + j
                n3 = (i + 1) * ny + (j + 1)
                n4 = i * ny + (j + 1)
                elements[idx] = [n1, n2, n3, n4]
                idx += 1
        return nodes, elements

    def bilinear_interpolate_on_mesh(self, nodes: np.ndarray,
                                      elements: np.ndarray,
                                      field_1d: np.ndarray,
                                      y_coords: np.ndarray) -> np.ndarray:
        r"""
        将一维 y 方向物理场通过双线性插值映射到四边形网格节点上。
        使用四边形参考单元上的双线性形函数进行插值。

        形函数:
            N1 = (1-\xi)(1-\eta)/4
            N2 = (1+\xi)(1-\eta)/4
            N3 = (1+\xi)(1+\eta)/4
            N4 = (1-\xi)(1+\eta)/4
        """
        if len(field_1d) != len(y_coords):
            raise ValueError("field_1d 与 y_coords 长度不匹配")
        if nodes.shape[1] != 2:
            raise ValueError("nodes 必须是二维坐标")

        # 先对一维 y 场做安全插值（单调检查）
        y_coords = np.asarray(y_coords)
        if not np.all(np.diff(y_coords) >= 0):
            # 如果不是单调递增，排序
            sort_idx = np.argsort(y_coords)
            y_coords = y_coords[sort_idx]
            field_1d = field_1d[sort_idx]

        field_2d = np.zeros(len(nodes))
        for i, (xn, yn) in enumerate(nodes):
            # 边界截断
            yn_clip = np.clip(yn, y_coords[0], y_coords[-1])
            field_2d[i] = np.interp(yn_clip, y_coords, field_1d)
        return field_2d

    def compute_reconnection_rate(self, y: np.ndarray, eta: np.ndarray, v: np.ndarray) -> np.ndarray:
        """
        计算磁重联电场（重联率）。
        广义欧姆定律中的电场:
            E_z = eta * J_z + (v x B)_z
        在二维 X 点重联中，重联率通常定义为 E_z(X-point)。
        """
        y = np.asarray(y, dtype=float)
        eta = np.asarray(eta, dtype=float)
        v = np.asarray(v, dtype=float)
        if v.shape[-1] != 3:
            raise ValueError("v 必须是三维速度场")

        J = self.current_density(y)
        B = self.B_field(y)
        # v x B 的 z 分量 = v_x * B_y - v_y * B_x
        v_cross_B_z = v[..., 0] * B[..., 1] - v[..., 1] * B[..., 0]
        Jz = J[..., 2]
        E_rec = eta * Jz + v_cross_B_z
        return E_rec

    def magnetic_shear(self, y: np.ndarray) -> np.ndarray:
        """
        计算磁剪切率 dB_x/dy，表征电流片的电流集中程度。
        公式: shear(y) = (B_0 / lambda) * sech^2(y / lambda)
        """
        y = np.asarray(y, dtype=float)
        sech2 = 1.0 / np.cosh(y / self.lambda_cs) ** 2
        return (self.B0 / self.lambda_cs) * sech2


def demo_harris():
    """
    快速验证 Harris 电流片基本物理量。
    """
    eq = HarrisEquilibrium()
    y = np.linspace(-eq.y_max, eq.y_max, 129)
    B = eq.B_field(y)
    p = eq.pressure(y)
    J = eq.current_density(y)
    rho = eq.mass_density(y)
    va = eq.alfven_speed(y)
    beta = eq.plasma_beta(y)
    shear = eq.magnetic_shear(y)

    print("[HarrisEquilibrium] 电流片中心 (y=0) 物理量:")
    print(f"  B_x(0) = {B[len(y)//2, 0]:.3e} T")
    print(f"  p(0)   = {p[len(y)//2]:.3e} Pa")
    print(f"  J_z(0) = {J[len(y)//2, 2]:.3e} A/m^2")
    print(f"  rho(0) = {rho[len(y)//2]:.3e} kg/m^3")
    print(f"  v_A(0) = {va[len(y)//2]:.3e} m/s")
    print(f"  beta(0)= {beta[len(y)//2]:.3f}")
    print(f"  shear(0)= {shear[len(y)//2]:.3e} T/m")
    return eq, y, B, p, J, rho, va, beta, shear


if __name__ == "__main__":
    demo_harris()
