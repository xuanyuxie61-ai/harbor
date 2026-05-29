# -*- coding: utf-8 -*-
"""
transport_operator.py
平流层输运算子模块：包括平流、涡动扩散及温度廓线边界值问题。

控制方程（连续性方程）：

    \frac{\partial c}{\partial t}
    + \frac{u}{R\cos\phi}\frac{\partial c}{\partial\lambda}
    + \frac{v}{R}\frac{\partial c}{\partial\phi}
    + w\frac{\partial c}{\partial z}
    = \frac{1}{R^2\cos\phi}
      \frac{\partial}{\partial\phi}\!
        \left(K_{yy}\cos\phi\frac{\partial c}{\partial\phi}\right)
    + \frac{\partial}{\partial z}\!
        \left(K_{zz}\frac{\partial c}{\partial z}\right)

温度廓线（辐射平衡边界值问题）：

    K_T \frac{d^2 T}{dz^2}
    - \alpha_{\mathrm{IR}} \sigma T^4
    + Q_{\mathrm{SW}}(z) = 0

    T(z_{\mathrm{trop}}) = T_0, \quad
    \left.\frac{dT}{dz}\right|_{z_{\mathrm{top}}} = -\Gamma_{\mathrm{meso}}

融合来源：
  - 572_ill_bvp: 病态 BVP 离散化思想
"""

import numpy as np
from linear_solvers import conjugate_gradient, gmres_restart
from utils import safe_divide, clip_positive

R_EARTH = 6371.0e3  # m
G_GRAV = 9.80665    # m s^{-2}


def temperature_profile_bvp(z_nodes, T_tropopause=220.0, T_stratopause=270.0,
                            z_trop_km=15.0, z_strat_km=50.0):
    r"""
    求解平流层温度廓线的离散边界值问题。

    模型方程（简化热扩散-辐射平衡）：

        K_T \frac{d^2 T}{dz^2} - \alpha_{\mathrm{IR}} (T - T_{\mathrm{eq}}(z)) = 0

    其中等效辐射平衡温度廓线采用臭氧加热参数化：

        T_{\mathrm{eq}}(z) = T_{\mathrm{trop}}
            + (T_{\mathrm{strat}} - T_{\mathrm{trop}})
              \sin^2\!\left(\frac{\pi}{2} \frac{z - z_{\mathrm{trop}}}
                                    {z_{\mathrm{strat}} - z_{\mathrm{trop}}}\right)

    边界条件：
      - 下边界（对流层顶）：T(z_{\mathrm{trop}}) = T_{\mathrm{trop}}
      - 上边界（平流层顶）：T(z_{\mathrm{strat}}) = T_{\mathrm{strat}}

    离散后使用三对角直接求解。

    Parameters
    ----------
    z_nodes : ndarray, shape (n,)
        高度节点 [m]，假设单调递增。
    T_tropopause : float
        对流层顶温度 [K]。
    T_stratopause : float
        平流层顶温度 [K]。
    z_trop_km : float
        对流层顶高度 [km]。
    z_strat_km : float
        平流层顶高度 [km]。

    Returns
    -------
    T : ndarray, shape (n,)
        温度廓线 [K]。
    """
    z = np.asarray(z_nodes, dtype=float)
    n = z.size
    if n < 3:
        return np.full(n, T_tropopause)

    z_trop = z_trop_km * 1000.0
    z_strat = z_strat_km * 1000.0
    dz_total = z_strat - z_trop

    # 等效平衡温度
    frac = np.clip((z - z_trop) / dz_total, 0.0, 1.0)
    T_eq = T_tropopause + (T_stratopause - T_tropopause) * (np.sin(0.5 * np.pi * frac) ** 2)

    # 热扩散系数与辐射冷却系数
    K_t = 5.0    # m^2/s（有效涡动热扩散）
    alpha_ir = 1.0e-5  # s^{-1}（Newtonian 冷却系数）

    # 构造三对角系统
    A = np.zeros((n, n))
    rhs = np.zeros(n)

    for i in range(n):
        if i == 0:
            A[i, i] = 1.0
            rhs[i] = T_tropopause
        elif i == n - 1:
            A[i, i] = 1.0
            rhs[i] = T_stratopause
        else:
            dz_p = z[i + 1] - z[i]
            dz_m = z[i] - z[i - 1]
            denom = dz_m * dz_p * (dz_m + dz_p)
            A[i, i - 1] = 2.0 * dz_p / denom
            A[i, i] = -2.0 * (dz_p + dz_m) / denom - alpha_ir / K_t
            A[i, i + 1] = 2.0 * dz_m / denom
            rhs[i] = -alpha_ir * T_eq[i] / K_t

    # 直接求解
    try:
        x = np.linalg.solve(A, rhs)
    except np.linalg.LinAlgError:
        x = np.linalg.lstsq(A, rhs, rcond=None)[0]
    T = np.clip(x, 180.0, 300.0)
    return T


def eddy_diffusivity(z_m, lat_rad):
    r"""
    平流层涡动扩散系数参数化（基于 Holton-Massa 方案）：

        K_{zz}(z) = K_0 + K_1 \exp\!\left(-\frac{(z - z_{\max})^2}{2\sigma_z^2}\right)

        K_{yy}(\phi) = K_{yy0} \cos^2\phi

    Parameters
    ----------
    z_m : float
        高度 [m]。
    lat_rad : float
        纬度 [rad]。

    Returns
    -------
    Kyy : float [m^2 s^{-1}]
    Kzz : float [m^2 s^{-1}]
    """
    z_km = z_m / 1000.0
    K0 = 0.1      # m^2/s
    K1 = 1.0      # m^2/s
    z_max = 30.0  # km
    sigma_z = 8.0 # km
    Kzz = K0 + K1 * np.exp(-0.5 * ((z_km - z_max) / sigma_z) ** 2)
    Kyy0 = 1.0e5  # m^2/s
    Kyy = Kyy0 * (np.cos(lat_rad) ** 2 + 0.1)
    return Kyy, Kzz


def background_wind_field(lon, lat, z_km):
    r"""
    参数化背景风场（Brewer-Dobson 环流简化）。

    经向速度（Hadley 型）：

        v(\phi) = V_0 \sin(2\phi) \exp\!\left(-\frac{z - z_0}{H}\right)

    垂直速度（由连续性约束）：

        w = -\frac{1}{R\cos\phi} \int \frac{\partial(v\cos\phi)}{\partial\phi} dz

    这里采用简化的参数化形式：

        w(z) = W_0 \sin\left(\frac{\pi(z - 15)}{35}\right) \cos(2\phi)

    Parameters
    ----------
    lon, lat : float
        经度 [rad], 纬度 [rad]。
    z_km : float
        高度 [km]。

    Returns
    -------
    u, v, w : float
        风速分量 [m/s]。
    """
    V0 = 0.5  # m/s
    W0 = 3.0e-4  # m/s
    H = 7.0   # km scale height
    v = V0 * np.sin(2.0 * lat) * np.exp(-(z_km - 30.0) / H)
    if 15.0 <= z_km <= 50.0:
        w = W0 * np.sin(np.pi * (z_km - 15.0) / 35.0) * np.cos(2.0 * lat)
    else:
        w = 0.0
    u = 0.0  # 简化：无纬向平均风
    return u, v, w


class TransportOperator:
    r"""
    三维平流层输运算子。

    在球面坐标 (λ, φ, z) 的离散网格上，计算输运源汇项：

        S_{\mathrm{trans}} = -\mathbf{v} \cdot \nabla c
        + \nabla \cdot (\mathbf{K} \cdot \nabla c)

    使用有限体积法离散。
    """

    def __init__(self, mesh):
        r"""
        Parameters
        ----------
        mesh : StratosphericMesh
        """
        from stratospheric_mesh import StratosphericMesh
        if not isinstance(mesh, StratosphericMesh):
            raise TypeError("mesh 必须是 StratosphericMesh 实例")
        self.mesh = mesh
        self._compute_cell_properties()

    def _compute_cell_properties(self):
        r"""预计算每个单元的风场和扩散系数。"""
        self.cell_u = np.zeros(self.mesh.n_cells)
        self.cell_v = np.zeros(self.mesh.n_cells)
        self.cell_w = np.zeros(self.mesh.n_cells)
        self.cell_Kyy = np.zeros(self.mesh.n_cells)
        self.cell_Kzz = np.zeros(self.mesh.n_cells)

        for i in range(self.mesh.n_cells):
            cent = self.mesh.cell_centroids[i]
            lon, lat, z_km = cent[0], cent[1], cent[2]
            u, v, w = background_wind_field(lon, lat, z_km)
            self.cell_u[i] = u
            self.cell_v[i] = v
            self.cell_w[i] = w
            Kyy, Kzz = eddy_diffusivity(z_km * 1000.0, lat)
            self.cell_Kyy[i] = Kyy
            self.cell_Kzz[i] = Kzz

    def transport_source(self, c_cell):
        r"""
        计算每个单元内的输运源汇项 [molecules cm^{-3} s^{-1}]。

        采用一阶迎风差分近似平流，中心差分近似扩散。

        Parameters
        ----------
        c_cell : ndarray, shape (n_cells, n_species)
            单元平均浓度。

        Returns
        -------
        S : ndarray, shape (n_cells, n_species)
        """
        # TODO: 实现垂直扩散（中心差分）与垂直平流（一阶迎风）的离散计算
        # 需要从 mesh 获取相邻层浓度，结合 cell_Kzz、cell_w 计算 diff_z 和 adv_z
        raise NotImplementedError("Hole 3: 请实现输运算子的源汇项离散计算")

    def apply_matrix(self, x_flat):
        r"""
        将输运算子作为矩阵-向量乘积（用于 GMRES）。

        返回 (I - dt * T) x 的近似，其中 T 为输运算子离散矩阵。
        这里简化为返回 x + transport_source(x_reshaped) 的扁平化。

        Parameters
        ----------
        x_flat : ndarray

        Returns
        -------
        y_flat : ndarray
        """
        n_cells = self.mesh.n_cells
        n_spec = len(x_flat) // n_cells
        x = x_flat.reshape((n_cells, n_spec))
        S = self.transport_source(x)
        y = x + S  # 简化：单位时间步长
        return y.ravel()
