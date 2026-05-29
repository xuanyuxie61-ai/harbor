"""
insar_forward.py
InSAR 正演模型模块：从断层滑动分布计算地表 LOS 形变场。

融合种子项目:
  - 363_fd1d_poisson: 一维有限差分求解泊松方程（用于弹性半空间格林函数的数值近似）
  - 661_legendre_polynomial: Legendre 多项式（用于雷达入射角和方位角的球谐展开/角度插值）

在 InSAR 形变反演中的应用:
  1. 基于 Okada 弹性半空间位错模型，将断层面上的滑动矢量 Δu 投影到地表 LOS 方向；
  2. 包含地球曲率修正（Legendre 展开）和大气相位延迟模型；
  3. 生成含噪声的合成 InSAR 观测数据，用于验证反演算法。

核心公式:
  1. Okada 位错模型:
        u_i(x) = ∬_Σ G_{ij}(x, ξ) Δu_j(ξ) dΣ
     其中 G_{ij} 为弹性半空间格林函数。

  2. LOS 投影:
        d_LOS = u · n_LOS
     n_LOS = [sin(θ_inc) cos(α_az), -sin(θ_inc) sin(α_az), cos(θ_inc)]
     θ_inc: 雷达入射角（from nadir）
     α_az:  方位角（flight direction from North）

  3. 一维弹性半空间位移-应力关系（泊松方程近似）:
        d²u/dz² = -f(z)/μ
     用于验证 Okada 解析解与数值解的一致性。
"""

import numpy as np
from spectral_basis import legendre_polynomial_values
from utils import check_finite, normalize_vector


class OkadaGreenFunction:
    """
    Okada 矩形位错模型的弹性半空间格林函数。
    基于 Chinnery (1963) 和 Okada (1985, 1992) 的解析公式。

    断层面由以下参数定义:
        length: 走向长度 (m)
        width:  倾向宽度 (m)
        strike: 走向角 (度，从北顺时针)
        dip:    倾角 (度，从水平面向下)
        depth:  上边界深度 (m)
        rake:   滑动角 (度，从走向逆时针)
        slip:   滑动量 (m)
    """

    def __init__(self, nu=0.25):
        """
        nu: 泊松比（默认 0.25，弹性半空间）
        """
        self.nu = nu

    def _chinnery(self, f, x, y, p, q, L, W):
        """
        Chinnery 符号约定:
            f(x, p) - f(x, p-W) - f(x-L, p) + f(x-L, p-W)
        其中 x 沿走向，p 沿倾向（垂直于走向的水平投影），q 为深度。
        """
        return (f(x, y, p, q) - f(x, y, p - W, q) -
                f(x - L, y, p, q) + f(x - L, y, p - W, q))

    def _f_strike_slip(self, x, y, p, q):
        """
        Strike-slip (走滑) 分量对 u_x 的贡献（内积形式）。
        采用 Okada (1992) 的公式 28-30 的简化标量表示。
        此处直接计算三维位移矢量。
        """
        # 简化版本：使用近似公式
        R = np.sqrt(x ** 2 + y ** 2 + q ** 2)
        if R < 1e-10:
            R = 1e-10
        # 位移近似
        ux = (1.0 / (2.0 * np.pi)) * (x * y / (R * (R + q)) +
                                      np.arctan(x * y / (q * R)))
        uy = (1.0 / (2.0 * np.pi)) * (-q / R - y ** 2 / (R * (R + q)))
        uz = (1.0 / (2.0 * np.pi)) * (-y / R)
        return np.array([ux, uy, uz])

    def _f_dip_slip(self, x, y, p, q):
        """
        Dip-slip (倾滑) 分量近似。
        """
        R = np.sqrt(x ** 2 + y ** 2 + q ** 2)
        if R < 1e-10:
            R = 1e-10
        ux = (1.0 / (2.0 * np.pi)) * (q / R)
        uy = (1.0 / (2.0 * np.pi)) * (y * q / (R * (R + q)))
        uz = (1.0 / (2.0 * np.pi)) * (1.0 - y ** 2 / (R * (R + q)))
        return np.array([ux, uy, uz])

    def compute_displacement(self, obs_e, obs_n, obs_u,
                              fault_length, fault_width,
                              strike_deg, dip_deg, depth,
                              rake_deg, slip):
        """
        计算单个矩形位错在地表观测点处的三维位移 [E, N, U]。

        参数单位: 全部使用米 (m) 和度 (deg)。
        """
        strike = np.deg2rad(strike_deg)
        dip = np.deg2rad(dip_deg)
        rake = np.deg2rad(rake_deg)

        # 滑动分解为走滑和倾滑分量
        slip_strike = slip * np.cos(rake)
        slip_dip = slip * np.sin(rake)

        # 将观测点坐标转换到断层局部坐标系
        # 断层局部坐标: x1 沿走向，x2 垂直于走向的水平投影
        dx = obs_e
        dy = obs_n
        x1 = dx * np.cos(strike) + dy * np.sin(strike)
        x2 = -dx * np.sin(strike) + dy * np.cos(strike)

        # 倾向方向参数
        p = x2 * np.cos(dip) + depth * np.sin(dip)
        q = x2 * np.sin(dip) - depth * np.cos(dip)
        if q < 0:
            q = -q  # 取绝对值简化

        # 计算走滑和倾滑贡献
        u_strike = self._f_strike_slip(x1, x2, p, q)
        u_dip = self._f_dip_slip(x1, x2, p, q)

        u_local = slip_strike * u_strike + slip_dip * u_dip

        # 转换回 ENU 坐标系
        u_e = u_local[0] * np.cos(strike) - u_local[1] * np.sin(strike)
        u_n = u_local[0] * np.sin(strike) + u_local[1] * np.cos(strike)
        u_u = u_local[2]

        return np.array([u_e, u_n, u_u])

    def compute_displacements_vectorized(self, obs_points,
                                          fault_length, fault_width,
                                          strike_deg, dip_deg, depth,
                                          rake_deg, slip):
        """
        向量化计算多个观测点的位移。
        obs_points: (N, 3) 数组，每行为 [E, N, U]
        """
        N = obs_points.shape[0]
        displacements = np.zeros((N, 3))
        for i in range(N):
            displacements[i] = self.compute_displacement(
                obs_points[i, 0], obs_points[i, 1], obs_points[i, 2],
                fault_length, fault_width, strike_deg, dip_deg,
                depth, rake_deg, slip
            )
        return displacements


class InSARForwardModel:
    """
    InSAR 正演模型：从断层滑动分布生成 LOS 形变观测。
    """

    def __init__(self, los_vector=None, wavelength=0.056):
        """
        los_vector: 雷达视线方向单位矢量 [E, N, U] 分量
        wavelength: 雷达波长 (m)，默认 Sentinel-1 C-band: 5.6 cm
        """
        if los_vector is None:
            # 默认 Sentinel-1 降轨，入射角 ~34°，方位角 ~190°
            theta_inc = np.deg2rad(34.0)
            alpha_az = np.deg2rad(190.0)
            self.los_vector = np.array([
                np.sin(theta_inc) * np.cos(alpha_az),
                -np.sin(theta_inc) * np.sin(alpha_az),
                np.cos(theta_inc)
            ])
        else:
            self.los_vector = normalize_vector(np.asarray(los_vector))
        self.wavelength = wavelength
        self.okada = OkadaGreenFunction(nu=0.25)

    def project_to_los(self, displacement_enu):
        """
        将三维位移投影到 LOS 方向。
        d_LOS = d_E * n_E + d_N * n_N + d_U * n_U
        """
        return displacement_enu @ self.los_vector

    def forward(self, fault_mesh, slip_distribution, obs_points):
        """
        正演计算：对断层面上每个单元的滑动，计算其在观测点的 LOS 形变，
        并对所有单元积分（求和）。

        参数:
            fault_mesh: FaultMesh 对象
            slip_distribution: (n_nodes,) 或 (n_nodes, 2) 滑动分布
                               若为 1D，假设纯走滑；若为 2D，[strike_slip, dip_slip]
            obs_points: (N_obs, 3) 观测点 ENU 坐标 (m)

        返回:
            d_los: (N_obs,) LOS 形变 (m)
        """
        # HOLE 3: 需实现 InSAR 正演模型的核心计算逻辑。
        # 关键步骤：
        #   1. 获取断层单元形心 (centroids) 和面积 (areas)
        #   2. 将节点滑动分布插值/平均到单元级别，计算每个单元的 slip 和 rake
        #   3. 对每个单元，遍历所有观测点，计算 Okada 位错产生的三维位移
        #   4. 将三维位移投影到 LOS 方向：d_LOS = u_enu · los_vector
        #   5. 按单元面积加权累加所有单元的贡献
        #   6. 注意单位换算：fault_mesh 使用 km，Okada 使用 m
        # 提示：可调用 self.okada.compute_displacement() 和 self.project_to_los()
        raise NotImplementedError("forward: 待实现 InSAR 正演计算")

    def add_noise(self, d_los, sigma=0.01, atmospheric=False,
                  correlation_length=5000.0):
        """
        为 LOS 形变添加观测噪声和大气延迟。

        参数:
            sigma: 随机噪声标准差 (m)
            atmospheric: 是否添加大气相关噪声
            correlation_length: 大气相关长度 (m)
        """
        N = len(d_los)
        noise = np.random.normal(0.0, sigma, N)

        if atmospheric:
            # 简化的指数相关大气噪声模型
            # cov(i,j) = σ_atm² exp(-|r_i - r_j| / L)
            sigma_atm = 0.02
            # 为简化，直接添加平滑随机噪声模拟大气
            atm_noise = np.random.normal(0.0, sigma_atm, N)
            # 简单平滑
            if N > 3:
                atm_noise_smooth = np.convolve(
                    atm_noise, np.ones(3) / 3.0, mode='same')
            else:
                atm_noise_smooth = atm_noise
            noise += atm_noise_smooth

        return d_los + noise

    def los_vector_legendre_expansion(self, theta_min, theta_max, n_terms):
        """
        使用 Legendre 多项式展开雷达入射角相关的 LOS 向量。
        用于不同入射角数据集的联合反演。

        返回基函数矩阵 B(θ) 的系数。
        """
        theta_eval = np.linspace(theta_min, theta_max, 50)
        # 将 θ 归一化到 [-1, 1]
        x_norm = 2.0 * (theta_eval - theta_min) / (theta_max - theta_min) - 1.0
        P = legendre_polynomial_values(len(x_norm), n_terms, x_norm)
        return P


class ElasticHalfspacePoisson1D:
    """
    一维弹性半空间泊松方程数值求解器。
    融合 fd1d_poisson 的核心算法：有限差分法。

    方程:
        -d²u/dz² = f(z)/μ,   z ∈ [0, H]
        u(0) = 0 (地表固定), u(H) = 0 (深部固定)

    用于验证弹性半空间位移场的数值一致性。
    """

    def __init__(self, H, mu, nx):
        self.H = H
        self.mu = mu
        self.nx = nx
        self.hz = H / (nx - 1)
        self.z = np.linspace(0, H, nx)

    def solve(self, f_func, u_bottom=0.0):
        """
        有限差分求解一维泊松方程。
        离散格式:
            -(u_{i-1} - 2u_i + u_{i+1}) / h² = f_i / μ
        即:
            A u = rhs
        其中 A 为三对角矩阵，对角线 2/h²，次对角线 -1/h²。
        """
        nx = self.nx
        hz = self.hz
        A = np.zeros((nx, nx))
        rhs = np.zeros(nx)

        for i in range(nx):
            z_i = self.z[i]
            if i == 0:
                A[i, i] = 1.0
                rhs[i] = 0.0  # 地表位移固定为 0
            elif i == nx - 1:
                A[i, i] = 1.0
                rhs[i] = u_bottom
            else:
                A[i, i] = 2.0 / (hz * hz)
                A[i, i - 1] = -1.0 / (hz * hz)
                A[i, i + 1] = -1.0 / (hz * hz)
                rhs[i] = f_func(z_i) / self.mu

        u = np.linalg.solve(A, rhs)
        check_finite(u, "ElasticHalfspacePoisson1D solve")
        return self.z, u
