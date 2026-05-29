"""
bilayer_system.py
核心脂质双分子层系统建模模块

本模块基于粗粒化分子动力学（Coarse-Grained MD）思想，构建一个二维格点上的
脂质双分子层模型。每个脂质分子被简化为具有取向角 theta 的有效粒子，
通过类似 Maier-Saupe 的取向序相互作用与疏水/亲水相互作用描述系统能量。

参考种子项目: 512_heated_plate (热板稳态热方程)
"""

import numpy as np


class LipidBilayerSystem:
    """
    脂质双分子层粗粒化模型。

    物理模型:
    ---------
    将双层膜投影到 x-y 平面，采用 N_x × N_y 的矩形格点。
    每个格点 (i,j) 上放置一对脂质分子（上层、下层）。
    每个脂质分子具有取向单位向量 n = (sinθ cosφ, sinθ sinφ, cosθ)。

    系统的总哈密顿量:
        H = Σ_i H_orient(i) + Σ_<ij> H_nn(i,j) + H_compress + H_tail

    其中:
      - H_orient = -J * P_2(cosθ_i)  （Maier-Saupe 取向势，P_2 为二阶 Legendre 多项式）
      - H_nn     = -ε * (n_i · n_j)^2  （最近邻取向耦合）
      - H_compress = (κ_A/2) * (A_i - A_0)^2 / A_0  （面积压缩弹性）
      - H_tail   = 疏水链内聚能，与链序参数相关

    相变类型:
        凝胶相 (Gel, L_β') ↔ 液晶相 (Liquid-Crystalline, L_α)
        转变温度 T_m 由链熔化和取向无序化共同驱动。
    """

    def __init__(self, nx=24, ny=24, k_boltzmann=1.380649e-23,
                 j_coupling=2.5, epsilon_nn=1.0, kappa_a=25.0,
                 area0=0.64, dt_md=0.002, mass=1.0):
        """
        初始化脂质双分子层系统。

        Parameters
        ----------
        nx, ny : int
            格点尺寸（每层脂质数 = nx * ny）。
        k_boltzmann : float
            玻尔兹曼常数（单位: kJ/(mol·K) 的缩放形式，此处取无量纲化值）。
        j_coupling : float
            Maier-Saupe 耦合常数 J（单位: kJ/mol）。
        epsilon_nn : float
            最近邻取向耦合强度 ε。
        kappa_a : float
            面积压缩模量 κ_A（单位: mN/m 的分子尺度等价）。
        area0 : float
            单个脂质平衡占据面积 A_0（单位: nm²）。
        dt_md : float
            MD 时间步长（单位: ps）。
        mass : float
            有效质量（无量纲化）。
        """
        if nx < 4 or ny < 4:
            raise ValueError("格点尺寸 nx, ny 必须至少为 4 以保证边界处理有效。")
        if j_coupling <= 0 or epsilon_nn <= 0 or kappa_a <= 0:
            raise ValueError("耦合常数与模量必须为正。")
        if area0 <= 0:
            raise ValueError("平衡面积必须为正。")
        if dt_md <= 0:
            raise ValueError("时间步长必须为正。")

        self.nx = nx
        self.ny = ny
        self.n_lipids = nx * ny
        self.kb = k_boltzmann
        self.J = j_coupling
        self.eps_nn = epsilon_nn
        self.kappa_a = kappa_a
        self.area0 = area0
        self.dt = dt_md
        self.mass = mass

        # 取向角: theta (极角, 相对膜法线), phi (方位角)
        # 初始状态: 凝胶相，取向基本垂直于膜面
        self.theta = np.full((nx, ny), 0.1) + 0.05 * np.random.randn(nx, ny)
        self.phi = 2.0 * np.pi * np.random.rand(nx, ny)

        # 角速度
        self.omega_theta = np.zeros((nx, ny))
        self.omega_phi = np.zeros((nx, ny))

        # 每个格点的瞬时面积（热涨落导致）
        self.area = np.full((nx, ny), area0)

        # 温度场（受 heated_plate 启发，用于非均匀热浴）
        self.temperature_field = np.ones((nx, ny)) * 300.0  # K

        # 力缓存
        self.torque_theta = np.zeros((nx, ny))
        self.torque_phi = np.zeros((nx, ny))
        self.force_area = np.zeros((nx, ny))

    def _p2_legendre(self, cos_theta):
        """
        二阶 Legendre 多项式 P_2(x) = (3x^2 - 1)/2。
        用于 Maier-Saupe 取向势。
        """
        return 0.5 * (3.0 * cos_theta ** 2 - 1.0)

    def _safe_modulo(self, idx, max_idx):
        """周期边界条件安全取模。"""
        return idx % max_idx

    def compute_local_order_parameter(self):
        """
        计算局域取向序参数 S_2(i,j) = <P_2(cosθ)>_neighbors。

        S_2 的取值范围 [-0.5, 1]:
          S_2 ≈ 1   : 完全取向有序（凝胶相）
          S_2 ≈ 0   : 完全取向无序（液晶相）
          S_2 = -0.5: 平面内取向（非本征双层构象）
        """
        nx, ny = self.nx, self.ny
        s2 = np.zeros((nx, ny))
        cos_t = np.cos(self.theta)
        p2 = self._p2_legendre(cos_t)

        for i in range(nx):
            for j in range(ny):
                s = 0.0
                count = 0
                for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1),
                                (-1, -1), (-1, 1), (1, -1), (1, 1)]:
                    ii = self._safe_modulo(i + di, nx)
                    jj = self._safe_modulo(j + dj, ny)
                    s += p2[ii, jj]
                    count += 1
                s2[i, j] = s / count if count > 0 else 0.0
        return s2

    def compute_total_energy(self):
        """
        计算系统总能量。

        E_total = Σ_ij [ -J * P_2(cosθ_ij) * S_2_ij
                         - ε * Σ_nn (n_ij · n_nn)^2
                         + (κ_A/2) * ((A_ij - A_0)^2 / A_0)
                         + (k_B T_ij) * ln(A_ij/A_0) ]
        """
        cos_t = np.cos(self.theta)
        sin_t = np.sin(self.theta)
        p2 = self._p2_legendre(cos_t)
        s2 = self.compute_local_order_parameter()

        # Maier-Saupe 取向能
        e_orient = -self.J * np.sum(p2 * s2)

        # 最近邻耦合能
        e_nn = 0.0
        for i in range(self.nx):
            for j in range(self.ny):
                n_i = np.array([
                    sin_t[i, j] * np.cos(self.phi[i, j]),
                    sin_t[i, j] * np.sin(self.phi[i, j]),
                    cos_t[i, j]
                ])
                for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    ii = self._safe_modulo(i + di, self.nx)
                    jj = self._safe_modulo(j + dj, self.ny)
                    n_j = np.array([
                        sin_t[ii, jj] * np.cos(self.phi[ii, jj]),
                        sin_t[ii, jj] * np.sin(self.phi[ii, jj]),
                        cos_t[ii, jj]
                    ])
                    dot = np.clip(np.dot(n_i, n_j), -1.0, 1.0)
                    e_nn -= self.eps_nn * dot ** 2
        e_nn *= 0.5  # 避免双重计数

        # 面积压缩能
        dA = self.area - self.area0
        e_compress = 0.5 * self.kappa_a * np.sum(dA ** 2 / self.area0)

        # 熵贡献（简化 Flory-Huggins 形式）
        ratio = self.area / self.area0
        ratio = np.where(ratio > 1e-12, ratio, 1e-12)
        e_entropy = np.sum(self.kb * self.temperature_field * np.log(ratio))

        return e_orient + e_nn + e_compress + e_entropy

    def compute_forces(self):
        """
        计算取向力矩与面积力。

        力矩 τ_θ = -∂H/∂θ,  τ_φ = -∂H/∂φ
        面积力 F_A = -∂H/∂A
        """
        nx, ny = self.nx, self.ny
        self.torque_theta = np.zeros((nx, ny))
        self.torque_phi = np.zeros((nx, ny))
        self.force_area = np.zeros((nx, ny))

        cos_t = np.cos(self.theta)
        sin_t = np.sin(self.theta)
        s2 = self.compute_local_order_parameter()

        # Maier-Saupe 对 theta 的力矩
        # P_2'(cosθ) = 3 cosθ, d(cosθ)/dθ = -sinθ
        # τ = -∂E/∂θ = -J * P_2'(cosθ) * (-sinθ) * S_2 = -J * 3 cosθ * (-sinθ) * S_2
        dp2_dcos = 3.0 * cos_t
        dcos_dtheta = -sin_t
        self.torque_theta += -self.J * dp2_dcos * dcos_dtheta * s2

        # 最近邻耦合力矩（数值微分近似）
        eps = 1e-6
        for i in range(nx):
            for j in range(ny):
                e0 = self._local_nn_energy(i, j)
                self.theta[i, j] += eps
                e1 = self._local_nn_energy(i, j)
                self.theta[i, j] -= eps
                self.torque_theta[i, j] += -(e1 - e0) / eps

                e0 = self._local_nn_energy(i, j)
                self.phi[i, j] += eps
                e1 = self._local_nn_energy(i, j)
                self.phi[i, j] -= eps
                self.torque_phi[i, j] += -(e1 - e0) / eps

        # 面积力
        dA = self.area - self.area0
        self.force_area = -(self.kappa_a * dA / self.area0 +
                            self.kb * self.temperature_field / self.area)

    def _local_nn_energy(self, i, j):
        """计算格点 (i,j) 的局部最近邻耦合能。"""
        cos_t = np.cos(self.theta)
        sin_t = np.sin(self.theta)
        n_i = np.array([
            sin_t[i, j] * np.cos(self.phi[i, j]),
            sin_t[i, j] * np.sin(self.phi[i, j]),
            cos_t[i, j]
        ])
        e = 0.0
        for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ii = self._safe_modulo(i + di, self.nx)
            jj = self._safe_modulo(j + dj, self.ny)
            n_j = np.array([
                sin_t[ii, jj] * np.cos(self.phi[ii, jj]),
                sin_t[ii, jj] * np.sin(self.phi[ii, jj]),
                cos_t[ii, jj]
            ])
            dot = np.clip(np.dot(n_i, n_j), -1.0, 1.0)
            e -= self.eps_nn * dot ** 2
        return e

    def thermalize_temperature_field(self, boundary_temp_high=350.0,
                                      boundary_temp_low=250.0,
                                      epsilon_conv=1e-4,
                                      max_iter=5000):
        """
        使用稳态热方程迭代求解温度场分布。

        受种子项目 512_heated_plate 启发，在双层膜平面内求解:
            ∇²T = 0  （内部）
            T = T_high （上下边界，模拟热源）
            T = T_low  （左右边界，模拟热沉）

        离散形式:
            T[i,j] = 0.25 * (T[i-1,j] + T[i+1,j] + T[i,j-1] + T[i,j+1])

        采用 Jacobi 迭代直至收敛。
        """
        nx, ny = self.nx, self.ny
        T = self.temperature_field.copy()

        # 边界条件
        T[:, 0] = boundary_temp_low
        T[:, -1] = boundary_temp_low
        T[0, :] = boundary_temp_high
        T[-1, :] = boundary_temp_high

        diff = epsilon_conv + 1.0
        iteration = 0
        while diff > epsilon_conv and iteration < max_iter:
            T_old = T.copy()
            # 内部点 Jacobi 更新
            T[1:nx-1, 1:ny-1] = 0.25 * (
                T_old[0:nx-2, 1:ny-1] +
                T_old[2:nx, 1:ny-1] +
                T_old[1:nx-1, 0:ny-2] +
                T_old[1:nx-1, 2:ny]
            )
            # 保持边界
            T[:, 0] = boundary_temp_low
            T[:, -1] = boundary_temp_low
            T[0, :] = boundary_temp_high
            T[-1, :] = boundary_temp_high

            diff = np.max(np.abs(T - T_old))
            iteration += 1

        self.temperature_field = T
        return iteration, diff

    def global_order_parameter(self):
        """
        全局取向序参数 S_2^global = (1/N) Σ_i P_2(cosθ_i)。

        相变判据:
            S_2^global > 0.6  → 凝胶相
            S_2^global < 0.3  → 液晶相
            中间区域          → 共存/转变区
        """
        cos_t = np.cos(self.theta)
        p2 = self._p2_legendre(cos_t)
        return float(np.mean(p2))

    def get_positions(self):
        """
        返回格点坐标（用于外部模块的网格拓扑分析）。
        """
        x = np.arange(self.nx)
        y = np.arange(self.ny)
        X, Y = np.meshgrid(x, y, indexing='ij')
        return X, Y
