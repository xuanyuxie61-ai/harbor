"""
ecosystem_dynamics.py
=====================
海洋 NPZD（营养盐-浮游植物-浮游动物-碎屑）生态系统模块，
与热盐环流通过平流-扩散项耦合。

核心生物地球化学模型
--------------------
采用 Fasham et al. (1990) 型 NPZD 模型，控制方程如下：

  (1) 营养盐 N（硝酸盐，单位：mmol N/m³）：
      ∂N/∂t + u·∇N = κ_N∇²N - U(N,I)·P + γ·g·Z + r_D·D

  (2) 浮游植物 P（单位：mmol N/m³）：
      ∂P/∂t + u·∇P = κ_P∇²P + U(N,I)·P - m_P·P - G(P)·Z

  (3) 浮游动物 Z（单位：mmol N/m³）：
      ∂Z/∂t + u·∇Z = κ_Z∇²Z + β·G(P)·Z - m_Z·Z - g·Z

  (4) 碎屑 D（单位：mmol N/m³）：
      ∂D/∂t + u·∇D = κ_D∇²D + (1-β)·G(P)·Z + m_P·P + m_Z·Z - r_D·D - w_s·∂D/∂z

其中：
  U(N,I) = V_max · (N / (K_N + N)) · (I / √(I_opt² + I²))   [Monod-Michaelis 光限制生长]
  G(P)   = g_max · (P / (K_P + P))                           [Holling II 型 grazing]
  I(z)   = I_0 · exp(-k_w·z - k_c·∫P dz)                    [Beer-Lambert 光衰减]

参数说明：
  V_max : 最大营养盐吸收率 [1/day]
  K_N   : 半饱和常数 [mmol N/m³]
  g_max : 最大摄食率 [1/day]
  K_P   : 摄食半饱和常数 [mmol N/m³]
  m_P, m_Z : 浮游植物/动物死亡率 [1/day]
  g     : 浮游动物排泄率 [1/day]
  β     : 摄食同化效率 [-]
  γ     : 排泄物再矿化比例 [-]
  r_D   : 碎屑再矿化率 [1/day]
  w_s   : 碎屑沉降速度 [m/day]
"""

import numpy as np


class NPZDEcosystem:
    """
    海洋 NPZD 生态系统求解器。
    """

    def __init__(self, nx, nz, dx, dz, dt,
                 V_max=1.0, K_N=0.5, I_opt=50.0,
                 g_max=0.6, K_P=0.5, beta=0.75, gamma=0.3,
                 m_P=0.05, m_Z=0.05, g_zoo=0.1, r_D=0.05,
                 w_s=5.0, k_w=0.04, k_c=0.03,
                 kappa_bio=1.0e-5, I_0=200.0):
        """
        初始化生态系统参数。
        """
        if nx < 4 or nz < 4:
            raise ValueError("nx, nz >= 4")
        if dx <= 0 or dz <= 0 or dt <= 0:
            raise ValueError("空间步长与时间步长必须为正")

        self.nx = nx
        self.nz = nz
        self.dx = dx
        self.dz = dz
        self.dt = dt

        # 生物参数
        self.V_max = V_max
        self.K_N = K_N
        self.I_opt = I_opt
        self.g_max = g_max
        self.K_P = K_P
        self.beta = beta
        self.gamma = gamma
        self.m_P = m_P
        self.m_Z = m_Z
        self.g_zoo = g_zoo
        self.r_D = r_D
        self.w_s = w_s / 86400.0  # 转换为 m/s
        self.k_w = k_w
        self.k_c = k_c
        self.kappa_bio = kappa_bio
        self.I_0 = I_0

        # 状态场
        self.N = np.ones((nx, nz)) * 5.0   # 营养盐背景 5 mmol/m³
        self.P = np.ones((nx, nz)) * 0.1   # 浮游植物背景 0.1 mmol/m³
        self.Z = np.ones((nx, nz)) * 0.05  # 浮游动物背景 0.05 mmol/m³
        self.D = np.ones((nx, nz)) * 0.05  # 碎屑背景 0.05 mmol/m³

        # 深度网格（正向下）
        self.z_grid = np.linspace(0.0, (nz - 1) * dz, nz)

        # 历史项（Adams-Bashforth）
        self.N_old = None
        self.P_old = None
        self.Z_old = None
        self.D_old = None

    def light_profile(self, P_field):
        """
        根据 Beer-Lambert 定律计算垂向光强分布：
            I(z) = I_0 · exp(-k_w·z - k_c·∫_0^z P(z') dz')
        其中垂向积分采用梯形法则。
        """
        nx, nz = P_field.shape
        I = np.zeros((nx, nz))
        dz = self.dz

        # 对每一列（每个水平位置）计算垂向累积叶绿素
        for i in range(nx):
            cum_p = 0.0
            for k in range(nz):
                if k > 0:
                    cum_p += 0.5 * (P_field[i, k] + P_field[i, k - 1]) * dz
                atten = np.exp(-self.k_w * self.z_grid[k] - self.k_c * cum_p)
                I[i, k] = self.I_0 * atten
        return I

    def uptake_rate(self, N_field, P_field):
        """
        浮游植物营养盐吸收率（Monod-Michaelis 光限制）：
            U = V_max · (N / (K_N + N)) · (I / √(I_opt² + I²))
        """
        # TODO(Hole 1): 实现 Monod-Michaelis 光限制生长率计算
        # 提示: 需要调用 light_profile，计算营养盐限制和光限制，返回生长率 [1/s]
        raise NotImplementedError("Hole 1: 请实现 uptake_rate")

    def grazing_rate(self, P_field):
        """
        Holling II 型摄食函数：
            G = g_max · P / (K_P + P)
        """
        g = self.g_max * P_field / (self.K_P + P_field)
        g = np.where(P_field >= 0.0, g, 0.0)
        return g / 86400.0  # 转 1/s

    def biological_tendency(self, N, P, Z, D):
        """
        计算生物地球化学源汇项（局部反应项，不含平流/扩散）。

        返回四元组 (dNdt, dPdt, dZdt, dDdt)。
        """
        U = self.uptake_rate(N, P)
        G = self.grazing_rate(P)

        # 营养盐
        dNdt = -U * P + (self.gamma * self.g_zoo / 86400.0) * Z + (self.r_D / 86400.0) * D

        # 浮游植物
        dPdt = U * P - (self.m_P / 86400.0) * P - G * Z

        # 浮游动物
        dZdt = self.beta * G * Z - (self.m_Z / 86400.0) * Z - (self.g_zoo / 86400.0) * Z

        # 碎屑（包含沉降项的垂向梯度近似）
        dDdt = (1.0 - self.beta) * G * Z + (self.m_P / 86400.0) * P + (self.m_Z / 86400.0) * Z - (self.r_D / 86400.0) * D

        # 沉降项（碎屑向下沉降）
        dDdt_sinking = np.zeros_like(D)
        dDdt_sinking[:, 1:-1] = -self.w_s * (D[:, 2:] - D[:, :-2]) / (2 * self.dz)
        # 上边界无通量，下边界出流
        dDdt_sinking[:, 0] = -self.w_s * (D[:, 1] - D[:, 0]) / self.dz
        dDdt_sinking[:, -1] = -self.w_s * (D[:, -1] - D[:, -2]) / self.dz

        dDdt += dDdt_sinking

        return dNdt, dPdt, dZdt, dDdt

    def laplacian_bio(self, F):
        """
        生物场扩散（水平扩散为主，垂向扩散较弱）。
        """
        dx = self.dx
        dz = self.dz
        L = np.zeros_like(F)
        L[1:-1, 1:-1] = (
            (F[2:, 1:-1] - 2 * F[1:-1, 1:-1] + F[:-2, 1:-1]) / (dx ** 2) +
            (F[1:-1, 2:] - 2 * F[1:-1, 1:-1] + F[1:-1, :-2]) / (dz ** 2)
        )
        return L

    def advection_term(self, u, w, F):
        """
        速度场 (u, w) 对场 F 的平流项 u·∇F（二阶中心差分）。
        """
        dx = self.dx
        dz = self.dz
        adv = np.zeros_like(F)

        adv[1:-1, 1:-1] = (
            u[1:-1, 1:-1] * (F[2:, 1:-1] - F[:-2, 1:-1]) / (2 * dx) +
            w[1:-1, 1:-1] * (F[1:-1, 2:] - F[1:-1, :-2]) / (2 * dz)
        )
        return adv

    def step(self, u, w):
        """
        执行一个生态系统时间步。

        时间积分采用前向欧拉，生物过程使用子步（substepping）
        以维持数值稳定性（CFL 条件对生物反应更严格）。
        """
        dt = self.dt
        # 生物子步：将 dt 细分为 n_sub 步
        bio_dt_max = 3600.0  # 生物最大允许步长 1 小时
        n_sub = max(1, int(np.ceil(dt / bio_dt_max)))
        dt_bio = dt / n_sub

        # 平流与扩散（物理时间尺度较大，每大步计算一次）
        adv_N = self.advection_term(u, w, self.N)
        adv_P = self.advection_term(u, w, self.P)
        adv_Z = self.advection_term(u, w, self.Z)
        adv_D = self.advection_term(u, w, self.D)

        diff_N = self.kappa_bio * self.laplacian_bio(self.N)
        diff_P = self.kappa_bio * self.laplacian_bio(self.P)
        diff_Z = self.kappa_bio * self.laplacian_bio(self.Z)
        diff_D = self.kappa_bio * self.laplacian_bio(self.D)

        for _ in range(n_sub):
            bio_N, bio_P, bio_Z, bio_D = self.biological_tendency(
                self.N, self.P, self.Z, self.D)

            self.N += dt_bio * (-adv_N + diff_N + bio_N)
            self.P += dt_bio * (-adv_P + diff_P + bio_P)
            self.Z += dt_bio * (-adv_Z + diff_Z + bio_Z)
            self.D += dt_bio * (-adv_D + diff_D + bio_D)

            # 非负约束与上限（生态浓度不可为负，且防止爆增）
            self.N = np.clip(self.N, 0.0, 20.0)
            self.P = np.clip(self.P, 0.0, 5.0)
            self.Z = np.clip(self.Z, 0.0, 3.0)
            self.D = np.clip(self.D, 0.0, 5.0)

            # NaN/Inf 清洗
            self.N = np.where(np.isfinite(self.N), self.N, 0.0)
            self.P = np.where(np.isfinite(self.P), self.P, 0.1)
            self.Z = np.where(np.isfinite(self.Z), self.Z, 0.05)
            self.D = np.where(np.isfinite(self.D), self.D, 0.05)

        # 边界：保持侧边界营养盐供给
        self.N[0, :] = 5.0
        self.N[-1, :] = 2.0
        self.P[0, :] = 0.05
        self.P[-1, :] = 0.2
        self.Z[0, :] = 0.02
        self.Z[-1, :] = 0.08

    def total_nitrogen(self):
        """
        总氮守恒检验：∫(N + P + Z + D) dV
        """
        return np.sum(self.N + self.P + self.Z + self.D) * self.dx * self.dz

    def primary_production(self):
        """
        初级生产力总量 [mmol N/(m³·s)] 的空间积分。
        """
        U = self.uptake_rate(self.N, self.P)
        return np.sum(U * self.P) * self.dx * self.dz
