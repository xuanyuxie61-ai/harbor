# -*- coding: utf-8 -*-
"""
boundary_conditions.py
======================

磁层粒子输运边界条件模块.

本模块实现辐射带 Fokker-Planck 方程的各类物理边界条件:
  - 损失锥边界 (内边界, L ~ 1): 粒子沉降到大气
  - 磁顶边界 (外边界, L ~ 10): 磁顶阴影效应 (magnetopause shadowing)
  - 零通量边界 (对称条件)
  - 注入边界 (源项驱动)

物理背景:

1. 损失锥边界 (Loss Cone Boundary):
   在低 L 壳层, 粒子的投掷角若小于损失锥角 alpha_lc,
   则其镜点位于大气层内 (~100 km), 与中性大气碰撞而损失.

   数学表达: f(L_min, alpha < alpha_lc) = 0
   或者等价地, 在投掷角空间施加吸收边界.

   在纯 (L, E) 扩散框架下, 这等效于在 L_min 处施加:
     D_LL * df/dL |_{L_min} = -v_loss * f(L_min)

   其中 v_loss 为有效损失速率.

2. 磁顶阴影 (Magnetopause Shadowing):
   当粒子的漂移壳与磁顶相交时, 粒子逃逸到星际空间.
   磁顶位置取决于太阳风动压:
     r_mp = (2*B_0^2 / (mu_0 * P_sw))^{1/6}

   在 L_max 处施加吸收边界:
     f(L_max) = 0  (当 L_max > r_mp/R_E)

3. 注入边界 (Injection Boundary):
   磁暴期间, 等离子体片粒子通过对流注入到内磁层.
   在外边界 L_max 处施加持续源:
     f(L_max, E) = f_injection(E)

   其中 f_injection 通常取为 kappa 分布或 Maxwellian.

4. 控制障碍函数 (Control Barrier Function, CBF):
   借鉴安全控制理论 (Masonniu 项目), 对分布函数施加约束:
     h(f) = f_max - f >= 0  (相空间密度上限)
     h(f) = f >= 0           (非负性)

   这些约束通过 CBF 条件嵌入到时间推进中:
     dh/dt + lambda * h >= 0

参考文献:
  [1] Kim, H.-J. & Chan, A.A., "Fully dimensional simulations of wave-particle
      interactions", JGR (1997)
  [2] Tu, W. et al., "Radiation belt electron losses due to magnetopause shadowing",
      JGR (2009)
  [3] Ames III, A. et al., "Control barrier functions", IEEE TAC (2019)
"""

import numpy as np
import physical_constants as pc


class BoundaryConditions:
    """
    磁层辐射带边界条件管理器.

    参数
    ----
    L_min : float
        内边界 L 值
    L_max : float
        外边界 L 值
    n_L : int
        L 方向网格点数
    inner_type : str
        内边界类型: 'loss_cone', 'zero_flux', 'absorbing'
    outer_type : str
        外边界类型: 'magnetopause', 'injection', 'zero_flux'
    loss_rate : float
        内边界损失速率 [1/s]
    injection_spectrum : callable
        注入能谱函数 f_inj(E_MeV)
    f_max_constraint : float
        相空间密度上限 (CBF 约束)
    """

    def __init__(self, L_min=None, L_max=None, n_L=None,
                 inner_type='loss_cone', outer_type='magnetopause',
                 loss_rate=1.0, injection_spectrum=None,
                 f_max_constraint=1.0e10):
        self.L_min = pc.L_MIN if L_min is None else L_min
        self.L_max = pc.L_MAX if L_max is None else L_max
        self.n_L = n_L if n_L is not None else pc.L_SHELL_NUM
        self.inner_type = inner_type
        self.outer_type = outer_type
        self.loss_rate = loss_rate
        self.f_max_constraint = f_max_constraint

        # 默认注入能谱: kappa 分布
        if injection_spectrum is None:
            self.injection_spectrum = self._default_injection_spectrum
        else:
            self.injection_spectrum = injection_spectrum

        # 磁顶位置 (随太阳风动压变化)
        self.P_sw_nominal = 2.0e-9  # 典型太阳风动压 [Pa]
        self.r_mp = self._compute_magnetopause_distance(self.P_sw_nominal)

    def _compute_magnetopause_distance(self, P_sw):
        """
        计算磁顶日下点距离.

        物理公式 (压力平衡):
          B^2/(2*mu_0) = P_sw
          (B_0 * (R_E/r)^3)^2 / (2*mu_0) = P_sw
          r_mp = R_E * (B_0^2 / (2*mu_0*P_sw))^{1/6}

        参数
        ----
        P_sw : float
            太阳风动压 [Pa]

        返回
        -------
        r_mp : float
            磁顶日下点距离 [R_E]
        """
        P_sw = max(P_sw, pc.EPSILON_NUM)
        B0 = pc.B_EQUATORIAL
        # r_mp / R_E = (B_0^2 / (2*mu_0*P_sw))^{1/6}
        r_mp = (B0**2 / (2.0 * pc.MU_0 * P_sw))**(1.0/6.0)
        return r_mp

    def _default_injection_spectrum(self, E_MeV):
        """
        默认注入能谱: 修正 kappa 分布.

        物理公式 (kappa 分布):
          f_kappa(E) = n_0 * (1 + E/(kappa*kT))^{-(kappa+1)}
                     * E / (kT)^{3/2} * Gamma(kappa+1) / (Gamma(kappa-1/2) * sqrt(pi*kappa))

        简化形式:
          f(E) = A * E * exp(-E/E_0)  (E < E_c)
          f(E) = A * E * (E/E_0)^{-kappa}  (E > E_c)

        参数
        ----
        E_MeV : ndarray
            动能 [MeV]

        返回
        -------
        f_inj : ndarray
            注入分布函数
        """
        E = np.asarray(E_MeV, dtype=np.float64)
        E_0 = 0.3   # 特征能量 [MeV]
        E_c = 1.0   # 截断能量 [MeV]
        kappa = 4.0  # kappa 指数
        A = 1.0e6    # 归一化

        f_inj = np.where(E < E_c,
                         A * E * np.exp(-E / E_0),
                         A * E * (E / E_0)**(-kappa))
        return f_inj

    # -----------------------------------------------------------------
    #  内边界 (损失锥)
    # -----------------------------------------------------------------
    def apply_inner_boundary(self, f, dL, dt=None):
        """
        施加内边界条件.

        物理模型:
          - 'loss_cone': 损失锥吸收边界
            f[0] = 0 或  df/dL = -v_loss * f / D_LL

          - 'zero_flux': 零通量边界
            df/dL = 0  =>  f[0] = f[1]

          - 'absorbing': 完全吸收
            f[0] = 0

        参数
        ----
        f : ndarray, shape (n_L, ...)
            分布函数
        dL : float
            L 方向网格间距
        dt : float, optional
            时间步长 [s]

        返回
        -------
        f : ndarray
            施加边界条件后的分布函数
        """
        f = f.copy()

        if self.inner_type == 'loss_cone':
            # Robin 边界条件: D_LL * df/dL + v_loss * f = 0
            # 离散化: D_LL * (f[1] - f[0]) / dL + v_loss * f[0] = 0
            # => f[0] * (v_loss - D_LL/dL) = -D_LL * f[1] / dL
            # => f[0] = D_LL * f[1] / (D_LL + v_loss * dL)
            # 简化: f[0] = 0 (强损失极限)
            f[0] = 0.0
            # 近边界指数衰减 (半隐式)
            if dt is not None and self.loss_rate > 0:
                decay = np.exp(-self.loss_rate * dt)
                f[1] = f[1] * decay

        elif self.inner_type == 'zero_flux':
            # Neumann 边界: df/dL = 0
            f[0] = f[1]

        elif self.inner_type == 'absorbing':
            # Dirichlet 边界: f = 0
            f[0] = 0.0

        else:
            raise ValueError(f"未知内边界类型: {self.inner_type}")

        return f

    # -----------------------------------------------------------------
    #  外边界 (磁顶阴影 / 注入)
    # -----------------------------------------------------------------
    def apply_outer_boundary(self, f, dL, E_MeV=None, dt=None):
        """
        施加外边界条件.

        物理模型:
          - 'magnetopause': 磁顶阴影
            f[-1] = 0 (如果 L_max > r_mp)
            f[-1] = f[-2] (否则, 零通量)

          - 'injection': 持续注入
            f[-1] = f_injection(E)

          - 'zero_flux': 零通量
            f[-1] = f[-2]

        参数
        ----
        f : ndarray, shape (n_L, ...) 或 (n_L, n_E)
            分布函数
        dL : float
            L 方向网格间距
        E_MeV : ndarray, optional
            能量网格 (用于注入谱)
        dt : float, optional
            时间步长

        返回
        -------
        f : ndarray
            施加边界条件后的分布函数
        """
        f = f.copy()
        n_L = f.shape[0]

        if self.outer_type == 'magnetopause':
            # 检查 L_max 是否在磁顶之内
            L_max = self.L_min + (n_L - 1) * dL
            if L_max > self.r_mp:
                # 磁顶阴影: 吸收边界
                f[-1] = 0.0
                # 近边界指数衰减 (模拟逃逸)
                if dt is not None:
                    tau_escape = 3600.0  # 1 小时逃逸时间
                    decay = np.exp(-dt / tau_escape)
                    f[-2] = f[-2] * decay
            else:
                # 零通量
                f[-1] = f[-2]

        elif self.outer_type == 'injection':
            # 持续注入
            if E_MeV is not None and f.ndim == 2:
                f[-1, :] = self.injection_spectrum(E_MeV)
            else:
                f[-1] = 1.0  # 标量情况

        elif self.outer_type == 'zero_flux':
            f[-1] = f[-2]

        else:
            raise ValueError(f"未知外边界类型: {self.outer_type}")

        return f

    # -----------------------------------------------------------------
    #  控制障碍函数 (CBF) 约束
    # -----------------------------------------------------------------
    def apply_cbf_constraint(self, f, dt):
        """
        施加控制障碍函数约束.

        物理意义:
          相空间密度 f 必须满足:
            h_1(f) = f >= 0           (非负性)
            h_2(f) = f_max - f >= 0   (上限)

        CBF 条件 (Ames et al. 2019):
          dh/dt + lambda * h >= 0

        对于 h_1 = f:
          f^{n+1} >= (1 - lambda*dt) * f^n

        对于 h_2 = f_max - f:
          f^{n+1} <= f_max - (1 - lambda*dt) * (f_max - f^n)

        参数
        ----
        f : ndarray
            分布函数
        dt : float
            时间步长

        返回
        -------
        f : ndarray
            满足 CBF 约束的分布函数
        """
        lambda_cbf = 1.0  # CBF 参数

        # 非负性约束
        f = np.maximum(f, 0.0)

        # 上限约束
        decay = np.exp(-lambda_cbf * dt)
        f = np.minimum(f, self.f_max_constraint * (1.0 - decay) + f * decay)

        # 硬截断
        f = np.clip(f, 0.0, self.f_max_constraint)

        return f

    # -----------------------------------------------------------------
    #  完整边界施加
    # -----------------------------------------------------------------
    def apply_all(self, f, dL, E_MeV=None, dt=None):
        """
        依次施加所有边界条件.

        顺序:
          1. 内边界
          2. 外边界
          3. CBF 约束

        参数
        ----
        f : ndarray
            分布函数
        dL : float
            L 方向网格间距
        E_MeV : ndarray, optional
            能量网格
        dt : float, optional
            时间步长

        返回
        -------
        f : ndarray
            施加边界条件后的分布函数
        """
        f = self.apply_inner_boundary(f, dL, dt)
        f = self.apply_outer_boundary(f, dL, E_MeV, dt)
        if dt is not None:
            f = self.apply_cbf_constraint(f, dt)
        return f

    # -----------------------------------------------------------------
    #  诊断
    # -----------------------------------------------------------------
    def boundary_diagnostics(self, f):
        """
        计算边界诊断信息.

        返回
        -------
        info : dict
            边界信息:
              - inner_value: 内边界值
              - outer_value: 外边界值
              - inner_gradient: 内边界梯度
              - outer_gradient: 外边界梯度
              - total_particles: 总粒子数 (积分)
              - max_value: 最大分布函数值
              - cbf_violation: CBF 违反量
        """
        dL = (self.L_max - self.L_min) / max(self.n_L - 1, 1)

        info = {
            'inner_value': float(np.max(np.abs(f[0]))),
            'outer_value': float(np.max(np.abs(f[-1]))),
            'inner_gradient': float(np.max(np.abs(f[1] - f[0])) / dL),
            'outer_gradient': float(np.max(np.abs(f[-1] - f[-2])) / dL),
            'total_particles': float(np.sum(f) * dL),
            'max_value': float(np.max(f)),
            'min_value': float(np.min(f)),
            'cbf_violation': float(max(0.0, np.max(f) - self.f_max_constraint)),
            'negativity': float(np.sum(f < 0)),
        }
        return info

    def summary(self):
        """打印边界条件摘要."""
        print(f"磁层辐射带边界条件:")
        print(f"  L 范围: [{self.L_min:.2f}, {self.L_max:.2f}], n_L = {self.n_L}")
        print(f"  内边界类型: {self.inner_type}")
        print(f"  外边界类型: {self.outer_type}")
        print(f"  磁顶距离: r_mp = {self.r_mp:.2f} R_E")
        print(f"  损失速率: {self.loss_rate:.4e} /s")
        print(f"  CBF 上限: {self.f_max_constraint:.4e}")


if __name__ == "__main__":
    bc = BoundaryConditions()
    bc.summary()

    # 测试边界施加
    n_L = 61
    f = np.ones((n_L, 41)) * 1.0e6
    bc_applied = bc.apply_all(f, dL=0.1, E_MeV=np.linspace(0.1, 8.0, 41))
    diag = bc.boundary_diagnostics(bc_applied)
    print(f"\n边界诊断: {diag}")
