"""
polymerization_kinetics.py
============================
自由基聚合反应动力学模型 (Radical Polymerization Kinetics)

基于种子项目 090_biochemical_linear_ode 与 1037_rk45 融合重构。

核心科学模型：
--------------
本模块实现非均相自由基链式聚合的完整动力学方程组，采用矩方法
(Method of Moments) 描述分子量分布(MWD)的时间演化。

符号约定：
  I    : 引发剂浓度  [mol/L]
  R    : 初级自由基浓度 [mol/L]
  P_n  : 长度为 n 的活性链浓度 [mol/L]
  D_n  : 长度为 n 的死链浓度 [mol/L]
  M    : 单体浓度 [mol/L]
  S    : 链转移剂浓度 [mol/L]

动力学机理：
  1. 引发剂分解 : I --k_d--> 2 R^ullet
  2. 引发        : R^ullet + M --k_i--> P_1^ullet
  3. 链增长      : P_n^ullet + M --k_p--> P_{n+1}^ullet
  4. 链终止(偶合): P_n^ullet + P_m^ullet --k_tc--> D_{n+m}
  5. 链终止(歧化): P_n^ullet + P_m^ullet --k_td--> D_n + D_m
  6. 链转移      : P_n^ullet + S --k_tr--> D_n + P_1^ullet

矩定义 (k-th moment):
  \lambda_k = \sum_{n=1}^{\infty} n^k P_n
  \mu_k     = \sum_{n=1}^{\infty} n^k D_n

守恒关系：
  总活性链浓度 : \lambda_0 = \sum P_n
  总死链浓度   : \mu_0 = \sum D_n
  总单体消耗   : -dM/dt = k_p M \lambda_0 + k_i M R + R_p^{other}

稳态假设(SSA)下活性链矩方程：
  d\lambda_0/dt = k_i M R + k_tr S (\lambda_0 - \lambda_0) - 2 k_t \lambda_0^2
                = k_i M R - 2 k_t \lambda_0^2  \approx 0

  d\lambda_1/dt = k_i M R + k_p M \lambda_0 + k_tr S (\lambda_0 - \lambda_1)
                 - 2 k_t \lambda_0 \lambda_1  \approx 0

  d\lambda_2/dt = k_i M R + k_p M (2\lambda_1 + \lambda_0)
                 + k_tr S (\lambda_0 - \lambda_2)
                 - 2 k_t \lambda_0 \lambda_2  \approx 0

其中 k_t = k_tc + k_td 为总终止速率常数。

死链矩演化方程（非稳态）：
  d\mu_0/dt = (k_tc + 0.5 k_td) \lambda_0^2 + 0.5 k_td \lambda_0^2
            = k_t \lambda_0^2 - 0.5 k_tc \lambda_0^2

  d\mu_1/dt = k_tc \lambda_0 \lambda_1 + k_td \lambda_0 \lambda_1
            = k_t \lambda_0 \lambda_1

  d\mu_2/dt = k_tc (\lambda_0 \lambda_2 + \lambda_1^2) + k_td \lambda_0 \lambda_2
            = k_t \lambda_0 \lambda_2 + k_tc \lambda_1^2

数均与重均聚合度：
  DP_n = (\lambda_1 + \mu_1) / (\lambda_0 + \mu_0)
  DP_w = (\lambda_2 + \mu_2) / (\lambda_1 + \mu_1)
  PDI  = DP_w / DP_n
"""

import numpy as np
from typing import Tuple, Dict, Callable, Optional


class PolymerizationParameters:
    """
    聚合反应参数管理类
    基于 biochemical_linear_parameters.m 的持久化参数设计理念
    """

    def __init__(self,
                 kd: float = 1.0e-4,      # 1/s  引发剂分解速率常数
                 ki: float = 1.0e3,       # L/(mol·s) 引发速率常数
                 kp: float = 3.0e3,       # L/(mol·s) 链增长速率常数
                 ktc: float = 1.0e7,      # L/(mol·s) 偶合终止速率常数
                 ktd: float = 1.0e7,      # L/(mol·s) 歧化终止速率常数
                 ktr: float = 1.0e-1,     # L/(mol·s) 链转移速率常数
                 f: float = 0.6,          # 引发剂效率因子
                 M0: float = 8.0,         # mol/L 初始单体浓度
                 I0: float = 1.0e-2,      # mol/L 初始引发剂浓度
                 S0: float = 0.0,         # mol/L 初始链转移剂浓度
                 T: float = 333.15,       # K 反应温度
                 Ea_p: float = 32.0e3,    # J/mol 链增长活化能
                 Ea_t: float = 8.0e3,     # J/mol 终止活化能
                 R_gas: float = 8.314,    # J/(mol·K) 气体常数
                 T_ref: float = 333.15,   # K 参考温度
                 t0: float = 0.0,         # s 初始时间
                 tstop: float = 7200.0):  # s 终止时间 (2小时)
        self.kd = float(kd)
        self.ki = float(ki)
        self.kp = float(kp)
        self.ktc = float(ktc)
        self.ktd = float(ktd)
        self.ktr = float(ktr)
        self.f = float(f)
        self.M0 = float(M0)
        self.I0 = float(I0)
        self.S0 = float(S0)
        self.T = float(T)
        self.Ea_p = float(Ea_p)
        self.Ea_t = float(Ea_t)
        self.R_gas = float(R_gas)
        self.T_ref = float(T_ref)
        self.t0 = float(t0)
        self.tstop = float(tstop)
        self._validate()

    def _validate(self) -> None:
        assert self.kd > 0.0, "kd must be positive"
        assert self.kp > 0.0, "kp must be positive"
        assert self.f > 0.0 and self.f <= 1.0, "f must be in (0,1]"
        assert self.M0 > 0.0, "M0 must be positive"
        assert self.I0 >= 0.0, "I0 must be non-negative"
        assert self.T > 0.0, "T must be positive"
        assert self.tstop > self.t0, "tstop must exceed t0"

    def temperature_correction(self, rate_const: float, Ea: float) -> float:
        """
        Arrhenius 温度修正：
            k(T) = k_ref * exp[ -Ea/R * (1/T - 1/T_ref) ]
        """
        return rate_const * np.exp(-Ea / self.R_gas * (1.0 / self.T - 1.0 / self.T_ref))

    def effective_rate_constants(self) -> Dict[str, float]:
        """返回经温度修正后的有效速率常数"""
        kp_eff = self.temperature_correction(self.kp, self.Ea_p)
        ki_eff = self.temperature_correction(self.ki, self.Ea_p * 0.8)
        ktc_eff = self.temperature_correction(self.ktc, self.Ea_t)
        ktd_eff = self.temperature_correction(self.ktd, self.Ea_t)
        ktr_eff = self.temperature_correction(self.ktr, self.Ea_p * 1.1)
        return {
            'kd': self.kd,
            'ki': ki_eff,
            'kp': kp_eff,
            'ktc': ktc_eff,
            'ktd': ktd_eff,
            'ktr': ktr_eff,
        }


def polymerization_deriv(t: float, y: np.ndarray, params: PolymerizationParameters) -> np.ndarray:
    """
    聚合动力学ODE右端项 (基于 biochemical_linear_deriv.m 的矩阵形式)

    状态向量 y = [M, I, S, lambda_0, lambda_1, lambda_2, mu_0, mu_1, mu_2]^T
    共9个方程，描述反应器内平均浓度与矩的时间演化。

    各组分平衡：
      dM/dt = - (kp * lambda_0 + ki * R) * M
      dI/dt = - kd * I
      dS/dt = - ktr * lambda_0 * S

    活性链矩（准稳态近似下代数耦合，此处保留ODE形式以确保数值稳定性）：
      dλ0/dt = 2f kd I - kt λ0^2 - ktr S λ0 + ktr S λ0   (注：转移对λ0无净贡献)
             = 2f kd I - kt λ0^2
      dλ1/dt = ki M R + kp M λ0 - kt λ0 λ1 - ktr S λ1 + ktr S λ0
      dλ2/dt = ki M R + kp M (2λ1 + λ0) - kt λ0 λ2 - ktr S λ2 + ktr S λ0

    死链矩：
      dμ0/dt = 0.5*ktc*λ0^2 + ktd*λ0^2
             = (0.5*ktc + ktd) * λ0^2
      dμ1/dt = kt λ0 λ1
      dμ2/dt = kt λ0 λ2 + ktc λ1^2
    """
    if y.ndim != 1:
        y = y.flatten()

    # 状态解包
    M, I_conc, S = y[0], y[1], y[2]
    lam0, lam1, lam2 = y[3], y[4], y[5]
    mu0, mu1, mu2 = y[6], y[7], y[8]

    # 获取温度修正速率常数
    k = params.effective_rate_constants()
    kd, ki, kp = k['kd'], k['ki'], k['kp']
    ktc, ktd, ktr = k['ktc'], k['ktd'], k['ktr']
    kt = ktc + ktd
    f = params.f

    # 初级自由基浓度（准稳态近似）
    # R_ss = 2 f kd I / (ki M + ktr S + small_stabilization)
    denom = ki * M + ktr * S + 1.0e-12
    R = 2.0 * f * kd * I_conc / denom

    # 边界处理：防止负浓度
    M = max(M, 1.0e-15)
    I_conc = max(I_conc, 1.0e-15)
    S = max(S, 0.0)
    lam0 = max(lam0, 1.0e-15)

    # 单体消耗速率
    dMdt = -(kp * lam0 + ki * R) * M

    # 引发剂分解
    dIdt = -kd * I_conc

    # 链转移剂消耗
    dSdt = -ktr * lam0 * S

    # 活性链矩
    dlam0dt = 2.0 * f * kd * I_conc - kt * lam0 ** 2
    dlam1dt = ki * M * R + kp * M * lam0 - kt * lam0 * lam1 - ktr * S * lam1 + ktr * S * lam0
    dlam2dt = ki * M * R + kp * M * (2.0 * lam1 + lam0) - kt * lam0 * lam2 - ktr * S * lam2 + ktr * S * lam0

    # TODO(Hole 1): 实现死链矩演化方程与状态向量构造
    # 自由基聚合动力学矩方法中，死链矩的演化方程为：
    #   dμ₀/dt = (0.5·ktc + ktd) · λ₀²
    #   dμ₁/dt = kt · λ₀ · λ₁
    #   dμ₂/dt = kt · λ₀ · λ₂ + ktc · λ₁²
    # 其中 kt = ktc + ktd 为总终止速率常数。
    # 最终构造 9 维状态导数向量 dydt = [dMdt, dIdt, dSdt, dλ₀dt, dλ₁dt, dλ₂dt, dμ₀dt, dμ₁dt, dμ₂dt]
    raise NotImplementedError("Hole 1: 请实现死链矩方程与 dydt 构造")


def polymerization_initial_state(params: PolymerizationParameters) -> np.ndarray:
    """
    构造初始状态向量 y0 = [M0, I0, S0, λ0_0, λ1_0, λ2_0, μ0_0, μ1_0, μ2_0]
    初始时刻无聚合物链存在。
    """
    y0 = np.array([
        params.M0,
        params.I0,
        params.S0,
        1.0e-12,   # λ0
        1.0e-12,   # λ1
        1.0e-12,   # λ2
        1.0e-12,   # μ0
        1.0e-12,   # μ1
        1.0e-12,   # μ2
    ])
    return y0


def rk45_step(yprime: Callable, t: float, y: np.ndarray, dt: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    单步 RK4(5) Cash-Karp 嵌入式方法
    基于 rk45.m 的 Butcher 表重构，采用五阶解与四阶解之差作为局部误差估计。

    Butcher 表 (Cash-Karp):
      0          |
      1/5        | 1/5
      3/10       | 3/40        9/40
      3/5        | 3/10       -9/10        6/5
      1          | -11/54      5/2        -70/27      35/27
      7/8        | 1631/55296  175/512    575/13824  44275/110592  253/4096
      -----------|-----------------------------------------------------------
      5th        | 37/378      0          250/621    125/594       0         512/1771
      4th        | 2825/27648  0          18575/48384  13525/55296  277/14336  1/4
    """
    a = np.array([
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0 / 5.0, 0.0, 0.0, 0.0, 0.0],
        [3.0 / 40.0, 9.0 / 40.0, 0.0, 0.0, 0.0],
        [3.0 / 10.0, -9.0 / 10.0, 6.0 / 5.0, 0.0, 0.0],
        [-11.0 / 54.0, 5.0 / 2.0, -70.0 / 27.0, 35.0 / 27.0, 0.0],
        [1631.0 / 55296.0, 175.0 / 512.0, 575.0 / 13824.0,
         44275.0 / 110592.0, 253.0 / 4096.0]
    ], dtype=float)

    b5 = np.array([37.0 / 378.0, 0.0, 250.0 / 621.0,
                   125.0 / 594.0, 0.0, 512.0 / 1771.0], dtype=float)
    b4 = np.array([2825.0 / 27648.0, 0.0, 18575.0 / 48384.0,
                   13525.0 / 55296.0, 277.0 / 14336.0, 1.0 / 4.0], dtype=float)
    c = np.array([0.0, 1.0 / 5.0, 3.0 / 10.0, 3.0 / 5.0, 1.0, 7.0 / 8.0], dtype=float)

    k = np.zeros((6, y.size), dtype=float)
    k[0, :] = dt * yprime(t + c[0] * dt, y)
    for i in range(1, 6):
        yi = y.copy()
        for j in range(i):
            yi += a[i, j] * k[j, :]
        k[i, :] = dt * yprime(t + c[i] * dt, yi)

    y5 = y + np.dot(b5, k)
    y4 = y + np.dot(b4, k)
    error = np.abs(y5 - y4)
    return y5, error


def integrate_polymerization(params: PolymerizationParameters,
                             n_steps: int = 2000) -> Tuple[np.ndarray, np.ndarray]:
    """
    使用自适应步长 RK45 对聚合动力学方程组进行时间积分。
    返回时间向量 t 与状态矩阵 Y (n_steps+1, 9)。
    """
    t0 = params.t0
    tstop = params.tstop
    y0 = polymerization_initial_state(params)

    dt_initial = (tstop - t0) / n_steps
    t_vec = np.zeros(n_steps + 1)
    y_mat = np.zeros((n_steps + 1, y0.size))
    t_vec[0] = t0
    y_mat[0, :] = y0

    t = t0
    y = y0.copy()
    dt = dt_initial
    step = 0

    # 简单的自适应步长控制
    atol = 1.0e-8
    rtol = 1.0e-6
    safety = 0.9
    min_dt = 1.0e-6
    max_dt = (tstop - t0) / 10.0

    while t < tstop and step < n_steps:
        dt = min(dt, tstop - t)

        def yprime(tau, yy):
            return polymerization_deriv(tau, yy, params)

        y_next, err = rk45_step(yprime, t, y, dt)

        # 误差估计
        scale = atol + rtol * np.maximum(np.abs(y), np.abs(y_next))
        err_norm = np.sqrt(np.mean((err / scale) ** 2))

        if err_norm <= 1.0 or dt <= min_dt * 1.01:
            t += dt
            step += 1
            y = y_next.copy()
            # 边界处理
            y = np.maximum(y, 1.0e-15)
            if step <= n_steps:
                t_vec[step] = t
                y_mat[step, :] = y

            # 步长增长
            if err_norm > 0.0:
                dt = min(max_dt, safety * dt * err_norm ** (-0.2))
        else:
            # 步长缩减
            dt = max(min_dt, safety * dt * err_norm ** (-0.25))

    # 若步数不足，用最后一个有效状态填充
    if step < n_steps:
        y_mat[step + 1:, :] = y_mat[step, :]
        t_vec[step + 1:] = tstop

    return t_vec, y_mat


def compute_conversion_and_pdi(t_vec: np.ndarray,
                               y_mat: np.ndarray,
                               params: PolymerizationParameters) -> Dict[str, np.ndarray]:
    """
    由状态矩阵计算工程指标：
      单体转化率 X = (M0 - M) / M0
      数均聚合度 DP_n = (λ1 + μ1) / (λ0 + μ0)
      重均聚合度 DP_w = (λ2 + μ2) / (λ1 + μ1)
      多分散指数 PDI = DP_w / DP_n
      数均分子量 Mn = DP_n * M0_monomer (假设单体分子量为104 g/mol, 苯乙烯)
      重均分子量 Mw = DP_w * M0_monomer
    """
    M = y_mat[:, 0]
    lam0, lam1, lam2 = y_mat[:, 3], y_mat[:, 4], y_mat[:, 5]
    mu0, mu1, mu2 = y_mat[:, 6], y_mat[:, 7], y_mat[:, 8]

    M0_monomer = 104.15  # g/mol, styrene
    conversion = (params.M0 - M) / params.M0
    conversion = np.clip(conversion, 0.0, 1.0)

    total_active = lam0 + mu0
    total_active = np.where(total_active < 1.0e-14, 1.0e-14, total_active)

    DP_n = (lam1 + mu1) / total_active
    DP_w = (lam2 + mu2) / np.where(lam1 + mu1 < 1.0e-14, 1.0e-14, lam1 + mu1)
    PDI = DP_w / np.where(DP_n < 1.0e-14, 1.0e-14, DP_n)

    Mn = DP_n * M0_monomer
    Mw = DP_w * M0_monomer

    return {
        't': t_vec,
        'conversion': conversion,
        'DP_n': DP_n,
        'DP_w': DP_w,
        'PDI': PDI,
        'Mn': Mn,
        'Mw': Mw,
        'M': M,
        'lam0': lam0,
        'mu0': mu0,
    }


def exact_solution_batch(t_array: np.ndarray, params: PolymerizationParameters) -> np.ndarray:
    """
    基于 biochemical_linear_exact.m 的思想，给出简化模型的解析解：
    仅对引发剂和单体做准解析估计（忽略链长分布细节），用于验证数值解的定性行为。

    简化模型假设：
      I(t) = I0 * exp(-kd * t)
      引发速率 Ri = 2 f kd I(t)
      稳态活性链 λ0_ss = sqrt(Ri / kt)
      单体一级近似: dM/dt ≈ -kp λ0_ss M

    解析解：
      M(t) ≈ M0 * exp( -kp * ∫_0^t λ0_ss(τ) dτ )
            = M0 * exp( -kp * sqrt(2f kd I0 / kt) * (2/kd) * (1 - exp(-kd t/2)) )
    """
    t = np.asarray(t_array)
    kd = params.kd
    kt = params.ktc + params.ktd
    f = params.f
    I0 = params.I0
    kp_eff = params.effective_rate_constants()['kp']

    A = np.sqrt(2.0 * f * kd * I0 / kt) * (2.0 / kd)
    integral = A * (1.0 - np.exp(-kd * t / 2.0))
    M_approx = params.M0 * np.exp(-kp_eff * integral)
    return M_approx
