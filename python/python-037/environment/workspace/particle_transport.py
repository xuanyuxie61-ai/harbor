r"""
particle_transport.py
粒子输运与信号产生模块

本模块实现：
1. 电子在探测器电场中的漂移轨迹 ODE 积分（参考 ode_euler_system）
2. 闪烁光/声子脉冲形状的 ODE 系统（参考 glycolysis_ode）
3. 能量沉积分布与 quenching factor 模型
4. 电子-离子对产生统计（Fano 因子）

核心方程：

A. 电子漂移（显式 Euler）：
    \frac{d\vec{r}}{dt} = \mu_e \vec{E}(\vec{r})
    \mu_e: 电子迁移率 [m^2/(V·s)]

B. 闪烁脉冲 ODE（改进 Sel'kov 模型）：
    \frac{dP}{dt} = -\gamma_P P + \alpha_Q Q \cdot E_{\rm dep}
    \frac{dQ}{dt} = \beta_R - \alpha_Q Q \cdot E_{\rm dep} - \kappa_{PQ} P \cdot Q

    P: 初级闪烁光子数密度
    Q: 电离电子数密度
    E_dep: 沉积能量密度 [keV/cm^3]

C. Quenching factor（Lindhard 模型近似）：
    Q(E_R) = \frac{k \cdot g(\epsilon)}{1 + k \cdot g(\epsilon)}
    \epsilon = 11.5 \, Z^{-7/3} \, E_R \, [\text{keV}]
    g(\epsilon) = 3 \epsilon^{0.15} + 0.7 \epsilon^{0.6} + \epsilon

参考文献：
- Lindhard, J., et al. (1963). Mat. Fys. Medd. Dan. Vid. Selsk. 33, 1.
- Sel'kov, E. E. (1968). Eur. J. Biochem. 4, 79.
- Doke, T. (1981). Portugal Phys. 12, 9.
"""

import numpy as np
from typing import Callable, Tuple
from utils import r8_uniform_01


# ============================================================================
# 电子漂移 ODE 求解器（显式 Euler）
# ============================================================================

def electron_drift_euler(
    e_field_fn: Callable[[np.ndarray], np.ndarray],
    r0: np.ndarray,
    t_span: Tuple[float, float],
    n_steps: int,
    mobility: float = 3.0e-4,  # m^2/(V·s) 典型液氙值
) -> Tuple[np.ndarray, np.ndarray]:
    """
    使用显式 Euler 方法积分电子漂移轨迹。

    微分方程：
        \frac{d\vec{r}}{dt} = \mu_e \vec{E}(\vec{r}(t))

    算法：
        \vec{r}_{n+1} = \vec{r}_n + h \cdot \mu_e \vec{E}(\vec{r}_n)
        h = (t_1 - t_0) / n_{\rm steps}

    参数：
        e_field_fn: 电场函数，输入位置 (3,) 返回电场 (3,) [V/m]
        r0: 初始位置 (3,) [m]
        t_span: (t0, t1) 时间区间 [s]
        n_steps: 积分步数
        mobility: 电子迁移率 [m^2/(V·s)]

    返回：
        t_array: (n_steps+1,) 时间节点
        r_array: (n_steps+1, 3) 轨迹
    """
    t0, t1 = t_span
    if n_steps < 1:
        raise ValueError("electron_drift_euler: n_steps 必须 >= 1")
    h = (t1 - t0) / n_steps

    t_array = np.linspace(t0, t1, n_steps + 1)
    r_array = np.zeros((n_steps + 1, len(r0)))
    r_array[0] = r0

    for n in range(n_steps):
        E = e_field_fn(r_array[n])
        r_array[n + 1] = r_array[n] + h * mobility * E

    return t_array, r_array


# ============================================================================
# 闪烁/电离脉冲 ODE 系统
# ============================================================================

class ScintillationODESystem:
    """
    闪烁光-电离电子耦合 ODE 系统。

    模型方程：
        dP/dt = -γ_P · P + α_Q · Q · E_dep_norm
        dQ/dt = β_R - α_Q · Q · E_dep_norm - κ_PQ · P · Q

    物理意义：
        P(t): 激发态密度（产生闪烁光子）
        Q(t): 电离电子密度
        E_dep_norm: 归一化沉积能量（作为外部驱动项）
    """

    def __init__(
        self,
        gamma_p: float = 5.0e6,      # s^{-1}
        alpha_q: float = 1.0e5,      # s^{-1}
        beta_r: float = 1.0e4,       # s^{-1}
        kappa_pq: float = 2.0e-3,    # (s·cm^3)^{-1}
        e_dep_norm: float = 1.0,     # 归一化沉积能量
    ):
        self.gamma_p = gamma_p
        self.alpha_q = alpha_q
        self.beta_r = beta_r
        self.kappa_pq = kappa_pq
        self.e_dep_norm = e_dep_norm

    def deriv(self, t: float, y: np.ndarray) -> np.ndarray:
        """
        计算 dy/dt。

        参数：
            t: 时间 [s]
            y: (2,) 状态向量 [P, Q]

        返回：
            dydt: (2,) 时间导数
        """
        P, Q = y
        dP = -self.gamma_p * P + self.alpha_q * Q * self.e_dep_norm
        dQ = self.beta_r - self.alpha_q * Q * self.e_dep_norm - self.kappa_pq * P * Q
        return np.array([dP, dQ])

    def solve_euler(
        self,
        y0: np.ndarray,
        t_span: Tuple[float, float],
        n_steps: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        使用显式 Euler 求解。

        参数：
            y0: (2,) 初始条件
            t_span: (t0, t1)
            n_steps: 步数

        返回：
            t: (n_steps+1,)
            y: (n_steps+1, 2)
        """
        t0, t1 = t_span
        h = (t1 - t0) / n_steps
        t = np.linspace(t0, t1, n_steps + 1)
        y = np.zeros((n_steps + 1, 2))
        y[0] = y0
        for i in range(n_steps):
            dydt = self.deriv(t[i], y[i])
            y[i + 1] = y[i] + h * dydt
            # 非负约束（物理意义）
            y[i + 1] = np.maximum(y[i + 1], 0.0)
        return t, y

    def equilibrium(self) -> np.ndarray:
        """
        计算稳态解（令 dy/dt = 0）。

        稳态方程：
            0 = -γ_P P^* + α_Q Q^* E^*
            0 = β_R - α_Q Q^* E^* - κ_PQ P^* Q^*

        解析解：
            P^* = \frac{\beta_R}{\gamma_P + \kappa_{PQ} Q^*}
            Q^* 满足二次方程：
            \kappa_{PQ} \alpha_Q E^* (Q^*)^2
            + \gamma_P \alpha_Q E^* Q^*
            - \gamma_P \beta_R = 0
        """
        a = self.kappa_pq * self.alpha_q * self.e_dep_norm
        b = self.gamma_p * self.alpha_q * self.e_dep_norm
        c = -self.gamma_p * self.beta_r
        if abs(a) < 1e-20:
            Q_star = -c / b if abs(b) > 1e-20 else 0.0
        else:
            discriminant = b * b - 4.0 * a * c
            if discriminant < 0.0:
                discriminant = 0.0
            Q_star = (-b + np.sqrt(discriminant)) / (2.0 * a)
        P_star = (self.alpha_q * Q_star * self.e_dep_norm) / self.gamma_p
        return np.array([P_star, Q_star])


# ============================================================================
# Quenching Factor（Lindhard 模型）
# ============================================================================

def lindhard_quenching_factor(er_kev: float, Z: int, A: int) -> float:
    """
    Lindhard 理论 quenching factor Q(E_R)。

    公式：
        Q(E_R) = \frac{k \cdot g(\epsilon)}{1 + k \cdot g(\epsilon)}

    其中：
        \epsilon = 11.5 \, Z^{-7/3} \, E_R \, [\text{keV}]
        g(\epsilon) = 3 \epsilon^{0.15} + 0.7 \epsilon^{0.6} + \epsilon
        k = 0.133 \, Z^{2/3} \, A^{-1/2}

    参数：
        er_kev: 核反冲能量 [keV]
        Z: 原子序数
        A: 质量数

    返回：
        Q: 无量纲，范围 [0, 1]
    """
    if er_kev <= 0.0:
        return 0.0
    if Z <= 0 or A <= 0:
        raise ValueError("lindhard_quenching_factor: Z, A 必须为正")

    eps = 11.5 * (Z ** (-7.0 / 3.0)) * er_kev
    k = 0.133 * (Z ** (2.0 / 3.0)) * (A ** (-0.5))
    g_eps = 3.0 * (eps ** 0.15) + 0.7 * (eps ** 0.6) + eps
    Q = (k * g_eps) / (1.0 + k * g_eps)
    return float(np.clip(Q, 0.0, 1.0))


def ionization_yield(er_kev: float, Z: int, A: int, fano_factor: float = 0.15, epsilon_eV: float = 3.0) -> Tuple[float, float]:
    """
    计算电离产额（电子-离子对数目）及其统计涨落。

    公式：
        N_{e^-} = \frac{Q(E_R) \cdot E_R}{\varepsilon}
        \sigma_{N}^2 = F \cdot N_{e^-}

    参数：
        er_kev: 反冲能量 [keV]
        Z, A: 原子序数与质量数
        fano_factor: Fano 因子
        epsilon_eV: 产生一对电子-离子所需平均能量 [eV]

    返回：
        (N_e, sigma_N): 平均电子数及标准差
    """
    if er_kev <= 0.0:
        return 0.0, 0.0
    Q = lindhard_quenching_factor(er_kev, Z, A)
    energy_eV = er_kev * 1000.0
    N_e = (Q * energy_eV) / epsilon_eV
    sigma = np.sqrt(fano_factor * N_e) if N_e > 0.0 else 0.0
    return float(N_e), float(sigma)


# ============================================================================
# 能量沉积分布
# ============================================================================

def energy_deposition_profile(
    er_kev: float,
    detector_thickness_m: float,
    n_bins: int = 50,
    interaction_depth_m: float = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    生成核反冲能量在探测器厚度方向上的沉积分布。

    物理模型：
        假设能量在反冲核径迹长度内近似均匀沉积。
        径迹长度近似：L \approx C \cdot E_R^{1.7} [nm]（经验公式）。

    参数：
        er_kev: 反冲能量 [keV]
        detector_thickness_m: 探测器厚度 [m]
        n_bins: 离散化层数
        interaction_depth_m: 相互作用深度（None 则随机均匀分布）

    返回：
        z: (n_bins,) 深度网格 [m]
        edep: (n_bins,) 能量沉积密度 [keV/m]
    """
    if detector_thickness_m <= 0.0:
        raise ValueError("energy_deposition_profile: 厚度必须为正")
    z = np.linspace(0.0, detector_thickness_m, n_bins)
    dz = z[1] - z[0]

    if interaction_depth_m is None:
        # 均匀随机深度
        depth = detector_thickness_m * np.random.rand()
    else:
        depth = np.clip(interaction_depth_m, 0.0, detector_thickness_m)

    # 径迹长度（简化经验模型）[m]
    track_length_m = 1.0e-9 * (er_kev ** 1.7) * 10.0
    track_length_m = min(track_length_m, detector_thickness_m * 0.5)

    # 高斯型沉积分布
    sigma_track = track_length_m / 2.355  # FWHM → sigma
    if sigma_track < dz:
        sigma_track = dz

    edep = np.exp(-0.5 * ((z - depth) / sigma_track) ** 2)
    edep = edep / (np.sum(edep) * dz) * er_kev  # 归一化并缩放至总能量
    return z, edep


# ============================================================================
# 自测
# ============================================================================

if __name__ == "__main__":
    # 测试电子漂移
    def const_e_field(r):
        return np.array([0.0, 0.0, 1.0e3])  # 1 kV/m 沿 z

    t, r = electron_drift_euler(const_e_field, np.array([0.0, 0.0, 0.0]), (0.0, 1.0e-6), 100, mobility=3.0e-4)
    expected_z = 3.0e-4 * 1.0e3 * 1.0e-6
    assert abs(r[-1, 2] - expected_z) < 1e-12, f"漂移距离偏差: {r[-1, 2]} vs {expected_z}"

    # 测试闪烁 ODE
    sys = ScintillationODESystem(e_dep_norm=1.0)
    eq = sys.equilibrium()
    assert np.all(eq >= 0.0), "稳态解出现负值"
    t, y = sys.solve_euler(eq * 0.1, (0.0, 1.0e-5), 500)
    assert np.all(y >= 0.0), "ODE 解出现负值"

    # 测试 Lindhard QF
    Q = lindhard_quenching_factor(10.0, 32, 73)  # Ge
    assert 0.0 <= Q <= 1.0, f"QF 超出范围: {Q}"
    Q_low = lindhard_quenching_factor(0.1, 32, 73)
    assert Q_low < Q, "低能 QF 应小于高能 QF"

    # 测试电离产额
    N_e, sig = ionization_yield(10.0, 32, 73)
    assert N_e >= 0.0 and sig >= 0.0

    print("particle_transport.py: 所有自测通过")
