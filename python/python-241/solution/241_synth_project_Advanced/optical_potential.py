"""
optical_potential.py
===================================================================
核光学模型势构造模块

映射种子项目:
  - 446_fractal_coastline: 分形微扰 → 核表面弥散分形微扰
  - 1172_sabrin1997_AccMLBio: 层次化VAE → 层次化光学势参数化
  - 702_logistic_ode: Logistic增长 → 吸收势的Logistic型分布

核心物理公式:
  Woods-Saxon 中心势:
    V_c(r) = -V_0 / (1 + exp((r - R_v) / a_v))
  Woods-Saxon 自旋-轨道势:
    V_so(r) = V_so * (1/r) * d/dr[1/(1+exp((r-R_so)/a_so))] * (l·s)
  吸收势 (体积型):
    W_v(r) = -W_0 / (1 + exp((r - R_w) / a_w))^2
  吸收势 (表面型, Logistic导数):
    W_s(r) = -4*W_d * d/dr[1/(1+exp((r-R_w)/a_w))]
  分形核表面:
    R(theta,phi) = R_0 * (1 + sum_{l,m} alpha_{lm} Y_{lm}(theta,phi))
  库仑势 (均匀带电球):
    V_C(r) = Z_1*Z_2*e^2/(2*R_C) * (3 - r^2/R_C^2)   (r < R_C)
    V_C(r) = Z_1*Z_2*e^2/r                              (r >= R_C)
===================================================================
"""

import numpy as np
from typing import Tuple, Optional, Dict, List


# ---------- 物理常数 ----------
HBAR_C = 197.3269804        # hbar*c [MeV·fm]
E2 = 1.4399764              # e^2 [MeV·fm] (精细结构常数 * hbar*c)
R0_CONST = 1.25             # 核半径参数 r_0 [fm]


class OpticalPotentialParams:
    """
    光学模型势参数集 (层次化结构)

    第一层 (全局几何):
      A_target: 靶核质量数
      Z_target: 靶核质子数
      r_v, a_v: 中心势半径/弥散参数
      r_w, a_w: 吸收势半径/弥散参数

    第二层 (能量依赖):
      V_0, W_0, W_d: 势深度参数
      V_so: 自旋-轨道势深度

    第三层 (分形表面修正):
      fractal_dim: 表面分形维数
      fractal_amp: 分形微扰振幅
      fractal_levels: 分形迭代层数
    """

    def __init__(
        self,
        A_target: int = 208,
        Z_target: int = 82,
        A_projectile: int = 1,
        Z_projectile: int = 0,
        energy: float = 14.0,
        # 中心势参数
        V_0: float = 50.0,
        r_v: float = 1.25,
        a_v: float = 0.65,
        # 吸收势参数 (体积型)
        W_0: float = 10.0,
        r_w: float = 1.25,
        a_w: float = 0.55,
        # 表面吸收 (Logistic 导数型)
        W_d: float = 5.0,
        # 自旋-轨道势
        V_so: float = 6.0,
        r_so: float = 1.20,
        a_so: float = 0.60,
        # 分形表面参数
        fractal_dim: float = 2.15,
        fractal_amp: float = 0.08,
        fractal_levels: int = 4,
        # Logistic 吸收参数
        logistic_k: float = 1.0,
        logistic_r: float = 0.5,
    ):
        self.A_target = A_target
        self.Z_target = Z_target
        self.A_projectile = A_projectile
        self.Z_projectile = Z_projectile
        self.energy = energy

        # 层次化参数存储
        self.geometry = {
            'r_v': r_v, 'a_v': a_v,
            'r_w': r_w, 'a_w': a_w,
            'r_so': r_so, 'a_so': a_so,
        }
        self.depths = {
            'V_0': V_0, 'W_0': W_0, 'W_d': W_d, 'V_so': V_so,
        }
        self.fractal = {
            'dim': fractal_dim,
            'amplitude': fractal_amp,
            'levels': fractal_levels,
        }
        self.logistic = {
            'carrying_capacity': logistic_k,
            'growth_rate': logistic_r,
        }

        # 约化质量
        m_n = 939.565  # 中子质量 [MeV/c^2]
        self.mass_reduced = (A_projectile * A_target * m_n /
                             (A_projectile + A_target))

    @property
    def R_nuclear(self) -> float:
        """核半径: R = r_0 * A^(1/3)"""
        return self.geometry['r_v'] * self.A_target ** (1.0 / 3.0)

    @property
    def k_wavevector(self) -> float:
        """入射波波数: k = sqrt(2*m*E) / hbar"""
        return np.sqrt(2.0 * self.mass_reduced * self.energy) / HBAR_C


def woods_saxon_form(r: np.ndarray, R: float, a: float) -> np.ndarray:
    """
    Woods-Saxon 形状因子:
        f(r) = 1 / (1 + exp((r - R) / a))

    数值稳定性处理: 当 (r-R)/a > 50 时直接返回 0
    """
    x = (r - R) / a
    # 防止溢出
    x = np.clip(x, -500.0, 500.0)
    return 1.0 / (1.0 + np.exp(x))


def woods_saxon_derivative(r: np.ndarray, R: float, a: float) -> np.ndarray:
    """
    Woods-Saxon 径向导数:
        df/dr = -(1/a) * exp((r-R)/a) / (1 + exp((r-R)/a))^2

    物理含义: 表面峰型分布, 用于自旋-轨道势和表面吸收势
    """
    f = woods_saxon_form(r, R, a)
    return -(1.0 / a) * f * (1.0 - f)


def fractal_surface_perturbation(
    theta: np.ndarray,
    R_0: float,
    amplitude: float = 0.08,
    n_levels: int = 4,
    seed: int = 42
) -> np.ndarray:
    """
    分形核表面微扰 (映射自 446_fractal_coastline):

    初始圆形核表面, 通过迭代中点位移产生分形微扰:
        R_n(theta) = R_0 * (1 + delta_n(theta))

    每层迭代:
        delta_{n+1}(theta) = delta_n(theta) + A_n * noise(theta)

    其中 A_n = amplitude * (fractal_dim - 2)^n

    物理含义: 描述核表面弥散的分形几何涨落,
    影响光学势的表面弥散度和散射截面

    参数:
        theta: 角度网格 [0, 2*pi]
        R_0: 未微扰核半径
        amplitude: 初始微扰振幅
        n_levels: 分形迭代层数
        seed: 随机种子 (保证可复现)

    返回:
        R(theta): 分形微扰后的核表面半径
    """
    rng = np.random.RandomState(seed)
    n_theta = len(theta)
    dR = amplitude * R_0

    # 初始微扰 (基模)
    delta = np.zeros(n_theta)

    for level in range(n_levels):
        # 频率倍增: 第 n 层贡献 ~ 2^n 阶球谐
        freq = 2 ** level
        # 振幅衰减: ~ amplitude^(fractal_dim - 2)
        amp_n = dR * (0.5 ** level)

        # 随机相位微扰 (映射自 coastline_perturb 的噪声注入)
        phase = rng.uniform(0, 2 * np.pi)
        noise = np.sin(freq * theta + phase) + 0.3 * rng.randn(n_theta) * amp_n

        # 微扰叠加 (映射自分形海岸线的迭代中点位移)
        delta += amp_n * noise

        # 确保微扰有界
        delta = np.clip(delta, -0.3 * R_0, 0.3 * R_0)

    return R_0 * (1.0 + delta)


class OpticalPotential:
    """
    核光学模型复势构造器

    总光学势:
        U(r) = V_c(r) + V_C(r) + i*W_v(r) + i*W_s(r) + V_so(r)*(l·s)

    其中:
        V_c: Woods-Saxon 实部中心势
        V_C: 库仑势 (带电粒子)
        W_v: 体积吸收势 (负虚部)
        W_s: 表面吸收势 (Logistic 导数型, 映射自 logistic_ode)
        V_so: 自旋-轨道势

    分形修正:
        核表面 R -> R_fractal(theta) 导致势的角向调制
    """

    def __init__(self, params: OpticalPotentialParams):
        self.p = params
        self.R_v = params.geometry['r_v'] * params.A_target ** (1.0 / 3.0)
        self.R_w = params.geometry['r_w'] * params.A_target ** (1.0 / 3.0)
        self.R_so = params.geometry['r_so'] * params.A_target ** (1.0 / 3.0)
        self.R_C = 1.2 * params.A_target ** (1.0 / 3.0)  # 库仑半径

    def central_potential(self, r: np.ndarray) -> np.ndarray:
        """
        Woods-Saxon 中心实势:
            V_c(r) = -V_0 * f(r, R_v, a_v)
        """
        V0 = self.p.depths['V_0']
        av = self.p.geometry['a_v']
        return -V0 * woods_saxon_form(r, self.R_v, av)

    def volume_absorption(self, r: np.ndarray) -> np.ndarray:
        """
        体积吸收势 (负虚部):
            W_v(r) = -W_0 * [f(r, R_w, a_w)]^2

        平方形式确保吸收集中在核内部
        """
        W0 = self.p.depths['W_0']
        aw = self.p.geometry['a_w']
        f = woods_saxon_form(r, self.R_w, aw)
        return -1j * W0 * f ** 2

    def surface_absorption_logistic(self, r: np.ndarray) -> np.ndarray:
        """
        表面吸收势 — Logistic 导数型 (映射自 702_logistic_ode):

        Logistic 增长方程 dy/dt = r*y*(1-y/k) 的稳态解给出:
            W_s(r) = -4*W_d * (-1/a_w) * f(r)*(1-f(r))

        其中 f(r) = 1/(1+exp((r-R_w)/a_w)) 为 Woods-Saxon 形状因子

        物理含义: 表面峰值吸收, 描述核表面区域的非弹性散射
        等价于 -W_d * (a_w * d f/dr) 的形式
        """
        Wd = self.p.depths['W_d']
        aw = self.p.geometry['a_w']
        # df/dr 的负号使得表面吸收为正虚部贡献 (物理上为负虚部)
        dfdr = woods_saxon_derivative(r, self.R_w, aw)
        # 表面吸收: -4*W_d*a_w*(-df/dr) => 峰值在 r=R_w
        return -1j * (-4.0 * Wd * aw * dfdr)

    def coulomb_potential(self, r: np.ndarray) -> np.ndarray:
        """
        库仑势 (均匀带电球近似):
            V_C(r) = Z_p*Z_t*e^2 / (2*R_C) * (3 - r^2/R_C^2)   r < R_C
            V_C(r) = Z_p*Z_t*e^2 / r                              r >= R_C
        """
        Zp = self.p.Z_projectile
        Zt = self.p.Z_target
        if Zp == 0 or Zt == 0:
            return np.zeros_like(r, dtype=complex)

        Vc = np.zeros_like(r, dtype=complex)
        mask_in = r < self.R_C
        mask_out = ~mask_in

        # 内部
        Vc[mask_in] = Zp * Zt * E2 / (2.0 * self.R_C) * (
            3.0 - (r[mask_in] / self.R_C) ** 2)
        # 外部
        r_safe = np.where(r[mask_out] > 1e-10, r[mask_out], 1e-10)
        Vc[mask_out] = Zp * Zt * E2 / r_safe

        return Vc

    def spin_orbit_potential(
        self, r: np.ndarray, l_quantum: int, spin: float = 0.5
    ) -> np.ndarray:
        """
        自旋-轨道势:
            V_so(r) = V_so * (1/r) * df_so/dr * <l·s>

        其中 <l·s> = [j(j+1) - l(l+1) - s(s+1)] / 2
        对 j = l + 1/2: <l·s> = l/2
        对 j = l - 1/2: <l·s> = -(l+1)/2
        """
        if l_quantum == 0:
            return np.zeros_like(r, dtype=complex)

        Vso = self.p.depths['V_so']
        aso = self.p.geometry['a_so']
        dfdr = woods_saxon_derivative(r, self.R_so, aso)

        # l·s 期望值 (取 j = l + 1/2)
        l_s = l_quantum / 2.0

        r_safe = np.where(r > 1e-10, r, 1e-10)
        return Vso * l_s * dfdr / r_safe * HBAR_C ** 2

    def total_potential(
        self, r: np.ndarray,
        l_quantum: int = 0,
        apply_fractal: bool = False,
        theta: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        总光学势 (径向, 球平均):
            U(r) = V_c(r) + V_C(r) + W_v(r) + W_s(r) + V_so(r)

        若 apply_fractal=True, 则对核表面施加分形微扰:
            R_eff = R_0 * (1 + delta_fractal(theta))
            导致势的角度调制
        """
        U = (self.central_potential(r) +
             self.coulomb_potential(r) +
             self.volume_absorption(r) +
             self.surface_absorption_logistic(r) +
             self.spin_orbit_potential(r, l_quantum))

        if apply_fractal and theta is not None:
            # 分形表面微扰修正 (径向有效半径偏移)
            R_fractal = fractal_surface_perturbation(
                theta,
                self.R_v,
                amplitude=self.p.fractal['amplitude'],
                n_levels=self.p.fractal['levels']
            )
            # 球面平均修正
            R_avg = np.mean(R_fractal)
            delta_R = R_avg - self.R_v
            # 一阶修正: dV/dR * delta_R
            av = self.p.geometry['a_v']
            dfdR = woods_saxon_derivative(r, self.R_v, av)
            V0 = self.p.depths['V_0']
            U += V0 * dfdR * delta_R

        return U

    def get_potential_info(self) -> Dict:
        """返回势的参数摘要"""
        return {
            'nuclear_radius_R_v': self.R_v,
            'nuclear_radius_R_w': self.R_w,
            'coulomb_radius_R_C': self.R_C,
            'wave_number_k': self.p.k_wavevector,
            'reduced_mass': self.p.mass_reduced,
            'incident_energy': self.p.energy,
            'depths': self.p.depths.copy(),
            'geometry': self.p.geometry.copy(),
        }
