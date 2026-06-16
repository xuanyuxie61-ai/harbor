"""
随机中微子源与蒙特卡罗输运 (from 1013_GARCH + 320_duel_simulation).

中微子光度 L_ν(t) 在超新星爆发中呈现显著的随机起伏，
来源包括：
  - PNS 表面对流
  - SASI 调制
  - 中微子驱动对流 (LED)

采用 GARCH(1,1) 过程 (from 1013_jaewansh_KMOU-RDP-GARCH-model)
模拟光度残差的波动率聚集 (volatility clustering)：
  ε_t = σ_t z_t,  z_t ~ N(0, 1)
  σ_t^2 = ω + α ε_{t-1}^2 + β σ_{t-1}^2
其中 ω, α, β 为 GARCH 参数，需满足 α+β < 1 (平稳性), ω > 0.

物理光度 = L_base(t) + ε_t,
L_base(t) 为基础光度 (幂律衰减):
  L_ν,0 ≈ 3e52 erg/s,  τ_ν ≈ 3 s
  L_ν(t) = L_ν,0 exp(-t/τ_ν)

中微子能量谱近似 Fermi-Dirac:
  f(ε) ∝ ε^2 / (exp(ε / T_ν) + 1)
平均能量 <ε_ν> ≈ 3.15 T_ν.
典型温度 T_νe ≈ 4 MeV, T_ν̄e ≈ 5 MeV, T_νx ≈ 8 MeV.

**蒙特卡罗中微子包输运** (from 320_duel_simulation 交替思想):
  类比决斗模拟中 Player 1 / Player 2 的交替存活概率：
  中微子在壳层间传播时与物质交替作用：
    - 吸收 (Player 1 击中): 概率 p_abs = 1 - exp(-χ_a Δs)
    - 散射 (Player 2 击中): 概率 p_sca = 1 - exp(-χ_s Δs)
    - 穿透: 概率 p_trans = exp(-(χ_a + χ_s) Δs)
  包存活概率 = Π_{壳层} p_trans_i + 散射重分配贡献.
"""
from __future__ import annotations
import math
import numpy as np

import constants as C


class GARCHLuminosity:
    """GARCH(1,1) 中微子光度扰动.

  L_ν(t) = L_base(t) + σ_t z_t
  σ_t^2 = ω + α ε_{t-1}^2 + β σ_{t-1}^2
  """

    def __init__(self, L0: float = 3.0e52, tau: float = 3.0,
                 omega: float = 1.0e50, alpha: float = 0.15,
                 beta: float = 0.80, T_nu: float = 5.0 * C.MEV_TO_ERG / C.K_BOLTZMANN):
        if alpha + beta >= 1.0:
            raise ValueError("GARCH 平稳性要求 α + β < 1")
        if omega <= 0:
            raise ValueError("GARCH 参数 ω 必须 > 0")
        self.L0 = float(L0)
        self.tau = float(tau)
        self.omega = float(omega)
        self.alpha = float(alpha)
        self.beta_param = float(beta)
        self.T_nu = float(T_nu)  # K
        self._sigma2_prev = omega / (1.0 - alpha - beta)
        self._eps_prev = 0.0

    def base_luminosity(self, t: float) -> float:
        """基础中微子光度 (幂律/指数衰减)."""
        return self.L0 * math.exp(-max(t, 0.0) / self.tau)

    def conditional_variance(self) -> float:
        """σ_t^2 当前值."""
        return self._sigma2_prev

    def sample(self, t: float, rng: np.random.Generator) -> float:
        """采样一次中微子光度 L_ν(t).

        返回 L_ν(t) = L_base(t) + σ_t z_t, z_t ~ N(0,1).
        同时更新内部状态 (σ_t^2, ε_t).
        """
        sigma2_t = self.omega + self.alpha * self._eps_prev ** 2 \
                   + self.beta_param * self._sigma2_prev
        sigma2_t = max(sigma2_t, 1.0e-30)
        z = float(rng.standard_normal())
        eps_t = math.sqrt(sigma2_t) * z
        L = self.base_luminosity(t) + eps_t
        self._eps_prev = eps_t
        self._sigma2_prev = sigma2_t
        return max(L, 0.0)

    def generate_sequence(self, t_array: np.ndarray,
                          rng: np.random.Generator) -> np.ndarray:
        """生成时间序列 L_ν(t_i)."""
        out = np.empty_like(t_array, dtype=np.float64)
        for i, t in enumerate(t_array):
            out[i] = self.sample(float(t), rng)
        return out


class NeutrinoSpectrum:
    """中微子 Fermi-Dirac 能谱."""

    def __init__(self, T_MeV: float = 5.0):
        self.T_MeV = float(T_MeV)
        self.T_K = self.T_MeV * C.MEV_TO_ERG / C.K_BOLTZMANN

    def mean_energy_MeV(self) -> float:
        """平均能量 <ε> = (F_3 / F_2) T ≈ 3.15 T (η=0 时)."""
        # Fermi-Dirac 积分比
        # <ε>/T = 3! ζ(4) / (2! ζ(3)) = π^4 / (30 ζ(3)) ≈ 3.1514
        return 3.1514 * self.T_MeV

    def sample_energy(self, rng: np.random.Generator) -> float:
        """从 Fermi-Dirac 分布采样 (rejection method).

        f(ε) ∝ ε^2 / (exp(ε/T) + 1).
        包络：g(ε) = ε^2 exp(-ε/T) (Maxwell-Boltzmann), 采样容易.
        接受率 = 1 / (1 + exp(-ε/T)) ≈ 1 对大 ε.
        """
        while True:
            # 从 ε^2 exp(-ε/T) 采样：Γ(3, T)
            # ε = -T ln(U1 U2 U3)
            u = rng.uniform(size=3)
            u = np.maximum(u, 1.0e-30)
            eps = -self.T_MeV * float(np.sum(np.log(u)))
            # 接受率
            accept = 1.0 / (math.exp(eps / self.T_MeV) + 1.0)
            if rng.uniform() < accept:
                return eps

    def opacity_absorption(self, eps_MeV: float, rho: float,
                           ye: float = 0.42) -> float:
        """中微子吸收不透明度 χ_a (1/cm).

        ν_e + n → p + e^-  (吸收)
        ν̄_e + p → n + e^+
        截面 σ ≈ σ_0 (ε / m_e c^2)^2 (1 + 3 g_A^2)/4
        σ_0 = 1.76e-44 cm^2,  g_A = 1.26 (轴矢耦合).
        χ_a = n_target σ
        对 ν_e: 靶核为中子 n_n = ρ X_n / m_p, X_n ≈ 1 - Y_e
        """
        sigma_0 = 1.76e-44  # cm^2
        g_A = 1.26
        fac = (1.0 + 3.0 * g_A ** 2) / 4.0
        eps_erg = eps_MeV * C.MEV_TO_ERG
        mec2 = C.M_ELECTRON * C.C_LIGHT ** 2
        sigma = sigma_0 * (eps_erg / mec2) ** 2 * fac
        n_n = rho * (1.0 - ye) / C.M_PROTON
        return n_n * sigma

    def opacity_scattering(self, eps_MeV: float, rho: float,
                           ye: float = 0.42) -> float:
        """中微子散射不透明度 χ_s (1/cm).

        ν + e^- → ν + e^- (弹性散射)
        ν + N → ν + N (核子散射)
        简化：χ_s ≈ χ_a / 5 (典型).
        """
        return self.opacity_absorption(eps_MeV, rho, ye) * 0.2


class MonteCarloNeutrinoTransport:
    """蒙特卡罗中微子包输运 (from 320_duel_simulation).

  类比决斗模拟：每个壳层中，包以概率 p_abs 被吸收 (能量沉积),
  以概率 p_sca 散射 (方向改变), 以概率 p_trans 穿透到下一壳层.
  存活 (穿透全部壳层) 概率 = Π p_trans_i.
  能量沉积率 ε_dep = Σ L_i (1 - p_trans_i).
  """

    def __init__(self, n_packets: int = 200, seed: int = 42):
        self.n_packets = int(n_packets)
        self.rng = np.random.default_rng(seed)

    def run(self, r_shell_edges: np.ndarray, chi_a: np.ndarray,
            chi_s: np.ndarray, L_inj: float) -> dict:
        """运行蒙特卡罗输运，返回能量沉积剖面.

        r_shell_edges: 壳层边界 (n_shell+1,)
        chi_a, chi_s: 壳层内吸收/散射不透明度 (n_shell,)
        L_inj: 注入光度 (erg/s)
        """
        n_shell = len(chi_a)
        if len(r_shell_edges) != n_shell + 1:
            raise ValueError("壳层边数应为 n_shell + 1")
        deposit = np.zeros(n_shell, dtype=np.float64)
        absorbed = 0
        escaped = 0
        for _ in range(self.n_packets):
            # 初始能量
            energy = L_inj / self.n_packets
            # 当前方向 (径向向外，简化 1D)
            alive = True
            for i in range(n_shell):
                dr = r_shell_edges[i + 1] - r_shell_edges[i]
                tau_a = chi_a[i] * dr
                tau_s = chi_s[i] * dr
                p_abs = 1.0 - math.exp(-tau_a)
                p_sca = 1.0 - math.exp(-tau_s)
                # 类比 duel 中 Player 1 / Player 2 的命中
                r = self.rng.uniform()
                if r < p_abs:
                    deposit[i] += energy
                    alive = False
                    break
                elif r < p_abs + p_sca:
                    # 散射：各向同性，50% 向内 50% 向外
                    if self.rng.uniform() < 0.5:
                        # 向内：可能再次穿过内层 (简化：丢失)
                        deposit[i] += energy * 0.1
                        alive = False
                        break
                # 否则穿透
            if alive:
                escaped += 1
        total_abs = deposit.sum()
        return {'deposit': deposit, 'n_absorbed': self.n_packets - escaped,
                'n_escaped': escaped, 'total_deposited': total_abs,
                'fraction_deposited': total_abs / max(L_inj, 1.0e-30)}


def neutrino_heating_rate(L_nu: float, r: np.ndarray, r_gain: float,
                          eps_avg_MeV: float = 15.0) -> np.ndarray:
    """中微子加热率 Q_ν(r) (erg/g/s).

  在增益区 r > r_gain, 中微子加热超过冷却, 净加热:
    Q_ν ≈ (L_ν / (4π r^2)) (κ_a / c) (<ε_ν> / (m_p c^2))^2
          × exp(-τ(r))
  简化形式 (Janka 2017, Eq. 28):
    Q_ν ≈ 8.4e19 (L_ν / 1e52) (r / 100km)^{-2} (<ε_ν> / MeV / 4)^2
  """
    r_safe = np.maximum(r, 1.0e5)
    L52 = L_nu / 1.0e52
    eps4 = eps_avg_MeV / 4.0
    r100 = r_safe / 1.0e7  # 100 km
    Q = 8.4e19 * L52 / (r100 ** 2) * eps4 ** 2
    # 衰减 exp(-τ)
    tau = np.maximum((r_gain / r_safe) ** 2 - 1.0, 0.0) * 3.0
    Q *= np.exp(-tau)
    return Q
