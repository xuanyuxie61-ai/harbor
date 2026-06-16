"""
physical_model.py
=================

计算高能物理物理模型 —— H → γγ 型双光子共振态搜索简化模型。
种子项目 1220 (Unruh thermodynamics) 启发: 将探测器响应建模为
类 Unruh 温度效应 T_eff = ħ a / (2π c k_B) 对信号效率的修正。

核心物理量:
    μ       —— 信号强度乘子 (POI, Parameter of Interest)
    θ       ——  nuisance 参数向量 (对数正态约束)
    n_i     —— 第 i 个分析区间 (bin) 的期望事件数
    s_i(μ)  —— μ 倍信号期望
    b_i(θ)  —— 受系统误差调制的背景期望

期望事件率:
    ν_i(μ, θ) = μ · s_i · (1 + Σ_k γ_{ik} θ_k) + b_i · Π_k (1 + δ_{ik} θ_k)^{κ_{ik}}

其中 γ_{ik} 为信号系统误差导数, δ_{ik} 为背景系统误差导数,
κ_{ik} 控制非线性修正阶数 (取自 Unruh 热力学约束中的高阶展开)。

Unruh 温度修正信号效率 (seed 1220):
    ε_i(a) = ε_i^0 · [1 - (T_U / T_beam)^2 · c_2,i + (T_U / T_beam)^4 · c_4,i]
    T_U = ħ a / (2π c k_B)

探测器质量分辨率 (Voigt 卷积展宽, 取自 seed 1225 HEOM):
    σ_eff,i = σ_stat,i ⊕ σ_syst,i(θ) = √(σ_stat,i^2 + (θ_JES · σ_JES · m_i)^2)

Poisson 涨落的单发约束 (seed 1220 → Rényi 熵单发下界):
    N_0(α) = (1/(1-α)) · ln[ Σ_i ν_i^α / (Σ_i ν_i)^α ]
    用于诊断统计区间的灵敏度退化。
"""

from __future__ import annotations
import math
import numpy as np
from typing import Tuple, Dict, Optional


# ---------------------------------------------------------------------------
# 物理常数 (自然单位 ħ = c = k_B = 1 与 SI 并用)
# ---------------------------------------------------------------------------
HBAR_C = 197.3269804       # MeV · fm
PROTON_MASS = 938.27208816 # MeV
HIGGS_MASS_REF = 125.09    # GeV (PDG 2022)
HIGGS_WIDTH_SM = 4.07e-3   # GeV (SM @ mH = 125 GeV)
SQRT_S = 13000.0           # GeV (LHC Run-2)
LUMI_REF = 139.0           # fb^{-1}


# ---------------------------------------------------------------------------
# Unruh 型温度修正 (seed 1220)
# ---------------------------------------------------------------------------
def unruh_temperature(accel: float) -> float:
    """
    Unruh 等效温度 T_U = a / (2π)  [自然单位 ħ = c = k_B = 1]

    Parameters
    ----------
    accel : float
        探测器等效固有加速度 a [GeV] (参数化探测器响应非理想性)

    Returns
    -------
    float
        T_U [GeV]
    """
    if accel < 0.0:
        raise ValueError(f"加速度不能为负: accel={accel}")
    return accel / (2.0 * math.pi)


def signal_efficiency_unruh(
    epsilon_0: np.ndarray,
    accel: float,
    c2: np.ndarray,
    c4: np.ndarray,
    T_beam: float = 0.15,
) -> np.ndarray:
    """
    Unruh 修正后的信号效率 (seed 1220 → 单发热力学约束):

        ε_i(a) = ε_i^0 · [1 - (T_U/T_beam)^2 · c2_i + (T_U/T_beam)^4 · c4_i]

    当 T_U → 0 时退化为理想效率 ε_i^0。
    边界: 效率截断于 [0, 1] 区间。
    """
    TU = unruh_temperature(accel)
    ratio2 = (TU / T_beam) ** 2
    ratio4 = ratio2 ** 2
    corr = 1.0 - ratio2 * c2 + ratio4 * c4
    eps = epsilon_0 * corr
    return np.clip(eps, 0.0, 1.0)


# ---------------------------------------------------------------------------
# 探测器分辨率展宽 (seed 1225 → HEOM Padé 分解启发)
# ---------------------------------------------------------------------------
def effective_mass_resolution(
    sigma_stat: np.ndarray,
    sigma_JES: np.ndarray,
    masses: np.ndarray,
    theta_JES: float,
) -> np.ndarray:
    """
    Voigt 卷积近似: σ_eff = √(σ_stat^2 + (θ_JES · σ_JES · m)^2)

    等价于 HEOM 中 Drude 谱密度展宽 J(ω) = 2λ ω γ / (ω^2 + γ^2)
    在 γ → 0 (窄宽度) 极限下的方差叠加。
    """
    syst_component = (theta_JES * sigma_JES * masses) ** 2
    return np.sqrt(sigma_stat**2 + syst_component)


def gaussian_signal_shape(
    m_grid: np.ndarray,
    m_peak: float,
    sigma_eff: float,
) -> np.ndarray:
    """
    高斯型信号分布 (单位归一):
        S(m) = (1 / (σ√(2π))) · exp(-(m - m_peak)^2 / (2σ^2))
    """
    if sigma_eff <= 0.0:
        raise ValueError(f"分辨率必须为正: sigma_eff={sigma_eff}")
    z = (m_grid - m_peak) / sigma_eff
    return np.exp(-0.5 * z**2) / (sigma_eff * math.sqrt(2.0 * math.pi))


# ---------------------------------------------------------------------------
# 分析区间期望事件数构造
# ---------------------------------------------------------------------------
class BinModel:
    """
    多区间 Poisson 计数模型。

    Attributes
    ----------
    n_bins : int
        分析区间数
    s_nom : ndarray, shape (n_bins,)
        标称信号期望 (μ=1, θ=0)
    b_nom : ndarray, shape (n_bins,)
        标称背景期望
    gamma_sig : ndarray, shape (n_bins, n_nuis)
        信号对第 k 个 nuisance 的线性响应系数
    delta_bkg : ndarray, shape (n_bins, n_nuis)
        背景对第 k 个 nuisance 的线性响应系数
    kappa : ndarray, shape (n_bins, n_nuis)
        背景非线性修正阶数 (κ=1 → 线性; κ=2 → 二次)
    accel : float
        探测器等效 Unruh 加速度 (用于效率修正)
    masses : ndarray, shape (n_bins,)
        各区间中心质量 [GeV]
    """

    def __init__(
        self,
        s_nom: np.ndarray,
        b_nom: np.ndarray,
        gamma_sig: Optional[np.ndarray] = None,
        delta_bkg: Optional[np.ndarray] = None,
        kappa: Optional[np.ndarray] = None,
        accel: float = 0.0,
        masses: Optional[np.ndarray] = None,
        c2_corr: Optional[np.ndarray] = None,
        c4_corr: Optional[np.ndarray] = None,
        T_beam: float = 0.15,
    ):
        self.s_nom = np.asarray(s_nom, dtype=np.float64)
        self.b_nom = np.asarray(b_nom, dtype=np.float64)
        self.n_bins = len(self.s_nom)
        if len(self.b_nom) != self.n_bins:
            raise ValueError("信号与背景区间数不一致")

        # 默认: 2 个 nuisance (亮度, JES)
        n_nuis = 2
        self.n_nuis = n_nuis
        self.gamma_sig = (
            np.asarray(gamma_sig, dtype=np.float64)
            if gamma_sig is not None
            else np.zeros((self.n_bins, n_nuis))
        )
        self.delta_bkg = (
            np.asarray(delta_bkg, dtype=np.float64)
            if delta_bkg is not None
            else np.zeros((self.n_bins, n_nuis))
        )
        self.kappa = (
            np.asarray(kappa, dtype=np.float64)
            if kappa is not None
            else np.ones((self.n_bins, n_nuis))
        )
        self.accel = float(accel)
        self.masses = (
            np.asarray(masses, dtype=np.float64)
            if masses is not None
            else np.linspace(110.0, 140.0, self.n_bins)
        )
        # Unruh 效率修正系数
        self.c2 = (
            np.asarray(c2_corr, dtype=np.float64)
            if c2_corr is not None
            else 0.05 * np.ones(self.n_bins)
        )
        self.c4 = (
            np.asarray(c4_corr, dtype=np.float64)
            if c4_corr is not None
            else 0.01 * np.ones(self.n_bins)
        )
        self.T_beam = float(T_beam)

        # 预计算 Unruh 修正后的信号
        eps_0 = np.ones(self.n_bins)  # 标称效率归一
        self.s_eff = signal_efficiency_unruh(
            eps_0, self.accel, self.c2, self.c4, self.T_beam
        )

    # ------------------------------------------------------------------
    def expected_rate(
        self, mu: float, theta: np.ndarray
    ) -> np.ndarray:
        """
        期望事件率 ν_i(μ, θ):

            ν_i = μ · s_i^eff · (1 + Σ_k γ_{ik} θ_k)
                  + b_i · Π_k (1 + δ_{ik} θ_k)^{κ_{ik}}

        边界: 期望事件率截断于 [1e-12, ∞) 以避免 log(0)。
        """
        theta = np.asarray(theta, dtype=np.float64)
        if theta.shape != (self.n_nuis,):
            raise ValueError(
                f"theta shape {theta.shape} != expected ({self.n_nuis},)"
            )
        # 信号项
        sig_mod = 1.0 + self.gamma_sig @ theta
        sig_term = mu * self.s_eff * self.s_nom * sig_mod

        # 背景项 (非线性修正)
        bkg_factor = np.ones(self.n_bins)
        for k in range(self.n_nuis):
            base = 1.0 + self.delta_bkg[:, k] * theta[k]
            # 物理约束: 背景修正因子必须为正
            base = np.maximum(base, 1e-8)
            bkg_factor *= np.power(base, self.kappa[:, k])
        bkg_term = self.b_nom * bkg_factor

        nu = sig_term + bkg_term
        return np.maximum(nu, 1e-12)

    # ------------------------------------------------------------------
    def renyi_entropy_floor(
        self, mu: float, theta: np.ndarray, alpha: float = 2.0
    ) -> float:
        """
        Rényi 熵单发噪声下界 (seed 1220):

            N_0(α) = (1/(1-α)) · ln[ Σ_i (ν_i / Σ ν_j)^α ]

        当 α → 1 时趋近 Shannon 熵; 用于诊断统计灵敏度退化。
        """
        nu = self.expected_rate(mu, theta)
        p = nu / np.sum(nu)
        if abs(alpha - 1.0) < 1e-10:
            # Shannon 极限
            return -np.sum(p * np.log(np.maximum(p, 1e-300)))
        log_sum = np.sum(np.power(p, alpha))
        return (1.0 / (1.0 - alpha)) * math.log(max(log_sum, 1e-300))


# ---------------------------------------------------------------------------
# 默认小规模可复现实验配置
# ---------------------------------------------------------------------------
def make_default_binmodel() -> BinModel:
    """
    5-bin H → γγ 简化模型。
    信号峰在 m ≈ 125 GeV, 背景为指数衰减分布。

    区间:
        [110, 116, 122, 128, 134] GeV
    标称信号 (μ=1):
        [2.0, 8.0, 15.0, 7.0, 1.5]
    标称背景:
        [50.0, 45.0, 40.0, 38.0, 35.0]
    系统误差:
        θ_0: 亮度不确定性 2.5% (对信号+背景均有影响)
        θ_1: JES 不确定性 (仅背景, 非线性 κ=2)
    """
    s_nom = np.array([2.0, 8.0, 15.0, 7.0, 1.5])
    b_nom = np.array([50.0, 45.0, 40.0, 38.0, 35.0])

    gamma_sig = np.array(
        [[0.025, 0.0], [0.025, 0.005], [0.025, 0.01],
         [0.025, 0.005], [0.025, 0.0]]
    )
    delta_bkg = np.array(
        [[0.025, 0.01], [0.025, 0.02], [0.025, 0.03],
         [0.025, 0.02], [0.025, 0.01]]
    )
    kappa = np.array(
        [[1.0, 2.0], [1.0, 2.0], [1.0, 2.0], [1.0, 2.0], [1.0, 2.0]]
    )
    masses = np.array([110.0, 116.0, 122.0, 128.0, 134.0])

    return BinModel(
        s_nom=s_nom,
        b_nom=b_nom,
        gamma_sig=gamma_sig,
        delta_bkg=delta_bkg,
        kappa=kappa,
        accel=0.02,  # 小幅 Unruh 修正
        masses=masses,
    )


# ---------------------------------------------------------------------------
# 生成伪观测数据 (Poisson 涨落)
# ---------------------------------------------------------------------------
def generate_observed_data(
    model: BinModel,
    mu_true: float = 0.0,
    theta_true: Optional[np.ndarray] = None,
    seed: int = 42,
) -> np.ndarray:
    """
    从模型生成伪数据 n_obs ~ Poisson(ν(μ_true, θ_true))。

    Parameters
    ----------
    model : BinModel
    mu_true : float
        真实信号强度 (0.0 → 纯背景)
    theta_true : ndarray, optional
        真实 nuisance 参数值
    seed : int
        RNG 种子 (可复现)

    Returns
    -------
    ndarray, shape (n_bins,)
        每个区间的观测事件数 (整数)
    """
    rng = np.random.default_rng(seed)
    if theta_true is None:
        theta_true = np.zeros(model.n_nuis)
    nu = model.expected_rate(mu_true, theta_true)
    return rng.poisson(lam=nu).astype(np.float64)


# ---------------------------------------------------------------------------
# 快速自检
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    mdl = make_default_binmodel()
    theta0 = np.zeros(mdl.n_nuis)
    nu0 = mdl.expected_rate(1.0, theta0)
    print(f"标称期望 (μ=1, θ=0): {nu0}")
    nu_bg = mdl.expected_rate(0.0, theta0)
    print(f"纯背景期望 (μ=0, θ=0): {nu_bg}")
    data = generate_observed_data(mdl, mu_true=0.0, seed=230)
    print(f"伪观测数据: {data}")
    S2 = mdl.renyi_entropy_floor(1.0, theta0, alpha=2.0)
    print(f"Rényi 噪声下界 (α=2): {S2:.6f}")
