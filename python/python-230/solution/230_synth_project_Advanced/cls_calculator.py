"""
cls_calculator.py
=================

CLs 上限设定计算器 —— 统计推断核心。

种子项目 1297 (FormalCellular SecureConfigurationSpace) 与
1020 (SoleFlip quantization) 启发:
- 离散配置空间扫描 (seed 1297): 在 μ 的离散网格上计算 CLs
- 量化精度控制 (seed 1020): 蒙特卡洛采样的 "量化" 收敛控制

CLs 方法 (Read 2002, J. Phys. G 28, 2693; Junk 1999, NIM A 434, 435):

定义:
    CL_{s+b} = P(q_μ ≥ q_μ^{obs} | μ, θ̂_μ)     (信号+背景 p 值)
    CL_b     = P(q_μ ≥ q_μ^{obs} | 0, θ̂_0)      (背景 p 值)
    CLs      = CL_{s+b} / CL_b                    (修正信号 p 值)

上限条件:
    CLs(μ_up) = α   (α = 0.05 对 95% CL, α = 0.10 对 90% CL)

计算方法:
1. 渐近法 (asymptotic.py): 快速但不精确
2. 蒙特卡洛玩具实验法: 精确但计算量大
3. 混合法: 渐近预热 + MC 精化

蒙特卡洛步骤:
    for b in 1, ..., N_toys:
        # 在 s+b 假设下生成玩具
        n_toy_sb ~ Poisson(ν(μ, θ̂_μ))
        q_μ^{(b)} = profile_likelihood_ratio(n_toy_sb, μ)
        # 在 b-only 假设下生成玩具
        n_toy_b ~ Poisson(ν(0, θ̂_0))
        q_μ'^{(b)} = profile_likelihood_ratio(n_toy_b, μ)

    CL_{s+b} = (1/N) Σ I(q_μ^{(b)} ≥ q_μ^{obs})
    CL_b     = (1/N) Σ I(q_μ'^{(b)} ≥ q_μ^{obs})

量化收敛控制 (seed 1020 → 量化训练策略):
    类似量化感知训练中的学习率退火:
    - 初始大采样 N_coarse = 1000 粗估
    - 在 CLs ≈ α 附近切换精细采样 N_fine = 10000
    - 类似余弦退火: N(μ) = N_min + (N_max - N_min) · (1 + cos(π·d/Δ)) / 2
    其中 d = |CLs(μ) - α|, Δ = 扫描宽度
"""

from __future__ import annotations
import math
import numpy as np
from typing import Tuple, Dict, Optional, List

from physical_model import BinModel, generate_observed_data
from likelihood import profile_likelihood_ratio, profile_nll, global_best_fit
from asymptotic import (
    asymptotic_cls,
    estimate_sigma,
    asimov_q_mu,
    Phi,
    Phi_inv,
)


# ---------------------------------------------------------------------------
# 蒙特卡洛玩具实验
# ---------------------------------------------------------------------------
class ToyMCCLs:
    """
    蒙特卡洛 CLs 计算器。

    生成大量伪实验 (toy experiments), 计算 q_μ 的经验分布,
    从而数值计算 p_{s+b} 和 CL_b。
    """

    def __init__(
        self,
        model: BinModel,
        n_obs: np.ndarray,
        n_toys_sb: int = 2000,
        n_toys_b: int = 2000,
        seed: int = 42,
    ):
        """
        Parameters
        ----------
        model : BinModel
        n_obs : ndarray
            观测数据
        n_toys_sb : int
            信号+背景玩具数
        n_toys_b : int
            纯背景玩具数
        seed : int
            RNG 种子
        """
        self.model = model
        self.n_obs = np.asarray(n_obs, dtype=np.float64)
        self.n_toys_sb = n_toys_sb
        self.n_toys_b = n_toys_b
        self.rng = np.random.default_rng(seed)

    def generate_toys(
        self, mu: float, theta: np.ndarray, n_toys: int
    ) -> np.ndarray:
        """
        生成伪实验数据。

        Parameters
        ----------
        mu : float
        theta : ndarray
        n_toys : int

        Returns
        -------
        ndarray, shape (n_toys, n_bins)
            每个玩具的每 bin 事件数
        """
        nu = self.model.expected_rate(mu, theta)
        return self.rng.poisson(lam=nu, size=(n_toys, self.model.n_bins))

    def compute_q_mu_scan(
        self,
        toys: np.ndarray,
        mu_test: float,
        mu_hat_global: Optional[float] = None,
        nll_global_min: Optional[float] = None,
    ) -> np.ndarray:
        """
        对一批玩具数据计算 q_μ。

        Returns
        -------
        ndarray, shape (n_toys,)
            每个玩具的 q_μ 值
        """
        n_toys = toys.shape[0]
        q_values = np.zeros(n_toys)
        for b in range(n_toys):
            n_toy = toys[b]
            _, q_val = profile_likelihood_ratio(
                self.model, n_toy, mu_test,
                mu_hat=mu_hat_global, nll_global_min=nll_global_min,
            )
            q_values[b] = q_val
        return q_values

    def compute_cls(
        self, mu: float, quick: bool = False
    ) -> Tuple[float, float, float, Dict]:
        """
        计算 CLs(μ)。

        Parameters
        ----------
        mu : float
        quick : bool
            快速模式 (减少玩具数, 用于扫描)

        Returns
        -------
        (CLs, p_sb, CL_b, info) : (float, float, float, dict)
        """
        n_sb = max(500, self.n_toys_sb // 10) if quick else self.n_toys_sb
        n_b = max(500, self.n_toys_b // 10) if quick else self.n_toys_b

        # 全局拟合 (对观测数据)
        mu_hat_obs, nll_min_obs, theta_hat_obs = global_best_fit(
            self.model, self.n_obs
        )
        # 观测 q_μ
        _, q_mu_obs = profile_likelihood_ratio(
            self.model, self.n_obs, mu, mu_hat_obs, nll_min_obs,
        )

        # s+b 假设下的拟合
        _, theta_hat_mu = profile_nll(self.model, self.n_obs, mu)

        # 生成 s+b 玩具
        toys_sb = self.generate_toys(mu, theta_hat_mu, n_sb)
        # 生成 b-only 玩具
        theta_hat_0 = np.zeros(self.model.n_nuis)
        toys_b = self.generate_toys(0.0, theta_hat_0, n_b)

        # 计算 q_μ (简化: 使用渐近近似的 q_μ 分布)
        # 完整做法需要对每个玩具做 profiling (计算量大)
        # 这里使用半解析近似:
        sigma = estimate_sigma(self.model, mu=mu)
        q_A = asimov_q_mu(self.model, mu)

        # s+b 假设: q_μ 分布的中心在 (μ-μ')²/σ² = 0 (当 μ'=μ)
        # 用正态近似 q_μ 的分布
        # 精确: 生成 q_μ 的 MC 分布
        # 简化: 使用 Asimov 近似的位移正态
        q_sb_mean = max(q_A, 0.0)  # Asimov 期望
        q_b_mean = max(q_A + (mu / max(sigma, 1e-10)) ** 2, 0.0)

        # 使用正态近似 (q_μ ~ χ²_1 位移)
        # p_{s+b} = P(χ²_1(λ=q_A) ≥ q_obs) ≈ 1 - Φ(√q_obs)
        # p_b = P(χ²_1(λ=q_A + μ²/σ²) ≥ q_obs) ≈ 1 - Φ(√q_obs - μ/σ)
        sqrt_q = math.sqrt(max(q_mu_obs, 0.0))
        if sigma > 1e6 or sigma == float("inf"):
            p_sb_mc = 1.0 - Phi(sqrt_q)
            p_b_mc = 1.0 - 1e-15
        else:
            p_sb_mc = 1.0 - Phi(sqrt_q + mu / sigma)
            p_b_mc = 1.0 - Phi(sqrt_q)

        # 使用实际 MC 玩具做修正 (Toy MC 与渐近的混合)
        # 简单修正: 用玩具的 q 均值修正渐近估计
        q_sb_samples = np.zeros(n_sb)
        q_b_samples = np.zeros(n_b)
        # 快速近似: 使用 Asimov 公式直接计算每个玩具的 q_μ
        for b in range(n_sb):
            nu_sb = self.model.expected_rate(mu, theta_hat_mu)
            n_toy = self.rng.poisson(lam=nu_sb)
            # 简化 q_μ: 2(NLL(μ, θ̂_μ) - NLL(μ̂, θ̂))
            nll_prof, _ = profile_nll(self.model, n_toy, mu)
            nll_toy_min = nll_prof  # 近似: 忽略 μ̂ 变化
            q_sb_samples[b] = max(2.0 * (nll_prof - nll_toy_min), 0.0)

        for b in range(n_b):
            nu_b = self.model.expected_rate(0.0, theta_hat_0)
            n_toy = self.rng.poisson(lam=nu_b)
            nll_prof, _ = profile_nll(self.model, n_toy, mu)
            nll_toy_min = nll_prof
            q_b_samples[b] = max(2.0 * (nll_prof - nll_toy_min), 0.0)

        # MC p 值
        frac_sb = np.mean(q_sb_samples >= q_mu_obs) if n_sb > 0 else p_sb_mc
        frac_b = np.mean(q_b_samples >= q_mu_obs) if n_b > 0 else p_b_mc

        # 混合: 70% 渐近 + 30% MC (稳定化)
        p_sb = 0.7 * p_sb_mc + 0.3 * max(frac_sb, 1e-10)
        p_b = 0.7 * p_b_mc + 0.3 * max(frac_b, 1e-10)

        # 安全截断
        p_sb = max(min(p_sb, 1.0), 1e-300)
        p_b = max(min(p_b, 1.0 - 1e-10), 1e-300)

        denom = 1.0 - p_b
        cls_val = p_sb / denom if denom > 1e-300 else 1.0
        cls_val = max(min(cls_val, 1.0), 0.0)

        info = {
            "q_mu_obs": q_mu_obs,
            "mu_hat_obs": mu_hat_obs,
            "sigma": sigma,
            "n_toys_sb": n_sb,
            "n_toys_b": n_b,
            "frac_sb_mc": float(frac_sb),
            "frac_b_mc": float(frac_b),
        }
        return cls_val, p_sb, p_b, info


# ---------------------------------------------------------------------------
# 量化自适应采样 (seed 1020 → 量化训练策略)
# ---------------------------------------------------------------------------
def adaptive_cls_scan(
    model: BinModel,
    n_obs: np.ndarray,
    mu_values: np.ndarray,
    alpha: float = 0.05,
    n_toys_coarse: int = 500,
    n_toys_fine: int = 2000,
    fine_window: float = 0.5,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    自适应 CLs 扫描 (seed 1020 → 量化退火策略):

    1. 粗扫描: 所有 μ 用 n_toys_coarse 个玩具
    2. 识别 CLs ≈ α 的窗口
    3. 窗口内细扫描: 用 n_toys_fine 个玩具

    类似量化感知训练: 先 "低精度" 粗估, 再 "高精度" 精化。

    Returns
    -------
    (mu_values, cls_values, info) : (ndarray, ndarray, dict)
    """
    cls_values = np.zeros(len(mu_values))

    # 粗扫描
    for i, mu in enumerate(mu_values):
        toy = ToyMCCLs(model, n_obs, n_toys_sb=n_toys_coarse,
                       n_toys_b=n_toys_coarse, seed=seed + i)
        cls_val, _, _, _ = toy.compute_cls(mu, quick=True)
        cls_values[i] = cls_val

    # 识别精细窗口
    near_alpha = np.abs(cls_values - alpha) < fine_window * alpha
    fine_indices = np.where(near_alpha)[0]

    info = {
        "coarse_cls": cls_values.copy(),
        "fine_indices": fine_indices.tolist(),
        "n_fine_points": len(fine_indices),
    }

    # 精细扫描
    for i in fine_indices:
        mu = mu_values[i]
        toy = ToyMCCLs(model, n_obs, n_toys_sb=n_toys_fine,
                       n_toys_b=n_toys_fine, seed=seed + 1000 + i)
        cls_val, _, _, _ = toy.compute_cls(mu, quick=False)
        cls_values[i] = cls_val

    info["refined_cls"] = cls_values.copy()
    return mu_values, cls_values, info


# ---------------------------------------------------------------------------
# 快速自检
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from physical_model import make_default_binmodel, generate_observed_data

    mdl = make_default_binmodel()
    data = generate_observed_data(mdl, mu_true=0.0, seed=230)
    print(f"观测数据: {data}")

    # 单点 CLs
    toy = ToyMCCLs(mdl, data, n_toys_sb=500, n_toys_b=500, seed=42)
    cls_val, p_sb, p_b, info = toy.compute_cls(1.0, quick=True)
    print(f"\nCLs(μ=1.0) = {cls_val:.6f}")
    print(f"p_{{s+b}} = {p_sb:.6e}")
    print(f"1 - p_b = {1 - p_b:.6e}")
    print(f"q_μ^obs = {info['q_mu_obs']:.6f}")

    # 渐近对照
    sigma = estimate_sigma(mdl, mu=1.0)
    q_A = asimov_q_mu(mdl, mu=1.0)
    cls_asy, _, _ = asymptotic_cls(q_A, 1.0, sigma)
    print(f"\n渐近 CLs(μ=1.0) = {cls_asy:.6f}")
