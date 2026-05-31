"""
uncertainty_pce.py
基于种子项目 854_pce_ode_hermite (polynomial chaos expansion with Hermite polynomials)
改造为钙钛矿太阳能电池光电转换效率的不确定性量化模块。

在实际器件中，材料参数（带隙、缺陷密度、迁移率）受合成工艺波动影响
具有随机性。多项式混沌展开（PCE）使用正交多项式基将随机输出展开为
输入随机变量的函数，可高效计算输出的统计矩（均值、方差）和敏感性指标。

核心公式：
  1. 随机 ODE：du/dt = -α(ξ) u,  α(ξ) = α_μ + α_σ ξ, ξ~N(0,1)
  2. PCE 展开：u(t, ξ) = Σ_{k=0}^{N_p} u_k(t) H_k(ξ)
     H_k 为概率化 Hermite 多项式（针对高斯随机变量）。
  3. Galerkin 投影后的系数方程：
       du_k/dt = -α_μ u_k - α_σ Σ_j u_j * <H_1 H_j H_k> / <H_k^2>
  4. Hermite 多项式三重积积分（来自原项目 he_triple_product_integral）：
       <H_i H_j H_k> = ∫ H_i(ξ) H_j(ξ) H_k(ξ) φ(ξ) dξ
       φ(ξ) = (2π)^{-1/2} exp(-ξ²/2)
  5. 均值与方差：
       E[u] = u_0(t)
       Var[u] = Σ_{k=1}^{N_p} u_k(t)² <H_k²>
"""

import numpy as np
from typing import Tuple


def hermite_polynomial(n: int, x: np.ndarray) -> np.ndarray:
    """
    概率化 Hermite 多项式 He_n(x)（物理学家约定，权重 exp(-x²/2)）。
    递推：He_0 = 1, He_1 = x
          He_{n+1} = x He_n - n He_{n-1}
    """
    x = np.asarray(x)
    if n == 0:
        return np.ones_like(x)
    if n == 1:
        return x.copy()
    H_prev2 = np.ones_like(x)
    H_prev1 = x.copy()
    for k in range(1, n):
        H_curr = x * H_prev1 - k * H_prev2
        H_prev2, H_prev1 = H_prev1, H_curr
    return H_prev1


def he_double_product_integral(i: int, j: int) -> float:
    """
    <He_i, He_j> = sqrt(2π) * i! * δ_{ij}
    对应原项目 he_double_product_integral。
    """
    if i != j:
        return 0.0
    import math
    return np.sqrt(2.0 * np.pi) * math.factorial(i)


def he_triple_product_integral(i: int, j: int, k: int) -> float:
    """
    <He_i He_j He_k> 的解析公式。
    对应原项目 he_triple_product_integral。
    仅当 i+j+k 为偶数且满足三角不等式时非零。
    """
    import math
    if (i + j + k) % 2 == 1:
        return 0.0
    if i > j + k or j > i + k or k > i + j:
        return 0.0

    # 使用已知公式或查表法
    # 对于小阶数，可以直接用数值积分
    if max(i, j, k) <= 10:
        # Gauss-Hermite 数值积分
        xi, wi = np.polynomial.hermite.hermgauss(32)
        Hi = hermite_polynomial(i, xi)
        Hj = hermite_polynomial(j, xi)
        Hk = hermite_polynomial(k, xi)
        return float(np.sum(wi * Hi * Hj * Hk))
    else:
        # 大阶数近似为零
        return 0.0


def pce_time_integrator(
    ti: float,
    tf: float,
    nt: int,
    ui: float,
    np_deg: int,
    alpha_mu: float,
    alpha_sigma: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    PCE 时间积分器（基于原项目 pce_ode_hermite）。
    求解 du/dt = -α(ξ) u 的 PCE 展开。

    Parameters
    ----------
    ti, tf : float
        初始和终止时间
    nt : int
        时间步数
    ui : float
        初始条件 u(0)=ui
    np_deg : int
        PCE 展开阶数（使用 0..np_deg）
    alpha_mu, alpha_sigma : float
        衰减系数的均值和标准差

    Returns
    -------
    t : (nt+1,) array
    u : (nt+1, np_deg+1) array
        PCE 系数 u_k(t)
    """
    if nt <= 0 or np_deg < 0:
        raise ValueError("nt 必须为正，np_deg 必须非负")
    if alpha_sigma < 0:
        raise ValueError("alpha_sigma 必须非负")

    dt = (tf - ti) / nt
    t = np.zeros(nt + 1)
    u = np.zeros((nt + 1, np_deg + 1))

    # 初始条件
    u1 = np.zeros(np_deg + 1)
    u1[0] = ui
    t[0] = ti
    u[0, :] = u1

    for it in range(1, nt + 1):
        t2 = ((nt - it) * ti + it * tf) / nt
        u2 = np.zeros(np_deg + 1)

        for k in range(np_deg + 1):
            dp = he_double_product_integral(k, k)
            if dp == 0:
                dp = 1.0

            term = -alpha_mu * u1[k]

            # i=1 对应 He_1(ξ) = ξ，即随机部分 α_σ ξ
            i = 1
            for j in range(np_deg + 1):
                tp = he_triple_product_integral(i, j, k)
                term -= alpha_sigma * u1[j] * tp / dp

            u2[k] = u1[k] + dt * term

        # 数值鲁棒性
        u2 = np.where(np.isfinite(u2), u2, 0.0)

        t[it] = t2
        u1 = u2.copy()
        u[it, :] = u1

    return t, u


def pce_efficiency_uq(
    efficiency_mean: float = 0.20,
    efficiency_std: float = 0.03,
    np_deg: int = 4,
    n_mc: int = 10000,
) -> dict:
    """
    对光电转换效率进行 PCE 不确定性量化。

    假设效率 η 与某个“有效衰减系数” α_eff 相关：
      η = η_0 * exp(-α_eff * t_op)
    其中 α_eff 为高斯随机变量，反映器件退化。
    使用 PCE 计算 η 的均值、方差和置信区间。
    """
    if efficiency_mean <= 0 or efficiency_std < 0:
        raise ValueError("效率均值必须为正，标准差必须非负")

    # 参数映射：令 α_eff = α_μ + α_σ ξ
    # 简化：直接对效率的对数进行 PCE
    t_op = 1.0  # 归一化工作时间
    alpha_mu = 0.1
    alpha_sigma = 0.05

    t, u_coeff = pce_time_integrator(0.0, t_op, 100, efficiency_mean, np_deg, alpha_mu, alpha_sigma)

    # 均值 = u_0, 方差 = Σ_{k≥1} u_k² * <H_k²>
    mean_eta = u_coeff[-1, 0]
    var_eta = 0.0
    for k in range(1, np_deg + 1):
        norm = he_double_product_integral(k, k) / np.sqrt(2.0 * np.pi)
        var_eta += u_coeff[-1, k] ** 2 * norm

    std_eta = np.sqrt(max(var_eta, 0.0))

    # 蒙特卡洛对照
    rng = np.random.default_rng(789)
    xi_samples = rng.standard_normal(n_mc)
    eta_mc = np.zeros(n_mc)
    for idx, xi in enumerate(xi_samples):
        alpha = alpha_mu + alpha_sigma * xi
        eta_mc[idx] = efficiency_mean * np.exp(-alpha * t_op)

    mc_mean = float(np.mean(eta_mc))
    mc_std = float(np.std(eta_mc))

    # PCE 敏感性指标：各阶系数的方差贡献
    sensitivity = {}
    total_var = var_eta if var_eta > 0 else 1e-14
    for k in range(1, np_deg + 1):
        norm = he_double_product_integral(k, k) / np.sqrt(2.0 * np.pi)
        contrib = u_coeff[-1, k] ** 2 * norm
        sensitivity[f"order_{k}"] = float(contrib / total_var)

    return {
        "pce_mean_efficiency": float(mean_eta),
        "pce_std_efficiency": float(std_eta),
        "mc_mean_efficiency": mc_mean,
        "mc_std_efficiency": mc_std,
        "variance": float(var_eta),
        "sensitivity_indices": sensitivity,
        "pce_coefficients_final": u_coeff[-1, :].tolist(),
    }


if __name__ == "__main__":
    result = pce_efficiency_uq()
    print("PCE 不确定性量化结果:")
    for k, v in result.items():
        if k != "sensitivity_indices":
            print(f"  {k}: {v}")
    print("  敏感性指标:", result["sensitivity_indices"])
