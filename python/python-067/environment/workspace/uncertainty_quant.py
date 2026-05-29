# -*- coding: utf-8 -*-
"""
uncertainty_quant.py
裂隙介质渗流不确定性量化模块

融合种子项目：
    - 082_beta_nc: 非中心 Beta 分布 CDF 计算
    - 053_asa266: Gamma、Digamma、Trigamma 函数及相关统计分布

在水文地质参数估计中，不确定性来源包括：
    1. 测量误差（水头、浓度、流量）
    2. 地质异质性（裂隙几何、开度分布）
    3. 模型结构不确定性（等效连续介质 vs 离散裂隙网络）

核心统计模型：

    非中心 Beta 分布（渗透率置信区间）：
        设渗透率 K 的先验为 Beta(α, β, λ)，其中 λ 为非中心参数
        
        PDF:
            f(x; α, β, λ) = Σ_{j=0}^∞ exp(-λ/2) (λ/2)^j / j! 
                            * x^{α+j-1} (1-x)^{β-1} / B(α+j, β)
        
        CDF 使用级数展开计算：
            F(x) = Σ_{j=0}^∞ p_j I_x(α+j, β)
        
        其中 p_j = exp(-λ/2) (λ/2)^j / j!
              I_x 为不完全 Beta 函数比

    置信区间传播：
        若 C = f(K)，则 C 的不确定性通过 Delta 方法传播：
            σ_C² ≈ (∂f/∂K)² σ_K²

    Gamma 分布（滞留时间）：
        f(t; α, β) = β^α t^{α-1} exp(-β t) / Γ(α)
        
        矩：E[t] = α/β, Var[t] = α/β²
"""

import numpy as np
from typing import Tuple, List
from scipy.special import betainc, gammaln, gamma, digamma, polygamma


class UncertaintyQuantification:
    """
    裂隙介质渗流参数不确定性量化器

    提供置信区间估计、敏感性分析和参数不确定性传播功能。
    """

    def __init__(self):
        pass

    @staticmethod
    def alogam(x: float) -> float:
        """
        计算 ln Γ(x)，使用 Lanczos 近似

        基于 asa266 的 alogam 函数
        """
        if x <= 0:
            raise ValueError("x 必须为正")
        return float(gammaln(x))

    @staticmethod
    def noncentral_beta_cdf(x: float, a: float, b: float,
                            lam: float, error_max: float = 1.0e-10) -> float:
        """
        非中心 Beta 分布 CDF

        基于 beta_noncentral_cdf 的级数展开算法：
            F(x) = Σ_{i=0}^∞ p_i * I_x(a+i, b)
            
            其中 p_i = exp(-λ/2) (λ/2)^i / i!
                  I_x 为不完全 Beta 函数比

        递推关系：
            p_{i+1} = (λ/2) * p_i / (i+1)
            I_{i+1} = I_i - B_i
            B_{i+1} = x (a+b+i) B_i / (a+i+1)

        Parameters
        ----------
        x : float
            自变量，0 ≤ x ≤ 1
        a, b : float
            形状参数
        lam : float
            非中心参数
        error_max : float
            截断误差控制

        Returns
        -------
        float
            CDF 值
        """
        if not (0.0 <= x <= 1.0):
            raise ValueError("x 必须在 [0, 1] 范围内")
        if a <= 0 or b <= 0:
            raise ValueError("a 和 b 必须为正")
        if lam < 0:
            raise ValueError("lam 必须为非负")

        if x == 0.0:
            return 0.0
        if x == 1.0:
            return 1.0

        # 初始项
        i = 0
        pi_val = np.exp(-lam / 2.0)

        beta_log = gammaln(a) + gammaln(b) - gammaln(a + b)
        bi = betainc(a, b, x)

        # 递推初始项
        si = np.exp(a * np.log(x) + b * np.log(1.0 - x) - beta_log - np.log(a))

        p_sum = pi_val
        pb_sum = pi_val * bi

        max_iter = 10000
        for _ in range(max_iter):
            if p_sum >= 1.0 - error_max:
                break

            i += 1
            pi_val = 0.5 * lam * pi_val / i
            bi = bi - si
            si = x * (a + b + i - 1.0) * si / (a + i)

            p_sum += pi_val
            pb_sum += pi_val * bi

            if pi_val < 1e-20:
                break

        return min(max(pb_sum, 0.0), 1.0)

    @staticmethod
    def gamma_sample_stats(alpha: float, beta_param: float) -> dict:
        """
        Gamma 分布统计量

        基于 asa266 的 gamma_sample 相关统计：
            PDF: f(x) = β^α x^{α-1} exp(-βx) / Γ(α)
            
            E[X] = α/β
            Var[X] = α/β²
            Mode = (α-1)/β  (当 α > 1)

        Parameters
        ----------
        alpha : float
            形状参数
        beta_param : float
            速率参数

        Returns
        -------
        dict
            统计量
        """
        if alpha <= 0 or beta_param <= 0:
            raise ValueError("参数必须为正")

        mean = alpha / beta_param
        variance = alpha / (beta_param ** 2)
        std = np.sqrt(variance)
        mode = (alpha - 1.0) / beta_param if alpha > 1.0 else 0.0

        return {
            'mean': mean,
            'variance': variance,
            'std': std,
            'mode': mode,
            'skewness': 2.0 / np.sqrt(alpha),
            'kurtosis': 6.0 / alpha
        }

    @staticmethod
    def monte_carlo_uncertainty(forward_model: callable,
                                 param_distributions: List[dict],
                                 n_samples: int = 1000,
                                 seed: int = 42) -> dict:
        """
        蒙特卡洛不确定性传播

        Parameters
        ----------
        forward_model : callable
            前向模型函数 f(params) -> result
        param_distributions : list
            参数分布列表，每个元素为 {'name': str, 'dist': str, 'params': dict}
        n_samples : int
            蒙特卡洛样本数
        seed : int
            随机种子

        Returns
        -------
        dict
            统计结果
        """
        rng = np.random.default_rng(seed)
        results = []

        for _ in range(n_samples):
            sample_params = {}
            for pd in param_distributions:
                name = pd['name']
                dist = pd['dist']
                p = pd['params']

                if dist == 'uniform':
                    sample_params[name] = rng.uniform(p['low'], p['high'])
                elif dist == 'normal':
                    sample_params[name] = rng.normal(p['mean'], p['std'])
                elif dist == 'lognormal':
                    sample_params[name] = rng.lognormal(p['mu'], p['sigma'])
                elif dist == 'gamma':
                    sample_params[name] = rng.gamma(p['alpha'], 1.0/p['beta'])
                else:
                    raise ValueError(f"不支持的分布: {dist}")

            try:
                result = forward_model(sample_params)
                results.append(result)
            except Exception:
                continue

        results = np.array(results)
        if len(results) == 0:
            return {'mean': 0.0, 'std': 0.0, 'ci_95': (0.0, 0.0)}

        return {
            'mean': float(np.mean(results)),
            'std': float(np.std(results)),
            'median': float(np.median(results)),
            'ci_95': (float(np.percentile(results, 2.5)),
                      float(np.percentile(results, 97.5))),
            'min': float(np.min(results)),
            'max': float(np.max(results)),
            'n_samples': len(results)
        }

    @staticmethod
    def permeability_confidence_interval(K_estimate: float,
                                          K_std: float,
                                          confidence: float = 0.95) -> Tuple[float, float]:
        """
        计算渗透率估计的置信区间

        假设渗透率服从对数正态分布：
            ln K ~ N(μ, σ²)
            
            其中 μ = ln K_est - σ²/2,  σ = ln(1 + (K_std/K_est)²)^{1/2}

        Parameters
        ----------
        K_estimate : float
            渗透率估计值 [m²]
        K_std : float
            渗透率标准差 [m²]
        confidence : float
            置信水平

        Returns
        -------
        tuple
            (下限, 上限)
        """
        if K_estimate <= 0 or K_std < 0:
            raise ValueError("K_estimate 必须为正，K_std 必须为非负")

        from scipy.stats import norm

        # 对数正态参数
        cv = K_std / K_estimate if K_estimate > 0 else 0.0
        sigma_ln = np.sqrt(np.log(1.0 + cv ** 2))
        mu_ln = np.log(K_estimate) - 0.5 * sigma_ln ** 2

        alpha = 1.0 - confidence
        z_low = norm.ppf(alpha / 2.0)
        z_high = norm.ppf(1.0 - alpha / 2.0)

        K_low = np.exp(mu_ln + z_low * sigma_ln)
        K_high = np.exp(mu_ln + z_high * sigma_ln)

        return K_low, K_high

    @staticmethod
    def sensitivity_analysis(forward_model: callable,
                              base_params: dict,
                              perturbation: float = 0.01) -> dict:
        """
        参数敏感性分析（有限差分法）

        计算各参数的敏感性系数：
            S_i = (∂f/∂p_i) * (p_i / f)

        Parameters
        ----------
        forward_model : callable
            前向模型
        base_params : dict
            基准参数
        perturbation : float
            扰动比例

        Returns
        -------
        dict
            敏感性系数
        """
        f_base = forward_model(base_params)
        if abs(f_base) < 1e-20:
            f_base = 1e-20

        sensitivities = {}
        for name, value in base_params.items():
            if abs(value) < 1e-20:
                continue

            dp = value * perturbation
            params_plus = base_params.copy()
            params_plus[name] = value + dp

            f_plus = forward_model(params_plus)
            df_dp = (f_plus - f_base) / dp

            # 无量纲敏感性系数
            S = df_dp * value / f_base
            sensitivities[name] = float(S)

        return sensitivities

    @staticmethod
    def first_order_reliability(g_func: callable,
                                 mean_params: np.ndarray,
                                 cov_matrix: np.ndarray,
                                 tol: float = 1.0e-6,
                                 max_iter: int = 100) -> dict:
        """
        一阶可靠性方法 (FORM)

        计算失效概率：
            P_f ≈ Φ(-β)
        
        其中 β 为可靠性指标，Φ 为标准正态 CDF。

        Parameters
        ----------
        g_func : callable
            极限状态函数 g(x) < 0 表示失效
        mean_params : np.ndarray
            参数均值向量
        cov_matrix : np.ndarray
            协方差矩阵
        tol : float
            收敛容差
        max_iter : int
            最大迭代次数

        Returns
        -------
        dict
            可靠性分析结果
        """
        from scipy.stats import norm

        n = len(mean_params)
        u = np.zeros(n)  # 标准正态空间

        # Cholesky 分解
        try:
            L = np.linalg.cholesky(cov_matrix)
        except np.linalg.LinAlgError:
            # 添加小正则化
            L = np.linalg.cholesky(cov_matrix + 1e-10 * np.eye(n))

        for _ in range(max_iter):
            x = mean_params + L @ u
            g_val = g_func(x)

            # 数值梯度
            grad = np.zeros(n)
            h = 1e-6
            for i in range(n):
                x_pert = x.copy()
                x_pert[i] += h
                grad[i] = (g_func(x_pert) - g_val) / h

            # 转换到标准正态空间的梯度
            grad_u = L.T @ grad
            grad_norm = np.linalg.norm(grad_u)
            if grad_norm < 1e-20:
                break

            # 更新设计点
            u_new = (grad_u @ u - g_val) / grad_norm ** 2 * grad_u

            if np.linalg.norm(u_new - u) < tol:
                u = u_new
                break
            u = u_new

        beta = np.linalg.norm(u)
        pf = norm.cdf(-beta)

        return {
            'reliability_index': float(beta),
            'failure_probability': float(pf),
            'design_point': mean_params + L @ u,
            'converged': True
        }
