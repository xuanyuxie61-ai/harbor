"""
parameter_calibration.py  —  宇宙学模拟参数自动标定 (CMA-ES)
=========================================================

科学来源种子:
  - 1201_de-ranit_GPP-ModelParamVariation / main_opti_and_run_model.py,
    get_best_param_from_cmaes_history.py, optimize_lbgfs/forward_run_lbfgs_opti.py
    直接使用其 CMA-ES (Covariance Matrix Adaptation Evolution Strategy)
    优化框架,将 GPP 模型参数标定替换为 N 体模拟参数标定。
    核心借鉴:
      - 多站点/多时刻参数组分离 (add_keys_for_group)
      - 从 CMA-ES 历史中提取最优参数
      - LBFGS 精调 (hybrid CMA-ES → L-BFGS)

物理背景:
  N 体模拟中,两个关键自由参数影响结果精度:
    1. 引力软化长度 ε: 平衡力分辨率与数值噪声
    2. 时间步长 Δt: 平衡精度与计算成本
  代价函数: 以解析 halo 质量函数 (Sheth-Tormen) 或功率谱为基准,
        C(ε, Δt) = || P_num(k) - P_ref(k) ||² / σ_P²
                + λ · N_steps(Δt) / N_max
  约束: ε ∈ [ε_min, ε_max], Δt ∈ [Δt_min, Δt_max]
  优化目标: min C(ε, Δt)
"""

from __future__ import annotations
import numpy as np
from numpy.typing import NDArray
from typing import Tuple, List, Dict, Callable, Any


# ---------------------------------------------------------------------------- #
#              CMA-ES 核心 (简化版,源自 GPP 项目的 cma 调用)
# ---------------------------------------------------------------------------- #
class CMAESOptimizer:
    """
    简化版 CMA-ES (Hansen & Ostermeier 2001):
        m_{g+1} = Σ_i w_i x_{i:λ}
        C_{g+1} = (1-c_1-c_μ) C_g + c_1 p_c p_c^T + ...
        σ_{g+1} = σ_g · exp( (||p_σ||/E||N|| - 1) · damping )
    """
    def __init__(self, x0: NDArray, sigma0: float,
                 bounds: Tuple[NDArray, NDArray] = None,
                 pop_size: int = None, seed: int = 42):
        self.x0 = np.asarray(x0, dtype=float)
        self.n = self.x0.size
        self.sigma = float(sigma0)
        self.bounds = bounds
        self.rng = np.random.default_rng(seed)
        # 种群大小 (默认: 4 + floor(3 log n)):
        self.lam = pop_size or (4 + int(3 * np.log(max(1, self.n))))
        # 权重:
        mu = self.lam // 2
        self.mu = mu
        w_raw = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        self.weights = w_raw / w_raw.sum()
        self.mueff = 1.0 / np.sum(self.weights ** 2)
        # 学习率:
        n = self.n
        self.cc = (4.0 + self.mueff / n) / (n + 4.0 + 2.0 * self.mueff / n)
        self.cs = (self.mueff + 2.0) / (n + self.mueff + 5.0)
        self.c1 = 2.0 / ((n + 1.3) ** 2 + self.mueff)
        self.cmu = min(1.0 - self.c1,
                       2.0 * (self.mueff - 2.0 + 1.0 / self.mueff)
                       / ((n + 2.0) ** 2 + self.mueff))
        self.damps = 1.0 + 2.0 * max(0.0, np.sqrt((self.mueff - 1.0) /
                                                    (n + 1.0)) - 1.0) + self.cs
        # 状态:
        self.mean = self.x0.copy()
        self.C = np.eye(n)
        self.pc = np.zeros(n)
        self.ps = np.zeros(n)
        self.chiN = np.sqrt(n) * (1.0 - 1.0 / (4.0 * n) + 1.0 / (21.0 * n * n))
        self.eigenevery = 10
        self.generation = 0
        self.best_x = self.x0.copy()
        self.best_f = np.inf
        self.history: List[Dict[str, Any]] = []

    def _decompose_C(self) -> Tuple[NDArray, NDArray]:
        eigvals, eigvecs = np.linalg.eigh(self.C)
        eigvals = np.clip(eigvals, 1e-20, None)
        invsqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
        sqrtC = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
        return sqrtC, invsqrt

    def sample_population(self) -> NDArray:
        sqrtC, _ = self._decompose_C()
        pop = np.zeros((self.lam, self.n))
        for i in range(self.lam):
            z = self.rng.standard_normal(self.n)
            x = self.mean + self.sigma * (sqrtC @ z)
            # 边界处理:
            if self.bounds is not None:
                low, high = self.bounds
                x = np.clip(x, low, high)
            pop[i] = x
        return pop

    def update(self, population: NDArray, fitness: NDArray) -> None:
        """基于适应度更新均值、协方差、步长。"""
        order = np.argsort(fitness)
        selected = population[order[: self.mu]]
        # 更新均值:
        mean_old = self.mean.copy()
        self.mean = np.sum(self.weights[:, None] * selected, axis=0)
        # 演化路径 (cumulation):
        _, invsqrt = self._decompose_C()
        self.ps = ((1.0 - self.cs) * self.ps
                   + np.sqrt(self.cs * (2.0 - self.cs) * self.mueff)
                   * (invsqrt @ (self.mean - mean_old)) / self.sigma)
        hsig = (np.linalg.norm(self.ps)
                / np.sqrt(1.0 - (1.0 - self.cs) ** (2 * (self.generation + 1)))
                / self.chiN
                < 1.4 + 2.0 / (self.n + 1.0))
        self.pc = ((1.0 - self.cc) * self.pc
                   + hsig * np.sqrt(self.cc * (2.0 - self.cc) * self.mueff)
                   * (self.mean - mean_old) / self.sigma)
        # 协方差更新:
        delta = (selected - mean_old) / self.sigma
        rank_mu = np.zeros_like(self.C)
        for i in range(self.mu):
            rank_mu += self.weights[i] * np.outer(delta[i], delta[i])
        rank_one = np.outer(self.pc, self.pc)
        self.C = ((1.0 - self.c1 - self.cmu) * self.C
                  + self.c1 * (rank_one + (1.0 - hsig) * self.cc * (2.0 - self.cc) * self.C)
                  + self.cmu * rank_mu)
        # 步长更新:
        self.sigma *= np.exp((self.cs / self.damps)
                              * (np.linalg.norm(self.ps) / self.chiN - 1.0))
        self.generation += 1
        if fitness[order[0]] < self.best_f:
            self.best_f = fitness[order[0]]
            self.best_x = population[order[0]].copy()
        self.history.append({
            "generation": self.generation,
            "best_f": float(self.best_f),
            "best_x": self.best_x.tolist(),
            "sigma": float(self.sigma),
        })


# ---------------------------------------------------------------------------- #
#                     代价函数 (模拟 benchmark)
# ---------------------------------------------------------------------------- #
def sheth_tormen_mass_function(M: NDArray, z: float,
                                omega_m: float = 0.308,
                                omega_l: float = 0.692) -> NDArray:
    """
    Sheth-Tormen halo 质量函数 (Sheth et al. 2001):
        dn/dM = (ρ̄/M) · (d ln σ^{-1}/dM) · f_ST(ν)
    其中 ν = δ_c / σ(M, z),
        f_ST(ν) = A √(2a/π) [1 + (a ν²)^{-p}] ν exp(-a ν²/2)
        A = 0.3222, a = 0.707, p = 0.3
    """
    A, a, p = 0.3222, 0.707, 0.3
    delta_c = 1.686
    # 简化 σ(M) 关系: σ(M) = σ_8 · (M/M_*)^{-1/6}
    M_star = 1e13  # M_sun/h
    sigma = 0.8159 * (M / M_star) ** (-1.0 / 6.0)
    sigma *= _growth_factor_st(z, omega_m, omega_l)
    nu = delta_c / np.maximum(sigma, 1e-30)
    f_ST = (A * np.sqrt(2.0 * a / np.pi)
            * (1.0 + (a * nu ** 2) ** (-p))
            * nu * np.exp(-a * nu ** 2 / 2.0))
    # d ln σ^{-1} / dM:
    dlnsiginv_dM = (1.0 / 6.0) / M
    rho_bar = 2.775e11 * omega_m  # ρ_crit h² Ω_m
    return (rho_bar / M) * dlnsiginv_dM * f_ST


def _growth_factor_st(z: float, omega_m: float, omega_l: float) -> float:
    a = 1.0 / (1.0 + z)
    x = (omega_l / omega_m) ** (1.0 / 3.0) * a
    D = a / (1.0 + x ** 1.5 + (1.0 + omega_l / omega_m) ** 0.2 * a)
    x1 = (omega_l / omega_m) ** (1.0 / 3.0)
    D1_now = 1.0 / (1.0 + x1 ** 1.5 + (1.0 + omega_l / omega_m) ** 0.2)
    return D / D1_now


def calibration_cost_function(params: NDArray, benchmark_M: NDArray,
                               benchmark_dn: NDArray) -> float:
    """
    参数标定代价函数:
        C(ε, Δt) = χ² + λ_reg · (ε / 0.1 + Δt / 0.01)
    其中 χ² = Σ_k (n_model(M_k) - n_benchmark(M_k))² / n_benchmark(M_k)
    """
    eps, dt = float(params[0]), float(params[1])
    # 简化: 模拟的 halo 质量函数受 ε 和 Δt 影响
    # 用解析形式 + 数值误差模型:
    M = benchmark_M
    dn_true = benchmark_dn
    # 数值误差模型:  ε → force smoothing bias, dt → time integration error
    bias_eps = np.exp(- (M / 1e12) ** (1.0 / 3.0) * eps / 0.1)
    err_dt = dt ** 2 * (M / 1e12) ** (1.0 / 6.0)
    dn_model = dn_true * bias_eps * (1.0 - err_dt)
    # χ²:
    chi2 = float(np.sum(((dn_model - dn_true) ** 2)
                         / np.maximum(dn_true, 1e-30)))
    # 正则化 (偏好小参数):
    reg = 0.01 * (eps / 0.1 + dt / 0.01)
    return chi2 + reg


# ---------------------------------------------------------------------------- #
#                      运行 CMA-ES 优化
# ---------------------------------------------------------------------------- #
def calibrate_parameters(x0: NDArray = None, sigma0: float = 0.1,
                         bounds: Tuple[NDArray, NDArray] = None,
                         max_gen: int = 30, seed: int = 42
                         ) -> Dict[str, Any]:
    """
    运行 CMA-ES 寻找最优 (ε, Δt)。

    Returns
    -------
    dict:
        'best_params' : (2,) array
        'best_cost'   : float
        'history'     : list of dict
        'n_evals'     : int
    """
    if x0 is None:
        x0 = np.array([0.05, 0.01])
    if bounds is None:
        low = np.array([0.01, 1e-3])
        high = np.array([0.5, 1.0])
        bounds = (low, high)
    # 基准质量函数:
    M = np.logspace(10, 15, 32)
    dn_bench = sheth_tormen_mass_function(M, z=0.0)
    # 目标函数:
    def f(x):
        return calibration_cost_function(x, M, dn_bench)
    opt = CMAESOptimizer(x0, sigma0, bounds=bounds, seed=seed)
    total_evals = 0
    for gen in range(max_gen):
        pop = opt.sample_population()
        fitness = np.array([f(x) for x in pop])
        total_evals += len(pop)
        opt.update(pop, fitness)
    return {
        "best_params": opt.best_x,
        "best_cost": opt.best_f,
        "history": opt.history,
        "n_evals": total_evals,
    }


# ---------------------------------------------------------------------------- #
#              LBFGS 精调 (源自 optimize_lbgfs 子目录)
# ---------------------------------------------------------------------------- #
def lbfgs_refine(x0: NDArray, benchmark_M: NDArray,
                  benchmark_dn: NDArray, max_iter: int = 50,
                  bounds: Tuple[NDArray, NDArray] = None
                  ) -> Dict[str, Any]:
    """
    L-BFGS-B 精调 CMA-ES 结果。
    """
    from scipy.optimize import minimize
    def f(x):
        return calibration_cost_function(x, benchmark_M, benchmark_dn)
    bnds = list(zip(bounds[0], bounds[1])) if bounds is not None else None
    res = minimize(f, x0, method="L-BFGS-B", bounds=bnds,
                   options={"maxiter": max_iter, "ftol": 1e-10})
    return {
        "best_params": res.x,
        "best_cost": float(res.fun),
        "success": bool(res.success),
        "n_iter": int(res.nit),
    }


# ---------------------------------------------------------------------------- #
#              从历史提取最佳 (源自 get_best_param_from_cmaes_history.py)
# ---------------------------------------------------------------------------- #
def extract_best_from_history(history: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    从 CMA-ES 历史记录中提取最佳参数 (复现 get_best_param_from_cmaes_history.py)。
    """
    if not history:
        return {"best_params": None, "best_cost": np.inf}
    best = min(history, key=lambda h: h["best_f"])
    return {
        "best_params": np.array(best["best_x"]),
        "best_cost": best["best_f"],
        "generation": best["generation"],
    }
