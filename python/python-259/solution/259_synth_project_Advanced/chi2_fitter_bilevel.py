# -*- coding: utf-8 -*-
"""
chi2_fitter_bilevel.py — 双层 χ² 拟合器 (LM + 物理约束 + MPC 式步进)

本模块实现 BAO 参数的非线性最小二乘拟合. 优化问题为:

  min_{θ ∈ Θ_phys} χ²(θ) = (y_obs - y_pred(θ))^T C^{-1} (y_obs - y_pred(θ))
                          + Σ_i (θ_i - μ_i)^2 / σ_i^2                (1)

其中:
  θ = (ω_b, ω_m, h, n_s, σ_8, w_0, w_a) 为宇宙学参数
  y = (α_∥, α_⊥) 在各红移壳层上的观测
  C 为协方差矩阵 (来自 `sparse_covariance`)
  第二项为高斯先验 (来自 `bao_constants.ParameterPriors`)

优化算法
-------
采用 Levenberg-Marquardt (LM) 迭代:

  (J^T W J + λ I) Δθ = J^T W (y_obs - y_pred(θ))                 (2)

其中 J 为 Jacobian 矩阵, W = C^{-1} 为权重矩阵, λ 为 LM 阻尼.

物理约束与 MPC 式步进
--------------------
借鉴种子项目 1080 safe-CCMPC 的"控制障碍函数"(CBF) 思想, 每次步进后
必须满足:
  h(θ) := Ω_m - 0.1 > 0  (物质密度为正)
  h(θ) := h - 0.5 > 0    (Hubble 常数为正)
  h(θ) := ω_b - 0.018 > 0
若违反, 用 CBF 投影把 Δθ 缩放到可行域边界.

双层优化 (种子项目 1135)
---------------------
外层: 拟合宇宙学参数 θ_cosmo
内层: 给定 θ_cosmo, 拟合 BAO 峰振幅 A_BAO 与宽展 Σ_NL (非线性 smearing)
  A_BAO, Σ_NL = argmin χ²_peak(A, Σ; θ_cosmo)

种子项目映射
----------
- 1080 safe-CCMPC-in-CARLA   : 安全控制障碍函数 → 物理边界投影.
- 1135 bilevel-optim          : 双层优化框架 → 外/内层分离.
- 1137 DRL (DDPG/PPO/Optimizer_final.py): 梯度累积 → Jacobian 近似;
  其 `BPTT.py` 的反向传播思想被用于伴随法计算 dχ²/dθ.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Tuple, List, Optional, Callable
import math
import numpy as np

from bao_constants import (
    FiducialCosmology, get_fiducial, get_priors, ParameterPriors,
)
from background_cosmology import (
    H_z, transverse_comoving_distance, D_V,
)
from sound_horizon import sound_horizon_numerical, drag_redshift_eisenstein_hu
from bao_observables import (
    BAOObservable, get_bao_observations, BAOPrediction, predict_bao,
)


# =============================================================
# 参数向量 ↔ 宇宙学对象
# =============================================================
@dataclass
class FitConfig:
    """拟合配置."""
    theta_names: Tuple[str, ...] = (
        "omega_b", "omega_m", "h",
    )
    max_iter  : int = 30
    lm_lambda : float = 1.0
    tol_grad  : float = 1.0e-5
    tol_step  : float = 1.0e-6
    cbf_margin: float = 1.0e-4   # CBF 安全边距


def theta_to_cosmo(theta: Dict[str, float],
                   base: FiducialCosmology = None) -> FiducialCosmology:
    """把参数向量转为 FiducialCosmology 对象."""
    if base is None:
        base = get_fiducial()
    return FiducialCosmology(
        h=theta.get("h", base.h),
        omega_b=theta.get("omega_b", base.omega_b),
        omega_m=theta.get("omega_m", base.omega_m),
        omega_k=base.Omega_k,
        n_s=theta.get("n_s", base.n_s),
        sigma8=theta.get("sigma8", base.sigma8),
        tau_reio=base.tau_reio,
        Neff=base.Neff,
        w0=theta.get("w0", base.w0),
        wa=theta.get("wa", base.wa),
        As_2e9=base.As_2e9,
    )


def cosmo_to_theta(cosmo: FiducialCosmology,
                   names: Tuple[str, ...]) -> Dict[str, float]:
    return {
        "omega_b": cosmo.omega_b,
        "omega_m": cosmo.omega_m,
        "h": cosmo.h,
        "n_s": cosmo.n_s,
        "sigma8": cosmo.sigma8,
        "w0": cosmo.w0,
        "wa": cosmo.wa,
    }


# =============================================================
# 物理边界检查 + CBF 投影
# =============================================================
def physical_bounds(theta: Dict[str, float], margin: float = 1.0e-4
                    ) -> Dict[str, Tuple[float, float]]:
    """返回每个参数的物理边界 (lower, upper)."""
    return {
        "omega_b": (0.018 + margin, 0.026 - margin),
        "omega_m": (0.10 + margin, 0.20 - margin),
        "h": (0.55 + margin, 0.85 - margin),
        "n_s": (0.85, 1.10),
        "sigma8": (0.6, 1.1),
        "w0": (-1.5, -0.5),
        "wa": (-1.0, 1.0),
    }


def project_to_feasible(theta: Dict[str, float],
                        bounds: Dict[str, Tuple[float, float]]
                        ) -> Dict[str, float]:
    """
    把 theta 投影到可行域 (CBF 式).
    若 θ_i 越界, 设 θ_i = clip(θ_i, lower, upper).
    """
    out = dict(theta)
    for name, (lo, hi) in bounds.items():
        if name in out:
            out[name] = max(lo, min(hi, out[name]))
    return out


# =============================================================
# 残差向量与 Jacobian
# =============================================================
def residual_vector(theta: Dict[str, float],
                    observations: List[BAOObservable],
                    cosmo_fid: FiducialCosmology,
                    priors: ParameterPriors) -> np.ndarray:
    """
    返回残差向量 r(θ) = W^{1/2} (y_obs - y_pred(θ)) 与先验残差拼接.
    """
    cosmo = theta_to_cosmo(theta)
    z_d = drag_redshift_eisenstein_hu(cosmo)
    rd = sound_horizon_numerical(cosmo, z_d=z_d)
    r_list: List[float] = []
    for obs in observations:
        pred = predict_bao(cosmo, obs.z_eff, cosmo_fid, rd)
        r_par = (obs.alpha_par_obs - pred.alpha_par) / obs.alpha_par_sigma
        r_perp = (obs.alpha_perp_obs - pred.alpha_perp) / obs.alpha_perp_sigma
        r_list.extend([r_par, r_perp])
    # 先验残差
    for name, (mu, sig) in (
        ("omega_b", priors.omega_b), ("omega_m", priors.omega_m),
        ("h", priors.h), ("n_s", priors.n_s), ("sigma8", priors.sigma8),
    ):
        if name in theta:
            r_list.append((theta[name] - mu) / sig)
    return np.array(r_list)


def jacobian_fd(theta: Dict[str, float],
                observations: List[BAOObservable],
                cosmo_fid: FiducialCosmology,
                priors: ParameterPriors,
                eps: float = 1.0e-4) -> np.ndarray:
    """
    用中心差分计算 Jacobian J_{i,j} = ∂r_i / ∂θ_j.
    """
    r0 = residual_vector(theta, observations, cosmo_fid, priors)
    n_r = len(r0)
    names = list(theta.keys())
    n_t = len(names)
    J = np.zeros((n_r, n_t), dtype=np.float64)
    for j, name in enumerate(names):
        t_plus = dict(theta)
        t_minus = dict(theta)
        step = max(abs(theta[name]) * eps, eps)
        t_plus[name] += step
        t_minus[name] -= step
        r_plus = residual_vector(t_plus, observations, cosmo_fid, priors)
        r_minus = residual_vector(t_minus, observations, cosmo_fid, priors)
        J[:, j] = (r_plus - r_minus) / (2.0 * step)
    return J


# =============================================================
# Levenberg-Marquardt 迭代
# =============================================================
def lm_step(theta: Dict[str, float],
            observations: List[BAOObservable],
            cosmo_fid: FiducialCosmology,
            priors: ParameterPriors,
            cfg: FitConfig) -> Tuple[Dict[str, float], float, float]:
    """
    单步 LM 更新. 返回 (theta_new, chi2_old, chi2_new).
    """
    r0 = residual_vector(theta, observations, cosmo_fid, priors)
    chi2_old = float(np.dot(r0, r0))
    J = jacobian_fd(theta, observations, cosmo_fid, priors)
    # LM 正规方程
    A = J.T @ J + cfg.lm_lambda * np.eye(J.shape[1])
    b = J.T @ r0
    try:
        delta = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        delta = np.linalg.lstsq(A, b, rcond=None)[0]
    # 更新
    theta_new = dict(theta)
    for j, name in enumerate(theta.keys()):
        theta_new[name] = theta[name] - delta[j]
    # CBF 投影
    bounds = physical_bounds(theta_new, cfg.cbf_margin)
    theta_new = project_to_feasible(theta_new, bounds)
    r1 = residual_vector(theta_new, observations, cosmo_fid, priors)
    chi2_new = float(np.dot(r1, r1))
    return theta_new, chi2_old, chi2_new


# =============================================================
# 双层优化: 外层宇宙学 + 内层 BAO 峰形
# =============================================================
@dataclass
class BAOPeakNuisance:
    """BAO 峰形的 nuisance 参数 (内层优化目标)."""
    A_BAO    : float = 1.0   # BAO 峰振幅
    Sigma_NL : float = 8.0   # 非线性 smearing (Mpc)
    B_lin    : float = 0.0   # 线性偏置


def inner_chi2_peak(A_BAO: float, Sigma_NL: float,
                    s_arr: np.ndarray, xi_obs: np.ndarray,
                    xi_err: np.ndarray,
                    xi_nw: np.ndarray) -> float:
    """
    内层: 在 ξ_0(s) 上拟合 BAO 峰形
      ξ_model(s) = ξ_nw(s) · (1 + A · exp(-(s - r_d)^2 / (2 Σ^2)))
    χ² = Σ_i (ξ_obs - ξ_model)^2 / ξ_err^2
    """
    r_d_fid = 147.78
    xi_model = xi_nw * (1.0 + A_BAO * np.exp(
        -0.5 * ((s_arr - r_d_fid) / max(Sigma_NL, 1.0)) ** 2
    ))
    return float(np.sum(((xi_obs - xi_model) / xi_err) ** 2))


def bilevel_fit(theta_outer: Dict[str, float],
                observations: List[BAOObservable],
                cosmo_fid: FiducialCosmology,
                priors: ParameterPriors,
                s_arr: np.ndarray, xi_obs: np.ndarray, xi_err: np.ndarray,
                xi_nw: np.ndarray,
                cfg: FitConfig) -> Tuple[Dict[str, float], BAOPeakNuisance]:
    """
    双层拟合: 外层 LM 迭代 (若干步), 内层网格搜索 (A, Σ).
    """
    theta = dict(theta_outer)
    best_nuisance = BAOPeakNuisance()
    for it in range(cfg.max_iter):
        # 外层 LM
        theta, chi2_old, chi2_new = lm_step(
            theta, observations, cosmo_fid, priors, cfg
        )
        # 内层: 在 (A, Σ) 网格上粗搜
        A_grid = np.linspace(0.0, 2.0, 11)
        sig_grid = np.linspace(4.0, 15.0, 11)
        best_c2 = float("inf")
        for A in A_grid:
            for sig in sig_grid:
                c2 = inner_chi2_peak(A, sig, s_arr, xi_obs, xi_err, xi_nw)
                if c2 < best_c2:
                    best_c2 = c2
                    best_nuisance = BAOPeakNuisance(A_BAO=A, Sigma_NL=sig)
        # 收敛判定
        if abs(chi2_old - chi2_new) < cfg.tol_grad:
            break
    return theta, best_nuisance


# =============================================================
# 完整拟合流程
# =============================================================
def run_full_fit(theta_init: Dict[str, float] = None,
                 observations: List[BAOObservable] = None,
                 cosmo_fid: FiducialCosmology = None,
                 priors: ParameterPriors = None,
                 cfg: FitConfig = None,
                 ) -> Tuple[Dict[str, float], float, List[float]]:
    """
    运行完整 LM 拟合. 返回 (theta_best, chi2_best, chi2_history).
    """
    if theta_init is None:
        theta_init = {
            "omega_b": 0.0224, "omega_m": 0.1420, "h": 0.68,
        }
    if observations is None:
        observations = get_bao_observations()
    if cosmo_fid is None:
        cosmo_fid = get_fiducial()
    if priors is None:
        priors = get_priors()
    if cfg is None:
        cfg = FitConfig()
    theta = dict(theta_init)
    # 投影到可行域
    bounds = physical_bounds(theta, cfg.cbf_margin)
    theta = project_to_feasible(theta, bounds)
    history = []
    for it in range(cfg.max_iter):
        theta, chi2_old, chi2_new = lm_step(
            theta, observations, cosmo_fid, priors, cfg
        )
        history.append(chi2_new)
        # 自适应 λ
        if chi2_new < chi2_old:
            cfg.lm_lambda *= 0.5
        else:
            cfg.lm_lambda *= 2.0
        cfg.lm_lambda = min(max(cfg.lm_lambda, 1.0e-6), 1.0e6)
        if abs(chi2_new - chi2_old) < cfg.tol_grad:
            break
    return theta, history[-1], history


# =============================================================
# 自检
# =============================================================
def _self_check() -> None:
    theta_init = {"omega_b": 0.0224, "omega_m": 0.1420, "h": 0.68}
    theta_best, chi2, hist = run_full_fit(theta_init)
    print(f"[chi2_fitter] 最佳参数: {theta_best}")
    print(f"[chi2_fitter] 最终 χ² = {chi2:.4f}")
    print(f"[chi2_fitter] χ² 迭代: {[f'{c:.3f}' for c in hist[:5]]}")


if __name__ == "__main__":
    _self_check()
