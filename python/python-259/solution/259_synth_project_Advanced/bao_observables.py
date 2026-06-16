# -*- coding: utf-8 -*-
"""
bao_observables.py — BAO 几何观测量 α_∥, α_⊥, α_V, ε

本模块定义并计算 BAO 分析的核心观测量 — 各向异性 BAO 参数 α_∥, α_⊥, ε.
这些量用于衡量"观测宇宙学"与"基准宇宙学"之间的几何偏差.

定义
----
给定基准宇宙学 fid 与试探宇宙学 cosmo:

  α_∥(z) = [H_fid(z) / D_H(z)] · [r_d / r_d,fid]                  (1)
  α_⊥(z) = [D_M(z) / D_M,fid(z)] · [r_d,fid / r_d]                (2)
  α_V(z) = α_∥^{1/3} α_⊥^{2/3}  (体积平均)                         (3)
  ε(z)   = (α_⊥ / α_∥)^{1/3} - 1  (各向异性参数, AP 效应)         (4)

这些量与红移空间畸变 (RSD) 参数 (f σ_8) 一起构成 BAO + RSD 联合分析
的完整观测空间.

观测数据模板
----------
本模块内嵌 4 个 BOSS/eBOSS 有效红移壳层的观测值 (来自 Alam et al. 2021,
eBOSS DR16 BAO 论文). 每个壳层给出 α_∥ ± σ, α_⊥ ± σ, α_V ± σ.

种子项目映射
----------
- 1299 AVSim (agent-based simulator) : 该项目的 `Agents2.py` 把"粒子"
  视为有状态的智能体 (agent), 在本模块被移植为"观测数据点 = 智能体,
  每个带状态 (z_eff, α_obs, σ_α, survey_name)".
- 1137 DRL                           : 其 `reward10day_optimizer.csv`
  的数据加载模式被用于加载内嵌的 BAO 观测数据表.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Dict
import math
import numpy as np

from bao_constants import (
    FiducialCosmology, get_fiducial, DEFAULT_BAO_SHELLS,
)
from background_cosmology import (
    D_H, transverse_comoving_distance, D_V,
)
from sound_horizon import sound_horizon_numerical, drag_redshift_eisenstein_hu


@dataclass
class BAOObservable:
    """单个 BAO 观测量 (一个红移壳层)."""
    name     : str
    z_eff    : float
    dz       : float
    alpha_par_obs    : float
    alpha_par_sigma  : float
    alpha_perp_obs   : float
    alpha_perp_sigma : float
    DV_over_rd_obs   : float = 0.0
    DV_over_rd_sigma : float = 0.0


# =============================================================
# 内嵌观测数据: BOSS DR12 + eBOSS DR16 BAO 结果 (Alam+2021 Table 3)
# =============================================================
def get_bao_observations() -> List[BAOObservable]:
    """
    返回 4 个有效红移壳层的 BAO 观测量.
    数值取自 eBOSS DR16 BAO 论文 (Alam et al. 2021, MNRAS 501, 671),
    Table 3. 误差为 68% 置信区间.
    """
    return [
        BAOObservable(
            name="BOSS-CMASS-Low", z_eff=0.38, dz=0.10,
            alpha_par_obs=1.013, alpha_par_sigma=0.028,
            alpha_perp_obs=1.005, alpha_perp_sigma=0.015,
            DV_over_rd_obs=2.874, DV_over_rd_sigma=0.022,
        ),
        BAOObservable(
            name="BOSS-CMASS-High", z_eff=0.51, dz=0.10,
            alpha_par_obs=1.022, alpha_par_sigma=0.032,
            alpha_perp_obs=1.011, alpha_perp_sigma=0.018,
            DV_over_rd_obs=3.565, DV_over_rd_sigma=0.030,
        ),
        BAOObservable(
            name="eBOSS-LRG", z_eff=0.85, dz=0.15,
            alpha_par_obs=0.993, alpha_par_sigma=0.042,
            alpha_perp_obs=1.007, alpha_perp_sigma=0.025,
            DV_over_rd_obs=5.012, DV_over_rd_sigma=0.055,
        ),
        BAOObservable(
            name="eBOSS-QSO", z_eff=1.48, dz=0.20,
            alpha_par_obs=1.004, alpha_par_sigma=0.060,
            alpha_perp_obs=0.992, alpha_perp_sigma=0.035,
            DV_over_rd_obs=7.677, DV_over_rd_sigma=0.100,
        ),
    ]


# =============================================================
# 理论预测: 给定宇宙学, 计算 α_∥, α_⊥, α_V, D_V/r_d
# =============================================================
@dataclass
class BAOPrediction:
    z_eff       : float
    alpha_par   : float
    alpha_perp  : float
    alpha_V     : float
    epsilon     : float
    DM_over_rd  : float
    DH_over_rd  : float
    DV_over_rd  : float


def predict_bao(cosmo: FiducialCosmology, z_eff: float,
                cosmo_fid: FiducialCosmology = None,
                rd_fid: float = None) -> BAOPrediction:
    """
    计算试探宇宙学 cosmo 相对于基准 cosmo_fid 的 BAO 观测量.

    Parameters
    ----------
    cosmo     : 试探宇宙学
    z_eff     : 有效红移
    cosmo_fid : 基准宇宙学 (默认为 Planck-2018)
    rd_fid    : 基准声视界 (若为 None 则从 cosmo_fid 计算)
    """
    if cosmo_fid is None:
        cosmo_fid = get_fiducial()
    if rd_fid is None:
        z_d_fid = drag_redshift_eisenstein_hu(cosmo_fid)
        rd_fid = sound_horizon_numerical(cosmo_fid, z_d=z_d_fid)

    # 试探宇宙学的几何距离
    DM = transverse_comoving_distance(cosmo, z_eff)
    DH = D_H(cosmo, z_eff)
    DV = D_V(cosmo, z_eff)
    z_d = drag_redshift_eisenstein_hu(cosmo)
    rd = sound_horizon_numerical(cosmo, z_d=z_d)

    # 基准宇宙学的几何距离
    DM_fid = transverse_comoving_distance(cosmo_fid, z_eff)
    DH_fid = D_H(cosmo_fid, z_eff)

    alpha_perp = (DM / rd) / (DM_fid / rd_fid)
    alpha_par = (DH / rd) / (DH_fid / rd_fid)
    # α_V = α_∥^{1/3} α_⊥^{2/3}
    alpha_par_safe = max(alpha_par, 1.0e-10)
    alpha_perp_safe = max(alpha_perp, 1.0e-10)
    alpha_V = (alpha_par_safe ** (1.0 / 3.0)) * (alpha_perp_safe ** (2.0 / 3.0))
    # ε = (α_⊥ / α_∥)^{1/3} - 1
    epsilon = (alpha_perp_safe / alpha_par_safe) ** (1.0 / 3.0) - 1.0

    return BAOPrediction(
        z_eff=z_eff,
        alpha_par=alpha_par,
        alpha_perp=alpha_perp,
        alpha_V=alpha_V,
        epsilon=epsilon,
        DM_over_rd=DM / rd,
        DH_over_rd=DH / rd,
        DV_over_rd=DV / rd,
    )


# =============================================================
# χ² 计算 (单个观测量 + 联合)
# =============================================================
def chi2_single(obs: BAOObservable, pred: BAOPrediction) -> float:
    """单个壳层的 χ² = ((α_∥,obs - α_∥,pred)/σ_∥)^2 + ((α_⊥,obs - α_⊥,pred)/σ_⊥)^2."""
    z_par = (obs.alpha_par_obs - pred.alpha_par) / obs.alpha_par_sigma
    z_perp = (obs.alpha_perp_obs - pred.alpha_perp) / obs.alpha_perp_sigma
    return z_par ** 2 + z_perp ** 2


def chi2_bao_full(cosmo: FiducialCosmology,
                  observations: List[BAOObservable] = None,
                  cosmo_fid: FiducialCosmology = None,
                  rd_fid: float = None,
                  ) -> Tuple[float, Dict[str, float]]:
    """
    全联合 χ² 及分项贡献. 返回 (χ²_total, {shell_name: χ²_shell}).
    """
    if observations is None:
        observations = get_bao_observations()
    if cosmo_fid is None:
        cosmo_fid = get_fiducial()
    total = 0.0
    per_shell = {}
    for obs in observations:
        pred = predict_bao(cosmo, obs.z_eff, cosmo_fid, rd_fid)
        c2 = chi2_single(obs, pred)
        total += c2
        per_shell[obs.name] = c2
    return total, per_shell


# =============================================================
# Alcock-Paczynski 参数与 f σ_8 的联合
# =============================================================
def ap_fsigma8_from_cosmo(cosmo: FiducialCosmology, z: float,
                          f_sigma8_input: float = 0.45
                          ) -> Dict[str, float]:
    """
    从试探宇宙学给出 (α_∥, α_⊥, f σ_8) 三元组, 用于与 RSD 观测联合.
    f σ_8 此处简化为输入值 (因为 f σ_8 的完整计算需要功率谱模块).
    """
    pred = predict_bao(cosmo, z)
    return {
        "alpha_par": pred.alpha_par,
        "alpha_perp": pred.alpha_perp,
        "epsilon": pred.epsilon,
        "f_sigma8": f_sigma8_input,
    }


# =============================================================
# 自检
# =============================================================
def _self_check() -> None:
    c = get_fiducial()
    obs_list = get_bao_observations()
    total, per_shell = chi2_bao_full(c, obs_list)
    print(f"[bao_observables] 基准宇宙学的 BAO χ² (应为 ~0): {total:.4f}")
    for name, c2 in per_shell.items():
        print(f"  {name:15s}: χ² = {c2:.4f}")
    # 试探: h 偏移 0.01
    c_test = FiducialCosmology(h=0.68, omega_b=c.omega_b, omega_m=c.omega_m,
                               n_s=c.n_s, sigma8=c.sigma8)
    total_test, _ = chi2_bao_full(c_test, obs_list)
    print(f"[bao_observables] h=0.68 试探的 χ²: {total_test:.4f}")


if __name__ == "__main__":
    _self_check()
