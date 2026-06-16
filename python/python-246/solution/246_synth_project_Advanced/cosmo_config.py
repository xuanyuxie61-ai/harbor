"""
cosmo_config.py  —  宇宙大尺度结构 N 体模拟配置加载器
=====================================================

科学来源种子:
  - 1068_jaweriaamjad_internalstate / config_utils.py
    借鉴其 JSON 配置加载、路径相对仓库根解析的思路,
    泛化为宇宙学模拟参数管理。

物理背景:
  本项目模拟 Einstein-de Sitter 宇宙中冷暗物质 (CDM) 密度扰动 δ(x,t)
  的演化,控制方程为 (共动坐标下):
      ∂δ/∂t + (1/a) ∇·[(1+δ) v] = 0                 (连续性方程)
      ∂v/∂t + H(a) v + (1/a)(v·∇)v = -(1/a) ∇Φ     (Euler 方程)
      ∇²Φ = 4 π G ρ̄ a² δ                           (Poisson 方程)
  其中 a(t) 为尺度因子, H(a) = ȧ/a 为 Hubble 参数。
"""

from __future__ import annotations
import json
import os
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, Any


_REPO_ROOT = Path(__file__).resolve().parent
_DEFAULT_CONFIG = "cosmo_config.json"
_DEFAULT_SIM = "cosmo_sim.json"


# ---------------------------------------------------------------------------- #
#                          物理 / 宇宙学常量
# ---------------------------------------------------------------------------- #
C_LIGHT_KMS = 2.99792458e5              # 光速 km/s
G_NEWTON_SI = 6.67430e-11               # 引力常数 m³ kg⁻¹ s⁻²
MPC_M = 3.085677581e22                  # 1 Mpc in meters
SOLAR_MASS_KG = 1.98892e30
RHO_CRIT_H2 = 2.77536627e11             # ρ_crit / h²  [M_sun / Mpc³]
H100_SI = 100.0e3 / MPC_M               # H0 = 100 km/s/Mpc in s⁻¹


# ---------------------------------------------------------------------------- #
#                          数据结构 (CosmoParams)
# ---------------------------------------------------------------------------- #
@dataclass
class CosmoParams:
    """可复现实验所需的所有物理与数值参数。"""
    box_length: float = 64.0
    n_grid: int = 32
    n_particles: int = 32768
    omega_m: float = 0.308
    omega_l: float = 0.692
    omega_b: float = 0.048
    hubble: float = 0.6781
    z_init: float = 49.0
    z_final: float = 0.0
    n_steps: int = 40
    fd_order: int = 4
    softening_eps: float = 0.05
    spectral_index: float = 0.9667
    sigma_8: float = 0.8159
    seed: int = 42
    max_cond_number: float = 1.0e10
    stability_tol: float = 1.0e-8
    cfl_safety: float = 0.35
    softening_bounds: tuple = (0.01, 0.5)
    timestep_bounds: tuple = (1.0e-3, 1.0)

    def __post_init__(self) -> None:
        if self.n_grid < 8:
            raise ValueError(f"n_grid={self.n_grid} 必须 ≥ 8 以保证有限差分模板")
        if not 0.0 < self.omega_m <= 1.0:
            raise ValueError("omega_m 必须位于 (0, 1]")
        if self.fd_order not in (2, 4, 6, 8):
            raise ValueError(f"不支持的 fd_order={self.fd_order}")
        if self.softening_eps <= 0.0:
            raise ValueError("引力软化长度必须为正")
        if self.sigma_8 <= 0.0:
            raise ValueError("sigma_8 必须为正")


# ---------------------------------------------------------------------------- #
#                              配置加载器
# ---------------------------------------------------------------------------- #
def _load_json(filename: str, required: bool = True) -> Dict[str, Any]:
    path = _REPO_ROOT / filename
    if not path.exists():
        if required:
            raise FileNotFoundError(
                f"未找到配置文件: {path}\n"
                f"请参考 {_DEFAULT_CONFIG}.example 创建之。"
            )
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_path(p: str) -> Path:
    path = Path(p)
    if not path.is_absolute():
        path = _REPO_ROOT / path
    return path.resolve()


def get_paths() -> Dict[str, Path]:
    """返回模拟数据、结果、诊断输出目录。"""
    cfg = _load_json(_DEFAULT_CONFIG, required=False)
    out = {
        "data_dir": _resolve_path(cfg.get("data_dir", "data")),
        "results_dir": _resolve_path(cfg.get("results_dir", "results")),
        "diag_dir": _resolve_path(cfg.get("diag_dir", "diagnostics")),
    }
    for v in out.values():
        v.mkdir(parents=True, exist_ok=True)
    return out


def load_cosmo_params() -> CosmoParams:
    """从 cosmo_config.json 加载参数,缺失字段使用默认值。"""
    cfg = _load_json(_DEFAULT_CONFIG, required=False)
    sim = _load_json(_DEFAULT_SIM, required=False)
    merged = {**cfg, **sim}
    known = {k: v for k, v in merged.items() if k in CosmoParams.__dataclass_fields__}
    # tuple 字段:
    for key in ("softening_bounds", "timestep_bounds"):
        if key in known and isinstance(known[key], list):
            known[key] = tuple(known[key])
    return CosmoParams(**known)


def save_params(params: CosmoParams, filename: str = "params_used.json") -> Path:
    out_dir = get_paths()["results_dir"]
    target = out_dir / filename
    # tuple 转 list 以便 JSON 序列化
    d = asdict(params)
    for k, v in d.items():
        if isinstance(v, tuple):
            d[k] = list(v)
    with open(target, "w", encoding="utf-8") as f:
        json.dump(d, f, indent=2, ensure_ascii=False)
    return target


# ---------------------------------------------------------------------------- #
#                          辅助: 红移 <-> 尺度因子
# ---------------------------------------------------------------------------- #
def z_to_a(z: float) -> float:
    if z < -0.999999:
        raise ValueError(f"红移 z={z} 不得 ≤ -1")
    return 1.0 / (1.0 + z)


def hubble_factor(a: float, p: "CosmoParams") -> float:
    """
    无量纲 Hubble 因子 E(a) = H(a)/H0:
        E(a) = sqrt( Ω_m a^{-3} + Ω_k a^{-2} + Ω_Λ )
    """
    if a <= 0.0:
        raise ValueError(f"尺度因子 a={a} 必须为正")
    omega_k = 1.0 - p.omega_m - p.omega_l
    e2 = p.omega_m / (a ** 3) + omega_k / (a ** 2) + p.omega_l
    if e2 < 0.0:
        raise ValueError("Hubble 因子平方为负,宇宙学参数不合理")
    return float(e2 ** 0.5)
