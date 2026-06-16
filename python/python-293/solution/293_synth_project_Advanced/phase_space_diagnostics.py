# -*- coding: utf-8 -*-
"""
phase_space_diagnostics.py
===========================
相空间诊断工具集

融合种子项目:
  1282_dapospisil_revisit_hid_geom — 特征矩 / 特征谱 / 幂律分析
  602_jaccard_distance              — 集合相似度度量

物理背景
--------
Vlasov-Poisson 模拟的核心诊断量:

1. 速度矩 (velocity moments):
   n(x) = ∫ f dv         (密度)
   J(x) = ∫ v f dv       (电流密度)
   P(x) = ∫ v² f dv      (压力)
   Q(x) = ∫ v³ f dv      (热流)
   E_kin = ∫∫ v² f dx dv / 2  (动能)

2. 熵:
   S = -∫∫ f ln f dx dv   (Gibbs 熵)
   守恒性: dS/dt = 0 (无碰撞 Vlasov)

3. Jaccard 相似度:
   J(A,B) = |A∩B| / |A∪B|
   用于比较不同时刻分布函数的支撑集差异

4. 特征谱分析:
   从电场时间序列提取本征模阻尼率 (Prony/Hankel 方法)
"""

import numpy as np
import math
from typing import Dict, List, Tuple, Optional


# ===== 速度矩计算 ==========================================================

def density(f: np.ndarray, dv: float) -> np.ndarray:
    """
    数密度: n(x) = ∫ f(x,v) dv

    物理含义: 位置 x 处的粒子数密度.
    """
    return np.sum(f, axis=1) * dv


def current_density(f: np.ndarray, v: np.ndarray,
                     dv: float) -> np.ndarray:
    """
    电流密度: J(x) = ∫ v f(x,v) dv

    与电场的关系: ∂J/∂x + ∂ρ/∂t = 0 (连续性方程)
    """
    return np.sum(f * v[np.newaxis, :], axis=1) * dv


def pressure_tensor(f: np.ndarray, v: np.ndarray,
                     dv: float) -> np.ndarray:
    """
    压力张量: P(x) = ∫ v² f(x,v) dv

    标量压力 p = P (1D 系统).
    温度: T(x) = P(x) / n(x) (设 k_B = 1)
    """
    return np.sum(f * (v ** 2)[np.newaxis, :], axis=1) * dv


def heat_flux(f: np.ndarray, v: np.ndarray,
               dv: float) -> np.ndarray:
    """
    热流: Q(x) = ∫ v³ f(x,v) dv

    决定能量输运速率.
    """
    return np.sum(f * (v ** 3)[np.newaxis, :], axis=1) * dv


def kurtosis(f: np.ndarray, v: np.ndarray,
              dv: float) -> np.ndarray:
    """
    速度分布峰度 (平坦度因子):

    .. math::
        K(x) = \\frac{\\int (v - u)^4 f dv}
                    {(\\int (v - u)^2 f dv)^2}

    K = 3: Maxwellian (高斯)
    K > 3: 重尾 (超热粒子)
    K < 3: 轻尾 (平板分布)
    """
    n = density(f, dv)
    n_safe = np.maximum(n, 1e-300)
    u = current_density(f, v, dv) / n_safe
    v_shift = v[np.newaxis, :] - u[:, np.newaxis]
    P = np.sum(f * v_shift ** 2, axis=1) * dv
    Q4 = np.sum(f * v_shift ** 4, axis=1) * dv
    return Q4 / (P ** 2 + 1e-300)


def kinetic_energy(f: np.ndarray, v: np.ndarray,
                    dx: float, dv: float) -> float:
    """
    总动能: E_kin = (1/2) ∫∫ v² f dx dv
    """
    return 0.5 * np.sum(f * (v ** 2)[np.newaxis, :]) * dx * dv


def field_energy(E: np.ndarray, dx: float) -> float:
    """
    电场能: E_field = (ε₀/2) ∫ E² dx  (设 ε₀ = 1)
    """
    return 0.5 * np.sum(E ** 2) * dx


def total_energy(f: np.ndarray, E: np.ndarray,
                  v: np.ndarray, dx: float, dv: float) -> float:
    """总能量 = 动能 + 电场能."""
    return kinetic_energy(f, v, dx, dv) + field_energy(E, dx)


# ===== 熵与散度 ============================================================

def gibbs_entropy(f: np.ndarray, dx: float, dv: float,
                   epsilon: float = 1.0e-300) -> float:
    """
    Gibbs 熵: S = -∫∫ f ln(f) dx dv

    Vlasov 方程的 Casimir 不变量: dS/dt = 0
    数值耗散会导致 S 单调增加 (H 定理).
    """
    f_safe = np.maximum(f, epsilon)
    return -float(np.sum(f * np.log(f_safe)) * dx * dv)


def kullback_leibler(f: np.ndarray, g: np.ndarray,
                      dx: float, dv: float,
                      epsilon: float = 1.0e-300) -> float:
    """
    KL 散度: D_KL(f || g) = ∫∫ f ln(f/g) dx dv

    度量两个分布函数的差异 (非对称).
    D_KL = 0 ⟺ f = g.
    """
    f_safe = np.maximum(f, epsilon)
    g_safe = np.maximum(g, epsilon)
    return float(np.sum(f * np.log(f_safe / g_safe)) * dx * dv)


# ===== Jaccard 相似度 ======================================================

def jaccard_index(A: np.ndarray, B: np.ndarray) -> float:
    """
    Jaccard 指数: J(A,B) = |A∩B| / |A∪B|

    在等离子体中用于比较相空间区域的相似度:
      J = 1: 完全相同
      J = 0: 无交集
    """
    A_set = set(np.asarray(A).ravel())
    B_set = set(np.asarray(B).ravel())
    inter = len(A_set & B_set)
    union = len(A_set | B_set)
    if union == 0:
        return 1.0
    return inter / union


def jaccard_distance(A: np.ndarray, B: np.ndarray) -> float:
    """Jaccard 距离: d_J(A,B) = 1 - J(A,B)."""
    return 1.0 - jaccard_index(A, B)


def distribution_support_similarity(
    f1: np.ndarray, f2: np.ndarray,
    threshold: float = 0.01
) -> float:
    """
    分布函数支撑集的 Jaccard 相似度

    将 f 二值化: A = {f > threshold}
    比较两个分布的有效支撑区域重叠程度。
    """
    A = f1 > threshold
    B = f2 > threshold
    inter = np.sum(A & B)
    union = np.sum(A | B)
    if union == 0:
        return 1.0
    return float(inter) / float(union)


# ===== 综合诊断器 ==========================================================

class PhaseSpaceDiagnostician:
    """
    集成相空间诊断器

    计算 Vlasov-Poisson 模拟的全部标准诊断量，
    并生成诊断报告。
    """

    def __init__(self, x: np.ndarray, v: np.ndarray):
        self.x = np.asarray(x, dtype=np.float64)
        self.v = np.asarray(v, dtype=np.float64)
        self.dx = self.x[1] - self.x[0] if len(self.x) > 1 else 1.0
        self.dv = self.v[1] - self.v[0] if len(self.v) > 1 else 1.0

    def compute_all(self, f: np.ndarray,
                     E: np.ndarray) -> Dict[str, float]:
        """
        计算全部诊断量

        Returns
        -------
        dict
            包含密度、电流、压力、能量、熵等.
        """
        return {
            'max_density': float(np.max(density(f, self.dv))),
            'min_density': float(np.min(density(f, self.dv))),
            'max_current': float(np.max(
                np.abs(current_density(f, self.v, self.dv)))),
            'max_pressure': float(np.max(pressure_tensor(f, self.v, self.dv))),
            'kinetic_energy': kinetic_energy(f, self.v, self.dx, self.dv),
            'field_energy': field_energy(E, self.dx),
            'total_energy': total_energy(f, E, self.v, self.dx, self.dv),
            'entropy': gibbs_entropy(f, self.dx, self.dv),
            'max_f': float(np.max(f)),
            'min_f': float(np.min(f)),
            'total_particles': float(np.sum(f) * self.dx * self.dv),
        }

    def conservation_check(self, f_history: List[np.ndarray],
                            E_history: List[np.ndarray]) -> Dict:
        """
        守恒量检查: 追踪粒子数、总能量的时间演化

        Returns
        -------
        dict
            各守恒量的相对漂移.
        """
        n_particles = []
        energies = []
        entropies = []

        for f, E in zip(f_history, E_history):
            n_particles.append(float(np.sum(f) * self.dx * self.dv))
            energies.append(total_energy(f, E, self.v, self.dx, self.dv))
            entropies.append(gibbs_entropy(f, self.dx, self.dv))

        n_particles = np.array(n_particles)
        energies = np.array(energies)
        entropies = np.array(entropies)

        return {
            'particle_drift': float(
                abs(n_particles[-1] - n_particles[0])
                / (abs(n_particles[0]) + 1e-300)),
            'energy_drift': float(
                abs(energies[-1] - energies[0])
                / (abs(energies[0]) + 1e-300)),
            'entropy_production': float(entropies[-1] - entropies[0]),
            'particle_conservation': n_particles,
            'energy_evolution': energies,
            'entropy_evolution': entropies,
        }

    def spectrum_analysis(self, f: np.ndarray) -> Dict:
        """
        分布函数的 Fourier 谱分析

        在速度方向做 FFT, 检查高分量是否被充分解析.
        """
        f_hat = np.fft.fft(f, axis=1)
        power = np.abs(f_hat) ** 2
        mean_power = np.mean(power, axis=0)

        # 谱衰减比 (最高频 / 低频)
        ratio = float(mean_power[-1] / (mean_power[0] + 1e-300))

        return {
            'spectral_power': mean_power,
            'high_freq_ratio': ratio,
            'effective_modes': int(np.sum(
                mean_power > 1e-6 * mean_power[0])),
        }
