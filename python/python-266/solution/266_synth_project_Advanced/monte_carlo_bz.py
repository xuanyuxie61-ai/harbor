"""
monte_carlo_bz.py — Brillouin 区 Monte Carlo 积分
====================================================

本模块实现 Brillouin 区上的 Monte Carlo 积分方法。
融合种子项目:
  - 683_line_monte_carlo: 随机/遍历采样, 单式积分
  - 559_hypercube_integrals: 高维超立方体积分
  - 302_disk01_rule: Gauss 求积对比
  - 1415_will_you_be_alive: Monte Carlo 概率估计

核心物理:
  BZ 积分:
    I = (1/|BZ|) ∫_{BZ} f(k) dk

  Monte Carlo 估计:
    I_MC = (1/N) Σ_{i=1}^{N} f(k_i)

  误差:
    σ_MC = σ_f / √N  (随机采样)
    σ_ergodic ~ O(1/N)  (低差异序列)

  应用:
  1. 态密度积分
  2. Fermi 面平均
  3. 输运系数 (电导率, 热导率)
  4. 光学矩阵元
"""

import numpy as np
from typing import Tuple, Callable, Dict
from physical_constants import PI, TWO_PI


# ============================================================
# BZ 采样方法
# ============================================================

def sample_bz_random(n_samples: int, a: float,
                      seed: int = 42) -> np.ndarray:
    """
    随机采样 BZ (源自 683_line_monte_carlo 的 line01_sample_random)。

    k_i ~ Uniform[-π/a, π/a]

    Parameters
    ----------
    n_samples : int
        采样点数
    a : float
        晶格常数
    seed : int
        随机种子

    Returns
    -------
    kpoints : np.ndarray, shape (n_samples,)
    """
    np.random.seed(seed)
    return -PI / a + np.random.rand(n_samples) * TWO_PI / a


def sample_bz_ergodic(n_samples: int, a: float) -> np.ndarray:
    """
    遍历 (低差异) 采样 BZ (源自 683_line_monte_carlo)。

    使用黄金比例生成 quasi-random 序列:
      k_i = ((i · φ) mod 1) · 2π/a - π/a

    其中 φ = (1+√5)/2 为黄金比例。

    这种序列的优势:
    - 低差异 (discrepancy ~ O(log(N)/N))
    - 均匀覆盖
    - 可复现 (确定性)

    Parameters
    ----------
    n_samples : int
    a : float

    Returns
    -------
    kpoints : np.ndarray
    """
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    kpoints = np.zeros(n_samples)
    shift = 0.0

    for i in range(n_samples):
        shift = (shift + 1.0 / phi) % 1.0
        kpoints[i] = -PI / a + shift * TWO_PI / a

    return kpoints


def sample_bz_halton(n_samples: int, a: float,
                      base: int = 2) -> np.ndarray:
    """
    Halton 序列采样 BZ (低差异序列)。

    Halton 序列基于 base-b 的 van der Corput 序列:
      x_i = Σ_{j=0}^{M} d_j(i) · b^{-(j+1)}

    其中 d_j(i) 为 i 的 base-b 展开的第 j 位。

    Parameters
    ----------
    n_samples : int
    a : float
    base : int
        序列基底 (常用 2 或 3)

    Returns
    -------
    kpoints : np.ndarray
    """
    kpoints = np.zeros(n_samples)
    for i in range(n_samples):
        val = 0.0
        f = 1.0 / base
        idx = i + 1
        while idx > 0:
            val += (idx % base) * f
            idx //= base
            f /= base
        kpoints[i] = -PI / a + val * TWO_PI / a
    return kpoints


# ============================================================
# Monte Carlo 积分
# ============================================================

def monte_carlo_bz_integral(func: Callable[[np.ndarray], np.ndarray],
                              n_samples: int,
                              a: float,
                              method: str = 'ergodic',
                              seed: int = 42
                              ) -> Tuple[float, float]:
    """
    Monte Carlo BZ 积分。

    I ≈ (1/N) Σ f(k_i)

    误差估计:
    - 随机: σ = √(Var(f)/N)
    - 遍历: σ ~ O(1/N) (理论上更快收敛)

    Parameters
    ----------
    func : callable
        被积函数 k → f(k)
    n_samples : int
        采样数
    a : float
        晶格常数
    method : str
        'random', 'ergodic', 'halton'
    seed : int
        随机种子

    Returns
    -------
    integral : float
        积分估计值
    error : float
        标准误差估计
    """
    if method == 'random':
        kpoints = sample_bz_random(n_samples, a, seed)
    elif method == 'ergodic':
        kpoints = sample_bz_ergodic(n_samples, a)
    elif method == 'halton':
        kpoints = sample_bz_halton(n_samples, a)
    else:
        raise ValueError(f"未知采样方法: {method}")

    # 计算函数值
    f_values = func(kpoints)

    # 积分 = BZ 长度 × 平均值
    bz_length = TWO_PI / a
    integral = bz_length * np.mean(f_values)

    # 误差估计
    if method == 'random':
        std_f = np.std(f_values)
        error = bz_length * std_f / np.sqrt(n_samples)
    else:
        # 遍历序列的误差用分块估计
        n_blocks = max(4, int(np.sqrt(n_samples)))
        block_size = n_samples // n_blocks
        block_means = []
        for b in range(n_blocks):
            start = b * block_size
            end = start + block_size
            block_means.append(np.mean(f_values[start:end]))
        error = bz_length * np.std(block_means) / np.sqrt(n_blocks)

    return integral, error


# ============================================================
# 收敛分析
# ============================================================

def monte_carlo_convergence(func: Callable,
                              n_samples_list: list,
                              a: float,
                              exact_value: float,
                              method: str = 'ergodic'
                              ) -> Dict[str, np.ndarray]:
    """
    Monte Carlo 积分的收敛分析。

    对不同采样数, 计算积分值并与精确值比较。

    Parameters
    ----------
    func : callable
    n_samples_list : list of int
    a : float
    exact_value : float
    method : str

    Returns
    -------
    results : dict
        包含积分值、误差、收敛阶数
    """
    integrals = []
    errors = []
    actual_errors = []

    for n in n_samples_list:
        val, err = monte_carlo_bz_integral(
            func, n, a, method=method)
        integrals.append(val)
        errors.append(err)
        actual_errors.append(abs(val - exact_value))

    return {
        'n_samples': np.array(n_samples_list),
        'integrals': np.array(integrals),
        'estimated_errors': np.array(errors),
        'actual_errors': np.array(actual_errors),
        'exact_value': exact_value,
    }


# ============================================================
# 高维 BZ Monte Carlo (源自 559_hypercube_integrals)
# ============================================================

def hypercube_bz_monte_carlo(func: Callable,
                               n_samples: int,
                               lattice_constants: np.ndarray,
                               seed: int = 42
                               ) -> Tuple[float, float]:
    """
    高维 BZ 的 Monte Carlo 积分 (源自 559_hypercube_integrals)。

    对于 d 维正交晶格:
      BZ = Π_i [-π/a_i, π/a_i]
      I = ∫_{BZ} f(k) d^d k

    Monte Carlo:
      I ≈ V_BZ · (1/N) Σ f(k_i)
    其中 V_BZ = Π_i (2π/a_i)

    Parameters
    ----------
    func : callable
        被积函数 (k_vector) → scalar
    n_samples : int
    lattice_constants : np.ndarray, shape (d,)
    seed : int

    Returns
    -------
    integral : float
    error : float
    """
    np.random.seed(seed)
    d = len(lattice_constants)

    # BZ 体积
    bz_volume = np.prod([TWO_PI / a for a in lattice_constants])

    # 随机采样
    kpoints = np.zeros((n_samples, d))
    for dim in range(d):
        kpoints[:, dim] = (-PI / lattice_constants[dim] +
                            np.random.rand(n_samples) *
                            TWO_PI / lattice_constants[dim])

    f_values = np.array([func(kpoints[i]) for i in range(n_samples)])

    integral = bz_volume * np.mean(f_values)
    error = bz_volume * np.std(f_values) / np.sqrt(n_samples)

    return integral, error


# ============================================================
# 精确参考值 (源自 681_line_integrals 和 559)
# ============================================================

def bz_monomial_reference(exponent: int, a: float) -> float:
    """
    BZ 上 k^e 的精确积分 (参考值)。

    源自 681_line_integrals 的 line01_monomial_integral,
    映射到 BZ [-π/a, π/a]。
    """
    from lattice import line_bz_monomial_integral
    return line_bz_monomial_integral(exponent, a)


def hypercube_bz_monomial_reference(exponents: np.ndarray,
                                      lattice_constants: np.ndarray
                                      ) -> float:
    """高维 BZ 单式积分精确值 (源自 559_hypercube_integrals)"""
    from lattice import hypercube_bz_monomial_integral
    return hypercube_bz_monomial_integral(exponents, lattice_constants)
