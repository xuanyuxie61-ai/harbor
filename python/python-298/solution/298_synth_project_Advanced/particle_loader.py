"""
粒子加载模块

负责 PIC 模拟中初始粒子分布的生成:
- 空间均匀加载
- 麦克斯韦速度分布加载
- 扰动加载 (用于激发等离子体波)
- 权重粒子加载
"""

import numpy as np
from typing import Tuple
try:
    from plasma_constants import ELECTRON_MASS, ELECTRON_CHARGE, PI
except ImportError:
    from plasma_constants import ELECTRON_MASS, ELECTRON_CHARGE, PI
try:
    from velocity_quadrature import hyperball_sample_maxwellian
except ImportError:
    from velocity_quadrature import hyperball_sample_maxwellian


def load_particles_uniform_1d(Np: int, L: float,
                                 seed: int = 42) -> np.ndarray:
    """
    一维均匀空间加载:
        x_i ~ Uniform(0, L)
    """
    rng = np.random.RandomState(seed)
    return rng.uniform(0.0, L, Np)


def load_particles_regular_1d(Np: int, L: float) -> np.ndarray:
    """
    规则空间加载 (无声 PIC):
        x_i = (i + 0.5) * L / Np
    """
    return (np.arange(Np) + 0.5) * L / Np


def load_velocity_maxwellian_1d(Np: int, v_th: float,
                                   seed: int = 42) -> np.ndarray:
    """
    一维麦克斯韦速度加载:
        v_i ~ N(0, v_th^2/2)
    """
    rng = np.random.RandomState(seed)
    sigma = v_th / np.sqrt(2.0)
    return rng.randn(Np) * sigma


def load_velocity_maxwellian_3d(Np: int, v_th: float,
                                   seed: int = 42) -> np.ndarray:
    """
    三维麦克斯韦速度加载:
        (vx, vy, vz) 各独立 ~ N(0, v_th^2/2)
    """
    return hyperball_sample_maxwellian(3, Np, v_th, seed)


def load_particles_with_perturbation(Np: int, L: float, v_th: float,
                                        perturbation_amplitude: float = 0.01,
                                        perturbation_mode: int = 1,
                                        seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    带扰动的粒子加载 (用于激发等离子体波):

    空间分布:
        f(x) = n_0 * (1 + alpha * cos(k * x))
    其中 k = 2*pi*perturbation_mode / L

    使用 rejection sampling 或 transform method

    Returns:
        (x, v) : 位置和速度
    """
    rng = np.random.RandomState(seed)
    k_mode = 2.0 * PI * perturbation_mode / L
    alpha = perturbation_amplitude

    # Rejection sampling for spatial distribution
    x = np.zeros(Np)
    count = 0
    max_iter = 100 * Np
    n_iter = 0
    while count < Np and n_iter < max_iter:
        x_try = rng.uniform(0.0, L)
        f_val = 1.0 + alpha * np.cos(k_mode * x_try)
        f_max = 1.0 + abs(alpha)
        if rng.uniform() < f_val / f_max:
            x[count] = x_try
            count += 1
        n_iter += 1

    if count < Np:
        # 填充剩余
        x[count:] = rng.uniform(0.0, L, Np - count)

    # 麦克斯韦速度
    v = load_velocity_maxwellian_1d(Np, v_th, seed + 1)

    return x, v


def load_particles_quiet_start(Np: int, L: float, v_th: float,
                                  seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
     quiet-start 加载 (减少初始噪声):

    1. 规则空间网格
    2. 按麦克斯韦 CDF 加载速度

    比随机加载噪声低 O(1/Np) 而非 O(1/sqrt(Np))
    """
    rng = np.random.RandomState(seed)
    x = load_particles_regular_1d(Np, L)

    # 按 CDF 排序加载
    v_random = rng.randn(Np) * v_th / np.sqrt(2.0)
    v_sorted = np.sort(v_random)
    # 随机分配 (shuffle) 以避免与空间位置关联
    rng.shuffle(v_sorted)
    v = v_sorted

    return x, v


def particle_weight(Np: int, n0: float, L: float) -> float:
    """
    计算粒子权重:
        weight = n_0 * L / N_p

    使得 sum_p weight = n_0 * L
    """
    return n0 * L / Np


def total_charge_from_particles(weight: float, q: float, Np: int) -> float:
    """
    总电荷:
        Q = N_p * weight * q
    """
    return Np * weight * q


def kinetic_energy_particles(v: np.ndarray, weight: float,
                               m: float = ELECTRON_MASS) -> float:
    """
    粒子总动能:
        W = weight * m * sum(v^2) / 2
    """
    return 0.5 * weight * m * np.sum(v**2)


def momentum_particles(v: np.ndarray, weight: float,
                         m: float = ELECTRON_MASS) -> float:
    """
    总动量:
        P = weight * m * sum(v)
    """
    return weight * m * np.sum(v)


def center_of_mass(x: np.ndarray, weight: np.ndarray = None) -> float:
    """
    质心位置:
        x_cm = sum(w_i * x_i) / sum(w_i)
    """
    if weight is None:
        return np.mean(x)
    return np.sum(weight * x) / (np.sum(weight) + 1e-300)


def velocity_dispersion(v: np.ndarray, weight: np.ndarray = None) -> float:
    """
    速度色散:
        sigma_v = sqrt(<v^2> - <v>^2)
    """
    if weight is None:
        return np.std(v)
    w_sum = np.sum(weight)
    v_mean = np.sum(weight * v) / (w_sum + 1e-300)
    v2_mean = np.sum(weight * v**2) / (w_sum + 1e-300)
    return np.sqrt(max(v2_mean - v_mean**2, 0.0))


def load_two_stream(Np_each: int, L: float, v_beam: float,
                      v_th: float, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    双流不稳定性初始加载:

    两束电子, 速度分别为 +v_beam 和 -v_beam
    每束具有热速度 v_th
    """
    Np = 2 * Np_each
    x = load_particles_regular_1d(Np, L)

    rng = np.random.RandomState(seed)
    sigma = v_th / np.sqrt(2.0)
    v = np.zeros(Np)
    v[:Np_each] = v_beam + rng.randn(Np_each) * sigma
    v[Np_each:] = -v_beam + rng.randn(Np_each) * sigma

    return x, v


def load_bump_on_tail(Np: int, L: float, v_th: float,
                        v_beam: float, n_beam_frac: float = 0.1,
                        seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    尾 bump 不稳定性加载:

    主体麦克斯韦 + 少量高速束
    f(v) = (1-alpha)*f_M(v) + alpha*f_beam(v-v_beam)
    """
    rng = np.random.RandomState(seed)
    x = load_particles_regular_1d(Np, L)

    sigma = v_th / np.sqrt(2.0)
    sigma_beam = sigma * 0.5

    v = np.zeros(Np)
    Np_beam = int(Np * n_beam_frac)
    Np_main = Np - Np_beam

    v[:Np_main] = rng.randn(Np_main) * sigma
    v[Np_main:] = v_beam + rng.randn(Np_beam) * sigma_beam

    return x, v
