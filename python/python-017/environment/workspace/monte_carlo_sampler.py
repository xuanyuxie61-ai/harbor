"""
蒙特卡洛热涨落采样模块
融合来源: 1092_snakes_and_ladders_simulation (随机游走统计)
         + 449_full_deck_simulation (批量蒙特卡洛统计)
         + 292_disk_distance (圆盘上随机采样)

功能:
- 对多铁性材料序参量场进行 Metropolis-Hastings 蒙特卡洛热采样
- 模拟热涨落导致的畴壁运动与极化/磁化反转
- 统计磁电耦合响应的涨落-耗散特性
- 计算关联函数与磁电系数的温度依赖

科学背景:
    在有限温度下，Landau 自由能需加入熵贡献:
        F_eff = F_landau - T S
    Metropolis 准则:
        接受概率 p = min(1, exp(-ΔE / (k_B T)))
    其中 ΔE 为单次蒙特卡洛步的能量变化。

    磁电响应系数的涨落公式 (Kubo 型):
        α_{ME} = (1/(k_B T V)) [ ⟨P M⟩ - ⟨P⟩⟨M⟩ ]
"""

import numpy as np
from typing import Callable, Tuple, Optional, List


def disk_unit_sample(n: int = 1, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """
    在单位圆盘内均匀随机采样点。
    源自 disk_distance 中 disk_unit_sample 的算法思想:
    用极坐标 (r, θ)，其中 r = √u, u~U(0,1)。
    """
    if rng is None:
        rng = np.random.default_rng()
    u = rng.random(n)
    theta = 2.0 * np.pi * rng.random(n)
    r = np.sqrt(u)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    if n == 1:
        return np.array([x[0], y[0]])
    return np.column_stack([x, y])


def disk_distance_stats(n_samples: int = 1000,
                        rng: Optional[np.random.Generator] = None) -> Tuple[float, float]:
    """
    估计单位圆盘上两随机点距离的均值与方差。
    源自 disk_distance 中 disk_distance_stats。
    在多铁性模拟中，用于标定畴壁热运动的空间尺度。
    """
    if rng is None:
        rng = np.random.default_rng()
    distances = np.zeros(n_samples)
    for i in range(n_samples):
        p = disk_unit_sample(1, rng)
        q = disk_unit_sample(1, rng)
        distances[i] = np.linalg.norm(p - q)
    mu = np.mean(distances)
    var = np.var(distances, ddof=1) if n_samples > 1 else 0.0
    return mu, var


class MetropolisMCSampler:
    """
    Metropolis-Hastings 蒙特卡洛采样器，用于多铁性序参量场。
    """

    def __init__(self, temperature: float, kB: float = 1.380649e-23,
                 rng_seed: Optional[int] = None):
        self.T = temperature
        self.kB = kB
        self.beta = 1.0 / (kB * temperature) if temperature > 0 else 1e20
        self.rng = np.random.default_rng(rng_seed)

    def propose_move(self, state: np.ndarray, amplitude: float) -> np.ndarray:
        """
        提出一个新状态: 在圆盘形邻域内扰动一个随机像素。
        融合 snakes_and_ladders 中随机移动 + disk_distance 中圆盘采样思想。
        """
        new_state = state.copy()
        idx = self.rng.integers(0, len(state))
        # 圆盘型扰动幅度
        delta = disk_unit_sample(1, self.rng)[0] * amplitude
        new_state[idx] += delta
        return new_state

    def acceptance_probability(self, dE: float) -> float:
        """Metropolis 接受概率。"""
        if dE < 0:
            return 1.0
        return np.exp(-self.beta * dE)

    def sample(self, initial_state: np.ndarray,
               energy_func: Callable[[np.ndarray], float],
               n_steps: int = 1000,
               amplitude: float = 0.1,
               burn_in: int = 100) -> Tuple[np.ndarray, List[float], List[float]]:
        """
        执行蒙特卡洛采样。

        参数:
            initial_state: 初始状态 (展平)
            energy_func:   能量函数 E(state)
            n_steps:       MC 步数
            amplitude:     扰动幅度
            burn_in:        burn-in 步数

        返回:
            final_state: 最终状态
            energies:    能量历史
            observables: 观测量历史（这里用状态总和作为简单观测量）
        """
        state = initial_state.copy()
        E_current = energy_func(state)
        energies = []
        observables = []

        accepted = 0
        for step in range(n_steps + burn_in):
            new_state = self.propose_move(state, amplitude)
            E_new = energy_func(new_state)
            dE = E_new - E_current

            if self.rng.random() < self.acceptance_probability(dE):
                state = new_state
                E_current = E_new
                accepted += 1

            if step >= burn_in:
                energies.append(E_current)
                observables.append(float(np.sum(state)))

        acceptance_rate = accepted / (n_steps + burn_in)
        # 边界鲁棒性: 若接受率过低，记录警告但不中断
        if acceptance_rate < 0.01:
            pass  # 可在日志中记录

        return state, energies, observables


def batch_monte_carlo_statistics(
    n_batches: int, n_per_batch: int,
    initial_state: np.ndarray,
    energy_func: Callable[[np.ndarray], float],
    temperature: float,
    amplitude: float = 0.1
) -> dict:
    """
    批量蒙特卡洛统计，融合 full_deck_simulation 中批量统计思想
    与 snakes_and_ladders 中多批次模拟思想。

    返回统计字典，包含:
        - energy_mean, energy_std
        - susceptibility (磁化率)
        - specific_heat
    """
    sampler = MetropolisMCSampler(temperature)
    batch_energies = []
    batch_obs = []

    for b in range(n_batches):
        state, energies, obs = sampler.sample(
            initial_state, energy_func, n_steps=n_per_batch,
            amplitude=amplitude, burn_in=n_per_batch // 5
        )
        batch_energies.append(np.mean(energies))
        batch_obs.append(np.mean(obs))

    stats = {
        'energy_mean': np.mean(batch_energies),
        'energy_std': np.std(batch_energies, ddof=1),
        'obs_mean': np.mean(batch_obs),
        'obs_std': np.std(batch_obs, ddof=1),
        'susceptibility': np.var(batch_obs, ddof=1) / (sampler.kB * temperature),
        'specific_heat': np.var(batch_energies, ddof=1) / (sampler.kB * temperature ** 2),
    }
    return stats


def compute_correlation_function(field: np.ndarray, max_r: int = 20) -> np.ndarray:
    """
    计算二维场的径向关联函数:
        C(r) = ⟨u(0) u(r)⟩ / ⟨u(0)^2⟩
    用于分析多铁性畴结构的关联长度。
    """
    ny, nx = field.shape
    center_y, center_x = ny // 2, nx // 2
    C = np.zeros(max_r + 1)
    counts = np.zeros(max_r + 1)

    for dy in range(-max_r, max_r + 1):
        for dx in range(-max_r, max_r + 1):
            r = int(np.round(np.sqrt(dx * dx + dy * dy)))
            if r > max_r:
                continue
            y = center_y + dy
            x = center_x + dx
            if 0 <= y < ny and 0 <= x < nx:
                C[r] += field[center_y, center_x] * field[y, x]
                counts[r] += 1

    counts = np.maximum(counts, 1)
    C /= counts
    norm = C[0] if abs(C[0]) > 1e-20 else 1.0
    C /= norm
    return C
