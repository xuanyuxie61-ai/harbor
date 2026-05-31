"""
filament_kinetics.py
核蛋白丝组装动力学模块

融合原项目:
  - 517_henon_orbit: Henon映射非线性动力系统 → DNA-蛋白质复合体的构象混沌采样

科学背景:
  RAD51核蛋白丝的组装是一个协同结合过程，可用非线性动力学描述。
  每个RAD51单体结合后，改变DNA构象，促进下一个单体结合——正反馈。

  将核蛋白丝组装建模为离散映射系统:
    状态变量 x_n = [结合覆盖率, 局部曲率, 构象序参数]

  借鉴Henon映射的二次非线性:
    x_{n+1} = x_n cos(α) - (y_n - x_n²) sin(α)
    y_{n+1} = x_n sin(α) + (y_n - x_n²) cos(α)

  在DNA修复语境下，重新诠释为:
    θ_{n+1} = θ_n·c - (κ_n - a·θ_n²)·s
    κ_{n+1} = θ_n·s + (κ_n - a·θ_n²)·c
    其中 θ 为局部结合覆盖率，κ 为DNA曲率，c=cos(α), s=sin(α), a 为协同性参数
"""

import numpy as np


def henon_filament_map(theta: float, kappa: float,
                       c: float = 0.95, a: float = 0.3) -> tuple:
    """
    基于Henon映射的核蛋白丝构象映射 (单步)

    参数:
        theta: 局部结合覆盖率 [0, 1]
        kappa: DNA局部曲率 (nm⁻¹)
        c: cos(α), 动力学参数
        a: 协同性参数

    Returns:
        theta_new, kappa_new
    """
    if c < -1.0 or c > 1.0:
        raise ValueError("c must be in [-1, 1]")
    s = np.sqrt(max(1.0 - c * c, 0.0))

    theta_new = theta * c - (kappa - a * theta ** 2) * s
    kappa_new = theta * s + (kappa - a * theta ** 2) * c

    # 边界处理
    theta_new = np.clip(theta_new, 0.0, 1.0)
    return theta_new, kappa_new


def simulate_filament_assembly(num_monomers: int = 50,
                               num_realizations: int = 100,
                               c: float = 0.95,
                               a: float = 0.3,
                               theta_init: float = 0.05) -> dict:
    """
    模拟核蛋白丝的随机组装过程

    多个初始条件下运行Henon型映射，统计平均组装动力学

    参数:
        num_monomers: 单体步数
        num_realizations: 随机实现次数
        c, a: 映射参数
        theta_init: 初始覆盖率

    Returns:
        dict 包含 theta_mean, theta_std, kappa_mean, kappa_std
    """
    if num_monomers <= 0 or num_realizations <= 0:
        raise ValueError("Counts must be positive")

    theta_traj = np.zeros((num_realizations, num_monomers), dtype=float)
    kappa_traj = np.zeros((num_realizations, num_monomers), dtype=float)

    for r in range(num_realizations):
        theta = theta_init + 0.02 * np.random.randn()
        theta = np.clip(theta, 0.0, 1.0)
        kappa = 0.1 + 0.05 * np.random.randn()

        theta_traj[r, 0] = theta
        kappa_traj[r, 0] = kappa

        for n in range(1, num_monomers):
            theta, kappa = henon_filament_map(theta, kappa, c, a)
            theta_traj[r, n] = theta
            kappa_traj[r, n] = kappa

    return {
        'theta_mean': np.mean(theta_traj, axis=0),
        'theta_std': np.std(theta_traj, axis=0),
        'kappa_mean': np.mean(kappa_traj, axis=0),
        'kappa_std': np.std(kappa_traj, axis=0),
        'coverage_final': np.mean(theta_traj[:, -1]),
        'cooperativity': a
    }


def compute_filament_stability_energy(coverage: float,
                                       bending_modulus: float = 200.0,
                                       binding_energy_per_monomer: float = -35.0) -> float:
    """
    计算核蛋白丝的弹性稳定性能量

    总能量:
        E_total = N·E_bind + E_bend
    其中 E_bend = (1/2)·B·L·κ², B 为弯曲模量, κ 为曲率

    假设 κ ∝ coverage² (结合越满，DNA越直，曲率越小)
        κ = κ_0 (1 - coverage)
    则:
        E_bend = (1/2) B L κ_0² (1 - coverage)²

    Returns:
        energy: 单位长度的能量 (kJ/mol/nm)
    """
    if coverage < 0 or coverage > 1:
        raise ValueError("coverage must be in [0, 1]")
    if bending_modulus < 0:
        raise ValueError("bending modulus must be non-negative")

    kappa_0 = 1.0  # 1/nm (未结合时的曲率)
    kappa = kappa_0 * (1.0 - coverage)
    e_bend = 0.5 * bending_modulus * kappa ** 2
    e_bind = coverage * binding_energy_per_monomer
    return e_bind + e_bend


def cooperativity_index_from_trajectory(theta_mean: np.ndarray) -> float:
    """
    从组装轨迹计算协同性指数

    Hill系数近似:
        n_H ≈ log(θ/(1-θ)) / log([L]/[L]_{50})

    简化: 计算覆盖率从0.1到0.9所需的步数比例
    """
    if len(theta_mean) < 2:
        return 1.0

    idx_10 = -1
    idx_90 = -1
    for i, th in enumerate(theta_mean):
        if idx_10 < 0 and th >= 0.1:
            idx_10 = i
        if th >= 0.9:
            idx_90 = i
            break

    if idx_10 < 0 or idx_90 < 0 or idx_90 == idx_10:
        return 1.0

    # 步数比越小，协同性越高
    ratio = (idx_90 - idx_10) / len(theta_mean)
    n_hill = max(1.0, 5.0 * (1.0 - ratio))
    return n_hill


def logistic_growth_model(t: np.ndarray, K: float = 1.0,
                          r: float = 0.1, t0: float = 50.0) -> np.ndarray:
    """
    Logistic增长模型描述核蛋白丝组装动力学

    公式:
        θ(t) = K / (1 + exp(-r(t-t0)))

    参数:
        t: 时间数组
        K: 饱和覆盖率
        r: 增长率
        t0: 半饱和时间
    """
    t = np.asarray(t, dtype=float)
    return K / (1.0 + np.exp(-r * (t - t0)))
