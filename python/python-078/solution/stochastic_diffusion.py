"""
stochastic_diffusion.py
血浆中微粒布朗运动与有效扩散系数计算

融合来源:
- 119_brownian_motion_simulation: M维布朗运动轨迹生成、均方位移统计、爱因斯坦关系验证
- 345_exm (brownian3): 3D布朗运动随机游走

科学背景:
在动脉血流中，血浆（占血液体积约55%）作为连续相，其中的脂蛋白、血小板、
药物纳米颗粒等微粒经历布朗运动。这种随机扩散影响：
1. 血液的有效粘度（Einstein粘度修正: μ_eff = μ_0(1 + 2.5φ)）
2. 药物在血管壁的传输与沉积
3. 低密度脂蛋白（LDL）向血管壁内皮的渗透

爱因斯坦关系（1905）:
    <r²> = 2 M D t
其中M为空间维数，D为扩散系数，t为时间。

在剪切流中，布朗扩散与对流输运的竞争可用Peclet数刻画：
    Pe = γ̇ R² / D
其中γ̇为剪切率，R为粒子半径。
"""

import numpy as np


# ======================================================================
# 来自 119_brownian_motion_simulation 的核心算法
# ======================================================================

def brownian_motion_simulation(m_dim: int, n_steps: int,
                               diffusion_coeff: float,
                               total_time: float,
                               seed: int = None) -> np.ndarray:
    """
    模拟M维空间中N步的布朗运动轨迹。

    数学模型:
    每一步的位移 dX 服从多维正态分布:
        dX ~ N(0, σ² I_M),  σ² = 2 M D Δt

    其中:
        Δt = T / (N-1)
        D: 扩散系数 [m²/s]
        M: 空间维数

    参数:
        m_dim: 空间维数（1, 2, 或 3）
        n_steps: 时间步数
        diffusion_coeff: 扩散系数 D
        total_time: 总模拟时间 T [s]
        seed: 随机种子（可重复性）

    返回:
        x: (M, N) 位置矩阵，x[:,0]=0
    """
    if m_dim < 1 or n_steps < 2:
        raise ValueError("Invalid simulation parameters")
    if diffusion_coeff <= 0 or total_time <= 0:
        raise ValueError("Physical parameters must be positive")

    if seed is not None:
        np.random.seed(seed)

    dt = total_time / (n_steps - 1)
    # 步长标准差: s = sqrt(2 * M * D * dt)
    step_std = np.sqrt(2.0 * m_dim * diffusion_coeff * dt)

    # 生成各向同性随机方向 + 步长
    if m_dim == 1:
        dx = step_std * np.random.randn(1, n_steps - 1)
    else:
        # 先生成标准正态随机向量，再归一化方向并缩放步长
        a = np.random.randn(m_dim, n_steps - 1)
        norms = np.linalg.norm(a, axis=0, keepdims=True)
        norms = np.where(norms < 1e-15, 1.0, norms)
        directions = a / norms
        step_sizes = step_std * np.random.randn(1, n_steps - 1)
        dx = directions * step_sizes

    x = np.zeros((m_dim, n_steps))
    x[:, 1:] = np.cumsum(dx, axis=1)
    return x


def brownian_displacement_simulation(k_trials: int, n_steps: int,
                                     m_dim: int, diffusion_coeff: float,
                                     total_time: float,
                                     seed: int = None) -> np.ndarray:
    """
    重复K次布朗运动模拟，计算位移平方统计量 DSQ(K, N)。

    用于验证爱因斯坦关系: mean(DSQ(:,j)) ≈ 2 M D t_j

    参数:
        k_trials: 重复实验次数
        n_steps: 每实验的步数
        m_dim: 空间维数
        diffusion_coeff: 扩散系数
        total_time: 总时间
        seed: 随机种子

    返回:
        dsq: (K, N) 每次实验每个时间点的位移平方
    """
    if k_trials < 1:
        raise ValueError("k_trials must be positive")

    if seed is not None:
        np.random.seed(seed)

    dsq = np.zeros((k_trials, n_steps))
    for k in range(k_trials):
        traj = brownian_motion_simulation(m_dim, n_steps, diffusion_coeff,
                                          total_time, seed=None)
        dsq[k, :] = np.sum(traj ** 2, axis=0)
    return dsq


def verify_einstein_relation(dsq: np.ndarray, m_dim: int,
                             diffusion_coeff: float, total_time: float) -> dict:
    """
    验证爱因斯坦关系 <r²> = 2 M D t。

    返回统计指标:
        - slope: 均方位移-时间曲线的斜率
        - theoretical_slope: 理论斜率 2MD
        - relative_error: 相对误差
    """
    k_trials, n_steps = dsq.shape
    mean_dsq = np.mean(dsq, axis=0)
    t = np.linspace(0, total_time, n_steps)

    # 线性回归（排除t=0）
    valid = t > 1e-12
    if np.count_nonzero(valid) < 2:
        return {"slope": 0.0, "theoretical_slope": 0.0, "relative_error": 1.0}

    slope, intercept = np.polyfit(t[valid], mean_dsq[valid], 1)
    theoretical = 2.0 * m_dim * diffusion_coeff
    rel_err = abs(slope - theoretical) / (theoretical + 1e-15)

    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "theoretical_slope": float(theoretical),
        "relative_error": float(rel_err)
    }


# ======================================================================
# 有效扩散与粘度修正（科学应用层）
# ======================================================================

def effective_diffusion_plasma(temperature_kelvin: float = 310.15,
                               particle_radius_nm: float = 100.0,
                               plasma_viscosity_pa_s: float = 0.0012) -> float:
    """
    使用Stokes-Einstein关系计算血浆中微粒的扩散系数。

    Stokes-Einstein方程:
        D = k_B T / (6 π η R)

    其中:
        k_B = 1.380649×10^{-23} J/K（玻尔兹曼常数）
        T: 绝对温度 [K]（人体 310.15 K = 37°C）
        η: 血浆粘度 [Pa·s]
        R: 粒子半径 [m]

    参数:
        temperature_kelvin: 温度 [K]
        particle_radius_nm: 粒子半径 [nm]
        plasma_viscosity_pa_s: 血浆动力粘度 [Pa·s]

    返回:
        D: 扩散系数 [m²/s]
    """
    k_B = 1.380649e-23
    R_m = particle_radius_nm * 1e-9
    if R_m <= 0 or plasma_viscosity_pa_s <= 0 or temperature_kelvin <= 0:
        raise ValueError("Physical parameters must be positive")
    return k_B * temperature_kelvin / (6.0 * np.pi * plasma_viscosity_pa_s * R_m)


def einstein_viscosity_correction(hematocrit: float) -> float:
    """
    Einstein粘度修正因子。

    对于稀悬浮液（红细胞体积分数 φ = 红细胞压积）:
        μ_eff / μ_0 = 1 + 2.5 φ + O(φ²)

    对于正常血液（Hct ≈ 0.40-0.45），采用二阶修正:
        μ_eff / μ_0 = 1 + 2.5 φ + 6.2 φ²

    参数:
        hematocrit: 红细胞压积（体积分数，0-1）

    返回:
        粘度比 μ_eff / μ_0
    """
    if not (0.0 <= hematocrit <= 1.0):
        raise ValueError("Hematocrit must be in [0, 1]")
    phi = hematocrit
    return 1.0 + 2.5 * phi + 6.2 * phi * phi


def peclet_number(shear_rate: float, particle_radius_nm: float,
                  diffusion_coeff: float) -> float:
    """
    计算Peclet数，表征布朗扩散与对流剪切之间的竞争。

    定义:
        Pe = γ̇ R² / D

    物理意义:
        Pe << 1: 布朗扩散主导，粒子分布均匀
        Pe >> 1: 对流剪切主导，粒子向低压区迁移

    参数:
        shear_rate: 剪切率 γ̇ [s^{-1}]
        particle_radius_nm: 粒子半径 [nm]
        diffusion_coeff: 扩散系数 [m²/s]
    """
    R = particle_radius_nm * 1e-9
    if diffusion_coeff <= 0:
        raise ValueError("Diffusion coefficient must be positive")
    return shear_rate * R * R / diffusion_coeff


def ldl_wall_flux_estimate(wss_pa: float, diffusion_coeff: float,
                           wall_permeability: float = 1e-8) -> float:
    """
    估算低密度脂蛋白（LDL）向血管壁的通量。

    模型假设:
    1. WSS增加会改变内皮细胞间隙大小，从而改变渗透性
    2. 通量 J = P_w · C_lumen，其中P_w为有效壁面渗透性
    3. WSS对渗透性的影响: P_w(WSS) = P_0 · (1 + k · WSS)

    参数:
        wss_pa: 壁面剪切应力 [Pa]
        diffusion_coeff: LDL扩散系数 [m²/s]
        wall_permeability: 基础壁面渗透性 [m/s]

    返回:
        有效渗透性 [m/s]
    """
    k_wss = 0.05  # WSS对渗透性影响系数 [Pa^{-1}]
    P_eff = wall_permeability * (1.0 + k_wss * wss_pa)
    return max(P_eff, 0.0)
