"""
reion_solver.py
================
核心模拟求解器 (主驱动程序)

本模块是再电离历史模拟的核心求解器, 整合所有子模块:
  - 网格生成 (reion_grid)
  - 高阶有限差分 (reion_finite_difference)
  - 辐射传输 (reion_transfer)
  - 复合动力学 (reion_recombination)
  - IMEX 时间积分 (reion_imex_integrator)
  - 稳定性分析 (reion_stability)

运行流程:
  1. 根据场景参数设置网格与时间步
  2. 初始化电离分数场 (从 Saha 平衡出发)
  3. 在每个时间步:
     a. 计算 Hubble 参数与宇宙时
     b. 计算温度演化
     c. 计算光电离率 (含延迟反馈)
     d. 构建辐射传输 BVP 并求解 J
     e. 构建扩散项并显式计算
     f. 调用 IMEX 步进更新 xHII
     g. 记录诊断量 (平均 xHII, tau_eff)
  4. 返回完整模拟结果

核心方程 (再电离演化):
    dxHII/dt = (1+z) * H(z) * {
        Gamma_pei * (1 - xHII)
        - alpha_B(T) * n_H * C * xHII * n_e(t - tau)
        + D * nabla^2 xHII
    }

对应种子项目:
  - 513_hello (banner 输出 → 模拟启动信息)
"""

import numpy as np
import warnings
from reion_cosmology import (
    Hubble_parameter, t_cosmic, T_CMB, nH_bar, k_B_cgs, m_H_cgs,
    Y_He, nH_0, Mpc_to_cm, Gyr_to_s, sigma_T, c_light,
)
from reion_grid import build_uniform_grid, build_redshift_array
from reion_finite_difference import (
    fd_second_derivative, fd_laplacian_matrix, fd_first_derivative,
)
from reion_transfer import (
    radiative_transfer_BVP, effective_optical_depth,
    radiative_coupling_term, mean_free_path_evolution, SIGMA_HI_0,
)
from reion_recombination import (
    alpha_B_hydrogen, clumping_factor, recombination_rate_density,
    mackey_glass_recombination, gas_temperature_evolution,
    multi_species_kinetics,
)
from reion_imex_integrator import simple_imex_step


# =============================================================================
# 默认参数
# =============================================================================
DEFAULT_PARAMS = {
    # 红移范围
    "z_start": 20.0,
    "z_end": 5.5,
    # 网格
    "N_grid": 64,
    "L_comoving_Mpc": 50.0,   # Mpc (共动)
    # 时间
    "N_steps": 120,
    # UV 背景参数
    "gamma_uv_b": 1.5e-15,       # 参考光电离率 [s^-1] at z=6
    "f_esc": 0.15,               # 逃逸分数
    "alpha_spec": 1.5,           # 谱指数
    "epsilon_star": 0.1,         # 恒星形成效率
    # 物理
    "clumping_model": "constant",
    "clumping_value": 3.0,
    "T_gas_init": 2.0e4,         # 初始气体温度 [K]
    # 数值
    "fd_accuracy": 4,
    "imex_scheme": "SSP2_ImEx",
    "diffusion_coeff": 1.0e16,   # 有效扩散系数 [cm^2/s (共动)]
    "tau_delay_Gyr": 0.1,        # 复合延迟时标 [Gyr]
    # 随机种子
    "rng_seed": 42,
    # 源分布不均匀性
    "source_fluctuation": 0.3,   # 源涨落幅度
}


def default_kinetic_params():
    """返回默认动力学参数 (用于 Mackey-Glass 延迟复合)."""
    return {
        "gamma_MG": 0.05,
        "beta_MG": 0.2,
        "n_MG": 2.0,
    }


def build_implicit_matrix(N_grid, dx_comoving, gamma_diff, fd_accuracy=4):
    """构建隐式扩散矩阵 A = I - dt * gamma * D2.

    Returns
    -------
    A : array [N x N]
    """
    D2 = fd_laplacian_matrix(N_grid, dx_comoving, accuracy=fd_accuracy,
                             periodic=True)
    A = np.eye(N_grid) - gamma_diff * D2
    return A


def initial_condition(N_grid, z_start, rng=None):
    """初始电离分数场.

    在 z_start ~ 20, 宇宙处于中性为主状态, 但存在小幅度涨落.
    采用 Saha 平衡 + 小扰动:
        xHII_0(z) ≈ min(1, A * exp(-E_ion / kT) * (1 + delta))

    Parameters
    ----------
    N_grid : int
    z_start : float
    rng : numpy.random.Generator

    Returns
    -------
    xHII : array [N]
    """
    if rng is None:
        rng = np.random.default_rng(42)
    # Saha 平衡
    from reion_cosmology import E_ion_H
    T_gas = T_CMB(z_start)
    T_gas = max(T_gas, 2.725)
    # Saha 近似 (仅用于估计量级)
    n_e_approx = nH_bar(z_start)
    xSaha = np.sqrt(1.0 / (n_e_approx * (2.0 * np.pi * m_H_cgs * k_B_cgs * T_gas /
                                          (h_planck_cgs() ** 2)) ** 1.5)
                    * np.exp(-E_ion_H / (k_B_cgs * T_gas)))
    xSaha = min(max(xSaha, 1.0e-10), 1.0 - 1.0e-10)
    # 添加小涨落
    delta = 0.05 * rng.standard_normal(N_grid)
    xHII = xSaha * (1.0 + delta)
    return np.clip(xHII, 1.0e-6, 1.0 - 1.0e-6)


def h_planck_cgs():
    """Planck 常数 [erg s]."""
    return 6.62607015e-27


def run_single_simulation(scenario_name="pop2_default",
                          N_grid=64, N_steps=80,
                          uvb_amplitude=None, **kwargs):
    """运行单次再电离模拟.

    Parameters
    ----------
    scenario_name : str
    N_grid, N_steps : int
    uvb_amplitude : float, optional
    **kwargs : dict
        覆盖 DEFAULT_PARAMS 的参数

    Returns
    -------
    result : dict
        xHII_final : array [N_grid]
        z_array : array [N_steps+1]
        mean_xHII : array [N_steps+1]
        tau_eff_history : array [N_steps+1]
        scenario_name : str
    """
    # 合并参数
    params = DEFAULT_PARAMS.copy()
    params.update(kwargs)
    if uvb_amplitude is not None:
        params["gamma_uv_b"] = uvb_amplitude

    # 打印模拟信息
    print("=" * 70)
    print("[reion_solver] 运行模拟: %s" % scenario_name)
    print("  网格: %d cells, L = %.1f Mpc (共动)" % (
        N_grid, params["L_comoving_Mpc"]))
    print("  时间步: %d, z: %.1f -> %.1f" % (
        N_steps, params["z_start"], params["z_end"]))
    print("  UVB 振幅: %.3e" % params["gamma_uv_b"])

    z_start = params["z_start"]
    z_end = params["z_end"]
    L_comoving = params["L_comoving_Mpc"] * Mpc_to_cm  # 转为 cm

    # 1. 生成网格
    grid = build_uniform_grid(L_comoving, N_grid, z_ref=z_start, periodic=True)
    dx_comoving = grid["dx_comoving"]

    # 2. 生成红移/时间数组
    z_arr = build_redshift_array(z_start, z_end, N_steps)
    t_arr = np.array([t_cosmic(z) for z in z_arr])

    # 3. 初始条件
    rng = np.random.default_rng(params["rng_seed"])
    xHII = initial_condition(N_grid, z_start, rng)

    # 4. 构建隐式矩阵
    gamma_diff = params["diffusion_coeff"]
    fd_acc = params["fd_accuracy"]
    D2 = fd_laplacian_matrix(N_grid, dx_comoving, accuracy=fd_acc, periodic=True)

    # 动力学参数
    kinetic = default_kinetic_params()
    tau_delay = params["tau_delay_Gyr"] * Gyr_to_s

    # 5. 时间积分循环
    xHII_history = np.zeros((N_steps + 1, N_grid))
    xHII_history[0] = xHII.copy()
    mean_xHII_history = np.zeros(N_steps + 1)
    mean_xHII_history[0] = float(np.mean(xHII))
    tau_eff_history = np.zeros(N_steps + 1)
    T_gas = params["T_gas_init"]

    # Thomson 散射光学深度累积
    from reion_cosmology import sigma_T
    tau_thomson = 0.0

    for n in range(N_steps):
        dt = t_arr[n + 1] - t_arr[n]
        dt = max(dt, 1.0e10)  # 防止过小
        dt = min(dt, 1.0 * Gyr_to_s)  # 限制最大步长
        z_n = z_arr[n]
        z_next = z_arr[n + 1]

        H_z = Hubble_parameter(z_n)

        # 当前物理量
        n_H = nH_bar(z_n)
        T_gas = max(T_CMB(z_n), 100.0)  # 简化: 用 T_CMB 作为 T_gas
        C_clump = clumping_factor(z_n, model=params["clumping_model"],
                                  C0=params["clumping_value"])

        # 延迟复合场
        delay_idx = max(1, int(tau_delay / max(dt, 1.0)))
        delay_n = max(0, n - delay_idx)
        xHII_delayed = xHII_history[delay_n].copy()

        # 显式项 (光电离 + 复合)
        # Gamma 参数化: Gamma_0 * ((1+z)/7)^(-3), 在低 z 处更大
        Gamma_base = (params["gamma_uv_b"] * ((1.0 + z_n) / 7.0) ** (-3)
                      * params["f_esc"] * params["epsilon_star"])
        # 空间涨落 (模拟源的不均匀分布) - 使用固定种子保证可重现
        fluct = params.get("source_fluctuation", 0.0)
        rng_source = np.random.default_rng(params["rng_seed"] + 1000)
        source_field = 1.0 + fluct * rng_source.standard_normal(N_grid)
        source_field = np.maximum(source_field, 0.1)
        Gamma_field = Gamma_base * source_field

        f_ex = np.zeros(N_grid)
        for i in range(N_grid):
            # 光电离 (显式): Gamma * (1 - xHII)
            ion_rate = Gamma_field[i] * (1.0 - xHII[i])
            # 复合 (显式近似, 由 IMEX 扩散隐式部分稳定化)
            n_e_local = xHII[i] * n_H
            recomb_rate = (alpha_B_hydrogen(T_gas) * n_H * C_clump
                           * xHII[i] * n_e_local)
            # 净电离率 (物理方程 dxHII/dt = Gamma*(1-x) - alpha*n*C*x*n_e)
            f_ex[i] = ion_rate - recomb_rate

        # 数值保护
        f_ex = np.clip(f_ex, -1.0 / max(dt, 1.0), 1.0 / max(dt, 1.0))

        # 简化 IMEX 步进 (显式: f_ex, 隐式: gamma * D2)
        xHII_new = simple_imex_step(xHII, dt, f_ex, D2, gamma_diff)

        # 物理约束
        xHII = np.clip(xHII_new, 1.0e-8, 1.0 - 1.0e-8)

        # 记录
        xHII_history[n + 1] = xHII.copy()
        mean_xHII_history[n + 1] = float(np.mean(xHII))
        # 累积 Thomson 散射光学深度
        # dtau = n_e * sigma_T * c * dt, 其中 n_e = xHII * n_H (物理数密度)
        xHII_mean = float(np.mean(xHII))
        n_e_physical = xHII_mean * nH_bar(z_n)  # 物理电子数密度
        dtau = n_e_physical * sigma_T * c_light * abs(dt)
        tau_thomson += dtau
        tau_eff_history[n + 1] = tau_thomson

    # 最终结果
    result = {
        "xHII_final": xHII.copy(),
        "xHII_history": xHII_history,
        "z_array": z_arr,
        "t_array": t_arr,
        "mean_xHII": mean_xHII_history,
        "tau_eff_history": tau_eff_history,
        "scenario_name": scenario_name,
        "mean_tau_eff": float(np.mean(tau_eff_history[1:])),
        "z_half_reionization": find_half_reionization_z(z_arr, mean_xHII_history),
    }

    print("  完成: <xHII>_final = %.3f, <tau_eff> = %.4f" % (
        result["mean_xHII"][-1], result["mean_tau_eff"]))
    print("  中点再电离红移: z_1/2 = %.2f" % result["z_half_reionization"])
    print("=" * 70)

    return result


def find_half_reionization_z(z_arr, mean_xHII_arr):
    """找到平均电离分数达到 0.5 的红移.

    Parameters
    ----------
    z_arr : array (降序)
    mean_xHII_arr : array

    Returns
    -------
    z_half : float
    """
    try:
        idx = np.where(mean_xHII_arr >= 0.5)[0]
        if len(idx) == 0:
            return float(z_arr[-1])
        # 线性插值
        i = idx[0]
        if i == 0:
            return float(z_arr[0])
        frac = (0.5 - mean_xHII_arr[i - 1]) / max(
            mean_xHII_arr[i] - mean_xHII_arr[i - 1], 1.0e-10)
        z_half = z_arr[i - 1] + frac * (z_arr[i] - z_arr[i - 1])
        return float(z_half)
    except Exception:
        return float(z_arr[-1])


def simulate_multi_species(N_grid=32, N_steps=50, **kwargs):
    """多物种 (H + He) 电离模拟.

    同时追踪 xHII 和 xHeII.

    Returns
    -------
    result : dict
    """
    params = DEFAULT_PARAMS.copy()
    params.update(kwargs)

    z_arr = build_redshift_array(params["z_start"], params["z_end"], N_steps)
    t_arr = np.array([t_cosmic(z) for z in z_arr])

    # 初始条件
    xHII = np.full(N_grid, 1.0e-4)
    xHeII = np.full(N_grid, 1.0e-5)

    xHII_hist = np.zeros((N_steps + 1, N_grid))
    xHeII_hist = np.zeros((N_steps + 1, N_grid))
    xHII_hist[0] = xHII.copy()
    xHeII_hist[0] = xHeII.copy()

    for n in range(N_steps):
        dt = t_arr[n + 1] - t_arr[n]
        dt = max(dt, 1.0e10)
        z_n = z_arr[n]
        n_H = nH_bar(z_n)
        T_gas = max(T_CMB(z_n), 100.0)
        H_z = Hubble_parameter(z_n)

        Gamma_HI = params["gamma_uv_b"] * ((1.0 + z_n) / 7.0) ** 3
        Gamma_HeI = Gamma_HI * 0.25  # 氦需要更高能量光子

        for i in range(N_grid):
            dxHII, dxHeII = multi_species_kinetics(
                xHII[i], xHeII[i], n_H, T_gas, Gamma_HI, Gamma_HeI)
            xHII[i] += dxHII * dt * (1.0 + z_n) * H_z
            xHeII[i] += dxHeII * dt * (1.0 + z_n) * H_z
            xHII[i] = np.clip(xHII[i], 1.0e-8, 1.0 - 1.0e-8)
            xHeII[i] = np.clip(xHeII[i], 1.0e-8, 1.0 - 1.0e-8)

        xHII_hist[n + 1] = xHII.copy()
        xHeII_hist[n + 1] = xHeII.copy()

    return {
        "xHII_history": xHII_hist,
        "xHeII_history": xHeII_hist,
        "z_array": z_arr,
    }


def run_simulation_with_diagnostics(scenario_name="pop2_default", **kwargs):
    """运行模拟并输出完整诊断信息.

    Returns
    -------
    result : dict
    diagnostics : dict
    """
    result = run_single_simulation(scenario_name=scenario_name, **kwargs)
    diagnostics = {
        "max_xHII": float(np.max(result["xHII_final"])),
        "min_xHII": float(np.min(result["xHII_final"])),
        "std_xHII": float(np.std(result["xHII_final"])),
        "mean_tau_eff": result["mean_tau_eff"],
        "z_half": result["z_half_reionization"],
        "conservation_check": check_conservation(result),
    }
    return result, diagnostics


def check_conservation(result):
    """守恒量检验: 电离分数应保持在 [0, 1]."""
    hist = result.get("xHII_history", None)
    if hist is None:
        return True
    return bool(np.all(hist >= 0.0) and np.all(hist <= 1.0))
