#!/usr/bin/env python3
"""
main.py
=======
计算宇宙学: 再电离历史数值模拟 —— 统一入口

本项目模拟宇宙再电离时期 (z ~ 20 → 6) 的电离分数场演化, 采用:
  - 高阶有限差分 (4阶精度) 空间离散化
  - IMEX Runge-Kutta 时间积分 (处理刚性的复合项)
  - Mackey-Glass 型延迟复合反馈
  - 辐射传输 BVP 求解器
  - Cholesky 分解参数不确定性量化
  - von Neumann 稳定性分析

运行方式:
    python main.py

输出:
    - 终端: 模拟结果汇总, 稳定性报告, 优化结果
    - 文件: results/reion_output.txt, results/reion_profile.txt

作者: 计算宇宙学合成项目
日期: 2026-06
"""

import os
import sys
import time
import numpy as np

# 保证当前目录在 Python path 中
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from reion_cosmology import (
    Hubble_parameter, t_cosmic, T_CMB, nH_bar,
    tau_e_obs, sigma_tau, Mpc_to_cm, Gyr_to_s,
)
from reion_grid import (
    build_uniform_grid, build_redshift_array, grid_vonNeumann_wavenumbers,
)
from reion_finite_difference import (
    fd_laplacian_matrix, fd_first_derivative, fd_second_derivative,
    fornberg_weights, spectral_derivative,
)
from reion_transfer import (
    radiative_transfer_BVP, effective_optical_depth,
    mean_free_path_evolution, SIGMA_HI_0,
)
from reion_recombination import (
    alpha_B_hydrogen, alpha_B_helium, clumping_factor,
    mackey_glass_recombination, multi_species_kinetics,
)
from reion_imex_integrator import (
    get_imex_scheme, simple_imex_step, thomas_periodic,
)
from reion_cholesky import (
    cholesky_factor_simple, log_determinant, sample_from_covariance,
    fisher_information_reionization,
)
from reion_stability import (
    compare_schemes, cfl_critical, stability_growth_rate,
    laplacian_symbol_4th, spectral_radius_stability,
)
from reion_brent_optimizer import (
    brent_optimize, chi_squared_reionization,
    optimize_reionization_params,
)
from reion_solver import (
    run_single_simulation, run_simulation_with_diagnostics,
    simulate_multi_species, default_kinetic_params,
)
from reion_scenarios import (
    run_all_scenarios, analyze_parameter_sensitivity, list_scenarios,
)
from reion_io import (
    print_simulation_banner, print_result_summary,
    save_results_to_text, save_final_profile,
    print_stability_report, print_optimization_report,
    final_report,
)


def phase_banner(phase_name):
    """打印阶段横幅."""
    print()
    print("=" * 70)
    print("  阶段: %s" % phase_name)
    print("=" * 70)


def run_stability_analysis():
    """运行数值稳定性分析."""
    phase_banner("数值稳定性分析")

    N_grid = 64
    L_comoving = 50.0 * Mpc_to_cm
    dx_comoving = L_comoving / N_grid
    gamma_diff = 1.0e23  # 扩散系数 [cm^2/s]

    # 构建波数数组
    k_arr = grid_vonNeumann_wavenumbers(N_grid, L_comoving)

    # 估计临界时间步
    dt_max, cfl_max = cfl_critical(gamma_diff, dx_comoving,
                                   order=4, scheme="explicit")
    dt_use = 0.5 * min(dt_max, 1.0 * Gyr_to_s)
    dt_use = max(dt_use, 1.0e10)

    print("  网格: N=%d, dx=%.3e cm (共动)" % (N_grid, dx_comoving))
    print("  扩散系数: gamma=%.3e cm^2/s" % gamma_diff)
    print("  临界时间步: dt_max=%.3e s" % dt_max)
    print("  使用 dt = %.3e s" % dt_use)
    print()

    # 多方案稳定性比较
    results = compare_schemes(k_arr, dx_comoving, dt_use, gamma_diff)

    # 矩阵谱半径分析
    D2 = fd_laplacian_matrix(N_grid, dx_comoving, accuracy=4, periodic=True)
    rho, dt_crit = spectral_radius_stability(D2, dt_use, gamma_diff,
                                             scheme="explicit")
    print("  矩阵谱半径 rho = %.4f" % rho)
    print("  谱分析临界 dt = %.3e s" % dt_crit)

    # 增长率
    omega_i = stability_growth_rate(k_arr, dx_comoving, dt_use,
                                    gamma_diff, scheme="explicit")
    print("  最大扰动增长率 = %.4e s^-1" % float(np.max(omega_i)))
    print()

    print_stability_report(results)
    return results


def run_parameter_optimization():
    """运行参数优化."""
    phase_banner("参数优化 (Brent 搜索)")

    # 观测数据
    z_obs = np.array([6.0, 7.0, 8.0])
    tau_obs = np.array([tau_e_obs])
    sigma = sigma_tau

    # Brent 优化 gamma_uv_b
    def objective(log_gamma):
        gamma = np.exp(log_gamma)
        chi2 = chi_squared_reionization(
            gamma, z_obs, tau_obs, sigma,
            None, N_grid=32, N_steps=30)
        return chi2

    # 在 log 空间搜索
    log_range = (np.log(1.0e-15), np.log(1.0e-11))
    log_opt, chi2_min, conv, n_iter = brent_optimize(
        log_range[0], log_range[1], objective, tol=1.0e-4, max_iter=30)

    gamma_opt = np.exp(log_opt)

    result = {
        "gamma_uv_b": gamma_opt,
        "chi2_min": chi2_min,
        "converged": conv,
        "n_iter": n_iter,
    }

    print("  最优 gamma_uv_b = %.4e" % gamma_opt)
    print("  最小 chi^2 = %.4f" % chi2_min)
    print("  收敛: %s, 迭代: %d" % ("是" if conv else "否", n_iter))
    print()

    print_optimization_report(result)
    return result


def run_cholesky_analysis():
    """运行 Cholesky 统计推断."""
    phase_banner("Cholesky 统计推断")

    # 构造 Fisher 信息矩阵
    def model_func(z, params):
        # 简化的 xHII 模型
        gamma = params[0]
        f_esc = params[1]
        # 经验模型: xHII(z) = 1 - exp(-gamma * f_esc * ((7/(1+z))^3 - 1))
        ratio = (7.0 / (1.0 + np.maximum(z, 5.5))) ** 3
        xHII = 1.0 - np.exp(-gamma * f_esc * 1.0e14 * (ratio - 0.5))
        return np.clip(xHII, 0.0, 1.0)

    z_data = np.array([5.5, 6.0, 6.5, 7.0, 7.5, 8.0])
    # 模拟观测 (带噪声)
    true_params = np.array([5.0e-14, 0.1])
    xHII_obs = model_func(z_data, true_params)
    rng = np.random.default_rng(42)
    xHII_obs += 0.02 * rng.standard_normal(len(z_data))
    xHII_obs = np.clip(xHII_obs, 0.01, 0.99)
    sigma_obs = np.full(len(z_data), 0.03)

    # Fisher 矩阵
    F = fisher_information_reionization(
        true_params, z_data, xHII_obs, model_func, sigma_obs)
    print("  Fisher 信息矩阵:")
    print("  [[%.4e, %.4e]," % (F[0, 0], F[0, 1]))
    print("   [%.4e, %.4e]]" % (F[1, 0], F[1, 1]))

    # 协方差矩阵 = F^{-1}
    try:
        cov = np.linalg.inv(F)
        # 使对称
        cov = 0.5 * (cov + cov.T)
        # 正则化
        cov += 1.0e-12 * np.eye(len(cov))
        # Cholesky 分解
        L = cholesky_factor_simple(cov)
        log_det = log_determinant(cov)
        print("  协方差矩阵对数行列式: %.4f" % log_det)

        # 参数不确定性
        uncertainties = {
            "gamma_uv_b": float(np.sqrt(max(cov[0, 0], 0.0))),
            "f_esc": float(np.sqrt(max(cov[1, 1], 0.0))),
        }
        print("  参数不确定性 (1-sigma):")
        for name, val in uncertainties.items():
            print("    %-15s: %.4e" % (name, val))

        # 抽样
        samples = sample_from_covariance(true_params, cov, 5, rng=rng)
        print("  参数后验抽样 (5个):")
        for i in range(5):
            print("    #%d: gamma=%.3e, f_esc=%.4f" % (
                i + 1, samples[i, 0], samples[i, 1]))

        cholesky_info = {
            "log_det": float(log_det),
            "uncertainties": uncertainties,
            "covariance": cov,
        }
    except Exception as e:
        print("  Cholesky 分解失败: %s" % str(e))
        cholesky_info = {"log_det": 0.0, "uncertainties": {}}

    return cholesky_info


def run_main_simulation():
    """运行主模拟."""
    phase_banner("主模拟 (Pop II 标准场景)")

    result, diagnostics = run_simulation_with_diagnostics(
        scenario_name="pop2_standard",
        N_grid=64,
        N_steps=80,
    )
    print_result_summary(result, diagnostics)

    # 保存结果
    save_results_to_text(result, output_dir="results")
    save_final_profile(result, output_dir="results")

    return result, diagnostics


def run_all_scenario_simulation():
    """运行所有场景."""
    phase_banner("多场景比较模拟")

    all_results, all_summaries, comparison = run_all_scenarios(
        N_grid=32,
        N_steps=40,
    )
    print("\n  最佳拟合场景: %s" % comparison.get("best_fit_scenario", "N/A"))
    print("  最佳 chi^2: %.4f" % comparison.get("best_chi2", 0.0))

    return all_results, all_summaries, comparison


def run_multi_species_simulation():
    """运行多物种模拟."""
    phase_banner("多物种电离动力学 (H + He)")

    result = simulate_multi_species(N_grid=32, N_steps=40)
    xHII_final = float(np.mean(result["xHII_history"][-1]))
    xHeII_final = float(np.mean(result["xHeII_history"][-1]))
    print("  <xHII>_final = %.4f" % xHII_final)
    print("  <xHeII>_final = %.4f" % xHeII_final)
    return result


def run_fd_verification():
    """验证有限差分精度."""
    phase_banner("有限差分精度验证")

    N = 64
    L = 2.0 * np.pi
    x = np.linspace(0, L, N, endpoint=False)
    dx = L / N

    # 测试函数: f(x) = sin(x) + 0.5*cos(2x)
    f = np.sin(x) + 0.5 * np.cos(2.0 * x)
    # 精确一阶导数
    df_exact = np.cos(x) - np.sin(2.0 * x)
    # 精确二阶导数
    d2f_exact = -np.sin(x) - 2.0 * np.cos(2.0 * x)

    # 2阶精度
    df_2 = fd_first_derivative(f, dx, accuracy=2, periodic=True)
    d2f_2 = fd_second_derivative(f, dx, accuracy=2, periodic=True)
    err_df_2 = float(np.max(np.abs(df_2 - df_exact)))
    err_d2f_2 = float(np.max(np.abs(d2f_2 - d2f_exact)))

    # 4阶精度
    df_4 = fd_first_derivative(f, dx, accuracy=4, periodic=True)
    d2f_4 = fd_second_derivative(f, dx, accuracy=4, periodic=True)
    err_df_4 = float(np.max(np.abs(df_4 - df_exact)))
    err_d2f_4 = float(np.max(np.abs(d2f_4 - d2f_exact)))

    # 谱方法 (参考解)
    df_spec = spectral_derivative(f, L)
    err_df_spec = float(np.max(np.abs(df_spec - df_exact)))

    # 4阶 Fornberg 权重验证
    pts = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    w = fornberg_weights(pts, 0.0, 1)
    print("  Fornberg 一阶导数权重 (5点, 4阶):")
    print("    " + ", ".join(["%.6f" % wi for wi in w]))
    print("  理论值: [0.0833, -0.6667, 0, 0.6667, -0.0833]")

    print()
    print("  有限差分精度 (N=%d, L=2*pi):" % N)
    print("    df (2阶): max err = %.4e" % err_df_2)
    print("    df (4阶): max err = %.4e" % err_df_4)
    print("    df (谱):  max err = %.4e" % err_df_spec)
    print("    d2f(2阶): max err = %.4e" % err_d2f_2)
    print("    d2f(4阶): max err = %.4e" % err_d2f_4)

    return {
        "err_df_2": err_df_2,
        "err_df_4": err_df_4,
        "err_df_spec": err_df_spec,
    }


def main():
    """主入口: 串联所有模拟阶段."""
    t_start = time.time()

    # 启动横幅
    print()
    print("#" * 70)
    print("#  计算宇宙学: 再电离历史数值模拟")
    print("#  高阶有限差分与 IMEX 稳定性分析 (小规模可复现实验)")
    print("#" * 70)
    print()
    print("  项目结构:")
    print("    reion_cosmology.py       : 物理与宇宙学常数")
    print("    reion_grid.py            : 计算网格生成")
    print("    reion_finite_difference.py: 高阶有限差分算子")
    print("    reion_transfer.py        : 辐射传输 BVP")
    print("    reion_recombination.py   : 复合动力学 (含延迟反馈)")
    print("    reion_imex_integrator.py : IMEX Runge-Kutta 积分")
    print("    reion_cholesky.py        : Cholesky 统计推断")
    print("    reion_stability.py       : von Neumann 稳定性分析")
    print("    reion_brent_optimizer.py : 参数优化")
    print("    reion_solver.py          : 核心求解器")
    print("    reion_scenarios.py       : 多场景模拟")
    print("    reion_io.py              : 输入/输出")
    print()

    # ============================================================
    # 阶段 1: 有限差分精度验证
    # ============================================================
    fd_verification = run_fd_verification()

    # ============================================================
    # 阶段 2: 稳定性分析
    # ============================================================
    stability = run_stability_analysis()

    # ============================================================
    # 阶段 3: 主模拟
    # ============================================================
    result, diagnostics = run_main_simulation()

    # ============================================================
    # 阶段 4: 多场景比较
    # ============================================================
    all_results, all_summaries, comparison = run_all_scenario_simulation()

    # ============================================================
    # 阶段 5: 参数优化
    # ============================================================
    opt_result = run_parameter_optimization()

    # ============================================================
    # 阶段 6: Cholesky 统计推断
    # ============================================================
    cholesky_info = run_cholesky_analysis()

    # ============================================================
    # 阶段 7: 多物种电离
    # ============================================================
    multi_species_result = run_multi_species_simulation()

    # ============================================================
    # 最终报告
    # ============================================================
    final_report(
        all_summaries=all_summaries,
        stability=stability,
        optimization=opt_result,
        cholesky_info=cholesky_info,
    )

    t_total = time.time() - t_start
    print("\n总计算时间: %.2f s" % t_total)
    print("所有模拟已成功完成.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
