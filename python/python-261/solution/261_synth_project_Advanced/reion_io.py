"""
reion_io.py
===========
输入/输出工具与结果汇总

本模块提供:
  1. 模拟结果的文本输出 (终端汇总)
  2. 数值结果保存为文本文件 (无图像)
  3. 结果摘要报告

对应种子项目:
  - 1155_anhkiet120206-lgtm_Supplementary-Materials-for-EEIO-Analysis
    (IO 分析汇总 → 模拟结果输出)
"""

import os
import numpy as np
from reion_cosmology import tau_e_obs, sigma_tau


def print_simulation_banner(scenario_name):
    """打印模拟启动横幅."""
    print()
    print("*" * 70)
    print("*  计算宇宙学: 再电离历史数值模拟")
    print("*  高阶有限差分与 IMEX 稳定性分析")
    print("*  场景: %s" % scenario_name)
    print("*" * 70)
    print()


def print_result_summary(result, diagnostics=None):
    """打印模拟结果汇总.

    Parameters
    ----------
    result : dict
    diagnostics : dict, optional
    """
    print("\n" + "-" * 60)
    print("模拟结果汇总:")
    print("-" * 60)
    print("  场景名称          : %s" % result.get("scenario_name", "unknown"))
    print("  最终红移 z_final  : %.3f" % float(result["z_array"][-1]))
    print("  平均电离 <xHII>   : %.4f" % float(result["mean_xHII"][-1]))
    print("  有效光学深度 <tau>: %.4f" % float(result["mean_tau_eff"]))
    print("  中点再电离 z_{1/2}: %.2f" % float(result["z_half_reionization"]))
    print("  Planck tau 观测值 : %.4f ± %.4f" % (tau_e_obs, sigma_tau))

    # 与 Planck 比较
    tau_model = result["mean_tau_eff"]
    chi2 = ((tau_model - tau_e_obs) / max(sigma_tau, 1.0e-10)) ** 2
    print("  chi^2 (vs Planck) : %.3f" % chi2)

    if diagnostics:
        print("  max(xHII)          : %.4f" % diagnostics.get("max_xHII", 0.0))
        print("  min(xHII)          : %.4f" % diagnostics.get("min_xHII", 0.0))
        print("  std(xHII)          : %.4f" % diagnostics.get("std_xHII", 0.0))
        print("  守恒性检查         : %s" % (
            "通过" if diagnostics.get("conservation_check", False) else "失败"))
    print("-" * 60)


def save_results_to_text(result, output_dir="results", filename="reion_output.txt"):
    """将模拟结果保存为文本文件.

    Parameters
    ----------
    result : dict
    output_dir : str
    filename : str
    """
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    filepath = os.path.join(output_dir, filename)
    with open(filepath, "w") as f:
        f.write("# 再电离历史模拟结果\n")
        f.write("# 场景: %s\n" % result.get("scenario_name", "unknown"))
        f.write("# 列: z  mean_xHII  tau_eff\n")
        z_arr = result["z_array"]
        mean_x = result["mean_xHII"]
        tau_h = result["tau_eff_history"]
        for i in range(len(z_arr)):
            f.write("%.6f  %.8e  %.8e\n" % (z_arr[i], mean_x[i], tau_h[i]))
    print("  结果已保存到: %s" % filepath)


def save_final_profile(result, output_dir="results",
                       filename="reion_profile.txt"):
    """保存最终电离分数分布."""
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    filepath = os.path.join(output_dir, filename)
    xHII = result["xHII_final"]
    with open(filepath, "w") as f:
        f.write("# 最终电离分数分布\n")
        f.write("# 列: cell_index  xHII\n")
        for i in range(len(xHII)):
            f.write("%d  %.8e\n" % (i, xHII[i]))
    print("  分布已保存到: %s" % filepath)


def print_scenario_comparison(summaries):
    """打印场景比较表."""
    print("\n" + "=" * 70)
    print("场景比较:")
    print("-" * 70)
    print("%-25s %10s %10s %10s %12s" % (
        "场景", "<xHII>", "tau_eff", "z_{1/2}", "chi^2"))
    print("-" * 70)
    for name, s in summaries.items():
        if "error" not in s:
            chi2 = ((s["mean_tau_eff"] - tau_e_obs) / sigma_tau) ** 2
            print("%-25s %10.4f %10.4f %10.2f %12.3f" % (
                name[:25], s["mean_xHII_final"], s["mean_tau_eff"],
                s["z_half_reionization"], chi2))
    print("=" * 70)


def print_stability_report(stability_results):
    """打印稳定性分析报告."""
    print("\n" + "=" * 70)
    print("稳定性分析报告:")
    print("-" * 70)
    for scheme, info in stability_results.items():
        status = "稳定" if info["is_stable"] else "不稳定"
        print("  方案 %-15s: %s  (max|G|=%.4f, growth=%.4e)" % (
            scheme, status, info["max_abs_G"], info["max_growth_rate"]))
    print("=" * 70)


def print_optimization_report(opt_result):
    """打印参数优化报告."""
    print("\n" + "=" * 70)
    print("参数优化报告:")
    print("-" * 70)
    print("  最优 gamma_uv_b    : %.4e" % opt_result["gamma_uv_b"])
    print("  最小 chi^2         : %.4f" % opt_result["chi2_min"])
    print("  收敛               : %s" % (
        "是" if opt_result["converged"] else "否"))
    print("  迭代次数           : %d" % opt_result["n_iter"])
    print("=" * 70)


def final_report(all_summaries=None, stability=None, optimization=None,
                 cholesky_info=None):
    """生成最终综合报告."""
    print("\n" + "#" * 70)
    print("#  最终综合报告: 计算宇宙学再电离模拟")
    print("#" * 70)

    if all_summaries:
        print_scenario_comparison(all_summaries)

    if stability:
        print_stability_report(stability)

    if optimization:
        print_optimization_report(optimization)

    if cholesky_info:
        print("\n" + "-" * 70)
        print("Cholesky 统计推断:")
        print("  参数协方差矩阵对数行列式: %.4f" % cholesky_info.get("log_det", 0.0))
        print("  参数不确定性 (1-sigma):")
        for name, val in cholesky_info.get("uncertainties", {}).items():
            print("    %-20s: %.4e" % (name, val))
        print("-" * 70)

    print("\n" + "#" * 70)
    print("#  模拟完成")
    print("#" * 70)
