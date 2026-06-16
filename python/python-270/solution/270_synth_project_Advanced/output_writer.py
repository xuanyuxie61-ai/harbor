"""
output_writer.py
================

结果输出到文本文件 (无可视化)。

本模块将所有计算结果输出为格式化的文本文件,
便于后续分析和复现。

输出文件格式
------------
1. parameters.txt: 模拟参数
2. energy_magnetization.txt: 能量/磁化时间序列
3. overlap_distribution.txt: P(q) 分布
4. stability_report.txt: 稳定性分析报告
5. spectral_analysis.txt: Hessian 谱分析
6. replica_exchange.txt: 副本交换统计
7. umbrella_wham.txt: WHAM 自由能结果
8. disorder_average.txt: 无序平均结果
"""

import os
import numpy as np
from typing import Dict, Optional
from datetime import datetime


# =====================================================================
#  输出目录管理
# =====================================================================

def ensure_output_dir(base_dir: str = "output") -> str:
    """确保输出目录存在"""
    os.makedirs(base_dir, exist_ok=True)
    return base_dir


# =====================================================================
#  参数输出
# =====================================================================

def write_parameters(params: Dict, filepath: str):
    """
    写入模拟参数。
    """
    with open(filepath, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("  Spin Glass Monte Carlo Simulation - Parameters\n")
        f.write("=" * 70 + "\n")
        f.write(f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("-" * 70 + "\n")
        for key, value in params.items():
            if isinstance(value, np.ndarray):
                f.write(f"  {key}: {value.tolist()}\n")
            elif isinstance(value, dict):
                f.write(f"  {key}:\n")
                for k2, v2 in value.items():
                    f.write(f"    {k2}: {v2}\n")
            else:
                f.write(f"  {key}: {value}\n")
        f.write("=" * 70 + "\n")


# =====================================================================
#  能量/磁化时间序列
# =====================================================================

def write_mc_results(results: Dict, filepath: str, label: str = ""):
    """
    写入 MC 模拟结果。
    """
    with open(filepath, 'w') as f:
        f.write(f"# Spin Glass MC Results {label}\n")
        f.write("# Columns: sweep  energy_density  magnetization_density  overlap  overlap_squared\n")

        n_sweeps = results["n_sweeps"]
        for i in range(n_sweeps):
            e = results["energy_density"][i]
            m = results["magnetization_density"][i]
            q = results["overlap"][i] if "overlap" in results else 0.0
            q2 = results["overlap_squared"][i] if "overlap_squared" in results else 0.0
            f.write(f"{i:6d}  {e:15.8f}  {m:15.8f}  {q:15.8f}  {q2:15.8f}\n")

        f.write("\n# Summary Statistics\n")
        f.write(f"# beta = {results['beta']:.6f}\n")
        f.write(f"# <E>/N = {results['mean_energy']:.8f}\n")
        f.write(f"# Var(E)/N = {results['var_energy']:.8f}\n")
        f.write(f"# <|m|> = {results['mean_mag']:.8f}\n")
        f.write(f"# <q> = {results['mean_q']:.8f}\n")
        f.write(f"# <q^2> = {results['mean_q2']:.8f}\n")
        f.write(f"# Binder cumulant g = {results['binder_cumulant']:.8f}\n")
        f.write(f"# chi_SG = {results['chi_SG']:.8f}\n")
        f.write(f"# C_V = {results['C_V']:.8f}\n")
        f.write(f"# Acceptance ratio = {results['acceptance_ratio']:.6f}\n")
        if "tau_int_energy" in results:
            f.write(f"# tau_int(E) = {results['tau_int_energy']:.4f}\n")
        if "tau_int_q" in results:
            f.write(f"# tau_int(q) = {results['tau_int_q']:.4f}\n")


# =====================================================================
#  稳定性分析报告
# =====================================================================

def write_stability_report(stability_results: Dict, filepath: str):
    """
    写入稳定性分析结果。
    """
    with open(filepath, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("  Stability Analysis Report\n")
        f.write("=" * 70 + "\n\n")

        for key, value in stability_results.items():
            f.write(f"  {key}: {value}\n")

        f.write("\n" + "-" * 70 + "\n")
        f.write("  Von Neumann Stability Analysis\n")
        f.write("-" * 70 + "\n")

        if "amplification_factors" in stability_results:
            af = stability_results["amplification_factors"]
            f.write(f"  Max |G(k)| (2nd order): {af.get('rho_max_2nd', 'N/A')}\n")
            f.write(f"  Max |G(k)| (4th order): {af.get('rho_max_4th', 'N/A')}\n")

        if "stability_limits" in stability_results:
            sl = stability_results["stability_limits"]
            f.write(f"  dt_crit (2nd order): {sl.get('dt_crit_2nd', 'N/A')}\n")
            f.write(f"  dt_crit (4th order): {sl.get('dt_crit_4th', 'N/A')}\n")

        f.write("=" * 70 + "\n")


# =====================================================================
#  谱分析输出
# =====================================================================

def write_spectral_analysis(spectral_result: Dict, filepath: str):
    """
    写入 Hessian 谱分析结果。
    """
    with open(filepath, 'w') as f:
        f.write("# Hessian Spectral Analysis\n")
        f.write("# \n")
        f.write(f"# lambda_min = {spectral_result['lambda_min']:.8f}\n")
        f.write(f"# lambda_max = {spectral_result['lambda_max']:.8f}\n")
        f.write(f"# n_negative_modes = {spectral_result['n_negative']}\n")
        f.write(f"# condition_number = {spectral_result['condition_number']:.4e}\n")
        f.write("#\n")
        f.write(f"# DOS moments:\n")
        for n, m in spectral_result['dos_moments'].items():
            f.write(f"#   M_{n} = {m:.8f}\n")
        f.write("#\n")

        f.write("# Eigenvalues (sorted ascending):\n")
        for i, ev in enumerate(spectral_result['eigenvalues']):
            f.write(f"{i:6d}  {ev:18.10f}\n")

        f.write("\n# DOS: omega  g(omega)\n")
        omega_grid = spectral_result['omega_grid']
        dos = spectral_result['dos']
        for i in range(len(omega_grid)):
            f.write(f"{omega_grid[i]:12.6f}  {dos[i]:15.8f}\n")


# =====================================================================
#  Replica Exchange 输出
# =====================================================================

def write_replica_exchange_results(re_results: Dict, filepath: str):
    """
    写入 replica exchange 结果。
    """
    with open(filepath, 'w') as f:
        f.write("# Replica Exchange (Parallel Tempering) Results\n")
        f.write("# \n")
        f.write(f"# Number of replicas R = {len(re_results['temperatures'])}\n")
        f.write(f"# Overall exchange ratio = {re_results['overall_exchange_ratio']:.4f}\n")
        f.write("#\n")
        f.write("# Temperatures:\n")
        for r, T in enumerate(re_results['temperatures']):
            f.write(f"#   T[{r}] = {T:.4f}  (beta = {re_results['betas'][r]:.4f})\n")
        f.write("#\n")

        f.write("# Exchange rate per round:\n")
        for i, rate in enumerate(re_results['exchange_rate_series']):
            f.write(f"{i:6d}  {rate:.4f}\n")

        f.write("\n# Energy per replica (last 10 rounds):\n")
        n_rounds = len(re_results['exchange_rate_series'])
        n_show = min(10, n_rounds)
        f.write("# round  " + "  ".join(f"T={T:.2f}" for T in re_results['temperatures']) + "\n")
        for i in range(n_rounds - n_show, n_rounds):
            energies = [re_results['energies'][r][i] for r in range(len(re_results['temperatures']))]
            f.write(f"{i:6d}  " + "  ".join(f"{e:10.4f}" for e in energies) + "\n")


# =====================================================================
#  Umbrella Sampling + WHAM 输出
# =====================================================================

def write_wham_results(wham_result: Dict, filepath: str):
    """
    写入 WHAM 分析结果。
    """
    with open(filepath, 'w') as f:
        f.write("# WHAM (Weighted Histogram Analysis Method) Results\n")
        f.write("#\n")
        f.write(f"# Converged: {wham_result['converged']}\n")
        f.write(f"# Iterations: {wham_result['n_iterations']}\n")
        f.write("#\n")

        f.write("# Window free energies F_j:\n")
        for j, Fj in enumerate(wham_result['F_windows']):
            f.write(f"#   F[{j}] = {Fj:.8f}\n")
        f.write("#\n")

        f.write("# q_center  P(q)  free_energy_F(q)\n")
        q_centers = wham_result['q_centers']
        P_q = wham_result['P_q']
        F_q = wham_result['free_energy']
        for i in range(len(q_centers)):
            f.write(f"{q_centers[i]:10.6f}  {P_q[i]:15.8e}  {F_q[i]:15.8f}\n")


# =====================================================================
#  无序平均输出
# =====================================================================

def write_disorder_average(averaged_results: Dict, filepath: str):
    """
    写入多个 disorder 实现的平均结果。
    """
    with open(filepath, 'w') as f:
        f.write("# Disorder-Averaged Results\n")
        f.write(f"# Number of realizations: {averaged_results.get('n_realizations', 'N/A')}\n")
        f.write("#\n")

        f.write("# beta  <E>/N  std(E)/N  <|m|>  <q>  <q^2>  g  chi_SG  C_V\n")
        for row in averaged_results.get('summary_table', []):
            f.write(f"{row['beta']:8.4f}  "
                    f"{row['mean_energy']:12.6f}  {row['std_energy']:12.6f}  "
                    f"{row['mean_mag']:8.4f}  {row['mean_q']:8.4f}  "
                    f"{row['mean_q2']:8.4f}  {row['binder']:8.4f}  "
                    f"{row['chi_SG']:10.4f}  {row['C_V']:10.4f}\n")

        f.write("\n# Quadrature-based disorder average (single-bond):\n")
        quad = averaged_results.get('quadrature_results', {})
        for key, value in quad.items():
            f.write(f"#   {key}: {value}\n")
