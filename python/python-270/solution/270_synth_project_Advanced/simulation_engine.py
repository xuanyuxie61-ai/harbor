"""
simulation_engine.py
====================

统一的模拟引擎：协调所有模块完成完整的自旋玻璃模拟流程。

流程概述
--------
1. 构建晶格 + 生成耦合 (淬火无序)
2. 稳定性分析 (有限差分 + von Neumann)
3. 多温度 Metropolis MC (扫描 beta)
4. Replica Exchange (parallel tempering)
5. Umbrella Sampling + WHAM (P(q) 与自由能)
6. Hessian 谱分析 (magnon DOS)
7. 无序平均 (多个耦合实现)
8. 结果输出
"""

import numpy as np
from typing import Dict, List, Optional
import time

from spin_lattice_geometry import CubicLattice3D
from spin_glass_couplings import (CouplingGenerator, verify_coupling_statistics,
                                   couplings_to_matrix)
from spin_glass_boundary import BoundaryHandler
from disorder_quadrature import (disorder_average_energy_density,
                                  disorder_average_susceptibility,
                                  gauss_hermite_nodes_weights)
from spin_glass_complex import (spectral_radius, max_amplification_over_bz,
                                 amplification_factor_symbol,
                                 gershgorin_eigenvalue_bounds)
from langevin_fd_integration import (LangevinIntegrator,
                                      laplacian_2nd_order, laplacian_4th_order)
from monte_carlo_core import (run_mc_simulation, MetropolisMC,
                               kmeans_overlap_clustering)
from replica_exchange import (ReplicaExchangeMC, geometric_temperature_ladder,
                               cvt_temperature_optimization)
from umbrella_sampling import (run_umbrella_sampling_windows, WHAMAnalyzer)
from hessian_magnon_dos import (spectral_analysis, de_almeida_thouless_criterion,
                                 build_hessian, hessian_eigenvalues)
from output_writer import (ensure_output_dir, write_parameters, write_mc_results,
                            write_stability_report, write_spectral_analysis,
                            write_replica_exchange_results, write_wham_results,
                            write_disorder_average)


# =====================================================================
#  模拟配置
# =====================================================================

class SimulationConfig:
    """
    模拟配置参数。

    所有参数都有默认值, 确保零参数可运行。
    """

    def __init__(self):
        # 晶格
        self.L = 4                    # 线性尺寸
        self.boundary_type = "periodic"

        # 耦合
        self.distribution = "gaussian"
        self.J_var = 1.0
        self.n_realizations = 3       # 无序实现数

        # 温度扫描
        self.T_list = [3.0, 2.0, 1.5, 1.0, 0.7]  # 从高到低
        self.n_sweeps = 300
        self.n_equil = 100
        self.n_replicas_mc = 2

        # Replica Exchange
        self.RE_T_min = 0.5
        self.RE_T_max = 4.0
        self.RE_n_replicas = 6
        self.RE_n_rounds = 100
        self.RE_sweeps_per_exchange = 5

        # Umbrella Sampling
        self.US_q0_list = np.linspace(-0.8, 0.8, 7)
        self.US_kappa = 30.0
        self.US_n_sweeps = 150
        self.US_n_equil = 50
        self.US_beta = 1.0

        # Langevin 稳定性
        self.langevin_dt = 0.005
        self.langevin_fd_order = 2
        self.langevin_method = "euler_maruyama"

        # Hessian
        self.hessian_continuous = True

        # 输出
        self.output_dir = "output"
        self.seed = 42

    def to_dict(self) -> Dict:
        d = {}
        for key in dir(self):
            if not key.startswith('_') and not callable(getattr(self, key)):
                val = getattr(self, key)
                d[key] = val
        return d


# =====================================================================
#  稳定性分析模块
# =====================================================================

def run_stability_analysis(config: SimulationConfig) -> Dict:
    """
    执行 von Neumann 稳定性分析。
    """
    lattice = CubicLattice3D(config.L)
    boundary = BoundaryHandler(config.L, config.boundary_type)

    # 扩散系数 (取 J_var 的平方根作为特征耦合强度)
    D = np.sqrt(config.J_var)

    # 临界时间步长
    dt_crit_2nd = boundary.max_stable_dt_2nd_order(D)
    dt_crit_4th = boundary.max_stable_dt_4th_order(D)

    # 放大因子扫描
    rho_max_2nd = max_amplification_over_bz(D, config.langevin_dt, config.L, order=2)
    rho_max_4th = max_amplification_over_bz(D, config.langevin_dt, config.L, order=4)

    # 放大因子详细 (沿高对称方向)
    n_k = 20
    kx_line = np.linspace(0, np.pi, n_k)
    G_2nd_line = []
    G_4th_line = []
    for kx in kx_line:
        G2 = amplification_factor_symbol(D, config.langevin_dt, kx, 0, 0, order=2)
        G4 = amplification_factor_symbol(D, config.langevin_dt, kx, 0, 0, order=4)
        G_2nd_line.append(abs(G2))
        G_4th_line.append(abs(G4))

    results = {
        "D": D,
        "dt_used": config.langevin_dt,
        "fd_order": config.langevin_fd_order,
        "stability_limits": {
            "dt_crit_2nd": dt_crit_2nd,
            "dt_crit_4th": dt_crit_4th,
            "dt_stable_2nd": config.langevin_dt < dt_crit_2nd,
            "dt_stable_4th": config.langevin_dt < dt_crit_4th,
        },
        "amplification_factors": {
            "rho_max_2nd": rho_max_2nd,
            "rho_max_4th": rho_max_4th,
            "stable_2nd": rho_max_2nd <= 1.0 + 1e-10,
            "stable_4th": rho_max_4th <= 1.0 + 1e-10,
        },
        "G_2nd_along_kx": list(zip(kx_line.tolist(), G_2nd_line)),
        "G_4th_along_kx": list(zip(kx_line.tolist(), G_4th_line)),
    }
    return results


# =====================================================================
#  Langevin 能量漂移测试
# =====================================================================

def run_langevin_test(config: SimulationConfig,
                      couplings: Dict,
                      lattice: CubicLattice3D,
                      rng: np.random.Generator) -> Dict:
    """
    测试 Langevin 积分器的能量守恒 (T=0)。
    """
    # 初始化: 小扰动从基态
    spins = np.ones((config.L, config.L, config.L), dtype=np.float64) * 0.9
    spins += 0.1 * rng.standard_normal(spins.shape)
    spins = np.clip(spins, -1.0, 1.0)

    integrator = LangevinIntegrator(
        lattice, temperature=0.0,
        dt=config.langevin_dt,
        fd_order=config.langevin_fd_order,
        method=config.langevin_method,
        rng=rng
    )

    energy_drift = integrator.check_energy_drift(spins, couplings, n_steps=50)

    return {
        "method": config.langevin_method,
        "fd_order": config.langevin_fd_order,
        "dt": config.langevin_dt,
        **energy_drift,
    }


# =====================================================================
#  主模拟引擎
# =====================================================================

def run_full_simulation(config: Optional[SimulationConfig] = None) -> Dict:
    """
    运行完整的自旋玻璃模拟流程。

    返回
    ----
    all_results : dict
        包含所有子模块的结果
    """
    if config is None:
        config = SimulationConfig()

    t_start = time.time()
    rng = np.random.default_rng(config.seed)
    output_dir = ensure_output_dir(config.output_dir)

    all_results = {"config": config.to_dict()}

    print("=" * 70)
    print("  Spin Glass Monte Carlo Simulation")
    print("  Edwards-Anderson Model + High-Order Finite Difference")
    print("=" * 70)

    # -----------------------------------------------------------------
    #  1. 晶格 + 耦合
    # -----------------------------------------------------------------
    print("\n[1/8] Building lattice and generating couplings...")
    lattice = CubicLattice3D(config.L)
    boundary = BoundaryHandler(config.L, config.boundary_type)

    print(f"  L = {config.L}, N = {lattice.N}, z = {lattice.z}")
    print(f"  N_bond = {lattice.n_bond}, boundary = {config.boundary_type}")

    # 验证晶格
    assert lattice.verify_consistency(), "晶格一致性检查失败!"
    assert boundary.verify_boundary_consistency(lattice), "边界条件一致性检查失败!"

    # 生成多个 disorder 实现
    generator = CouplingGenerator(
        distribution=config.distribution,
        J_var=config.J_var,
        rng=rng
    )

    all_couplings = []
    for r in range(config.n_realizations):
        couplings = generator.generate(lattice)
        stats = verify_coupling_statistics(couplings, generator)
        all_couplings.append(couplings)
        print(f"  Realization {r+1}: <J>={stats['sample_mean']:.4f}, "
              f"Var(J)={stats['sample_var']:.4f}")

    # -----------------------------------------------------------------
    #  2. 稳定性分析
    # -----------------------------------------------------------------
    print("\n[2/8] Running von Neumann stability analysis...")
    stability_results = run_stability_analysis(config)
    all_results["stability"] = stability_results

    sl = stability_results["stability_limits"]
    af = stability_results["amplification_factors"]
    print(f"  dt_crit (2nd) = {sl['dt_crit_2nd']:.6f}, "
          f"dt_crit (4th) = {sl['dt_crit_4th']:.6f}")
    print(f"  rho_max (2nd) = {af['rho_max_2nd']:.6f}, "
          f"rho_max (4th) = {af['rho_max_4th']:.6f}")
    print(f"  Stable (2nd): {af['stable_2nd']}, Stable (4th): {af['stable_4th']}")

    write_stability_report(stability_results,
                           f"{output_dir}/stability_report.txt")

    # Langevin 测试
    langevin_test = run_langevin_test(config, all_couplings[0], lattice, rng)
    all_results["langevin_test"] = langevin_test
    print(f"  Langevin drift: {langevin_test['relative_drift']:.2e}")

    # -----------------------------------------------------------------
    #  3. 多温度 MC 扫描 (对第一个 disorder 实现)
    # -----------------------------------------------------------------
    print("\n[3/8] Temperature scan (Metropolis MC)...")
    mc_results_by_T = {}
    for T in config.T_list:
        beta = 1.0 / T
        result = run_mc_simulation(
            lattice, all_couplings[0], beta,
            n_sweeps=config.n_sweeps,
            n_equil=config.n_equil,
            n_replicas=config.n_replicas_mc,
            seed=config.seed + int(T * 100)
        )
        mc_results_by_T[T] = result
        print(f"  T={T:.2f}: <E>/N={result['mean_energy']:.4f}, "
              f"<|m|>={result['mean_mag']:.4f}, "
              f"<q>={result['mean_q']:.4f}, "
              f"g={result['binder_cumulant']:.4f}")

        write_mc_results(result,
                         f"{output_dir}/mc_T{T:.2f}.txt",
                         label=f"T={T:.2f}")

    all_results["mc_temperature_scan"] = mc_results_by_T

    # -----------------------------------------------------------------
    #  4. Replica Exchange
    # -----------------------------------------------------------------
    print("\n[4/8] Replica Exchange (Parallel Tempering)...")
    temperatures = geometric_temperature_ladder(
        config.RE_T_min, config.RE_T_max, config.RE_n_replicas
    )
    print(f"  Temperatures: {temperatures.round(3).tolist()}")

    # CVT 优化温度
    temperatures_cvt = cvt_temperature_optimization(
        config.RE_T_min, config.RE_T_max, config.RE_n_replicas,
        lattice.N, config.J_var, lattice.z
    )
    print(f"  CVT-optimized: {temperatures_cvt.round(3).tolist()}")

    re_mc = ReplicaExchangeMC(
        lattice, all_couplings[0], temperatures_cvt,
        n_sweeps_per_exchange=config.RE_sweeps_per_exchange,
        rng=rng
    )
    re_results = re_mc.run(config.RE_n_rounds)
    all_results["replica_exchange"] = re_results
    print(f"  Exchange ratio: {re_results['overall_exchange_ratio']:.4f}")

    write_replica_exchange_results(re_results,
                                    f"{output_dir}/replica_exchange.txt")

    # -----------------------------------------------------------------
    #  5. Umbrella Sampling + WHAM
    # -----------------------------------------------------------------
    print("\n[5/8] Umbrella Sampling + WHAM...")
    windows = run_umbrella_sampling_windows(
        lattice, all_couplings[0], config.US_beta,
        q0_list=config.US_q0_list,
        kappa=config.US_kappa,
        n_sweeps=config.US_n_sweeps,
        n_equil=config.US_n_equil,
        seed=config.seed
    )

    q_bins = np.linspace(-1.05, 1.05, 43)
    wham = WHAMAnalyzer(windows, q_bins, config.US_beta)
    wham_result = wham.solve()
    all_results["wham"] = wham_result
    print(f"  WHAM converged: {wham_result['converged']}, "
          f"iterations: {wham_result['n_iterations']}")

    write_wham_results(wham_result, f"{output_dir}/wham_results.txt")

    # -----------------------------------------------------------------
    #  6. Hessian 谱分析
    # -----------------------------------------------------------------
    print("\n[6/8] Hessian spectral analysis...")
    # 对最低温 MC 的最终配置做谱分析
    lowest_T = config.T_list[-1]
    # 重新运行一次 MC 获得最终配置
    beta_low = 1.0 / lowest_T
    mc_low = MetropolisMC(lattice, beta_low, rng=rng)
    spins_final = rng.choice([-1.0, +1.0], size=lattice.N).reshape(
        (config.L,) * 3
    )
    for _ in range(config.n_equil):
        spins_final = mc_low.single_sweep(spins_final, all_couplings[0])

    spectral = spectral_analysis(spins_final, all_couplings[0], lattice)
    all_results["spectral"] = {
        "lambda_min": spectral["lambda_min"],
        "lambda_max": spectral["lambda_max"],
        "n_negative": spectral["n_negative"],
        "condition_number": spectral["condition_number"],
        "dos_moments": spectral["dos_moments"],
    }

    at_crit = de_almeida_thouless_criterion(
        spectral["eigenvalues"], beta_low, config.J_var
    )
    all_results["AT_criterion"] = at_crit
    print(f"  lambda_min = {spectral['lambda_min']:.6f}")
    print(f"  n_negative = {spectral['n_negative']}")
    print(f"  AT stable: {at_crit['is_stable']}")

    write_spectral_analysis(spectral, f"{output_dir}/spectral_analysis.txt")

    # -----------------------------------------------------------------
    #  7. 无序平均
    # -----------------------------------------------------------------
    print("\n[7/8] Disorder averaging...")
    disorder_summary = []
    for T in config.T_list:
        beta = 1.0 / T
        energies = []
        mags = []
        qs = []
        q2s = []
        binders = []
        chi_SGs = []
        C_Vs = []

        for r_idx, couplings in enumerate(all_couplings):
            result = run_mc_simulation(
                lattice, couplings, beta,
                n_sweeps=config.n_sweeps // 2,
                n_equil=config.n_equil // 2,
                n_replicas=2,
                seed=config.seed + r_idx * 100 + int(T * 10)
            )
            energies.append(result['mean_energy'])
            mags.append(result['mean_mag'])
            qs.append(result['mean_q'])
            q2s.append(result['mean_q2'])
            binders.append(result['binder_cumulant'])
            chi_SGs.append(result['chi_SG'])
            C_Vs.append(result['C_V'])

        disorder_summary.append({
            'beta': beta,
            'mean_energy': np.mean(energies),
            'std_energy': np.std(energies),
            'mean_mag': np.mean(mags),
            'mean_q': np.mean(qs),
            'mean_q2': np.mean(q2s),
            'binder': np.mean(binders),
            'chi_SG': np.mean(chi_SGs),
            'C_V': np.mean(C_Vs),
        })

    # 高斯求积基准
    quad_results = {}
    for T in config.T_list:
        beta = 1.0 / T
        e_quad = disorder_average_energy_density(
            n_quad=15, J_var=config.J_var, beta=beta, z=lattice.z
        )
        chi_quad = disorder_average_susceptibility(
            n_quad=20, J_var=config.J_var, beta=beta, z=lattice.z
        )
        quad_results[f"T={T:.2f}"] = {
            "e_mf": e_quad,
            "chi_SG_mf": chi_quad
        }

    all_results["disorder_average"] = {
        "summary_table": disorder_summary,
        "quadrature_results": quad_results,
        "n_realizations": config.n_realizations,
    }

    write_disorder_average(all_results["disorder_average"],
                           f"{output_dir}/disorder_average.txt")

    # -----------------------------------------------------------------
    #  8. K-means 聚类分析 (纯态识别)
    # -----------------------------------------------------------------
    print("\n[8/8] Cluster analysis of replica overlaps...")
    # 构建 overlap 矩阵
    n_cluster_replicas = 8
    overlap_matrix = np.zeros((n_cluster_replicas, n_cluster_replicas))
    cluster_replicas = []
    mc_cluster = MetropolisMC(lattice, 1.0 / config.T_list[-1], rng=rng)
    for r in range(n_cluster_replicas):
        s = rng.choice([-1.0, +1.0], size=lattice.N).reshape((config.L,) * 3)
        for _ in range(config.n_equil):
            s = mc_cluster.single_sweep(s, all_couplings[0])
        cluster_replicas.append(s)

    from monte_carlo_core import overlap as calc_overlap
    for i in range(n_cluster_replicas):
        for j in range(n_cluster_replicas):
            overlap_matrix[i, j] = calc_overlap(cluster_replicas[i],
                                                  cluster_replicas[j])

    centers, labels, wss = kmeans_overlap_clustering(
        overlap_matrix, n_clusters=2, rng=rng
    )
    all_results["cluster_analysis"] = {
        "overlap_matrix": overlap_matrix.tolist(),
        "labels": labels.tolist(),
        "wss": wss.tolist(),
        "n_clusters": 2,
    }
    print(f"  Cluster labels: {labels.tolist()}")
    print(f"  WSS: {wss.round(4).tolist()}")

    # -----------------------------------------------------------------
    #  写入参数文件
    # -----------------------------------------------------------------
    write_parameters(config.to_dict(), f"{output_dir}/parameters.txt")

    t_end = time.time()
    all_results["wall_time_seconds"] = t_end - t_start

    print("\n" + "=" * 70)
    print(f"  Simulation completed in {t_end - t_start:.2f} seconds")
    print(f"  Output directory: {output_dir}/")
    print("=" * 70)

    return all_results
