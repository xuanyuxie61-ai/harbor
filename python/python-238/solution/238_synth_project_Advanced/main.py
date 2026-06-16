"""
main.py
=======

格点 QCD 有限温相变模拟的统一入口.

科学问题:
---------
本项目研究纯 SU(3) 规范理论在有限温度下的退禁闭相变.
通过 HMC (混合蒙特卡罗) 算法采样规范场构型, 测量 Polyakov loop
等序参量, 分析相变特性, 并对高阶有限差分格式进行稳定性验证.

模拟流程:
---------
1. 参数设置: 格点尺寸, 耦合常数, 积分器参数
2. 冷启动: 初始化规范场
3. HMC 轨迹: 在多个 β 值处运行 HMC
4. 观测量测量: plaquette, Polyakov loop, 拓扑荷
5. 稳定性分析: 色散关系, 积分器稳定性
6. 相变分析: susceptibility, Binder cumulant, 重加权
7. 输出报告

本模拟为小规模可复现实验 (Ns=3, Nt=4),
用于演示算法框架与物理分析流程.

融合种子项目:
  - 1333_triangulation_boundary_nodes: 边界识别
  - 631_l4lib: 布尔运算 → 站点 parity
  - 977_r8col: 排序统计
  - 478_gradient_descent: CG 求解器
  - 020_artery_pde: PDE 演化
  - 017_area_under_curve: 热力学积分
  - 270_dfield9: 高阶 RK 积分器
  - 1136_SonyResearch_SVG_baseline: 基线配置
  - 431_filum: 文件管理
  - 1158_shoh5301_Quick-MSD: 关联函数
  - 545_house: 参考构型
  - 1044_nicsar2_FootlooseCalvingMechanism: 相变追踪
  - 1338_triangulation_l2q: Symanzik 改进
  - 351_fd_to_tec: 场数据导出
  - 1078_nec-research_alebrew: 主动学习 (配置选择)
"""

import os
import sys
import time
import numpy as np

# 将项目目录加入路径
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from lattice_geometry import LatticeGeometry
from gauge_field import GaugeField
from gauge_actions import GaugeAction, SYMANZIK_PRESETS
from high_order_fd import (
    fd_coefficients_central, naik_coefficients,
    dispersion_relation, dispersion_error, DispersionAnalyzer
)
from stability_analysis import (
    critical_r_fd, leapfrog_stability_bound,
    omelyan_stability_bound, forest_ruth_stability_bound,
    dirac_eigenvalue_bounds, condition_number, StepSizeSelector
)
from molecular_dynamics import HMCTrajectory
from fermion_solver import (
    StaggeredDirac, CGSolver, chiral_condensate
)
from phase_transition import (
    PhaseTransitionAnalyzer, HistogramReweighting,
    critical_temperature_estimate
)
from observables import (
    measure_plaquette, measure_plaquette_per_direction,
    autocorrelation_function, integrated_autocorrelation_time,
    sorted_statistics, topological_charge_diffusion,
    thermodynamic_integration
)
from config_io import save_configuration, next_config_number


# ============================================================
# 模拟参数
# ============================================================

class SimulationConfig:
    """模拟配置 (全部默认值, 零参数运行)."""

    # 格点参数
    Ns = 3          # 空间格点 (小规模)
    Nt = 4          # 时间格点 (T = 1/4a)
    fermion_apbc = True

    # 耦合常数扫描
    beta_values = [5.0, 5.5, 5.7, 6.0]  # 跨越相变

    # 作用量类型
    action_type = 'tree_level_symanzik'  # 改进作用量

    # HMC 参数
    n_trajectories = 5   # 每个 β 的轨迹数 (小规模)
    n_steps = 8           # 每条轨迹的 MD 步数
    step_size = 0.02      # MD 步长
    integrator = 'leapfrog'  # 积分器类型

    # 费米子参数
    quark_mass = 0.1      # 轻夸克质量 (格点单位)
    fd_order = 4          # 有限差分精度
    use_naik = True       # 是否 Naik 改进

    # 输出
    output_dir = os.path.join(PROJECT_DIR, 'output')
    save_configs = True

    def __str__(self):
        lines = ["=" * 60,
                 "格点 QCD 有限温相变模拟配置",
                 "=" * 60,
                 f"  格点尺寸: {self.Ns}^3 × {self.Nt}",
                 f"  β 值扫描: {self.beta_values}",
                 f"  作用量类型: {self.action_type}",
                 f"  积分器: {self.integrator}",
                 f"  轨迹数: {self.n_trajectories}",
                 f"  MD 步数: {self.n_steps}, 步长: {self.step_size}",
                 f"  夸克质量: {self.quark_mass}",
                 f"  有限差分精度: O(a^{self.fd_order})",
                 f"  Naik 改进: {self.use_naik}",
                 "=" * 60]
        return "\n".join(lines)


# ============================================================
# 主模拟函数
# ============================================================

def run_simulation(config: SimulationConfig = None):
    """运行完整的格点 QCD 模拟."""

    if config is None:
        config = SimulationConfig()
    print(config)

    # 创建输出目录
    os.makedirs(config.output_dir, exist_ok=True)

    # 1. 创建格点几何
    print("\n[阶段 1] 构建格点几何...")
    geom = LatticeGeometry(config.Ns, config.Nt, config.fermion_apbc)
    print(f"  体积: {geom.volume}, 空间体积: {geom.spatial_volume}")
    print(f"  温度: T = 1/(N_t a) = {1.0/config.Nt:.4f}/a")
    print(f"  边界节点数: {len(geom.boundary_nodes)}")
    print(f"  偶站点数: {len(geom.even_sites)}, 奇站点数: {len(geom.odd_sites)}")

    # 2. 有限差分分析
    print("\n[阶段 2] 高阶有限差分色散分析...")
    analyzer = DispersionAnalyzer(max_order=8)
    fd_comparison = analyzer.compare_methods()
    print("  各方法色散误差比较:")
    for name, data in sorted(fd_comparison.items()):
        print(f"    {name}: RMS = {data['rms_error']:.6e}, max = {data['max_deviation']:.6e}")

    # 临界 CFL 数
    for order in [2, 4, 6, 8]:
        r_c = critical_r_fd(order)
        print(f"  O(a^{order}) 差分最大稳定 r = {r_c:.6f}")

    # 3. 稳定性分析
    print("\n[阶段 3] MD 积分器稳定性分析...")
    lambda_max = 5.0  # 典型力矩阵本征值估计
    print(f"  Leapfrog 稳定性上界: ε < {leapfrog_stability_bound(lambda_max):.4f}")
    print(f"  Omelyan 稳定性上界:   ε < {omelyan_stability_bound(lambda_max):.4f}")
    print(f"  Forest-Ruth 稳定性上界: ε < {forest_ruth_stability_bound(lambda_max):.4f}")

    # Dirac 算子条件数
    l_min, l_max = dirac_eigenvalue_bounds(config.quark_mass, config.fd_order, config.use_naik)
    kappa = condition_number(config.quark_mass, config.fd_order)
    print(f"  Dirac 算子本征值: [{l_min:.4f}, {l_max:.4f}], 条件数 = {kappa:.2f}")

    # 步长选择
    selector = StepSizeSelector(safety_factor=0.8, integrator=config.integrator)
    opt_eps = selector.optimal_step_size(lambda_max)
    print(f"  推荐 MD 步长: ε = {opt_eps:.4f}")

    # 4. HMC 模拟 (在多个 β 值处)
    print("\n[阶段 4] HMC 模拟 (多 β 扫描)...")
    results_by_beta = {}

    for beta in config.beta_values:
        print(f"\n  === β = {beta} ===")
        action = GaugeAction(geom, beta, config.action_type)
        print(f"    作用量: {action.description}")
        print(f"    系数: c0={action.c0:.4f}, c1={action.c1:.4f}, c2={action.c2:.4f}")

        # 初始化规范场 (冷启动)
        gf = GaugeField(geom, initial='cold')
        initial_plaq = gf.avg_plaquette()
        initial_S = action.total_action(gf)
        print(f"    冷启动: P = {initial_plaq:.6f}, S = {initial_S:.4f}")

        # Phase transition analyzer
        pt_analyzer = PhaseTransitionAnalyzer(geom)

        # 收集数据
        plaq_series = []
        polyakov_series = []
        Q_series = []
        accept_count = 0
        dH_series = []

        # HMC 轨迹
        hmc = HMCTrajectory(
            geom, action,
            integrator=config.integrator,
            n_steps=config.n_steps,
            step_size=config.step_size,
            seed=42 + int(beta * 10)
        )

        for traj in range(config.n_trajectories):
            info = hmc.run_trajectory(gf)

            plaq_series.append(info['final_plaq'])
            L_val = gf.avg_polyakov_loop()
            polyakov_series.append(L_val)
            pt_analyzer.add_sample(L_val)
            dH_series.append(info['dH'])

            if info['accepted']:
                accept_count += 1

            # 每 2 条轨迹计算拓扑荷 (耗时较长)
            if traj % 2 == 0:
                Q = action.topological_charge(gf)
                Q_series.append(Q)

            if traj % 2 == 0:
                print(f"      轨迹 {traj}: P={info['final_plaq']:.6f}, "
                      f"|L|={abs(L_val):.6f}, "
                      f"dH={info['dH']:.4f}, "
                      f"accepted={info['accepted']}")

        # 统计
        accept_rate = accept_count / config.n_trajectories
        stats = pt_analyzer.get_statistics()
        plaq_stats = sorted_statistics(np.array(plaq_series))

        print(f"\n    β = {beta} 统计结果:")
        print(f"      接受率: {accept_rate:.2%}")
        print(f"      平均 plaquette: {plaq_stats['mean']:.6f} ± {plaq_stats['std']:.6f}")
        print(f"      ⟨|L|⟩ = {stats['mean_abs_L']:.6f} ± {stats['jackknife_error']:.6f}")
        print(f"      Susceptibility: χ_L = {stats['susceptibility']:.4f}")
        print(f"      Binder cumulant: B_4 = {stats['binder_cumulant']:.4f}")

        # 自关联时间
        if len(plaq_series) > 4:
            tau_plaq = integrated_autocorrelation_time(np.array(plaq_series))
            print(f"      Plaquette 自关联时间: τ = {tau_plaq:.2f}")

        # 保存构型
        if config.save_configs:
            cfg_num = next_config_number(config.output_dir, prefix=f'beta{beta:.1f}')
            save_configuration(gf, action, cfg_num, config.output_dir,
                               trajectory=config.n_trajectories,
                               prefix=f'beta{beta:.1f}')
            print(f"      保存配置: {cfg_num}")

        results_by_beta[beta] = {
            'plaquette': plaq_series,
            'polyakov': polyakov_series,
            'statistics': stats,
            'accept_rate': accept_rate,
        }

    # 5. 相变分析
    print("\n[阶段 5] 相变分析...")
    beta_arr = np.array(config.beta_values)
    L_means = np.array([np.mean(np.abs([complex(L) for L in results_by_beta[b]['polyakov']]))
                         for b in config.beta_values])

    print("  β 扫描结果:")
    for i, b in enumerate(config.beta_values):
        print(f"    β = {b:.2f}: ⟨|L|⟩ = {L_means[i]:.6f}")

    # 临界 β 估计
    beta_c, T_c = critical_temperature_estimate(
        config.beta_values, L_means.tolist(), config.Nt
    )
    print(f"  伪临界 β_c ≈ {beta_c:.3f} (N_t = {config.Nt})")
    print(f"  临界温度: T_c ≈ {T_c:.1f} MeV")

    # 热力学积分 (融合 017)
    plaq_means = np.array([np.mean(results_by_beta[b]['plaquette'])
                            for b in config.beta_values])
    delta_ln_Z = thermodynamic_integration(beta_arr, plaq_means)
    print(f"  热力学积分: Δ(ln Z)/(6V) = {delta_ln_Z:.6f}")

    # 6. 费米子 sector 分析 (小规模演示)
    print("\n[阶段 6] 费米子 sector 演示...")
    gf_test = GaugeField(geom, initial='cold')
    action_test = GaugeAction(geom, 6.0, 'wilson')

    dirac = StaggeredDirac(geom, gf_test, config.quark_mass,
                            use_naik=config.use_naik)
    print(f"  Staggered Dirac 算子:")
    print(f"    质量: m = {config.quark_mass}")
    print(f"    Naik 改进: {config.use_naik}")

    # CG 求解器测试
    rng = np.random.default_rng(42)
    phi = rng.standard_normal(geom.volume * 3) + 1j * rng.standard_normal(geom.volume * 3)
    phi = phi.reshape(geom.volume, 3)

    def normal_op(x_flat):
        return dirac.normal_operator(x_flat.reshape(-1, 3)).flatten()

    solver = CGSolver(max_iterations=100, tolerance=1e-6)
    chi_flat, info = solver.solve(normal_op, phi.flatten())
    print(f"  CG 求解器: 收敛 = {info['converged']}, "
          f"迭代数 = {info['iterations']}")

    # 7. 总结报告
    print("\n" + "=" * 60)
    print("模拟完成!")
    print("=" * 60)
    print(f"\n科学成果摘要:")
    print(f"  1. 格点 {config.Ns}^3 × {config.Nt} 上 SU(3) 纯规范理论模拟")
    print(f"  2. 使用 {config.action_type} 作用量 (c0={action.c0}, c1={action.c1})")
    print(f"  3. β = {config.beta_values} 范围扫描, 每个 β {config.n_trajectories} 条 HMC 轨迹")
    print(f"  4. 退禁闭相变临界 β_c ≈ {beta_c:.3f}, T_c ≈ {T_c:.1f} MeV")
    print(f"  5. 高阶有限差分 O(a^{config.fd_order}) 与 Naik 改进已实现")
    print(f"  6. {config.integrator} 积分器, 步长 ε = {config.step_size}")
    print(f"\n输出目录: {config.output_dir}")
    print("=" * 60)

    return results_by_beta


# ============================================================
# 入口
# ============================================================

if __name__ == '__main__':
    start_time = time.time()
    results = run_simulation()
    elapsed = time.time() - start_time
    print(f"\n总运行时间: {elapsed:.2f} 秒")
