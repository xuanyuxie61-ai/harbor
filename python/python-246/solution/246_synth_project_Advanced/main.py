#!/usr/bin/env python3
"""
main.py  —  宇宙大尺度结构 N 体模拟主程序
========================================

可复现的小规模实验: 使用高阶有限差分 Poisson 求解器,
leapfrog 时间积分,与 CMA-ES 参数标定。

运行方式:
    python main.py
    # 或: python -m 246_synth_project_Advanced

无参数,零配置即可运行;结果输出到 results/ 目录。
"""

from __future__ import annotations
import sys
import time
import numpy as np
from pathlib import Path

# 使当前目录作为包被导入
_HERE = Path(__file__).resolve().parent
if str(_HERE.parent) not in sys.path:
    sys.path.insert(0, str(_HERE.parent))

from cosmo_config import (
    CosmoParams, load_cosmo_params, save_params, get_paths,
    z_to_a, hubble_factor,
)
from legendre_kernel import (
    legendre_shifted_values, gauss_legendre_quadrature,
)
from finite_difference import (
    fd_stencil_2nd, compact_second_derivative_1d,
    laplacian_3d_periodic, gradient_3d_periodic,
)
from poisson_solver import (
    poisson_fft_3d, condition_number_estimate_1d,
)
from initial_conditions import (
    generate_density_field, sample_particle_positions,
    primordial_power_spectrum,
)
from mesh_topology import (
    MeshTopology, cube_surface_distance_stats,
    particle_pair_distance_stats,
)
from time_integrator import (
    kick_drift_kick_step, kdv_exact_sech, kdv_exact_rational,
    kdv_residual, kdv_parameters,
)
from stability_analysis import (
    zero_itp, power_method, power_method2,
    spectral_radius_leapfrog, critical_cfl_number,
)
from parameter_calibration import (
    calibrate_parameters, sheth_tormen_mass_function,
)
from nas_search import cic_deposit, neighbor_count_statistics
from diagnostics import (
    power_spectrum, two_point_correlation, full_diagnostics,
)


# ============================================================================ #
#                              工具: 打印分隔
# ============================================================================ #
def section(title: str) -> None:
    print("\n" + "=" * 72)
    print(f"  {title}")
    print("=" * 72)


# ============================================================================ #
#                             Stage 1: 配置加载
# ============================================================================ #
def stage_config() -> CosmoParams:
    section("Stage 1: 加载宇宙学参数 (seed 1068 internalstate/config_utils)")
    p = load_cosmo_params()
    print(f"  盒子边长 L = {p.box_length} Mpc/h")
    print(f"  网格 N = {p.n_grid} (总 {p.n_grid**3} 个网格单元)")
    print(f"  粒子数 N_p = {p.n_particles}")
    print(f"  (Ω_m, Ω_Λ, h) = ({p.omega_m}, {p.omega_l}, {p.hubble})")
    print(f"  红移范围 z = {p.z_init} → {p.z_final}")
    print(f"  时间步数 = {p.n_steps},  FD 阶数 = {p.fd_order}")
    print(f"  软化长度 ε = {p.softening_eps} Mpc/h")
    print(f"  σ_8 = {p.sigma_8},  n_s = {p.spectral_index}")
    # 保存实际使用参数:
    out = save_params(p)
    print(f"  参数已保存至: {out}")
    return p


# ============================================================================ #
#            Stage 2: Legendre 核验证与 Gauss-Legendre 积分
# ============================================================================ #
def stage_legendre() -> None:
    section("Stage 2: Legendre 多项式与求积验证 (seed 666)")
    # 验证正交性: ∫_{-1}^1 P_n P_m dx = 2/(2n+1) δ_{nm}
    n_max = 5
    def Pn(n):
        return lambda x: legendre_shifted_values(np.array([x]), n_max)[0, n]
    for n in range(n_max + 1):
        for m in range(n + 1):
            val = gauss_legendre_quadrature(
                lambda x: Pn(n)(x) * Pn(m)(x), -1.0, 1.0, n=32)
            expected = 2.0 / (2 * n + 1) if n == m else 0.0
            err = abs(val - expected)
            status = "OK" if err < 1e-10 else "FAIL"
            if n == m or err > 1e-10:
                print(f"  ∫ P_{n} P_{m} = {val:+.6e}  "
                      f"(期望 {expected:+.3e})  [{status}]")


# ============================================================================ #
#       Stage 3: 有限差分精度测试
# ============================================================================ #
def stage_finite_difference(p: CosmoParams) -> None:
    section("Stage 3: 高阶 FD 精度验证 (seeds 367 + 979 + 666)")
    # 测试函数: u(x) = sin(kx), u'' = -k² sin(kx)
    L = 2.0 * np.pi
    k_wave = 2.0
    n_test = 64
    x = np.linspace(0, L, n_test, endpoint=False)
    h = L / n_test
    u = np.sin(k_wave * x)
    u_xx_exact = -(k_wave ** 2) * u
    for p_order in (1, 2, 3, 4):
        fd_order = 2 * p_order
        # 显式 FD:
        s = fd_stencil_2nd(p_order)
        u_xx_fd = np.zeros_like(u)
        for m in range(-p_order, p_order + 1):
            u_xx_fd += s[m + p_order] * np.roll(u, -m)
        u_xx_fd /= h * h
        err = np.max(np.abs(u_xx_fd - u_xx_exact))
        # 紧致 FD:
        scheme = {1: "c4", 2: "c4", 3: "c6", 4: "c8"}[p_order]
        u_xx_compact = compact_second_derivative_1d(u, h, scheme=scheme)
        err_c = np.max(np.abs(u_xx_compact - u_xx_exact))
        print(f"  FD order {fd_order}:  "
              f"max|err_explicit| = {err:.3e},  "
              f"max|err_compact({scheme})| = {err_c:.3e}")
    # 3D Laplacian 验证:
    n3d = 16
    h3 = 1.0
    X = np.arange(n3d) * h3
    xx, yy, zz = np.meshgrid(X, X, X, indexing="ij")
    u3 = np.sin(2 * np.pi * xx) * np.cos(2 * np.pi * yy) * np.sin(np.pi * zz)
    lap_exact = (
        -((2 * np.pi) ** 2 + (2 * np.pi) ** 2 + np.pi ** 2) * u3
    )
    lap_fd = laplacian_3d_periodic(u3, h3, p=2)
    err3 = np.max(np.abs(lap_fd - lap_exact))
    print(f"  3D Laplacian (N={n3d}, 4阶) max|err| = {err3:.3e}")


# ============================================================================ #
#       Stage 4: Poisson 求解验证
# ============================================================================ #
def stage_poisson(p: CosmoParams) -> None:
    section("Stage 4: Poisson 求解器验证 (seeds 367 + 979)")
    n = p.n_grid
    L = p.box_length
    h = L / n
    # 精确解: Φ(x) = sin(2πx/L), ∇²Φ = -(2π/L)² sin(2πx/L)
    x = np.arange(n) * h
    xx, yy, zz = np.meshgrid(x, x, x, indexing="ij")
    phi_exact = np.sin(2.0 * np.pi * xx / L)
    source = -(2.0 * np.pi / L) ** 2 * phi_exact
    phi_num = poisson_fft_3d(source, L)
    # FFT 解差一个常数 (规范选择); 比较梯度:
    grad_e = gradient_3d_periodic(phi_exact, h, p=2)
    grad_n = gradient_3d_periodic(phi_num, h, p=2)
    err_grad = np.max(np.abs(grad_e - grad_n))
    print(f"  FFT Poisson 求解 N={n}:  max|∇Φ_err| = {err_grad:.3e}")
    # 条件数:
    kappa = condition_number_estimate_1d(n, h, p=2)
    print(f"  1D Laplace 条件数估计: κ ≈ {kappa:.3e}")
    # 1D 带状直接求解:
    src1d = source[0, 0, :]
    phi1d = np.zeros(n)
    from poisson_solver import poisson_1d_banded
    try:
        phi1d = poisson_1d_banded(src1d, h, p=1)
        print(f"  1D 带状直接求解: max|Φ| = {np.max(np.abs(phi1d)):.3e}")
    except Exception as e:
        print(f"  1D 带状求解跳过: {e}")


# ============================================================================ #
#        Stage 5: 初始条件生成 (seeds 201 + 1290 + 915)
# ============================================================================ #
def stage_initial_conditions(p: CosmoParams):
    section("Stage 5: 初始密度场生成 (seeds 201 + 1290 + 915)")
    rng = np.random.default_rng(p.seed)
    grid = (p.n_grid, p.n_grid, p.n_grid)
    delta = generate_density_field(grid, p.box_length, p, seed=p.seed)
    print(f"  δ(x) 生成完成: shape={delta.shape}")
    print(f"  <δ> = {delta.mean():+.4e},  σ_δ = {delta.std():.4e}")
    print(f"  δ_min = {delta.min():.4e},  δ_max = {delta.max():.4e}")
    # 功率谱:
    k_bins, Pk = power_spectrum(delta, p.box_length)
    nz = (k_bins > 0) & (Pk > 0)
    if nz.sum() > 2:
        logk = np.log10(k_bins[nz])
        logP = np.log10(Pk[nz])
        # 拟合斜率:
        slope, _ = np.polyfit(logk, logP, 1)
        print(f"  大尺度 P(k) 斜率 ≈ {slope:.2f}  "
              f"(理论: n_s = {p.spectral_index})")
    # 粒子位置 (QMC 素数采样):
    positions = sample_particle_positions(p.n_particles, p.box_length,
                                          seed=p.seed)
    print(f"  粒子位置采样完成: shape={positions.shape}")
    print(f"  位置范围: [{positions.min():.3f}, {positions.max():.3f}] Mpc/h")
    # 初始速度 (Zel'dovich 近似简化: 小随机扰动):
    velocities = 50.0 * rng.standard_normal(positions.shape)
    print(f"  初始速度: RMS = {np.sqrt(np.mean(velocities**2)):.2f} km/s")
    # 粒子质量 (均匀):
    masses = np.full(p.n_particles,
                     p.omega_m * 2.775e11 * (p.box_length ** 3) / p.n_particles)
    return delta, positions, velocities, masses


# ============================================================================ #
#          Stage 6: 网格拓扑与距离统计 (seeds 1059 + 884 + 236)
# ============================================================================ #
def stage_mesh_topology(p: CosmoParams, positions: np.ndarray) -> None:
    section("Stage 6: 网格拓扑与距离统计 (seeds 1059 + 884 + 236)")
    topo = MeshTopology(p.n_grid, p.box_length)
    print(f"  网格: {topo.n}³ = {topo.n_nodes} 节点")
    print(f"  网格间距 h = {topo.h:.4f} Mpc/h")
    # 表面距离统计:
    stats = cube_surface_distance_stats(2000, p.box_length, seed=p.seed)
    print(f"  立方体表面两点距离:")
    print(f"    mean = {stats['mean']:.3f},  std = {np.sqrt(stats['var']):.3f}  "
          f"[{stats['min']:.3f}, {stats['max']:.3f}] Mpc/h")
    # 粒子对距离:
    ps = particle_pair_distance_stats(positions, p.box_length,
                                      max_pairs=5000, seed=p.seed)
    print(f"  粒子对距离:")
    print(f"    mean = {ps['mean']:.3f},  std = {np.sqrt(ps['var']):.3f}  "
          f"[{ps['min']:.3f}, {ps['max']:.3f}] Mpc/h")
    # 2D 多边形三角剖分示例:
    theta = np.linspace(0, 2 * np.pi, 9)[:-1]
    poly = np.stack([np.cos(theta), np.sin(theta)], axis=1)
    from mesh_topology import polygon_triangulate
    tri = polygon_triangulate(poly)
    print(f"  八边形三角剖分: {len(tri)} 个三角形")


# ============================================================================ #
#         Stage 7: KdV 精确解与残差基准 (seed 615)
# ============================================================================ #
def stage_kdv_benchmark() -> None:
    section("Stage 7: KdV 精确解残差基准 (seed 615)")
    # sech² soliton:
    x = np.linspace(-10, 10, 200)
    u, u_t, u_x, u_xx, u_xxx = kdv_exact_sech(x, t=0.0, a_phase=0.0, v_vel=1.0)
    r = kdv_residual(u, u_t, u_x, u_xxx)
    print(f"  sech² soliton 残差: max|r| = {np.max(np.abs(r)):.3e}")
    # 有理精确解:
    x2 = np.linspace(-5, 5, 100)
    u2, ut2, ux2, uxx2, uxxx2 = kdv_exact_rational(x2, t=1.0)
    r2 = kdv_residual(u2, ut2, ux2, uxxx2)
    print(f"  有理精确解残差: max|r| = {np.max(np.abs(r2)):.3e}")
    # 参数查询:
    p_kdv = kdv_parameters(a_phase=0.0, v_vel=2.0, t0=0.0, tstop=5.0)
    print(f"  KdV 参数: a={p_kdv['a']}, v={p_kdv['v']}, "
          f"t=[{p_kdv['t0']}, {p_kdv['tstop']}]")


# ============================================================================ #
#     Stage 8: 稳定性分析 (seeds 1429 + 902)
# ============================================================================ #
def stage_stability() -> None:
    section("Stage 8: von Neumann 稳定性分析 (seeds 1429 + 902)")
    # 谱半径扫描:
    k_vals = np.logspace(-2, 1, 12)
    dt_vals = np.logspace(-3, 0, 10)
    from stability_analysis import von_neumann_stability_scan
    rho = von_neumann_stability_scan(k_vals, dt_vals, rho_bar=1.0, a=1.0)
    print(f"  谱半径矩阵 shape: {rho.shape}")
    print(f"    ρ_min = {rho.min():.4f},  ρ_max = {rho.max():.4f}")
    stable_frac = (rho <= 1.0 + 1e-6).mean()
    print(f"    稳定点比例 (ρ ≤ 1): {stable_frac * 100:.1f}%")
    # 临界 CFL:
    for k_test in [0.1, 1.0, 5.0]:
        dt_c = critical_cfl_number(k_test)
        print(f"    k={k_test:.2f}:  Δt_crit ≈ {dt_c:.4e}")
    # 幂法验证:
    A = np.array([[2.0, 1.0], [1.0, 3.0]], dtype=complex)
    lam, v, it = power_method(A, np.array([1.0, 0.0], dtype=complex))
    print(f"  幂法: 矩阵 [[2,1],[1,3]] 主特征值 λ = {lam.real:.6f}  "
          f"({it} 次迭代, 精确 3.618034)")
    # ITP 求根示例:
    def f_test(x):
        return x ** 3 - 2.0
    root, f_root, calls = zero_itp(f_test, 0.0, 2.0, epsi=1e-12)
    print(f"  ITP 求根 x³=2:  x = {root:.10f}  "
          f"(精确 2^{1/3:.0f} = {2**(1/3):.10f}), {calls} 次函数调用")


# ============================================================================ #
#         Stage 9: CMA-ES 参数标定 (seed 1201)
# ============================================================================ #
def stage_calibration() -> None:
    section("Stage 9: CMA-ES 参数标定 (seed 1201)")
    # 基准 halo 质量函数:
    M = np.logspace(10, 15, 32)
    dn_bench = sheth_tormen_mass_function(M, z=0.0)
    print(f"  基准 Sheth-Tormen 质量函数:")
    print(f"    M ∈ [{M[0]:.2e}, {M[-1]:.2e}] M_sun/h,  "
          f"dn/dM ∈ [{dn_bench.min():.3e}, {dn_bench.max():.3e}]")
    # CMA-ES 优化:
    res = calibrate_parameters(
        x0=np.array([0.05, 0.01]),
        sigma0=0.1,
        max_gen=15,
        seed=42,
    )
    print(f"  CMA-ES 完成: {res['n_evals']} 次评估")
    print(f"    最优 (ε, Δt) = ({res['best_params'][0]:.4f}, "
          f"{res['best_params'][1]:.4f})")
    print(f"    最小代价 = {res['best_cost']:.4e}")
    # LBFGS 精调:
    from parameter_calibration import lbfgs_refine
    res2 = lbfgs_refine(
        res["best_params"], M, dn_bench,
        max_iter=20,
        bounds=(np.array([0.01, 1e-3]), np.array([0.5, 1.0])),
    )
    print(f"  L-BFGS 精调:  最优 (ε, Δt) = "
          f"({res2['best_params'][0]:.4f}, {res2['best_params'][1]:.4f})")
    print(f"    最小代价 = {res2['best_cost']:.4e}  "
          f"(收敛: {res2['success']})")


# ============================================================================ #
#       Stage 10: 质量赋值与邻居搜索 (seed 786)
# ============================================================================ #
def stage_nas(p: CosmoParams, positions: np.ndarray,
              masses: np.ndarray) -> np.ndarray:
    section("Stage 10: 质量赋值与邻居搜索 (seed 786)")
    rho = cic_deposit(positions, masses, p.n_grid, p.box_length)
    print(f"  CIC 质量赋值: ρ_grid shape = {rho.shape}")
    print(f"    <ρ> = {rho.mean():.3e},  σ_ρ = {rho.std():.3e}")
    # 邻居统计:
    stats = neighbor_count_statistics(positions, p.box_length,
                                      r_smooth=p.box_length / 8.0,
                                      max_samples=500, seed=p.seed)
    print(f"  邻居统计 (r={p.box_length / 8.0:.2f} Mpc/h):")
    print(f"    mean={stats['mean']:.1f}, std={stats['std']:.1f}, "
          f"median={stats['median']:.1f}")
    return rho


# ============================================================================ #
#       Stage 11: 时间积分 (N 体演化)
# ============================================================================ #
def stage_evolution(p: CosmoParams, delta: np.ndarray,
                    positions: np.ndarray, velocities: np.ndarray,
                    masses: np.ndarray) -> dict:
    section("Stage 11: N 体时间积分 (PM leapfrog)")
    # 小规模: 仅演化少量步数:
    p_ev = CosmoParams(**{
        **{k: getattr(p, k) for k in p.__dataclass_fields__},
        "n_steps": min(p.n_steps, 8),
    })
    from time_integrator import run_nbody_simulation
    t0 = time.time()
    result = run_nbody_simulation(delta, positions, velocities, p_ev)
    elapsed = time.time() - t0
    snaps = result["trajectory_snapshots"]
    print(f"  完成 {len(snaps)} 步演化 (耗时 {elapsed:.2f}s)")
    if snaps:
        print(f"    初始 a = {snaps[0]['a']:.4f}  (z = {snaps[0]['z']:.2f})")
        print(f"    最终 a = {snaps[-1]['a']:.4f}  (z = {snaps[-1]['z']:.2f})")
        print(f"    KE 变化: {snaps[0]['KE']:.3e} → {snaps[-1]['KE']:.3e}")
    return result


# ============================================================================ #
#                          Stage 12: 完整诊断
# ============================================================================ #
def stage_diagnostics(p: CosmoParams, delta: np.ndarray,
                      positions: np.ndarray, velocities: np.ndarray,
                      masses: np.ndarray) -> dict:
    section("Stage 12: 完整诊断")
    diag = full_diagnostics(delta, positions, velocities, masses,
                            p.box_length, p.softening_eps)
    print(f"  密度场:  <δ> = {diag['delta_mean']:+.4e},  "
          f"σ_δ = {diag['delta_std']:.4e}")
    print(f"  能量:  KE = {diag['kinetic_energy']:.3e},  "
          f"PE = {diag['potential_energy']:.3e},  "
          f"E_tot = {diag['total_energy']:.3e}")
    # 功率谱斜率:
    k_bins = diag['power_spectrum']['k']
    Pk = diag['power_spectrum']['Pk']
    nz = (k_bins > 0) & (Pk > 0)
    if nz.sum() > 2:
        slope, _ = np.polyfit(np.log10(k_bins[nz]), np.log10(Pk[nz]), 1)
        print(f"  P(k) 斜率 ≈ {slope:.2f}")
    # 相关函数:
    r_bins = diag['correlation']['r']
    xi = diag['correlation']['xi']
    if xi[0] > 0:
        print(f"  ξ(r=0) = {xi[0]:.3e}")
    return diag


# ============================================================================ #
#                                  主函数
# ============================================================================ #
def main() -> None:
    print("\n" + "#" * 72)
    print("#  宇宙大尺度结构 N 体模拟 — 高阶 FD + 稳定性分析")
    print("#  Computational Astrophysics: LSS N-body with High-Order FD")
    print("#" * 72)
    t_start = time.time()
    # Stage 1: 配置
    p = stage_config()
    # Stage 2: Legendre
    stage_legendre()
    # Stage 3: FD
    stage_finite_difference(p)
    # Stage 4: Poisson
    stage_poisson(p)
    # Stage 5: 初始条件
    delta, positions, velocities, masses = stage_initial_conditions(p)
    # Stage 6: 拓扑
    stage_mesh_topology(p, positions)
    # Stage 7: KdV
    stage_kdv_benchmark()
    # Stage 8: 稳定性
    stage_stability()
    # Stage 9: 参数标定
    stage_calibration()
    # Stage 10: NAS
    rho = stage_nas(p, positions, masses)
    # Stage 11: 演化
    result = stage_evolution(p, delta, positions, velocities, masses)
    # Stage 12: 诊断
    diag = stage_diagnostics(p, delta, positions, velocities, masses)
    # 保存结果:
    paths = get_paths()
    results_file = paths["results_dir"] / "summary.txt"
    with open(results_file, "w", encoding="utf-8") as f:
        f.write(f"# Cosmo N-body Simulation Summary\n")
        f.write(f"# Generated at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"# Parameters: {p}\n")
        f.write(f"#\n")
        f.write(f"# Total runtime: {time.time() - t_start:.2f}s\n")
        f.write(f"#\n")
        f.write(f"# Final diagnostics:\n")
        f.write(f"delta_mean = {diag['delta_mean']:.6e}\n")
        f.write(f"delta_std  = {diag['delta_std']:.6e}\n")
        f.write(f"KE         = {diag['kinetic_energy']:.6e}\n")
        f.write(f"PE         = {diag['potential_energy']:.6e}\n")
        f.write(f"E_total    = {diag['total_energy']:.6e}\n")
    print(f"\n  结果已保存至: {results_file}")
    print("\n" + "#" * 72)
    print(f"#  模拟完成 — 总耗时: {time.time() - t_start:.2f}s")
    print("#" * 72 + "\n")


if __name__ == "__main__":
    main()
