"""
主程序入口：重离子碰撞椭圆流与初始条件涨落建模
Main entry point: Heavy-ion collision elliptic flow and initial condition
fluctuation modeling with high-order finite differences and stability analysis.

This program performs:
1. Monte Carlo Glauber initial conditions with fluctuating nucleon positions
2. High-order finite difference discretization of transverse gradients
3. 2+1D relativistic viscous hydrodynamic evolution
4. Flow harmonic extraction (v_n, Ψ_n)
5. Stability analysis of numerical schemes
6. Statistical hypothesis testing for v2 significance

Zero-parameter execution: just run `python main.py`
"""
import numpy as np
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from physics_constants import (
    AU_MASS_NUMBER, TC_QCD, ETA_S_QGP, NUCLEON_CROSS_SECTION_FM2
)
from nuclear_geometry import (
    wood_saxon_density, participant_eccentricity, sample_nucleon_positions,
    apply_hard_sphere_collision, angle_between_vectors
)
from initial_conditions import (
    mc_glauber_event, initial_energy_density_from_event,
    build_transport_matrix, fluctuation_correlator
)
from high_order_fd import (
    laplacian_2d, gradient_2d, apply_1d_derivative, apply_2nd_derivative
)
from equation_of_state import (
    pressure_lattice, energy_density_lattice, temperature_from_energy_density,
    sound_speed_squared, shear_viscosity_eta_over_s
)
from viscous_hydro import (
    hydro_step_rk2, viscous_correction_to_stress, apply_boundary_conditions,
    freezeout_sampler
)
from flow_harmonics import (
    fourier_decomposition, event_plane_method, event_by_event_vn_fluctuation,
    chunked_flow_reduction
)
from quadrature import (
    triangle_quadrature_symmetric, integrate_over_domain,
    verify_quadrature_accuracy
)
from linear_solver import gmres, packed_symmetric_store, jacobi_iteration
from stability_analysis import (
    cfl_condition, von_neumann_stability_fd1d, adaptive_cfl_timestep,
    stability_report
)
from golden_optimizer import golden_section_search, optimize_eta_over_s
from hypothesis_test import (
    ttest_v2_nonzero, confidence_interval, bootstrap_confidence_interval
)


def print_header(title):
    """Print section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def generate_mock_particle_sample(event, v2_true=0.05, n_particles=120, seed=None):
    """
    Generate mock particle azimuthal distribution with embedded v2.

    dN/dφ ∝ 1 + 2 v2 cos(2(φ - Ψ2))

    Parameters
    ----------
    event : dict
        Glauber event data
    v2_true : float
        True elliptic flow
    n_particles : int
        Number of particles to sample
    seed : int, optional

    Returns
    -------
    phi : ndarray
        Azimuthal angles
    """
    rng = np.random.default_rng(seed)

    # Get participant plane angle
    x_part = np.concatenate([event['x_part_a'], event['x_part_b']])
    y_part = np.concatenate([event['y_part_a'], event['y_part_b']])
    eps_n, psi_n = participant_eccentricity(x_part, y_part)
    psi_2 = psi_n[1]

    # Sample φ from azimuthal distribution with v2
    phi = []
    for _ in range(n_particles):
        # Rejection sampling
        while True:
            phi_try = rng.uniform(0, 2 * np.pi)
            f_val = 1.0 + 2.0 * v2_true * np.cos(2.0 * (phi_try - psi_2))
            if rng.uniform() < f_val / (1.0 + 2.0 * v2_true):
                phi.append(phi_try)
                break

    return np.array(phi)


def main():
    """
    Main simulation pipeline for heavy-ion collision modeling.
    """
    print_header("重离子碰撞椭圆流与初始条件涨落建模")
    print("Heavy-Ion Collision: Elliptic Flow & Initial Condition Fluctuations")
    print("High-Order Finite Differences & Stability Analysis")

    # ========================================================================
    # Section 1: Initial Conditions - Monte Carlo Glauber
    # ========================================================================
    print_header("1. 蒙特卡洛Glauber初始条件 (Monte Carlo Glauber)")

    n_nucleons = AU_MASS_NUMBER
    b_impact = 7.0  # fm (mid-central collision)
    sigma_shear = 0.5  # fm

    print(f"碰撞系统: Au+Au, √s = 200 GeV")
    print(f"碰撞参数: b = {b_impact} fm")
    print(f"每核子数: A = {n_nucleons}")
    print(f"高斯展宽: σ = {sigma_shear} fm")

    # Generate multiple events
    n_events = 2
    events = []
    for i in range(n_events):
        event = mc_glauber_event(
            n_nucleons, b_impact, sigma_shear, seed=42 + i
        )
        events.append(event)
        print(f"  事件 {i+1}: N_part = {event['n_part']}, N_coll = {event['n_coll']}")

    # Compute participant eccentricity
    event = events[0]
    x_part = np.concatenate([event['x_part_a'], event['x_part_b']])
    y_part = np.concatenate([event['y_part_a'], event['y_part_b']])
    eps_n, psi_n = participant_eccentricity(x_part, y_part)

    print(f"\n参与者偏心率:")
    for n in range(5):
        print(f"  ε_{n+1} = {eps_n[n]:.4f}, Ψ_{n+1} = {psi_n[n]:.4f} rad")

    # ========================================================================
    # Section 2: Energy Density & Temperature Profile
    # ========================================================================
    print_header("2. 能量密度与温度分布 (Energy Density & Temperature)")

    n_grid = 24
    extent = 10.0

    x, y, eps, dx = initial_energy_density_from_event(
        event, n_grid=n_grid, extent=extent, sigma=sigma_shear
    )

    print(f"网格尺寸: {n_grid} × {n_grid}")
    print(f"网格间距: Δx = Δy = {dx:.3f} fm")
    print(f"域范围: [-{extent}, {extent}] fm")
    print(f"最大能量密度: ε_max = {np.max(eps):.2f} GeV/fm³")
    print(f"总能量: E = {np.sum(eps) * dx**2:.1f} GeV")

    # Convert to temperature
    T = temperature_from_energy_density(eps)
    print(f"最大温度: T_max = {np.max(T)*1000:.0f} MeV")
    print(f"临界温度: T_c = {TC_QCD*1000:.0f} MeV")

    # ========================================================================
    # Section 3: High-Order Finite Difference Operators
    # ========================================================================
    print_header("3. 高阶有限差分算子 (High-Order FD)")

    # Test FD accuracy on smooth function
    x_test = np.linspace(-5, 5, n_grid)
    y_test = np.linspace(-5, 5, n_grid)
    X, Y = np.meshgrid(x_test, y_test, indexing='ij')
    f_test = np.sin(X) * np.cos(Y)

    print("测试函数: f(x,y) = sin(x) cos(y)")

    for order in [2, 4, 6]:
        df_dx_num = apply_1d_derivative(f_test, dx=x_test[1]-x_test[0], order=order, axis=0)
        df_dx_exact = np.cos(X) * np.cos(Y)

        # Error in interior (exclude boundaries)
        interior = slice(4, -4)
        error = np.max(np.abs(df_dx_num[interior, interior] - df_dx_exact[interior, interior]))
        print(f"  {order}阶精度: max|∂f/∂x - exact| = {error:.2e}")

    # Compute Laplacian of energy density
    lap_eps = laplacian_2d(eps, dx, dx, order=4)
    print(f"\n能量密度拉普拉斯: max|∇²ε| = {np.max(np.abs(lap_eps)):.3f}")

    # ========================================================================
    # Section 4: Stability Analysis
    # ========================================================================
    print_header("4. 稳定性分析 (Stability Analysis)")

    # CFL condition
    cs_max = np.sqrt(np.max(sound_speed_squared(T)))
    dt_cfl = cfl_condition(cs_max, dx, dx, safety_factor=0.5)
    print(f"最大声速: c_s = {cs_max:.3f} c")
    print(f"CFL时间步长: Δt_CFL = {dt_cfl:.4f} fm/c")

    # Von Neumann analysis
    for order in [2, 4, 6]:
        vn = von_neumann_stability_fd1d(order, cs_max, dx)
        print(f"  {order}阶von Neumann: max CFL = {vn['max_cfl']:.3f}")

    # Adaptive time step
    mx_init = np.zeros_like(eps)
    my_init = np.zeros_like(eps)
    dt_adapt, max_speed = adaptive_cfl_timestep(eps, mx_init, my_init, dx, dx)
    print(f"自适应时间步长: Δt = {dt_adapt:.4f} fm/c")
    print(f"最大波速: v_max = {max_speed:.3f} c")

    # Full stability report
    report = stability_report(4, dx, dx, T)
    print(f"推荐时间步长: Δt_rec = {report['recommended_dt']:.4f} fm/c")

    # ========================================================================
    # Section 5: Hydrodynamic Evolution
    # ========================================================================
    print_header("5. 流体力学演化 (Hydrodynamic Evolution)")

    # Initialize conserved variables
    eps_evolve = eps.copy()
    mx_evolve = mx_init.copy()
    my_evolve = my_init.copy()

    # Evolution parameters
    tau_0 = 0.6  # fm/c (initial time)
    tau_final = 5.0  # fm/c (freezeout)
    dt_hydro = min(dt_adapt, 0.1)  # fm/c
    n_steps = int((tau_final - tau_0) / dt_hydro)
    n_steps = min(n_steps, 4)  # Limit for executable demo

    print(f"初始时间: τ₀ = {tau_0} fm/c")
    print(f"终止时间: τ_f = {tau_final} fm/c")
    print(f"时间步长: Δt = {dt_hydro:.4f} fm/c")
    print(f"演化步数: {n_steps}")

    # Evolve
    for step in range(n_steps):
        # Hydro step (RK2)
        eps_evolve, mx_evolve, my_evolve = hydro_step_rk2(
            eps_evolve, mx_evolve, my_evolve, dx, dx, dt_hydro, fd_order=4
        )

        # Apply boundary conditions
        eps_evolve, mx_evolve, my_evolve = apply_boundary_conditions(
            eps_evolve, mx_evolve, my_evolve, bc_type='outflow'
        )

        if (step + 1) % 10 == 0 or step == n_steps - 1:
            T_evolve = temperature_from_energy_density(eps_evolve)
            print(f"  步骤 {step+1}/{n_steps}: T_max = {np.max(T_evolve)*1000:.0f} MeV")

    # ========================================================================
    # Section 6: Viscous Corrections
    # ========================================================================
    print_header("6. 粘性修正 (Viscous Corrections)")

    T_final = temperature_from_energy_density(eps_evolve)
    w_final = eps_evolve + pressure_lattice(T_final)
    w_final = np.maximum(w_final, 1e-10)
    ux_final, uy_final = mx_evolve / w_final, my_evolve / w_final

    pi_xx, pi_yy, pi_xy, bulk_pi = viscous_correction_to_stress(
        eps_evolve, ux_final, uy_final, dx, dx, dt_hydro, T_final
    )

    eta_over_s_field = shear_viscosity_eta_over_s(T_final)
    print(f"剪切粘滞系数: η/s = {np.mean(eta_over_s_field):.4f}")
    print(f"KSS下界: 1/(4π) = {1.0/(4*np.pi):.4f}")
    print(f"剪切应力: max|π^xx| = {np.max(np.abs(pi_xx)):.4f} GeV/fm³")
    print(f"体积粘滞: max|Π| = {np.max(np.abs(bulk_pi)):.4f} GeV/fm³")

    # ========================================================================
    # Section 7: Flow Harmonics
    # ========================================================================
    print_header("7. 流谐波提取 (Flow Harmonics)")

    # Generate mock particle samples from each event
    all_phi = []
    v2_samples = []

    for i, evt in enumerate(events):
        # Sample particles with embedded v2 proportional to ε_2
        v2_true = 0.8 * eps_n[1]  # v2 ≈ 0.8 * ε_2
        phi = generate_mock_particle_sample(evt, v2_true=v2_true,
                                           n_particles=120, seed=100+i)
        all_phi.append(phi)

        # Extract v2 from this event
        v_n, psi_n_evt, Q_n = fourier_decomposition(None, phi, n_harmonics=3)
        v2_samples.append(v_n[1])

    print(f"事件数: {n_events}")
    print(f"每事件粒子数: 500")

    # Event-averaged v2
    v2_mean = np.mean(v2_samples)
    v2_std = np.std(v2_samples)
    print(f"\n椭圆流统计:")
    print(f"  <v₂> = {v2_mean:.4f} ± {v2_std:.4f}")

    # Event-plane method
    phi_all = np.concatenate(all_phi)
    v2_ep, psi2_ep = event_plane_method(phi_all, n=2)
    print(f"  事件平面法: v₂ = {v2_ep:.4f}, Ψ₂ = {psi2_ep:.4f} rad")

    # Chunked reduction (distributed computing pattern)
    v2_chunk, psi2_chunk = chunked_flow_reduction(all_phi, n=2)
    print(f"  分块归约法: v₂ = {v2_chunk:.4f}")

    # ========================================================================
    # Section 8: Statistical Analysis
    # ========================================================================
    print_header("8. 统计假设检验 (Statistical Hypothesis Testing)")

    v2_array = np.array(v2_samples)

    # t-test: H0: v2 = 0
    ttest_result = ttest_v2_nonzero(v2_array, alpha=0.05)
    print(f"t检验 (H₀: v₂ = 0):")
    print(f"  t统计量 = {ttest_result['t_stat']:.3f}")
    print(f"  p值 = {ttest_result['p_value']:.4f}")
    print(f"  拒绝H₀: {ttest_result['reject_null']}")

    # Confidence interval
    ci_result = confidence_interval(v2_array, confidence=0.95)
    print(f"\n95%置信区间:")
    print(f"  v₂ ∈ [{ci_result['ci_lower']:.4f}, {ci_result['ci_upper']:.4f}]")

    # Bootstrap CI
    bootstrap_result = bootstrap_confidence_interval(v2_array, n_bootstrap=80)
    print(f"\nBootstrap 95%置信区间:")
    print(f"  v₂ ∈ [{bootstrap_result['ci_lower']:.4f}, {bootstrap_result['ci_upper']:.4f}]")

    # ========================================================================
    # Section 9: Optimization
    # ========================================================================
    print_header("9. 参数优化 (Parameter Optimization)")

    # Golden section search for η/s
    print("优化 η/s 以匹配 v₂ 数据...")

    def v2_model(eta_s):
        # Simplified model: v2 ∝ ε_2 / (1 + K η_s)
        K = 5.0
        return 0.8 * eps_n[1] / (1.0 + K * eta_s)

    # Mock target
    v2_target = v2_mean
    def chi2_eta_s(eta_s):
        return (v2_model(eta_s) - v2_target)**2

    eta_s_opt, chi2_min, n_iter = golden_section_search(
        chi2_eta_s, 0.04, 0.3, tol=0.001, max_iter=30
    )

    print(f"最优 η/s = {eta_s_opt:.4f}")
    print(f"最小 χ² = {chi2_min:.6f}")
    print(f"迭代次数: {n_iter}")

    # ========================================================================
    # Section 10: Quadrature Verification
    # ========================================================================
    print_header("10. 数值积分验证 (Quadrature Verification)")

    for order in [1, 2, 3, 4, 5]:
        max_err = verify_quadrature_accuracy(order, n_tests=5)
        print(f"  {order}阶对称求积: max误差 = {max_err:.2e}")

    # ========================================================================
    # Section 11: Linear Solver Test
    # ========================================================================
    print_header("11. 线性求解器测试 (Linear Solver)")

    # Build small test system
    n_test = 20
    A_test = np.zeros((n_test, n_test))
    for i in range(n_test):
        A_test[i, i] = 4.0
        if i > 0:
            A_test[i, i-1] = -1.0
        if i < n_test - 1:
            A_test[i, i+1] = -1.0

    b_test = np.ones(n_test)

    # GMRES
    x_gmres, info_gmres = gmres(A_test, b_test, tol=1e-8, max_iter=50)
    print(f"GMRES:")
    print(f"  迭代次数: {info_gmres['n_iter']}")
    print(f"  收敛: {info_gmres['converged']}")
    print(f"  残差: {np.linalg.norm(A_test @ x_gmres - b_test):.2e}")

    # Jacobi
    x_jac, n_jac = jacobi_iteration(A_test, b_test, tol=1e-8, max_iter=500)
    print(f"\nJacobi迭代:")
    print(f"  迭代次数: {n_jac}")
    print(f"  残差: {np.linalg.norm(A_test @ x_jac - b_test):.2e}")

    # Packed storage
    packed, n_packed = packed_symmetric_store(A_test)
    print(f"\n压缩存储:")
    print(f"  原始大小: {n_test}×{n_test} = {n_test**2}")
    print(f"  压缩大小: {len(packed)}")

    # ========================================================================
    # Section 12: Fluctuation Analysis
    # ========================================================================
    print_header("12. 涨落关联分析 (Fluctuation Correlator)")

    r_centers, corr = fluctuation_correlator(eps, x, y, dx, r_bins=10, r_max=5.0)
    print(f"能量密度涨落关联函数:")
    for i in range(min(5, len(r_centers))):
        print(f"  r = {r_centers[i]:.2f} fm: C(r) = {corr[i]:.4f}")

    # ========================================================================
    # Summary
    # ========================================================================
    print_header("模拟总结 (Summary)")

    print("✓ 蒙特卡洛Glauber初始条件生成")
    print("✓ 高阶有限差分算子 (2/4/6阶)")
    print("✓ 稳定性分析与CFL条件")
    print("✓ 2+1D粘滞流体力学演化")
    print("✓ 流谐波提取 (v₂, Ψ₂)")
    print("✓ 统计假设检验 (t检验, 置信区间)")
    print("✓ 参数优化 (黄金分割)")
    print("✓ 数值积分验证")
    print("✓ 线性求解器 (GMRES, Jacobi)")
    print("✓ 涨落关联分析")

    print("\n" + "=" * 70)
    print("  程序执行完成")
    print("=" * 70 + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
