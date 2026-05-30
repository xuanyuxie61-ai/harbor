
import numpy as np
import time


def run_section(title, func):
    print("=" * 70)
    print(f"  {title}")
    print("=" * 70)
    t0 = time.time()
    result = func()
    t1 = time.time()
    print(f"  Completed in {t1 - t0:.3f} seconds\n")
    return result


def stage_1_combinatorial_analysis():
    from utils import (
        stirling_numbers_2, bell_numbers, ion_channel_state_enumeration,
        compute_relative_error, convergence_rate, catastrophic_cancellation_test,
        generate_gray_code
    )
    
    print("  [1a] Stirling numbers S(5,2) =", stirling_numbers_2(5, 2))
    print("  [1b] Bell number B(5) =", bell_numbers(5))
    
    total, configs = ion_channel_state_enumeration(4, 2)
    print(f"  [1c] Ion channel states (4 gates, 2 open): {total} configurations")
    

    err_test = catastrophic_cancellation_test()
    print(f"  [1d] Catastrophic cancellation: R={err_test['computed']:.6f}, rel_err={err_test['relative_error']:.2e}")
    

    h_vals = np.array([0.1, 0.05, 0.025, 0.0125])
    errors = h_vals ** 2
    rate = convergence_rate(errors, h_vals)
    print(f"  [1e] Estimated convergence rate: {rate:.3f}")
    

    gray = generate_gray_code(3)
    print(f"  [1f] 3-bit Gray code: {gray}")
    
    return {'stirling_5_2': stirling_numbers_2(5, 2), 'bell_5': bell_numbers(5)}


def stage_2_quadrature_and_integration():
    from numerical_integration import (
        quadrilateral_witherden_rule, integrate_2d_quadrilateral,
        monte_carlo_integral_1d, monte_carlo_integral_2d
    )
    

    def f_test(x, y):
        return x ** 2 + y ** 2
    
    exact = 2.0 / 3.0
    
    for p in [1, 3, 5, 7, 9]:
        n, x, y, w = quadrilateral_witherden_rule(p)
        approx = np.sum(w * (x ** 2 + y ** 2))
        err = abs(approx - exact)
        print(f"  [2a] Witherden rule p={p:2d}: n={n:2d}, integral={approx:.10f}, error={err:.2e}")
    

    mc_est, mc_err = monte_carlo_integral_1d(lambda x: x ** 2, 0.0, 1.0, 10000)
    print(f"  [2b] MC integral of x^2 on [0,1]: {mc_est:.6f} ± {mc_err:.2e}")
    

    mc2_est, mc2_err = monte_carlo_integral_2d(lambda x, y: x * y, (0.0, 1.0), (0.0, 1.0), 5000)
    print(f"  [2c] MC integral of x*y on [0,1]^2: {mc2_est:.6f} ± {mc2_err:.2e}")
    
    return {'quadrature_test': 'passed'}


def stage_3_stochastic_sampling():
    from stochastic_sampler import (
        generate_niederreiter_sequence, estimate_area_qmc,
        sample_conductivity_parameters, compute_discrepancy
    )
    

    points = generate_niederreiter_sequence(2, 1000)
    disc = compute_discrepancy(points)
    print(f"  [3a] Niederreiter sequence L2 discrepancy: {disc:.6f}")
    

    polygon = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
    area_est, bbox_area = estimate_area_qmc(polygon, (0, 1, 0, 1), 5000)
    print(f"  [3b] QMC area estimate (unit square): {area_est:.6f} (exact=1.0)")
    

    samples = sample_conductivity_parameters(10, method='niederreiter')
    print(f"  [3c] Sampled conductivity parameters (σ_f, σ_t, σ_n):")
    for i in range(min(5, len(samples))):
        print(f"       Sample {i + 1}: ({samples[i, 0]:.4f}, {samples[i, 1]:.4f}, {samples[i, 2]:.4f})")
    
    return {'discrepancy': disc, 'area_estimate': area_est}


def stage_4_mesh_generation():
    from mesh_generator import generate_cardiac_mesh, polygon_contains_point
    
    nodes, polygon = generate_cardiac_mesh(200, model='ventricle', n_cvt_iter=20)
    print(f"  [4a] Generated {len(nodes)} mesh nodes for ventricle model")
    

    test_point = np.array([0.0, 0.0])
    inside = polygon_contains_point(polygon, test_point)
    print(f"  [4b] Point (0,0) inside ventricle polygon: {inside}")
    
    return {'n_nodes': len(nodes), 'polygon': polygon}


def stage_5_linear_algebra():
    from linear_algebra_core import (
        r8pbu_cg, build_laplacian_banded, power_method, solve_poisson_2d_cg
    )
    

    nx, ny = 16, 16
    n, mu, a = build_laplacian_banded(nx, ny, 0.1, 0.1)
    b = np.ones(n)
    x0 = np.zeros(n)
    x, res, iters = r8pbu_cg(n, mu, a, b, x0, tol=1e-10)
    print(f"  [5a] CG solve: residual={res:.2e}, iterations={iters}/{n}")
    

    A_test = np.diag([5.0, 3.0, 1.0, 0.5])
    y0 = np.random.randn(4)
    y, lam, it_num = power_method(A_test, y0, it_max=100, tol=1e-10)
    print(f"  [5b] Power method: λ_max={lam:.6f} (expected=5.0), iterations={it_num}")
    

    f = np.ones((nx, ny))
    phi, res_p, iters_p = solve_poisson_2d_cg(f, nx, ny, 0.1, 0.1, max_iter=500)
    print(f"  [5c] Poisson solve: residual={res_p:.2e}, iterations={iters_p}")
    
    return {'cg_residual': res, 'power_lambda': lam}


def stage_6_ion_channel_dynamics():
    from ion_channel_dynamics import (
        single_cell_ap_model, squircle_ode_integrate,
        gate_alpha_beta, aliev_panfilov_reaction
    )
    

    alpha_m, beta_m = gate_alpha_beta(-40.0, 'm')
    print(f"  [6a] Na+ m-gate at V=-40mV: α={alpha_m:.4f}, β={beta_m:.4f}")
    

    print("  [6b] Simulating single-cell action potential...")
    t_ap, v_ap, gates = single_cell_ap_model(t_max=400.0, dt=0.05, stim_period=300.0)
    v_max = np.max(v_ap)
    v_min = np.min(v_ap)
    print(f"  [6c] AP: V_max={v_max:.2f}mV, V_min={v_min:.2f}mV, duration={t_ap[-1]:.1f}ms")
    

    t_sq, u_sq, v_sq, H_sq = squircle_ode_integrate((1.0, 0.0), (0.0, 10.0), s=4.0, n_steps=1000)
    H_drift = np.max(np.abs(H_sq - H_sq[0]))
    print(f"  [6d] Squircle ODE: H_drift={H_drift:.2e} (conservation check)")
    

    u_test = np.array([[0.5, 0.8], [0.2, 0.1]])
    v_test = np.array([[0.3, 0.1], [0.5, 0.6]])
    f_r, g_r = aliev_panfilov_reaction(u_test, v_test)
    print(f"  [6e] Aliev-Panfilov reaction: f_avg={np.mean(f_r):.4f}, g_avg={np.mean(g_r):.4f}")
    
    return {'ap_vmax': v_max, 'ap_vmin': v_min, 'H_drift': H_drift}


def stage_7_tissue_simulation():
    from electrophysiology_simulator import run_full_simulation
    
    print("  [7a] Running small-scale tissue simulation for validation...")
    results = run_full_simulation(
        nx=48, ny=48, T=300.0, dt=0.05, dx=0.05,
        D_f=0.001, D_t=0.0002,
        a=0.1, k=8.0, mu1=0.2, mu2=0.3, eps=0.002,
        solver='adi',
        n_stimuli=2, stim_period=150.0,
        fiber_model='parallel',
        add_noise=True, noise_level=0.005
    )
    
    print(f"  [7b] Wavefront velocity: {results['velocity']:.4f} cm/ms")
    print(f"  [7c] Action Potential Duration (APD): {results['apd']:.2f} ms")
    print(f"  [7d] Wavelength: {results['wavelength']:.4f} cm")
    print(f"  [7e] Effective Refractory Period (ERP): {results['erp']:.2f} ms")
    print(f"  [7f] Reentrant activity detected: {results['reentrant_detected']}")
    print(f"  [7g] Arrhythmia risk index: {results['risk_index']:.4f}")
    print(f"  [7h] Stability eigenvalue: {results['lambda_max']:.4f}")
    print(f"  [7i] System stable: {results['is_stable']}")
    
    return results


def stage_8_scattered_interpolation():
    from mesh_generator import scattered_interpolation_2d
    

    n_data = 50
    data_points = np.random.rand(n_data, 2)
    data_values = np.sin(2 * np.pi * data_points[:, 0]) * np.cos(2 * np.pi * data_points[:, 1])
    

    query_points = np.array([[0.5, 0.5], [0.25, 0.75], [0.8, 0.2]])
    interpolated = scattered_interpolation_2d(data_points, data_values, query_points)
    
    print(f"  [8a] Scattered interpolation test:")
    for i, qp in enumerate(query_points):
        exact = np.sin(2 * np.pi * qp[0]) * np.cos(2 * np.pi * qp[1])
        print(f"       Point ({qp[0]:.2f},{qp[1]:.2f}): exact={exact:.4f}, interp={interpolated[i]:.4f}")
    
    return {'interpolated': interpolated}


def stage_9_parameter_study():
    from electrophysiology_simulator import run_full_simulation
    
    print("  [9a] Parameter sensitivity study (eps variation)...")
    
    eps_values = [0.001, 0.002, 0.005, 0.01]
    results_list = []
    
    for eps in eps_values:
        res = run_full_simulation(
            nx=32, ny=32, T=200.0, dt=0.1, dx=0.05,
            D_f=0.001, D_t=0.0002,
            eps=eps,
            solver='forward_euler',
            n_stimuli=1, stim_period=200.0,
            add_noise=False
        )
        results_list.append({
            'eps': eps,
            'velocity': res['velocity'],
            'apd': res['apd'],
            'risk': res['risk_index']
        })
        print(f"       eps={eps:.3f}: v={res['velocity']:.4f}, APD={res['apd']:.1f}, risk={res['risk_index']:.4f}")
    
    return results_list


def print_summary(results):
    print("\n" + "=" * 70)
    print("  SIMULATION SUMMARY")
    print("=" * 70)
    print(f"  Domain: {results['nx']} x {results['ny']} grid, dx={results['dx']}cm")
    print(f"  Simulation time: {results['T']}ms, dt={results['dt']}ms, solver={results['solver']}")
    print(f"  Fiber model: {results['fiber_model']}")
    print(f"  Conduction velocity: {results['velocity']:.4f} cm/ms")
    print(f"  Action Potential Duration: {results['apd']:.2f} ms")
    print(f"  Wavelength: {results['wavelength']:.4f} cm")
    print(f"  Effective Refractory Period: {results['erp']:.2f} ms")
    print(f"  Reentrant activity: {'YES' if results['reentrant_detected'] else 'NO'}")
    print(f"  Arrhythmia risk index: {results['risk_index']:.4f}")
    print(f"  Max eigenvalue: {results['lambda_max']:.4f}")
    print(f"  System stability: {'STABLE' if results['is_stable'] else 'UNSTABLE'}")
    print("=" * 70)


def main():
    print("\n" + "=" * 70)
    print("  CARDIAC ELECTROPHYSIOLOGY & ARRHYTHMIA SIMULATION")
    print("  Project 121: Biomedical Science - Heart Modeling")
    print("=" * 70 + "\n")
    
    total_start = time.time()
    

    run_section("Stage 1: Combinatorial Analysis & Error Estimation",
                stage_1_combinatorial_analysis)
    

    run_section("Stage 2: Quadrature Rules & Monte Carlo Integration",
                stage_2_quadrature_and_integration)
    

    run_section("Stage 3: Quasi-Random Sampling & Parameter Exploration",
                stage_3_stochastic_sampling)
    

    run_section("Stage 4: Cardiac Mesh Generation",
                stage_4_mesh_generation)
    

    run_section("Stage 5: Linear Algebra Solvers (CG & Power Method)",
                stage_5_linear_algebra)
    

    run_section("Stage 6: Ion Channel Dynamics & Single-Cell AP",
                stage_6_ion_channel_dynamics)
    

    tissue_results = run_section("Stage 7: Tissue-Level Reaction-Diffusion Simulation",
                                  stage_7_tissue_simulation)
    

    run_section("Stage 8: Scattered Data Interpolation",
                stage_8_scattered_interpolation)
    

    run_section("Stage 9: Parameter Sensitivity Study",
                stage_9_parameter_study)
    

    print_summary(tissue_results)
    
    total_end = time.time()
    print(f"\n  Total execution time: {total_end - total_start:.3f} seconds")
    print("  Simulation completed successfully.")
    print("=" * 70 + "\n")
    
    return tissue_results


if __name__ == "__main__":
    main()
