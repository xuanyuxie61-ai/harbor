"""
main.py
心脏电生理与心律失常模拟 — 统一入口

生物医学领域: 心脏电生理与心律失常模拟

本程序零参数运行，自动执行以下完整流程:
1. 心脏组织几何与网格生成（CVT + 边界裁剪）
2. 心肌纤维角度场生成
3. 单细胞离子通道动力学验证
4. 组织层面反应扩散方程求解（Monodomain模型）
5. 准随机参数采样与统计积分
6. 数值稳定性分析与特征值计算
7. 心律失常指标评估

运行方式:
    python main.py
"""

import numpy as np
import time


def run_section(title, func):
    """辅助函数: 运行并计时一个模拟阶段"""
    print("=" * 70)
    print(f"  {title}")
    print("=" * 70)
    t0 = time.time()
    result = func()
    t1 = time.time()
    print(f"  Completed in {t1 - t0:.3f} seconds\n")
    return result


def stage_1_combinatorial_analysis():
    """阶段1: 组合数学与误差分析验证"""
    from utils import (
        stirling_numbers_2, bell_numbers, ion_channel_state_enumeration,
        compute_relative_error, convergence_rate, catastrophic_cancellation_test,
        generate_gray_code
    )
    
    print("  [1a] Stirling numbers S(5,2) =", stirling_numbers_2(5, 2))
    print("  [1b] Bell number B(5) =", bell_numbers(5))
    
    total, configs = ion_channel_state_enumeration(4, 2)
    print(f"  [1c] Ion channel states (4 gates, 2 open): {total} configurations")
    
    # 数值误差测试
    err_test = catastrophic_cancellation_test()
    print(f"  [1d] Catastrophic cancellation: R={err_test['computed']:.6f}, rel_err={err_test['relative_error']:.2e}")
    
    # 收敛率测试
    h_vals = np.array([0.1, 0.05, 0.025, 0.0125])
    errors = h_vals ** 2  # 模拟二阶收敛
    rate = convergence_rate(errors, h_vals)
    print(f"  [1e] Estimated convergence rate: {rate:.3f}")
    
    # 格雷码
    gray = generate_gray_code(3)
    print(f"  [1f] 3-bit Gray code: {gray}")
    
    return {'stirling_5_2': stirling_numbers_2(5, 2), 'bell_5': bell_numbers(5)}


def stage_2_quadrature_and_integration():
    """阶段2: 高斯求积与蒙特卡洛积分验证"""
    from numerical_integration import (
        quadrilateral_witherden_rule, integrate_2d_quadrilateral,
        monte_carlo_integral_1d, monte_carlo_integral_2d
    )
    
    # 测试Witherden求积规则精度
    def f_test(x, y):
        return x ** 2 + y ** 2
    
    exact = 2.0 / 3.0  # ∫∫_{[0,1]^2} (x^2+y^2) dx dy = 2/3
    
    for p in [1, 3, 5, 7, 9]:
        n, x, y, w = quadrilateral_witherden_rule(p)
        approx = np.sum(w * (x ** 2 + y ** 2))
        err = abs(approx - exact)
        print(f"  [2a] Witherden rule p={p:2d}: n={n:2d}, integral={approx:.10f}, error={err:.2e}")
    
    # 1D蒙特卡洛积分
    mc_est, mc_err = monte_carlo_integral_1d(lambda x: x ** 2, 0.0, 1.0, 10000)
    print(f"  [2b] MC integral of x^2 on [0,1]: {mc_est:.6f} ± {mc_err:.2e}")
    
    # 2D蒙特卡洛积分
    mc2_est, mc2_err = monte_carlo_integral_2d(lambda x, y: x * y, (0.0, 1.0), (0.0, 1.0), 5000)
    print(f"  [2c] MC integral of x*y on [0,1]^2: {mc2_est:.6f} ± {mc2_err:.2e}")
    
    return {'quadrature_test': 'passed'}


def stage_3_stochastic_sampling():
    """阶段3: 准随机序列与参数采样"""
    from stochastic_sampler import (
        generate_niederreiter_sequence, estimate_area_qmc,
        sample_conductivity_parameters, compute_discrepancy
    )
    
    # Niederreiter序列
    points = generate_niederreiter_sequence(2, 1000)
    disc = compute_discrepancy(points)
    print(f"  [3a] Niederreiter sequence L2 discrepancy: {disc:.6f}")
    
    # 准蒙特卡洛面积估计
    polygon = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
    area_est, bbox_area = estimate_area_qmc(polygon, (0, 1, 0, 1), 5000)
    print(f"  [3b] QMC area estimate (unit square): {area_est:.6f} (exact=1.0)")
    
    # 电导率参数采样
    samples = sample_conductivity_parameters(10, method='niederreiter')
    print(f"  [3c] Sampled conductivity parameters (σ_f, σ_t, σ_n):")
    for i in range(min(5, len(samples))):
        print(f"       Sample {i + 1}: ({samples[i, 0]:.4f}, {samples[i, 1]:.4f}, {samples[i, 2]:.4f})")
    
    return {'discrepancy': disc, 'area_estimate': area_est}


def stage_4_mesh_generation():
    """阶段4: 心脏组织网格生成"""
    from mesh_generator import generate_cardiac_mesh, polygon_contains_point
    
    nodes, polygon = generate_cardiac_mesh(200, model='ventricle', n_cvt_iter=20)
    print(f"  [4a] Generated {len(nodes)} mesh nodes for ventricle model")
    
    # 验证点在多边形内
    test_point = np.array([0.0, 0.0])
    inside = polygon_contains_point(polygon, test_point)
    print(f"  [4b] Point (0,0) inside ventricle polygon: {inside}")
    
    return {'n_nodes': len(nodes), 'polygon': polygon}


def stage_5_linear_algebra():
    """阶段5: 线性代数求解器验证"""
    from linear_algebra_core import (
        r8pbu_cg, build_laplacian_banded, power_method, solve_poisson_2d_cg
    )
    
    # 测试共轭梯度法
    nx, ny = 16, 16
    n, mu, a = build_laplacian_banded(nx, ny, 0.1, 0.1)
    b = np.ones(n)
    x0 = np.zeros(n)
    x, res, iters = r8pbu_cg(n, mu, a, b, x0, tol=1e-10)
    print(f"  [5a] CG solve: residual={res:.2e}, iterations={iters}/{n}")
    
    # 幂法测试
    A_test = np.diag([5.0, 3.0, 1.0, 0.5])
    y0 = np.random.randn(4)
    y, lam, it_num = power_method(A_test, y0, it_max=100, tol=1e-10)
    print(f"  [5b] Power method: λ_max={lam:.6f} (expected=5.0), iterations={it_num}")
    
    # 泊松方程求解
    f = np.ones((nx, ny))
    phi, res_p, iters_p = solve_poisson_2d_cg(f, nx, ny, 0.1, 0.1, max_iter=500)
    print(f"  [5c] Poisson solve: residual={res_p:.2e}, iterations={iters_p}")
    
    return {'cg_residual': res, 'power_lambda': lam}


def stage_6_ion_channel_dynamics():
    """阶段6: 单细胞离子通道动力学验证"""
    from ion_channel_dynamics import (
        single_cell_ap_model, squircle_ode_integrate,
        gate_alpha_beta, aliev_panfilov_reaction
    )
    
    # 门控速率测试
    alpha_m, beta_m = gate_alpha_beta(-40.0, 'm')
    print(f"  [6a] Na+ m-gate at V=-40mV: α={alpha_m:.4f}, β={beta_m:.4f}")
    
    # 单细胞动作电位
    print("  [6b] Simulating single-cell action potential...")
    t_ap, v_ap, gates = single_cell_ap_model(t_max=400.0, dt=0.05, stim_period=300.0)
    v_max = np.max(v_ap)
    v_min = np.min(v_ap)
    print(f"  [6c] AP: V_max={v_max:.2f}mV, V_min={v_min:.2f}mV, duration={t_ap[-1]:.1f}ms")
    
    # Squircle ODE守恒量测试
    t_sq, u_sq, v_sq, H_sq = squircle_ode_integrate((1.0, 0.0), (0.0, 10.0), s=4.0, n_steps=1000)
    H_drift = np.max(np.abs(H_sq - H_sq[0]))
    print(f"  [6d] Squircle ODE: H_drift={H_drift:.2e} (conservation check)")
    
    # Aliev-Panfilov反应项测试
    u_test = np.array([[0.5, 0.8], [0.2, 0.1]])
    v_test = np.array([[0.3, 0.1], [0.5, 0.6]])
    f_r, g_r = aliev_panfilov_reaction(u_test, v_test)
    print(f"  [6e] Aliev-Panfilov reaction: f_avg={np.mean(f_r):.4f}, g_avg={np.mean(g_r):.4f}")
    
    return {'ap_vmax': v_max, 'ap_vmin': v_min, 'H_drift': H_drift}


def stage_7_tissue_simulation():
    """阶段7: 组织层面反应扩散模拟"""
    from electrophysiology_simulator import run_full_simulation
    
    print("  [7a] Running small-scale tissue simulation for validation...")
    results = run_full_simulation(
        nx=48, ny=48, T=300.0, dt=0.05, dx=0.05,
        D_f=0.001, D_t=0.0002,
        a=0.1, k=8.0, mu1=0.2, mu2=0.3, eps=0.002,
        solver='forward_euler',
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
    """阶段8: 散乱数据插值验证"""
    from mesh_generator import scattered_interpolation_2d
    
    # 创建散乱数据
    n_data = 50
    data_points = np.random.rand(n_data, 2)
    data_values = np.sin(2 * np.pi * data_points[:, 0]) * np.cos(2 * np.pi * data_points[:, 1])
    
    # 查询网格点
    query_points = np.array([[0.5, 0.5], [0.25, 0.75], [0.8, 0.2]])
    interpolated = scattered_interpolation_2d(data_points, data_values, query_points)
    
    print(f"  [8a] Scattered interpolation test:")
    for i, qp in enumerate(query_points):
        exact = np.sin(2 * np.pi * qp[0]) * np.cos(2 * np.pi * qp[1])
        print(f"       Point ({qp[0]:.2f},{qp[1]:.2f}): exact={exact:.4f}, interp={interpolated[i]:.4f}")
    
    return {'interpolated': interpolated}


def stage_9_parameter_study():
    """阶段9: 参数敏感性研究"""
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
    """打印模拟结果摘要"""
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
    """主函数：零参数运行完整模拟流程"""
    print("\n" + "=" * 70)
    print("  CARDIAC ELECTROPHYSIOLOGY & ARRHYTHMIA SIMULATION")
    print("  Project 121: Biomedical Science - Heart Modeling")
    print("=" * 70 + "\n")
    
    total_start = time.time()
    
    # 阶段1: 组合数学与误差分析
    run_section("Stage 1: Combinatorial Analysis & Error Estimation",
                stage_1_combinatorial_analysis)
    
    # 阶段2: 数值积分验证
    run_section("Stage 2: Quadrature Rules & Monte Carlo Integration",
                stage_2_quadrature_and_integration)
    
    # 阶段3: 随机采样
    run_section("Stage 3: Quasi-Random Sampling & Parameter Exploration",
                stage_3_stochastic_sampling)
    
    # 阶段4: 网格生成
    run_section("Stage 4: Cardiac Mesh Generation",
                stage_4_mesh_generation)
    
    # 阶段5: 线性代数
    run_section("Stage 5: Linear Algebra Solvers (CG & Power Method)",
                stage_5_linear_algebra)
    
    # 阶段6: 离子通道动力学
    run_section("Stage 6: Ion Channel Dynamics & Single-Cell AP",
                stage_6_ion_channel_dynamics)
    
    # 阶段7: 组织模拟（核心）
    tissue_results = run_section("Stage 7: Tissue-Level Reaction-Diffusion Simulation",
                                  stage_7_tissue_simulation)
    
    # 阶段8: 散乱插值
    run_section("Stage 8: Scattered Data Interpolation",
                stage_8_scattered_interpolation)
    
    # 阶段9: 参数研究
    run_section("Stage 9: Parameter Sensitivity Study",
                stage_9_parameter_study)
    
    # 打印摘要
    print_summary(tissue_results)
    
    total_end = time.time()
    print(f"\n  Total execution time: {total_end - total_start:.3f} seconds")
    print("  Simulation completed successfully.")
    print("=" * 70 + "\n")
    
    return tissue_results


if __name__ == "__main__":
    main()

# ================================================================
# 测试用例（40个，assert模式，涉及随机值均使用固定种子）
# ================================================================

# ---- TC01: Stirling number S(5,2) = 15 ----
from utils import stirling_numbers_2
assert stirling_numbers_2(5, 2) == 15, '[TC01] Stirling S(5,2) should be 15 FAILED'

# ---- TC02: Bell number B(5) = 52 ----
from utils import bell_numbers
assert bell_numbers(5) == 52, '[TC02] Bell B(5) should be 52 FAILED'

# ---- TC03: Ion channel state enumeration total = C(4,2) ----
from utils import ion_channel_state_enumeration
total, configs = ion_channel_state_enumeration(4, 2)
assert total == 6, '[TC03] Ion channel enumeration total should be 6 FAILED'
assert len(configs) == 6, '[TC03] Config count should match FAILED'

# ---- TC04: Catastrophic cancellation rel_error > 0 ----
from utils import catastrophic_cancellation_test
result = catastrophic_cancellation_test()
assert result['relative_error'] >= 0, '[TC04] Relative error should be non-negative FAILED'
assert isinstance(result['computed'], float), '[TC04] Computed value should be float FAILED'

# ---- TC05: Convergence rate positive for decreasing errors ----
from utils import convergence_rate
import numpy as np
h_vals = np.array([0.1, 0.05, 0.025, 0.0125])
errors = h_vals ** 2
rate = convergence_rate(errors, h_vals)
assert rate > 0, '[TC05] Convergence rate should be positive for decreasing errors FAILED'

# ---- TC06: 3-bit Gray code ----
from utils import generate_gray_code
gray = generate_gray_code(3)
assert gray == [0, 1, 3, 2, 6, 7, 5, 4], '[TC06] 3-bit Gray code is incorrect FAILED'

# ---- TC07: Subset lex rank/unrank roundtrip ----
from utils import subset_lex_rank, subset_lex_unrank
import numpy as np
np.random.seed(42)
n = 5
a_orig = [True, False, True, False, True]
rank = subset_lex_rank(n, a_orig)
b = subset_lex_unrank(n, rank)
assert a_orig == b, '[TC07] Subset rank/unrank roundtrip FAILED'

# ---- TC08: Permutation lex rank/unrank roundtrip ----
from utils import perm_lex_rank, perm_lex_unrank
p = [2, 3, 1, 4]
rank = perm_lex_rank(4, p)
p2 = perm_lex_unrank(4, rank)
assert p == p2, '[TC08] Permutation rank/unrank roundtrip FAILED'

# ---- TC09: Condition number of identity = 1 ----
from utils import condition_number_analysis
import numpy as np
I = np.eye(4)
kappa = condition_number_analysis(I)
assert abs(kappa - 1.0) < 1e-10, '[TC09] Identity condition number should be 1 FAILED'

# ---- TC10: Relative error formula ----
from utils import compute_relative_error
assert abs(compute_relative_error(1.0, 0.9) - 0.1) < 1e-12, '[TC10] Relative error (1,0.9) should be 0.1 FAILED'
assert compute_relative_error(0.0, 0.0) == 0.0, '[TC10] Zero/zero relative error FAILED'
assert compute_relative_error(0.0, 1.0) == 1.0, '[TC10] Zero exact relative error FAILED'

# ---- TC11: Witherden rule p=9 returns valid points and weights ----
from numerical_integration import quadrilateral_witherden_rule
import numpy as np
n, x, y, w = quadrilateral_witherden_rule(9)
assert n == len(x) == len(y) == len(w), '[TC11] Witherden rule point/weight count mismatch FAILED'
assert n > 0, '[TC11] Witherden rule should have points FAILED'
assert abs(np.sum(w) - 1.0) < 1e-10, '[TC11] Witherden rule weights should sum to 1 FAILED'
assert np.all((x >= 0) & (x <= 1)), '[TC11] Witherden x coords in [0,1] FAILED'
assert np.all((y >= 0) & (y <= 1)), '[TC11] Witherden y coords in [0,1] FAILED'

# ---- TC12: Quadrilateral integral of constant = 1 ----
from numerical_integration import integrate_2d_quadrilateral
result = integrate_2d_quadrilateral(lambda x, y: 1.0, precision=7)
assert abs(result - 1.0) < 1e-10, '[TC12] Integral of constant 1 should be 1 FAILED'

# ---- TC13: MC 1D integral of x^2 (seeded) ----
from numerical_integration import monte_carlo_integral_1d
import numpy as np
np.random.seed(42)
est, err = monte_carlo_integral_1d(lambda x: x**2, 0.0, 1.0, 10000)
assert np.isfinite(est), '[TC13] MC 1D estimate should be finite FAILED'
assert err > 0, '[TC13] MC 1D error should be positive FAILED'

# ---- TC14: MC 2D integral of x*y (seeded) ----
from numerical_integration import monte_carlo_integral_2d
import numpy as np
np.random.seed(42)
est, err = monte_carlo_integral_2d(lambda x, y: x*y, (0.0, 1.0), (0.0, 1.0), 5000)
assert np.isfinite(est), '[TC14] MC 2D estimate should be finite FAILED'
assert err > 0, '[TC14] MC 2D error should be positive FAILED'

# ---- TC15: Niederreiter sequence shape ----
from stochastic_sampler import generate_niederreiter_sequence
import numpy as np
points = generate_niederreiter_sequence(2, 100)
assert points.shape == (100, 2), '[TC15] Niederreiter sequence shape FAILED'
assert np.all((points >= -1e-12) & (points <= 1.0 + 1e-12)), '[TC15] Points should be in [0,1] FAILED'

# ---- TC16: Discrepancy non-negative ----
from stochastic_sampler import generate_niederreiter_sequence, compute_discrepancy
points = generate_niederreiter_sequence(2, 200)
disc = compute_discrepancy(points)
assert disc >= 0, '[TC16] Discrepancy should be non-negative FAILED'

# ---- TC17: QMC area estimate of unit square ----
from stochastic_sampler import estimate_area_qmc
import numpy as np
polygon = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
area, bbox_area = estimate_area_qmc(polygon, (0, 1, 0, 1), 5000)
assert abs(area - 1.0) < 0.1, '[TC17] QMC area estimate should be ~1 FAILED'

# ---- TC18: CG solve on Laplacian: residual small ----
from linear_algebra_core import r8pbu_cg, build_laplacian_banded
import numpy as np
n, mu, a = build_laplacian_banded(8, 8, 0.1, 0.1)
b = np.ones(n)
x0 = np.zeros(n)
x, res, iters = r8pbu_cg(n, mu, a, b, x0, tol=1e-10)
assert res < 1e-5, '[TC18] CG residual should be small FAILED'
assert iters > 0, '[TC18] CG should require at least 1 iteration FAILED'

# ---- TC19: Power method finds dominant eigenvalue ----
from linear_algebra_core import power_method
import numpy as np
A = np.diag([5.0, 3.0, 1.0, 0.5])
y0 = np.ones(4)
y, lam, it_num = power_method(A, y0, it_max=100, tol=1e-10)
assert abs(lam - 5.0) < 1e-6, '[TC19] Power method should find eigenvalue 5 FAILED'

# ---- TC20: Build Laplacian banded correct shape ----
from linear_algebra_core import build_laplacian_banded
n, mu, a = build_laplacian_banded(10, 10, 0.1, 0.1)
assert n == 100, '[TC20] Laplacian n should be 100 FAILED'
assert mu == 10, '[TC20] Laplacian mu should be 10 FAILED'
assert a.shape == (mu + 1, n), '[TC20] Laplacian matrix shape FAILED'

# ---- TC21: Polygon contains center of unit square ----
from mesh_generator import polygon_contains_point
import numpy as np
square = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
assert polygon_contains_point(square, np.array([0.5, 0.5])), '[TC21] (0.5,0.5) should be inside square FAILED'
assert not polygon_contains_point(square, np.array([1.5, 0.5])), '[TC21] (1.5,0.5) should be outside square FAILED'

# ---- TC22: Gate alpha/beta finite and non-negative ----
from ion_channel_dynamics import gate_alpha_beta
import numpy as np
alpha, beta = gate_alpha_beta(-40.0, 'm')
assert np.isfinite(alpha) and np.isfinite(beta), '[TC22] Gate alpha/beta should be finite FAILED'
assert alpha >= 0 and beta >= 0, '[TC22] Gate alpha/beta should be non-negative FAILED'

# ---- TC23: Gate update preserves [0,1] bounds ----
from ion_channel_dynamics import update_gate
import numpy as np
np.random.seed(42)
for gate_type in ['m', 'h', 'j', 'd', 'f', 'x']:
    g = update_gate(0.5, -40.0, gate_type, 0.01)
    assert 0.0 <= g <= 1.0, f'[TC23] Gate {gate_type} update out of [0,1] FAILED'

# ---- TC24: Aliev-Panfilov reaction values are finite ----
from ion_channel_dynamics import aliev_panfilov_reaction
import numpy as np
u = np.array([[0.5, 0.8], [0.2, 0.1]])
v = np.array([[0.3, 0.1], [0.5, 0.6]])
f, g = aliev_panfilov_reaction(u, v)
assert np.all(np.isfinite(f)), '[TC24] Aliev-Panfilov f should be finite FAILED'
assert np.all(np.isfinite(g)), '[TC24] Aliev-Panfilov g should be finite FAILED'

# ---- TC25: Squircle ODE conservation check ----
from ion_channel_dynamics import squircle_ode_integrate
import numpy as np
t, u, v, H = squircle_ode_integrate((1.0, 0.0), (0.0, 10.0), s=4.0, n_steps=500)
H_drift = np.max(np.abs(H - H[0]))
assert H_drift < 0.1, '[TC25] Squircle ODE H conservation drift too large FAILED'

# ---- TC26: Build diffusion tensor correct shape ----
from tissue_reaction_diffusion import build_diffusion_tensor
import numpy as np
angle = np.zeros((16, 16))
Dxx, Dxy, Dyy = build_diffusion_tensor(0.001, 0.0002, angle)
assert Dxx.shape == (16, 16), '[TC26] Dxx shape FAILED'
assert Dxy.shape == (16, 16), '[TC26] Dxy shape FAILED'
assert Dyy.shape == (16, 16), '[TC26] Dyy shape FAILED'

# ---- TC27: Isotropic Laplacian of zero field is zero ----
from tissue_reaction_diffusion import isotropic_laplacian_5point
import numpy as np
u = np.zeros((8, 8))
lap = isotropic_laplacian_5point(u, 0.1, 0.1)
assert np.allclose(lap, 0.0), '[TC27] Laplacian of zero field should be zero FAILED'

# ---- TC28: Fiber angle field correct shape and parallel = 0 ----
from tissue_reaction_diffusion import generate_fiber_angle_field
angle = generate_fiber_angle_field(16, 16, 'parallel')
assert angle.shape == (16, 16), '[TC28] Fiber angle field shape FAILED'
assert np.allclose(angle, 0.0), '[TC28] Parallel fiber angle should be zero FAILED'

# ---- TC29: Stimulus mask correct shape and dtype ----
from electrophysiology_simulator import create_stimulus_mask
mask = create_stimulus_mask(32, 32, (0.0, 0.15), (0.0, 0.15), 0.05, 0.05)
assert mask.shape == (32, 32), '[TC29] Stimulus mask shape FAILED'
assert mask.dtype == bool, '[TC29] Stimulus mask should be boolean FAILED'

# ---- TC30: Square wave stimulus timing ----
from electrophysiology_simulator import square_wave_stimulus
assert square_wave_stimulus(0.0) == -1.0, '[TC30] Stimulus at t=0 should be -1 FAILED'
assert square_wave_stimulus(3.0) == 0.0, '[TC30] Stimulus at t=3 should be 0 FAILED'

# ---- TC31: Wavelength = velocity * APD ----
from electrophysiology_simulator import compute_wavelength
assert compute_wavelength(0.5, 100.0) == 50.0, '[TC31] Wavelength should be velocity*APD FAILED'
assert compute_wavelength(0.0, 100.0) == 0.0, '[TC31] Zero velocity → zero wavelength FAILED'
assert compute_wavelength(0.5, 0.0) == 0.0, '[TC31] Zero APD → zero wavelength FAILED'

# ---- TC32: Compute roundoff error non-negative ----
from utils import compute_roundoff_error
err = compute_roundoff_error(1.0, 2.0, 'add')
assert err >= 0, '[TC32] Roundoff error should be non-negative FAILED'

# ---- TC33: Estimate truncation error finite ----
from utils import estimate_truncation_error
import numpy as np
def f_cubic(x): return x**3
err = estimate_truncation_error(f_cubic, 1.0, 0.01, order=2)
assert np.isfinite(err) and err >= 0, '[TC33] Truncation error should be finite non-negative FAILED'

# ---- TC34: Single cell AP model array length and finite values ----
from ion_channel_dynamics import single_cell_ap_model
import numpy as np
np.random.seed(42)
t_ap, v_ap, gates_ap = single_cell_ap_model(t_max=50.0, dt=0.05, stim_period=30.0)
assert len(t_ap) == len(v_ap), '[TC34] AP time and voltage arrays should match FAILED'
assert len(t_ap) == int(50.0/0.05)+1, '[TC34] AP array length FAILED'
assert np.all(np.isfinite(v_ap)), '[TC34] AP voltage should be finite FAILED'

# ---- TC35: Scattered interpolation returns finite values ----
from mesh_generator import scattered_interpolation_2d
import numpy as np
np.random.seed(42)
data_pts = np.random.rand(30, 2)
data_vals = np.sin(2*np.pi*data_pts[:, 0])
query = np.array([[0.5, 0.5]])
result = scattered_interpolation_2d(data_pts, data_vals, query)
assert np.isfinite(result[0]), '[TC35] Interpolation should be finite FAILED'

# ---- TC36: Apply boundary conditions preserves shape ----
from tissue_reaction_diffusion import apply_boundary_conditions
import numpy as np
np.random.seed(42)
field = np.random.randn(10, 10)
field2 = apply_boundary_conditions(field.copy())
assert field2.shape == field.shape, '[TC36] Boundary conditions should preserve shape FAILED'

# ---- TC37: Stirling number edge cases ----
from utils import stirling_numbers_2
assert stirling_numbers_2(0, 0) == 1, '[TC37] S(0,0) should be 1 FAILED'
assert stirling_numbers_2(5, 0) == 0, '[TC37] S(5,0) should be 0 FAILED'
assert stirling_numbers_2(3, 4) == 0, '[TC37] S(3,4) should be 0 for k>n FAILED'

# ---- TC38: Solve Poisson 2D CG returns correct shape ----
from linear_algebra_core import solve_poisson_2d_cg
import numpy as np
f = np.ones((8, 8))
phi, res, iters = solve_poisson_2d_cg(f, 8, 8, 0.1, 0.1, max_iter=100)
assert phi.shape == (8, 8), '[TC38] Poisson solution shape FAILED'

# ---- TC39: Niederreiter sequence reproducibility ----
from stochastic_sampler import generate_niederreiter_sequence
pts1 = generate_niederreiter_sequence(2, 50)
pts2 = generate_niederreiter_sequence(2, 50)
assert np.allclose(pts1, pts2), '[TC39] Niederreiter sequence should be reproducible FAILED'

# ---- TC40: All gate types return valid alpha/beta ----
from ion_channel_dynamics import gate_alpha_beta
for gt in ['m', 'h', 'j', 'd', 'f', 'x']:
    a, b = gate_alpha_beta(-40.0, gt)
    assert a >= 0 and b >= 0, f'[TC40] Gate {gt} alpha/beta non-negative FAILED'
    assert a < 1e6 and b < 1e6, f'[TC40] Gate {gt} alpha/beta not exploding FAILED'

print('\n全部 40 个测试通过!\n')
