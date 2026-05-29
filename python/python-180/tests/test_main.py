"""
main.py
统一入口：三维随机 Fisher-KPP 反应-扩散-对流方程的
自适应高阶数值积分、不确定性量化与扩散系数反演

运行方式:
    python main.py
无需任何命令行参数。
"""

import numpy as np
import os
import sys

# 将当前目录加入路径（保证模块导入）
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mesh_generation import generate_composite_mesh_1d, fibonacci_spiral_disk, cvt_lloyd_1d
from wiener_process import QWienerProcess, LEcuyerRNG
from spatial_operators import SpatialDiscretization1D
from stochastic_rk import StochasticIntegrator
from spde_core import SPDESolver1D
from monte_carlo_uq import MonteCarloEngine, hierarchical_distance_matrix
from parameter_calibration import PraxisOptimizer, rosenbrock, SPDEParameterCalibration
from strategy_selector import StrategySelector
from numerical_utils import r8_hypot


def print_section(title: str):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def experiment_1_spde_single_realization():
    """
    实验 1: 单条路径的 SPDE 求解，对比不同时间积分方法。
    """
    print_section("实验 1: SPDE 单路径求解与格式对比")

    # 空间参数
    x = generate_composite_mesh_1d(n_base=65, a=0.0, b=1.0, steepness=12.0, center=0.5)
    nx = len(x)

    # 物理参数
    epsilon = 0.008
    velocity = 0.6
    reaction_rate = 10.0
    carrying_capacity = 1.0
    sigma_noise = 0.15

    spatial = SpatialDiscretization1D(
        x, epsilon=epsilon, velocity=velocity,
        reaction_rate=reaction_rate, carrying_capacity=carrying_capacity
    )

    # 初始条件：高斯脉冲
    u0 = carrying_capacity * np.exp(-200.0 * (x - 0.2) ** 2)

    # Wiener 过程
    wiener = QWienerProcess(
        x, n_modes=32, alpha=1.2, sigma=1.0,
        rng=LEcuyerRNG(seed1=12345, seed2=67890), use_antithetic=False
    )

    methods = ["em", "srk_platen", "milstein"]
    results = {}

    for method in methods:
        integrator = StochasticIntegrator(method=method, dt=1e-4)
        solver = SPDESolver1D(
            spatial, wiener, integrator,
            sigma_noise=sigma_noise,
            dirichlet_bc=(np.array([0]), np.array([0.0])),
            neumann_bc=(np.array([nx - 1]), np.array([0.0]))
        )
        t_arr, u_hist = solver.solve(u0, (0.0, 0.05), store_every=10)
        results[method] = (t_arr, u_hist)
        energy_final = solver.compute_energy(u_hist[-1])
        mass_final = solver.compute_total_mass(u_hist[-1])
        print(f"  方法 {method:12s}: 步数={len(t_arr):4d}, 终态能量={energy_final:.6f}, 总质量={mass_final:.6f}")

    return results


def experiment_2_monte_carlo_uq():
    """
    实验 2: 蒙特卡洛不确定性量化与方差缩减。
    """
    print_section("实验 2: 蒙特卡洛不确定性量化")

    x = generate_composite_mesh_1d(n_base=65, a=0.0, b=1.0, steepness=10.0, center=0.5)
    nx = len(x)

    epsilon = 0.01
    velocity = 0.5
    reaction_rate = 8.0
    carrying_capacity = 1.0
    sigma_noise = 0.2

    spatial = SpatialDiscretization1D(x, epsilon, velocity, reaction_rate, carrying_capacity)
    u0 = carrying_capacity * np.exp(-150.0 * (x - 0.3) ** 2)

    def sampler(seed: int) -> np.ndarray:
        rng = LEcuyerRNG(seed1=seed, seed2=seed + 10000)
        wiener = QWienerProcess(x, n_modes=24, alpha=1.2, sigma=1.0, rng=rng, use_antithetic=True)
        integrator = StochasticIntegrator(method="srk_platen", dt=2e-4)
        solver = SPDESolver1D(
            spatial, wiener, integrator, sigma_noise,
            dirichlet_bc=(np.array([0]), np.array([0.0])),
            neumann_bc=(np.array([nx - 1]), np.array([0.0]))
        )
        _, u_hist = solver.solve(u0, (0.0, 0.04), store_every=9999)
        return u_hist[-1]

    def observable(u: np.ndarray) -> float:
        # 可观测：波前位置（质量中心）
        dx = np.diff(x)
        dx = np.append(dx, dx[-1])
        mass_center = np.sum(x * u * dx) / (np.sum(u * dx) + 1e-12)
        return float(mass_center)

    engine = MonteCarloEngine(n_samples=80, n_strata=4, use_antithetic=True, random_seed_base=100)
    stats = engine.run_ensemble(sampler, observable)

    print(f"  样本数 (有效): {stats['n_effective']}")
    print(f"  期望估计:      {stats['mean']:.6f}")
    print(f"  方差估计:      {stats['variance']:.8f}")
    print(f"  RMSE:          {stats['rmse']:.8f}")
    print(f"  分层后方差:    {stats['stratified_variance']:.8f}")
    print(f"  Gelman-Rubin R-hat: {stats['r_hat']:.4f}")

    # 层次聚类距离矩阵演示
    dist = hierarchical_distance_matrix(8)
    print(f"  层次距离矩阵 (8x8) 谱范数: {np.linalg.norm(dist, 2):.4f}")

    return stats


def experiment_3_parameter_calibration():
    """
    实验 3: 无梯度参数标定与测试函数验证。
    """
    print_section("实验 3: 无梯度参数标定 (PRAXIS)")

    # 3a: Rosenbrock 基准测试
    print("  --- Rosenbrock 测试 ---")
    opt = PraxisOptimizer(tol=1e-8, max_iter=300, h0=0.5)
    x0_rosen = np.array([-1.2, 1.0], dtype=np.float64)
    x_opt, f_opt, n_iter = opt.minimize(rosenbrock, x0_rosen)
    print(f"      初始值: {x0_rosen}, f0={rosenbrock(x0_rosen):.6f}")
    print(f"      最优值: {x_opt}, f_opt={f_opt:.10f}, 迭代={n_iter}")

    # 3b: SPDE 扩散系数 epsilon 的反演
    print("  --- SPDE 扩散系数反演 ---")
    x = generate_composite_mesh_1d(n_base=65, a=0.0, b=1.0)
    nx = len(x)
    epsilon_true = 0.01
    spatial_true = SpatialDiscretization1D(
        x, epsilon=epsilon_true, velocity=0.4, reaction_rate=6.0, carrying_capacity=1.0
    )
    u0 = np.exp(-100.0 * (x - 0.25) ** 2)
    wiener_ref = QWienerProcess(x, n_modes=16, alpha=1.2, sigma=1.0,
                                rng=LEcuyerRNG(111, 222), use_antithetic=False)
    integrator_ref = StochasticIntegrator("em", dt=1e-4)
    solver_ref = SPDESolver1D(
        spatial_true, wiener_ref, integrator_ref, sigma_noise=0.1,
        dirichlet_bc=(np.array([0]), np.array([0.0])),
        neumann_bc=(np.array([nx - 1]), np.array([0.0]))
    )
    _, u_ref_hist = solver_ref.solve(u0, (0.0, 0.03), store_every=9999)
    u_ref = u_ref_hist[-1]

    def solver_factory(theta: np.ndarray) -> np.ndarray:
        eps = float(np.clip(theta[0], 1e-6, 1.0))
        sp = SpatialDiscretization1D(x, epsilon=eps, velocity=0.4,
                                     reaction_rate=6.0, carrying_capacity=1.0)
        w = QWienerProcess(x, n_modes=16, alpha=1.2, sigma=1.0,
                           rng=LEcuyerRNG(111, 222), use_antithetic=False)
        intg = StochasticIntegrator("em", dt=1e-4)
        sol = SPDESolver1D(
            sp, w, intg, sigma_noise=0.1,
            dirichlet_bc=(np.array([0]), np.array([0.0])),
            neumann_bc=(np.array([nx - 1]), np.array([0.0]))
        )
        _, uh = sol.solve(u0, (0.0, 0.03), store_every=9999)
        return uh[-1]

    calibrator = SPDEParameterCalibration(
        u_ref, x, solver_factory,
        param_bounds=np.array([[1e-4, 0.5]])
    )
    theta0 = np.array([0.05], dtype=np.float64)
    theta_opt, f_opt = calibrator.calibrate(theta0)
    print(f"      真实 epsilon: {epsilon_true:.4f}")
    print(f"      反演 epsilon: {theta_opt[0]:.4f}, 失配泛函={f_opt:.6f}")

    return theta_opt, f_opt


def experiment_4_adaptive_strategy():
    """
    实验 4: 自适应数值格式选择策略。
    """
    print_section("实验 4: 自适应策略选择 (Reversi-风格决策)")

    x = generate_composite_mesh_1d(n_base=65, a=0.0, b=1.0, steepness=15.0, center=0.5)
    nx = len(x)

    # 构造具有激波特征的解场
    u = np.ones(nx, dtype=np.float64)
    u[x < 0.4] = 0.1
    u[x > 0.6] = 0.1
    # 在 0.4~0.6 之间保持 1.0，形成方波
    # 平滑过渡
    mask = (x >= 0.4) & (x <= 0.6)
    u[mask] = 1.0 - 0.9 * np.exp(-50.0 * (x[mask] - 0.5) ** 2)

    selector = StrategySelector()
    rec = selector.aggregate_recommendation(u, x, v=0.8, epsilon=0.005, dt=1e-4)
    print(f"  全局推荐格式: {rec}")

    states = selector.evaluate_state(u, x, v=0.8, epsilon=0.005, dt=1e-4)
    pe_vals = [s.peclet for s in states]
    cfl_vals = [s.cfl for s in states]
    print(f"  Peclet 数范围: [{min(pe_vals):.2f}, {max(pe_vals):.2f}]")
    print(f"  CFL 数范围:    [{min(cfl_vals):.4f}, {max(cfl_vals):.4f}]")

    return rec


def experiment_5_fibonacci_and_cvt():
    """
    实验 5: Fibonacci 螺旋采样与 CVT 网格优化。
    """
    print_section("实验 5: Fibonacci 螺旋与 CVT 网格")

    # Fibonacci 圆盘采样
    pts = fibonacci_spiral_disk(n=200, R=1.0)
    print(f"  Fibonacci 采样点数: {len(pts)}")
    print(f"  径向标准差: {np.std(np.linalg.norm(pts, axis=1)):.4f}")

    # CVT Lloyd 优化演示
    x_cvt = cvt_lloyd_1d(n=21, a=0.0, b=1.0, density=None, it_num=50, tol=1e-10)
    print(f"  CVT 节点数: {len(x_cvt)}")
    print(f"  最大间距: {np.max(np.diff(x_cvt)):.6f}")
    print(f"  最小间距: {np.min(np.diff(x_cvt)):.6f}")

    return pts, x_cvt


def write_summary_report(results: dict, filename: str = "summary_report.txt"):
    """
    将实验结果写入文本报告。
    """
    path = os.path.join(os.path.dirname(__file__), filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("  SPDE 数值积分合成项目实验报告\n")
        f.write("  领域: 计算数学 — 随机微分方程数值积分\n")
        f.write("=" * 70 + "\n")
        for key, val in results.items():
            f.write(f"\n{key}:\n")
            if isinstance(val, dict):
                for k2, v2 in val.items():
                    f.write(f"    {k2}: {v2}\n")
            else:
                f.write(f"    {val}\n")
    print(f"\n  报告已保存至: {path}")


def main():
    print("\n" + "#" * 70)
    print("#  随机 Fisher-KPP 反应-扩散-对流方程的数值积分与不确定性量化")
    print("#  合成项目: PROJECT_180 (Python)")
    print("#" * 70)

    results = {}

    try:
        results["exp1_spde"] = experiment_1_spde_single_realization()
    except Exception as e:
        print(f"  [警告] 实验 1 异常: {e}")
        results["exp1_spde"] = str(e)

    try:
        results["exp2_mc"] = experiment_2_monte_carlo_uq()
    except Exception as e:
        print(f"  [警告] 实验 2 异常: {e}")
        results["exp2_mc"] = str(e)

    try:
        results["exp3_calib"] = experiment_3_parameter_calibration()
    except Exception as e:
        print(f"  [警告] 实验 3 异常: {e}")
        results["exp3_calib"] = str(e)

    try:
        results["exp4_strategy"] = experiment_4_adaptive_strategy()
    except Exception as e:
        print(f"  [警告] 实验 4 异常: {e}")
        results["exp4_strategy"] = str(e)

    try:
        results["exp5_sampling"] = experiment_5_fibonacci_and_cvt()
    except Exception as e:
        print(f"  [警告] 实验 5 异常: {e}")
        results["exp5_sampling"] = str(e)

    write_summary_report(results)

    print("\n" + "#" * 70)
    print("#  所有实验执行完毕，main.py 正常结束。")
    print("#" * 70 + "\n")


if __name__ == "__main__":
    main()

# ================================================================
# 测试用例（30个，assert模式，涉及随机值均使用固定种子）
# ================================================================
# 补充测试所需导入（这些函数在 main.py 原始导入中未包含）
from mesh_generation import tetrahedron_grid_count, adaptive_density_function
from parameter_calibration import camel_back
from numerical_utils import band_solve, assemble_band_storage, apply_dirichlet_bc
from stochastic_rk import sde_euler_maruyama_step, sde_srk_platen_step, stiff_sde_semiimplicit_step, adaptive_rk12_sde_step
from strategy_selector import NumericalStrategy

# ---- TC01: r8_hypot 基本计算 (3-4-5直角三角形) ----
val = r8_hypot(3.0, 4.0)
assert abs(val - 5.0) < 1e-14, '[TC01] r8_hypot(3,4) should be 5 FAILED'
assert np.isfinite(val), '[TC01b] r8_hypot result should be finite FAILED'

# ---- TC02: r8_hypot 对称性 ----
assert abs(r8_hypot(5.0, 12.0) - r8_hypot(12.0, 5.0)) < 1e-14, '[TC02] r8_hypot symmetry FAILED'

# ---- TC03: r8_hypot 零值输入边界 ----
assert r8_hypot(0.0, 0.0) == 0.0, '[TC03] r8_hypot(0,0) should be 0 FAILED'
assert abs(r8_hypot(0.0, 5.0) - 5.0) < 1e-14, '[TC03b] r8_hypot(0,5) should be 5 FAILED'
assert abs(r8_hypot(7.0, 0.0) - 7.0) < 1e-14, '[TC03c] r8_hypot(7,0) should be 7 FAILED'

# ---- TC04: r8_hypot 极大值不溢出 ----
big = 1e200
assert np.isfinite(r8_hypot(big, big)), '[TC04] r8_hypot with large values should not overflow FAILED'

# ---- TC05: tetrahedron_grid_count 已知解析值 ----
assert tetrahedron_grid_count(0) == 1, '[TC05] tetrahedron_grid_count(0) should be 1 FAILED'
assert tetrahedron_grid_count(1) == 4, '[TC05b] tetrahedron_grid_count(1) should be 4 FAILED'
assert tetrahedron_grid_count(2) == 10, '[TC05c] tetrahedron_grid_count(2) should be 10 FAILED'

# ---- TC06: fibonacci_spiral_disk 输出形状与半径约束 ----
pts = fibonacci_spiral_disk(n=50, R=1.0)
assert pts.shape == (50, 2), '[TC06] fibonacci_spiral_disk shape FAILED'
radii = np.linalg.norm(pts, axis=1)
assert np.all(radii <= 1.0 + 1e-12), '[TC06b] fibonacci_spiral_disk radii exceed R FAILED'

# ---- TC07: fibonacci_spiral_disk 确定性（无随机成分） ----
pts1 = fibonacci_spiral_disk(n=20, R=1.5)
pts2 = fibonacci_spiral_disk(n=20, R=1.5)
assert np.allclose(pts1, pts2), '[TC07] fibonacci_spiral_disk determinism FAILED'

# ---- TC08: cvt_lloyd_1d 输出单调性与边界 ----
x_cvt = cvt_lloyd_1d(n=15, a=0.0, b=1.0, it_num=50, tol=1e-10)
assert len(x_cvt) == 15, '[TC08] cvt_lloyd_1d output length FAILED'
assert np.all(np.diff(x_cvt) > 0), '[TC08b] cvt_lloyd_1d monotonicity FAILED'
assert x_cvt[0] >= 0.0 and x_cvt[-1] <= 1.0, '[TC08c] cvt_lloyd_1d bounds FAILED'

# ---- TC09: adaptive_density_function 处处正值 ----
x_test = np.linspace(0.0, 1.0, 100)
rho = adaptive_density_function(x_test, steepness=10.0, center=0.5)
assert np.all(rho > 0), '[TC09] adaptive_density_function positivity FAILED'
assert np.all(np.isfinite(rho)), '[TC09b] adaptive_density_function finite FAILED'

# ---- TC10: generate_composite_mesh_1d 输出单调性与边界 ----
mesh = generate_composite_mesh_1d(n_base=33, a=0.0, b=1.0)
assert len(mesh) == 33, '[TC10] generate_composite_mesh_1d length FAILED'
assert np.all(np.diff(mesh) > 0), '[TC10b] generate_composite_mesh_1d monotonicity FAILED'
assert mesh[0] >= 0.0 and mesh[-1] <= 1.0, '[TC10c] generate_composite_mesh_1d bounds FAILED'

# ---- TC11: LEcuyerRNG 确定性（固定种子产生相同序列） ----
rng1 = LEcuyerRNG(seed1=42, seed2=99)
rng2 = LEcuyerRNG(seed1=42, seed2=99)
u1 = rng1.uniform(size=(10,))
u2 = rng2.uniform(size=(10,))
assert np.allclose(u1, u2), '[TC11] LEcuyerRNG determinism FAILED'
assert np.all((u1 > 0) & (u1 < 1)), '[TC11b] LEcuyerRNG uniform (0,1) range FAILED'

# ---- TC12: QWienerProcess 增量输出形状与有限性 ----
x_w = np.linspace(0.0, 1.0, 31)
wp = QWienerProcess(x_w, n_modes=8, alpha=1.2, sigma=1.0, rng=LEcuyerRNG(1, 2))
dw = wp.increment(0.01)
assert dw.shape == (31,), '[TC12] QWienerProcess increment shape FAILED'
assert np.all(np.isfinite(dw)), '[TC12b] QWienerProcess increment finite FAILED'

# ---- TC13: SpatialDiscretization1D 初始化与 pe_local 有限性 ----
x_sp = np.linspace(0.0, 1.0, 21)
sp = SpatialDiscretization1D(x_sp, epsilon=0.01, velocity=0.5, reaction_rate=8.0, carrying_capacity=1.0)
assert sp.nx == 21, '[TC13] SpatialDiscretization1D nx FAILED'
assert np.all(np.isfinite(sp.pe_local)), '[TC13b] SpatialDiscretization1D pe_local finite FAILED'

# ---- TC14: sde_euler_maruyama_step 基本单步 ----
y0 = np.array([1.0, 2.0], dtype=np.float64)
f_em = lambda y: -0.5 * y
g_em = lambda y: 0.1 * y
dW_em = np.array([0.05, -0.03], dtype=np.float64)
y1 = sde_euler_maruyama_step(y0, f_em, g_em, 0.01, dW_em)
assert y1.shape == y0.shape, '[TC14] sde_euler_maruyama_step shape FAILED'
assert np.all(np.isfinite(y1)), '[TC14b] sde_euler_maruyama_step finite FAILED'

# ---- TC15: rosenbrock 全局最小值 (x=[1,1], f=0) ----
x_opt_rb = np.array([1.0, 1.0], dtype=np.float64)
assert rosenbrock(x_opt_rb) < 1e-14, '[TC15] rosenbrock at minimum should be ~0 FAILED'

# ---- TC16: rosenbrock 正定性（非最优点 f>0） ----
x_test_rb = np.array([0.5, 0.8], dtype=np.float64)
assert rosenbrock(x_test_rb) > 0, '[TC16] rosenbrock positivity at non-optimum FAILED'

# ---- TC17: camel_back 原点值为零 ----
assert abs(camel_back(np.array([0.0, 0.0])) - 0.0) < 1e-14, '[TC17] camel_back at origin should be 0 FAILED'

# ---- TC18: hierarchical_distance_matrix 对称性与非负性 ----
dist = hierarchical_distance_matrix(12)
assert dist.shape == (12, 12), '[TC18] hierarchical_distance_matrix shape FAILED'
assert np.allclose(dist, dist.T), '[TC18b] hierarchical_distance_matrix symmetry FAILED'
assert np.all(np.isfinite(dist)), '[TC18c] hierarchical_distance_matrix finite FAILED'

# ---- TC19: band_solve 单位矩阵求解（应等于 b） ----
A_dense = np.eye(5, dtype=np.float64)
A_band = assemble_band_storage(A_dense, ml=2, mu=2)
b_bs = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
x_sol = band_solve(A_band, 2, 2, b_bs)
assert np.allclose(x_sol, b_bs), '[TC19] band_solve identity should return b FAILED'

# ---- TC20: apply_dirichlet_bc 边界值强制施加 ----
A_bc = np.eye(5, dtype=np.float64)
b_bc = np.ones(5, dtype=np.float64)
A_mod, b_mod = apply_dirichlet_bc(A_bc, b_bc, np.array([0, 4]), np.array([0.0, 0.0]))
assert b_mod[0] == 0.0 and b_mod[4] == 0.0, '[TC20] apply_dirichlet_bc boundary values FAILED'
assert A_mod[0, 0] == 1.0 and A_mod[4, 4] == 1.0, '[TC20b] apply_dirichlet_bc diagonal ones FAILED'

# ---- TC21: dg_numerical_flux 数值通量有限性 ----
x_fl = np.linspace(0.0, 1.0, 11)
sp_fl = SpatialDiscretization1D(x_fl, epsilon=0.01, velocity=0.5)
flux_val = sp_fl.dg_numerical_flux(1.0, 0.5)
assert np.isfinite(flux_val), '[TC21] dg_numerical_flux finite FAILED'

# ---- TC22: MonteCarloEngine 基本运行输出结构 ----
x_mc = np.linspace(0.0, 1.0, 21)
def simple_sampler(seed):
    return np.ones(len(x_mc)) * seed * 0.01
def simple_obs(u):
    return float(np.mean(u))
mc22 = MonteCarloEngine(n_samples=10, n_strata=2, use_antithetic=False, random_seed_base=100)
stats22 = mc22.run_ensemble(simple_sampler, simple_obs)
assert 'mean' in stats22, '[TC22] MonteCarloEngine output keys FAILED'
assert np.isfinite(stats22['mean']), '[TC22b] MonteCarloEngine mean finite FAILED'
assert stats22['n_effective'] > 0, '[TC22c] MonteCarloEngine n_effective positive FAILED'

# ---- TC23: PraxisOptimizer Rosenbrock 收敛进展 ----
opt23 = PraxisOptimizer(tol=1e-6, max_iter=200, h0=0.5)
x_init = np.array([-1.2, 1.0], dtype=np.float64)
f_init = rosenbrock(x_init)
x_opt23, f_opt23, n_iter = opt23.minimize(rosenbrock, x_init)
assert f_opt23 < f_init, '[TC23] PraxisOptimizer rosenbrock should reduce function value FAILED'
assert np.all(np.isfinite(x_opt23)), '[TC23b] PraxisOptimizer optimal x finite FAILED'
assert np.isfinite(f_opt23), '[TC23c] PraxisOptimizer optimal f finite FAILED'

# ---- TC24: StrategySelector 聚合推荐输出为合法策略名 ----
x_ss = np.linspace(0.0, 1.0, 31)
u_ss = np.exp(-100.0 * (x_ss - 0.5) ** 2)
selector24 = StrategySelector()
rec24 = selector24.aggregate_recommendation(u_ss, x_ss, v=0.5, epsilon=0.01, dt=1e-3)
assert rec24 in NumericalStrategy.STRATEGIES, '[TC24] StrategySelector valid strategy FAILED'

# ---- TC25: SPDESolver1D 完整求解流程（em 方法） ----
x_spde = generate_composite_mesh_1d(n_base=33, a=0.0, b=1.0)
nx_spde = len(x_spde)
sp_spde = SpatialDiscretization1D(x_spde, epsilon=0.01, velocity=0.3, reaction_rate=5.0, carrying_capacity=1.0)
u0_spde = np.exp(-150.0 * (x_spde - 0.3) ** 2)
wiener_spde = QWienerProcess(x_spde, n_modes=8, alpha=1.2, sigma=1.0, rng=LEcuyerRNG(42, 24))
integrator_spde = StochasticIntegrator(method="em", dt=5e-4)
solver_spde = SPDESolver1D(
    sp_spde, wiener_spde, integrator_spde, sigma_noise=0.1,
    dirichlet_bc=(np.array([0]), np.array([0.0])),
    neumann_bc=(np.array([nx_spde - 1]), np.array([0.0]))
)
t_arr25, u_hist25 = solver_spde.solve(u0_spde, (0.0, 0.01), store_every=10)
assert len(t_arr25) >= 1, '[TC25] SPDESolver1D solve returned time array FAILED'
assert u_hist25.shape[0] == len(t_arr25), '[TC25b] SPDESolver1D history shape FAILED'
energy25 = solver_spde.compute_energy(u_hist25[-1])
assert np.isfinite(energy25), '[TC25c] SPDESolver1D energy finite FAILED'
mass25 = solver_spde.compute_total_mass(u_hist25[-1])
assert np.isfinite(mass25), '[TC25d] SPDESolver1D mass finite FAILED'

# ---- TC26: QWienerProcess strong_error_estimate 正值 ----
x_w26 = np.linspace(0.0, 1.0, 21)
wp26 = QWienerProcess(x_w26, n_modes=8, alpha=1.2, sigma=1.0)
err_est = wp26.strong_error_estimate(dt=0.001, p=2)
assert err_est > 0, '[TC26] strong_error_estimate positivity FAILED'
assert np.isfinite(err_est), '[TC26b] strong_error_estimate finite FAILED'

# ---- TC27: sde_srk_platen_step 基本单步 ----
y027 = np.array([1.0, 0.5], dtype=np.float64)
f27 = lambda y: -y
g27 = lambda y: 0.2 * y
y127 = sde_srk_platen_step(y027, f27, g27, 0.01, np.array([0.02, -0.01]))
assert y127.shape == y027.shape, '[TC27] sde_srk_platen_step shape FAILED'
assert np.all(np.isfinite(y127)), '[TC27b] sde_srk_platen_step finite FAILED'

# ---- TC28: stiff_sde_semiimplicit_step 稳定性（输出有限） ----
n28 = 10
A28 = -2.0 * np.eye(n28)
y028 = np.ones(n28)
f_nl28 = lambda y: 0.1 * y
g28 = lambda y: 0.05 * y
np.random.seed(42)
dW28 = np.random.randn(n28) * 0.1
y128 = stiff_sde_semiimplicit_step(y028, A28, f_nl28, g28, 0.01, dW28)
assert y128.shape == y028.shape, '[TC28] stiff_sde_semiimplicit_step shape FAILED'
assert np.all(np.isfinite(y128)), '[TC28b] stiff_sde_semiimplicit_step finite FAILED'

# ---- TC29: adaptive_rk12_sde_step 自适应步长调整 ----
y029 = np.array([1.0], dtype=np.float64)
f29 = lambda y: -y
g29 = lambda y: 0.1 * y
np.random.seed(42)
dW29 = np.array([0.03])
y_new29, h_new29, acc29 = adaptive_rk12_sde_step(y029, f29, g29, 0.001, dW29, tol=1e-3)
assert np.isfinite(y_new29[0]), '[TC29] adaptive_rk12_sde_step finite FAILED'
assert h_new29 > 0, '[TC29b] adaptive_rk12_sde_step positive h_new FAILED'

# ---- TC30: MonteCarloEngine antithetic 方差缩减增加有效样本 ----
x_mc30 = np.linspace(0.0, 1.0, 11)
def sampler30(seed):
    return np.ones(len(x_mc30)) * (1.0 + 0.01 * seed)
def obs30(u):
    return float(np.mean(u))
mc30 = MonteCarloEngine(n_samples=20, n_strata=2, use_antithetic=True, random_seed_base=200)
stats30 = mc30.run_ensemble(sampler30, obs30)
assert stats30['n_effective'] >= 20, '[TC30] antithetic n_effective should be >= 20 FAILED'
assert np.isfinite(stats30['variance']), '[TC30b] antithetic variance finite FAILED'

print('\n全部 30 个测试通过!\n')
