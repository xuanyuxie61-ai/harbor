
import numpy as np
import os
import sys


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
    print_section("实验 1: SPDE 单路径求解与格式对比")


    x = generate_composite_mesh_1d(n_base=65, a=0.0, b=1.0, steepness=12.0, center=0.5)
    nx = len(x)


    epsilon = 0.008
    velocity = 0.6
    reaction_rate = 10.0
    carrying_capacity = 1.0
    sigma_noise = 0.15

    spatial = SpatialDiscretization1D(
        x, epsilon=epsilon, velocity=velocity,
        reaction_rate=reaction_rate, carrying_capacity=carrying_capacity
    )


    u0 = carrying_capacity * np.exp(-200.0 * (x - 0.2) ** 2)


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


    dist = hierarchical_distance_matrix(8)
    print(f"  层次距离矩阵 (8x8) 谱范数: {np.linalg.norm(dist, 2):.4f}")

    return stats


def experiment_3_parameter_calibration():
    print_section("实验 3: 无梯度参数标定 (PRAXIS)")


    print("  --- Rosenbrock 测试 ---")
    opt = PraxisOptimizer(tol=1e-8, max_iter=300, h0=0.5)
    x0_rosen = np.array([-1.2, 1.0], dtype=np.float64)
    x_opt, f_opt, n_iter = opt.minimize(rosenbrock, x0_rosen)
    print(f"      初始值: {x0_rosen}, f0={rosenbrock(x0_rosen):.6f}")
    print(f"      最优值: {x_opt}, f_opt={f_opt:.10f}, 迭代={n_iter}")


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
    print_section("实验 4: 自适应策略选择 (Reversi-风格决策)")

    x = generate_composite_mesh_1d(n_base=65, a=0.0, b=1.0, steepness=15.0, center=0.5)
    nx = len(x)


    u = np.ones(nx, dtype=np.float64)
    u[x < 0.4] = 0.1
    u[x > 0.6] = 0.1


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
    print_section("实验 5: Fibonacci 螺旋与 CVT 网格")


    pts = fibonacci_spiral_disk(n=200, R=1.0)
    print(f"  Fibonacci 采样点数: {len(pts)}")
    print(f"  径向标准差: {np.std(np.linalg.norm(pts, axis=1)):.4f}")


    x_cvt = cvt_lloyd_1d(n=21, a=0.0, b=1.0, density=None, it_num=50, tol=1e-10)
    print(f"  CVT 节点数: {len(x_cvt)}")
    print(f"  最大间距: {np.max(np.diff(x_cvt)):.6f}")
    print(f"  最小间距: {np.min(np.diff(x_cvt)):.6f}")

    return pts, x_cvt


def write_summary_report(results: dict, filename: str = "summary_report.txt"):
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
