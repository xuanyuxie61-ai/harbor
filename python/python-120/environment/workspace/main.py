
import os
import sys
import numpy as np
import time


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils import (
    grid_uniform_1d, kb_t_ev, arrhenius_rate,
    morse_potential, lennard_jones_potential,
    BOLTZMANN_KB, ELEMENTARY_CHARGE, AMU_TO_KG, FS_TO_S
)
from catalyst_surface import Pt111Surface
from potential_surface import PotentialEnergySurface, build_co_oxidation_pes_demo
from tight_binding import TightBindingSolver
from langevin_integrator import LangevinIntegrator, StochasticReactionDynamics
from reaction_diffusion import ReactionDiffusion1D, LangmuirHinshelwoodKinetics
from monte_carlo import MonteCarloSampler, QuadratureIntegrator, PiecewiseLinearProductIntegral
from markov_kinetics import SurfaceReactionNetwork
from svd_reaction_coords import ReactionCoordinateAnalyzer, generate_test_trajectory


def section(title: str):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def run_surface_structure():
    section("模块 1: Pt(111) 表面结构与 CVT 优化采样")
    surface = Pt111Surface(nx=4, ny=4, n_layers=2)
    surface.dump_site_info()


    cvt_points = surface.cvt_optimize_sites(n_generators=8, it_num=15)
    print(f"\n  CVT 优化后采样点数: {cvt_points.shape[0]}")
    print(f"  采样区域范围 (Å): X=[{cvt_points[:,0].min()*1e10:.2f}, {cvt_points[:,0].max()*1e10:.2f}], "
          f"Z=[{cvt_points[:,2].min()*1e10:.2f}, {cvt_points[:,2].max()*1e10:.2f}]")


    surface.site_occupancy[:surface.n_sites // 4] = 1
    surface.update_occupancy_ca(rule=30, steps=5)
    print(f"\n  CA 演化后 CO 覆盖率: {surface.surface_coverage(species=1):.4f}")
    return surface


def run_potential_surface():
    section("模块 2: 势能面 (PES) 构建与活化能计算")
    pes = build_co_oxidation_pes_demo()
    print("  PES 多项式拟合完成 (degree=4)")


    test_points = np.array([
        [0.0, 0.0, 1.5e-10],
        [0.5e-10, 0.0, 1.5e-10],
    ])
    v_vals = pes.evaluate(test_points)
    grads = pes.gradient(test_points)

    g1_eva = np.linalg.norm(grads[0]) * 1e-10
    g2_eva = np.linalg.norm(grads[1]) * 1e-10
    print(f"\n  测试点 1 势能: {v_vals[0]:.4f} eV, 梯度模: {g1_eva:.4f} eV/Å")
    print(f"  测试点 2 势能: {v_vals[1]:.4f} eV, 梯度模: {g2_eva:.4f} eV/Å")


    x0 = np.array([0.8e-10, 0.0, 1.5e-10])
    x_ts, is_saddle = pes.find_saddle_point_newton(x0, tol=1e-7, max_iter=100)
    print(f"\n  鞍点搜索: 位置=({x_ts[0]*1e10:.3f}, {x_ts[1]*1e10:.3f}, {x_ts[2]*1e10:.3f}) Å")
    print(f"  是否为鞍点: {is_saddle}")


    r_react = np.array([0.0, 0.0, 1.5e-10])
    r_prod = np.array([1.5e-10, 0.0, 0.8e-10])
    e_a = pes.estimate_activation_energy(r_react, r_prod)
    print(f"\n  CO + O → CO2 反应活化能 (NEB 近似): {e_a:.4f} eV")
    return pes


def run_tight_binding(surface):
    section("模块 3: 紧束缚 (Tight-Binding) 电子结构")
    n_atoms = surface.n_atoms
    tb = TightBindingSolver(n_atoms=n_atoms, n_orbitals_per_atom=1)


    onsite = np.full(n_atoms, -5.0)
    tb.build_hamiltonian_sk(surface.atoms, onsite,
                            v_ss_sigma=-2.0, r_cutoff=3.5e-10)
    print(f"  Hamiltonian 矩阵维度: {tb.n_basis} × {tb.n_basis}")


    eigvals, eigvecs = tb.solve_eigenvalues_dense()
    print(f"\n  电子本征值范围: [{eigvals.min():.3f}, {eigvals.max():.3f}] eV")


    e_grid = np.linspace(eigvals.min() - 2.0, eigvals.max() + 2.0, 200)
    dos = tb.compute_dos(e_grid, sigma=0.1)
    print(f"  DOS 峰值能量: {e_grid[np.argmax(dos)]:.3f} eV")


    e_band = tb.compute_band_energy(n_electrons=n_atoms, temperature_k=500.0)
    print(f"\n  500K 下能带能量: {e_band:.4f} eV")


    slap_file = "/tmp/tb_hamiltonian_slap.txt"
    tb.write_slap_format(slap_file)
    n_read, ia, ja, a_vals = TightBindingSolver.read_slap_format(slap_file)
    print(f"\n  SLAP 格式 I/O 测试: 读取 {len(a_vals)} 个非零元 (与写入一致: {np.sum(np.abs(tb.H)>1e-15)})")
    os.remove(slap_file)


    x_grid = grid_uniform_1d(0.0, 5e-10, 51)
    rho = np.exp(-x_grid / 1e-10) * 1e20
    phi = tb.solve_poisson_fd1d(rho, x_grid, epsilon_r=1.0)
    print(f"\n  Poisson 方程求解: 表面电势 φ(0) = {phi[0]:.4e} V")
    return tb


def run_langevin_dynamics(surface, pes):
    section("模块 4: Langevin 随机分子动力学 (SDE 积分)")


    n_particles = 4
    rng = np.random.default_rng(42)
    pos0 = rng.uniform(-1e-10, 1e-10, size=(n_particles, 3))
    pos0[:, 2] = 1.5e-10 + rng.normal(0.0, 0.1e-10, n_particles)



    ev_to_j = 1.602176634e-19
    k_spring = 10.0
    def force_func(positions):

        grad = np.zeros_like(positions)
        grad[:, :2] = positions[:, :2]
        grad[:, 2] = positions[:, 2] - 1.5e-10
        grad = grad * k_spring

        return -grad * ev_to_j


    mass = np.full(n_particles, 28.010)
    integrator = LangevinIntegrator(mass_amu=mass, gamma_ps=2.0,
                                    temperature_k=500.0, dt_fs=0.5)
    integrator.initialize(pos0)


    traj = [integrator.positions.copy()]
    n_steps = 500
    for step in range(n_steps):
        integrator.step(force_func)
        if step % 25 == 0:
            traj.append(integrator.positions.copy())


    msd = integrator.compute_mean_square_displacement(traj)
    D = integrator.diffusion_coefficient_from_msd(traj, time_interval_fs=10.0)
    print(f"  粒子数: {n_particles}, 时间步: {n_steps} fs (dt=0.5 fs)")
    print(f"  最终动能: {integrator.kinetic_energy()*6.241509074e18:.4f} eV")
    print(f"  瞬时温度: {integrator.temperature_instantaneous():.1f} K")
    print(f"\n  表面扩散系数 (MSD 拟合): {D*1e20:.4e} × 10^-20 m²/s")
    print(f"  实验值参考: ~10^{-9} m²/s (室温)")










    raise NotImplementedError("Hole_3: 请实现 Gillespie 随机反应动力学调用")
    return integrator


def run_reaction_diffusion():
    section("模块 5: 反应-扩散方程与 LH 动力学")


    x_grid = grid_uniform_1d(0.0, 10e-9, 101)
    diffusivity = 1.0e-9
    rd = ReactionDiffusion1D(x_grid, diffusivity, bc_type="dirichlet")

    def reaction_func(c):

        k = 1.0e3
        k_des = 2.0e2
        return k * c * (1.0 - c) - k_des * c

    c_ss = rd.solve_steady_state(reaction_func, bc_values=(0.8, 0.1))
    print(f"  稳态浓度范围: [{c_ss.min():.4f}, {c_ss.max():.4f}]")


    c0 = np.linspace(0.8, 0.1, len(x_grid))
    traj_rd, times_rd = rd.solve_time_dependent(c0, t_end=1.0e-6, n_steps=500, reaction_func=reaction_func)
    print(f"  时间演化完成: {len(times_rd)} 步, 最终 t={times_rd[-1]*1e6:.2f} μs")


    lh = LangmuirHinshelwoodKinetics(temperature_k=500.0, p_co_pa=1.0e3, p_o2_pa=5.0e2)
    theta_ss = lh.steady_state_coverage()
    print(f"\n  LH 稳态覆盖率: θ_CO={theta_ss[0]:.4f}, θ_O={theta_ss[1]:.4f}")

    k = lh._rate_constants()
    print(f"\n  速率常数 (500K):")
    for key, val in k.items():
        print(f"    {key:12s} = {val:.4e} s^-1 (或相应单位)")
    return rd, lh


def run_monte_carlo():
    section("模块 6: 蒙特卡洛采样与数值积分")

    mc = MonteCarloSampler(seed=42)


    def test_func_3d(points):

        r2 = np.sum(points ** 2, axis=1) / (1e-10) ** 2
        return np.exp(-0.5 * r2)

    bounds = np.array([[-3e-10, 3e-10], [-3e-10, 3e-10], [-3e-10, 3e-10]])
    sample_sizes = np.array([1000, 5000, 20000, 100000])
    estimates, errors = mc.convergence_test(test_func_3d, bounds, sample_sizes)
    print("  蒙特卡洛积分收敛性测试 (3D Gaussian):")
    for n, est, err in zip(sample_sizes, estimates, errors):
        print(f"    N={n:7d}: I={est:.6e} ± {err:.6e}")


    qi = QuadratureIntegrator()
    f_test = lambda x: np.sin(x) * np.exp(-x)
    a, b = 0.0, 5.0
    I_trap = qi.composite_trapezoidal(f_test, a, b, n=100)
    I_simp = qi.composite_simpson(f_test, a, b, n=100)
    I_gauss = qi.gauss_legendre_3point(f_test, a, b)
    print(f"\n  数值积分比较 (∫_0^5 sin(x) exp(-x) dx):")
    print(f"    复合梯形:   {I_trap:.8f}")
    print(f"    复合 Simpson: {I_simp:.8f}")
    print(f"    Gauss-Legendre (3点): {I_gauss:.8f}")


    pli = PiecewiseLinearProductIntegral()
    f_x = np.linspace(0, 1, 11)
    f_v = f_x ** 2
    g_x = np.linspace(0, 1, 11)
    g_v = np.sin(np.pi * g_x)
    I_pwl = pli.integrate(f_x, f_v, g_x, g_v, 0.0, 1.0)
    print(f"\n  分段线性乘积积分: ∫_0^1 x² sin(πx) dx ≈ {I_pwl:.8f}")
    print(f"  解析参考值: {(np.pi**2 - 4)/np.pi**3:.8f}")


    def dummy_energy(pos):
        r = np.sqrt(np.sum(pos ** 2, axis=1))
        return 0.5 * (r / 1e-10) ** 2 - 1.5

    p_ads = mc.adsorption_probability_monte_carlo(dummy_energy, n_samples=50000,
                                                   temperature_k=500.0)
    print(f"\n  热活化吸附概率 (500K): {p_ads:.6f}")
    return mc


def run_markov_kinetics():
    section("模块 7: 马尔可夫链主方程 (整合 778_monopoly_matrix)")

    network = SurfaceReactionNetwork(n_sites=3, max_occupancy=1)
    network.build_transition_matrix(rate_ads_co=1.0, rate_des_co=0.2,
                                     rate_ads_o=0.5, rate_des_o=0.05,
                                     rate_rxn=0.3)
    network.dump_transition_matrix()


    p_ss = network.steady_state_distribution()
    print("\n  稳态概率分布:")
    for i, (s, p) in enumerate(zip(network.states, p_ss)):
        print(f"    状态 {i}: {s} -> P={p:.6f}")


    tof = network.compute_turnover_frequency(p_ss)
    print(f"\n  转换频率 (TOF): {tof:.6f} s^-1")


    sigma = network.entropy_production_rate(p_ss)
    print(f"  稳态熵产生率: {sigma:.6f} k_B/s")


    p0 = np.zeros(network.n_states)
    p0[0] = 1.0
    traj_mk, times_mk = network.solve_master_equation_ode(p0, t_end=10.0, n_steps=2000)
    print(f"\n  主方程演化: t=0→10s, {len(times_mk)} 时间步")
    print(f"  t=10s 时状态 0 概率: {traj_mk[-1,0]:.6f}")


    if network.n_states > 1:
        mfpt = network.mean_first_passage_time(target_state=0, start_state=1)
        print(f"\n  状态 1 → 状态 0 平均首次通过时间: {mfpt:.4f} s")
    return network


def run_svd_analysis():
    section("模块 8: SVD 反应坐标分析 (整合 1191_svd_snowfall)")


    traj = generate_test_trajectory(n_atoms=12, n_frames=300)
    print(f"  轨迹维度: {traj.shape[0]} 帧 × {traj.shape[1]} 原子 × 3 坐标")


    analyzer = ReactionCoordinateAnalyzer()
    analyzer.fit(traj)

    var_ratio = analyzer.variance_explained(n_components=5)
    print(f"\n  前 5 个主成分方差贡献率:")
    for i, v in enumerate(var_ratio):
        print(f"    PC{i+1}: {v*100:.2f}%")


    rc = analyzer.reaction_coordinate(traj)
    print(f"\n  反应坐标范围: [{rc.min():.4f}, {rc.max():.4f}]")


    q_bins, f_profile = analyzer.free_energy_profile(rc, temperature_k=500.0, n_bins=30)
    barrier_idx = np.argmax(f_profile)
    print(f"\n  自由能势垒位置: q={q_bins[barrier_idx]:.4f}, ΔF={f_profile[barrier_idx]:.4f} eV")


    kappa = analyzer.collectivity_index(n_components=3)
    print(f"\n  集体性指数 κ: {kappa:.4f} (→1 表示高度集体运动)")
    return analyzer


def main():
    print("\n" + "#" * 70)
    print("#  分子动力学: 表面催化反应机理 (Pt(111) 表面 CO 氧化)")
    print("#" * 70)
    print("\n  项目基于 15 个种子代码项目合成")
    print("  科学领域: 分子动力学表面催化反应机理")
    print("  编程语言: Python 3")
    t_start = time.time()


    surface = run_surface_structure()
    pes = run_potential_surface()
    tb = run_tight_binding(surface)
    integrator = run_langevin_dynamics(surface, pes)
    rd, lh = run_reaction_diffusion()
    mc = run_monte_carlo()
    network = run_markov_kinetics()
    analyzer = run_svd_analysis()

    t_elapsed = time.time() - t_start
    section("模拟完成")
    print(f"  总执行时间: {t_elapsed:.3f} 秒")
    print(f"  所有模块运行正常，无报错")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
