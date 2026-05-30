
import numpy as np
import time
import sys

from polymer_chain import PolymerChain, generate_ellipse_cross_section
from force_field import ForceField
from integrator import VelocityVerletIntegrator, NoseHooverIntegrator
from thermostat import TemperatureProtocol, BerendsenThermostat
from cvt_sampler import CVTSampler
from heat_diffusion import HeatDiffusion1D, HeatDiffusion2DFEM
from sparse_solver import build_neighbor_sparse_matrix, conjugate_gradient, SparseCCS
from glass_transition import GlassTransitionAnalyzer, vft_viscosity
from numeric_utils import (
    integrate_2d_gauss,
    mean_squared_displacement,
    generate_primes,
    seeded_random,
)


def run_polymer_glass_transition_simulation():
    print("=" * 70)
    print("  聚合物玻璃化转变分子动力学模拟")
    print("  Polymer Glass Transition Molecular Dynamics Simulation")
    print("=" * 70)
    



    print("\n[1/7] 系统初始化...")
    

    n_chains = 4
    beads_per_chain = 20
    bond_length = 1.0
    dt = 0.002
    n_equil_steps = 100
    n_prod_steps_per_T = 80
    

    primes = generate_primes(20)
    print(f"  使用素数种子序列: {primes[:10]}...")
    

    chain = PolymerChain(
        n_chains=n_chains,
        beads_per_chain=beads_per_chain,
        bond_length=bond_length,
        random_seed=int(primes[5]),
    )
    print(f"  链数: {chain.n_chains}, 单体总数: {chain.n_total}")
    print(f"  模拟盒子: {chain.box}")
    print(f"  初始回转半径 R_g = {chain.radius_of_gyration():.4f}")
    

    cross_section = generate_ellipse_cross_section(
        n_points=8,
        semi_axes=(1.5, 1.0),
        center=(0.0, 0.0),
    )
    print(f"  椭圆截面网格点数: {cross_section.shape[0]}")
    

    ff = ForceField(
        epsilon=1.0,
        sigma=1.0,
        rcutoff=2.5,
        fene_k=30.0,
        fene_R0=1.5,
        angle_k=5.0,
        angle_theta0=np.pi,
    )
    print(f"  力场参数: ε={ff.epsilon}, σ={ff.sigma}, r_c={ff.rcutoff}")
    print(f"  FENE: k={ff.fene_k}, R0={ff.fene_R0}")
    

    integrator = VelocityVerletIntegrator(dt=dt)
    print(f"  时间步长 dt = {dt}")
    

    protocol = TemperatureProtocol(
        t0=0.0,
        T_initial=2.0,
        T_final=0.2,
        t_stop=200.0,
        protocol="linear",
    )
    T_schedule = np.linspace(protocol.T_initial, protocol.T_final, 8)
    print(f"  温度扫描点: {T_schedule}")
    


    alpha_rubber = 0.015

    alpha_glass = 0.003

    T_g_estimated = 1.0

    box_ref = chain.box.copy()
    volume_ref = np.prod(box_ref)
    

    cvt = CVTSampler(
        n_generators=16,
        n_samples=800,
        max_iter=15,
        box=chain.box,
    )
    print(f"  CVT 采样器: {cvt.n_generators} 生成器")
    

    analyzer = GlassTransitionAnalyzer()
    



    print("\n[2/7] 高温平衡化 (T = 2.0)...")
    
    T_high = 2.0
    berendsen = BerendsenThermostat(tau=0.5)
    

    def compute_forces(pos):
        return ff.compute_total_forces(pos, chain.box, chain.chain_starts)
    
    chain.forces = compute_forces(chain.positions)
    
    for step in range(n_equil_steps):
        T_inst = chain.instantaneous_temperature()
        
        chain.positions, chain.velocities, chain.forces = integrator.step(
            chain.positions,
            chain.velocities,
            chain.forces,
            chain.masses,
            chain.box,
            compute_forces,
        )
        

        chain.velocities = berendsen.apply(
            chain.velocities, chain.masses, T_high, dt
        )
        

        if step % 100 == 0 and not integrator.cfl_constraint(
            chain.positions, chain.velocities, chain.box
        ):

            v_norm = np.linalg.norm(chain.velocities, axis=1)
            max_v = np.min(chain.box) / (10 * dt)
            if np.max(v_norm) > max_v:
                scale = np.minimum(1.0, max_v / (v_norm + 1e-15))
                chain.velocities = chain.velocities * scale[:, np.newaxis]
    
    print(f"  平衡化完成，最终温度: {chain.instantaneous_temperature():.4f}")
    print(f"  平衡后 R_g = {chain.radius_of_gyration():.4f}")
    



    print("\n[3/7] 温度淬火模拟...")
    
    results = []
    trajectory = []
    
    for idx_T, T_target in enumerate(T_schedule):
        print(f"\n  --> T = {T_target:.3f}")
        






        if T_target > T_g_estimated:
            scale_factor = 1.0 + alpha_rubber * (T_target - T_g_estimated)
        else:
            scale_factor = 1.0 + alpha_glass * (T_target - T_g_estimated)
        

        scale_factor = max(0.5, min(2.0, scale_factor))
        
        new_box = box_ref * scale_factor

        if np.any(np.abs(new_box - chain.box) > 1e-10):
            chain.positions = chain.positions * (new_box / chain.box)
            chain.box = new_box
            cvt.box = new_box
        

        for step in range(n_prod_steps_per_T):
            T_inst = chain.instantaneous_temperature()
            
            chain.positions, chain.velocities, chain.forces = integrator.step(
                chain.positions,
                chain.velocities,
                chain.forces,
                chain.masses,
                chain.box,
                compute_forces,
            )
            

            chain.velocities = berendsen.apply(
                chain.velocities, chain.masses, T_target, dt
            )
            
            if step % 10 == 0:
                trajectory.append(chain.positions.copy())
        

        specific_volume = np.prod(chain.box) / chain.n_total
        rg = chain.radius_of_gyration()
        ke = chain.kinetic_energy()
        pe = ff.total_potential_energy(chain.positions, chain.box, chain.chain_starts)
        total_energy = ke + pe
        

        fv_fraction = cvt.free_volume_fraction(
            chain.positions,
            van_der_waals_radius=0.8,
        )
        order_param = cvt.structural_order_parameter()
        

        neighbor_mat = build_neighbor_sparse_matrix(
            chain.positions, chain.box, cutoff=2.5
        )
        sparsity = neighbor_mat.sparsity_ratio()
        

        if len(trajectory) >= 10:
            traj_arr = np.array(trajectory[-20:])
            msd = mean_squared_displacement(traj_arr)
            diff_coeff = np.mean(msd[-5:]) / (6.0 * dt * 10 * 5) if len(msd) > 5 else 0.0
        else:
            diff_coeff = 0.0
        

        def integrand_g_r(x, y):
            r = np.sqrt(x**2 + y**2)
            return np.exp(-r**2 / 2.0)
        
        g_r_integral = integrate_2d_gauss(
            integrand_g_r,
            xlim=(0.0, 3.0),
            ylim=(0.0, 3.0),
            nx=8,
            ny=8,
        )
        
        results.append({
            "T": T_target,
            "specific_volume": specific_volume,
            "R_g": rg,
            "KE": ke,
            "PE": pe,
            "total_energy": total_energy,
            "free_volume_fraction": fv_fraction,
            "order_parameter": order_param,
            "sparsity": sparsity,
            "diffusion_coefficient": diff_coeff,
            "g_r_integral": g_r_integral,
        })
        
        print(f"      v = {specific_volume:.4f}, R_g = {rg:.4f}, "
              f"f_v = {fv_fraction:.4f}, D = {diff_coeff:.6f}")
        

        analyzer.add_data_point(T_target, specific_volume, total_energy)
    



    print("\n[4/7] 玻璃化转变分析...")
    
    summary = analyzer.get_summary()
    
    if "error" not in summary:
        print(f"  T_g (切线法)     = {summary['Tg_tangent']:.4f}")
        print(f"  T_g (Regula Falsi) = {summary['Tg_regula_falsi']:.4f}")
        print(f"  比容在 T_g 处    = {summary['specific_volume_at_Tg']:.4f}")
        print(f"  橡胶态膨胀系数   = {summary['alpha_rubber']:.4f}")
        print(f"  玻璃态膨胀系数   = {summary['alpha_glass']:.4f}")
        print(f"  VFT 参数: A={summary['VFT_A']:.4f}, B={summary['VFT_B']:.4f}, T0={summary['VFT_T0']:.4f}")
        print(f"  脆性指数 m       = {summary['fragility_index_m']:.2f}")
    else:
        print(f"  警告: {summary['error']}")
    



    print("\n[5/7] 热扩散分析...")
    

    heat1d = HeatDiffusion1D(L=10.0, nx=51, alpha=0.1, dt=0.001)
    T_steady_1d = heat1d.solve_steady(T_left=2.0, T_right=0.2)
    print(f"  1D 稳态温度梯度: max = {np.max(T_steady_1d):.4f}, min = {np.min(T_steady_1d):.4f}")
    

    heat2d = HeatDiffusion2DFEM(Lx=10.0, Ly=10.0, nx=21, ny=21, alpha=0.1)
    
    def boundary_T(X, Y):
        T = np.ones_like(X) * 1.1
        T[0, :] = 2.0
        T[-1, :] = 0.2
        T[:, 0] = 1.1
        T[:, -1] = 1.1
        return T
    
    T_steady_2d = heat2d.solve_steady_jacobi(boundary_T=boundary_T)
    print(f"  2D 稳态温度场: max = {np.max(T_steady_2d):.4f}, min = {np.min(T_steady_2d):.4f}")
    
    kappa_eff = heat2d.effective_thermal_conductivity()
    print(f"  有效热导率 κ_eff = {kappa_eff:.4f}")
    



    print("\n[6/7] 稀疏矩阵与线性求解...")
    

    n_test = 20
    A_dense = np.zeros((n_test, n_test))
    for i in range(n_test):
        A_dense[i, i] = 2.0
        if i > 0:
            A_dense[i, i-1] = -1.0
        if i < n_test - 1:
            A_dense[i, i+1] = -1.0
    
    A_sparse = SparseCCS.from_dense(A_dense)
    b_test = np.ones(n_test)
    x_test, iters, residual = conjugate_gradient(A_sparse, b_test, tol=1e-10)
    print(f"  CG 求解: 维度={n_test}, 迭代={iters}, 残差={residual:.2e}")
    print(f"  稀疏度 = {A_sparse.sparsity_ratio()*100:.2f}%")
    



    print("\n[7/7] 结果汇总")
    print("-" * 70)
    print(f"{'T':>8} {'v':>10} {'R_g':>10} {'f_v':>10} {'D':>12} {'E_tot':>12}")
    print("-" * 70)
    for r in results:
        print(f"{r['T']:>8.3f} {r['specific_volume']:>10.4f} "
              f"{r['R_g']:>10.4f} {r['free_volume_fraction']:>10.4f} "
              f"{r['diffusion_coefficient']:>12.6f} {r['total_energy']:>12.4f}")
    print("-" * 70)
    
    if "error" not in summary:
        print(f"\n玻璃化转变温度 T_g = {summary['Tg_tangent']:.4f} (切线法)")
        print(f"玻璃化转变温度 T_g = {summary['Tg_regula_falsi']:.4f} (Regula Falsi)")
        print(f"脆性指数 m = {summary['fragility_index_m']:.2f}")
    
    print("\n模拟正常结束。")
    print("=" * 70)
    
    return results, summary


if __name__ == "__main__":
    np.random.seed(42)
    start_time = time.time()
    
    try:
        results, summary = run_polymer_glass_transition_simulation()
    except Exception as e:
        print(f"\n模拟过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    elapsed = time.time() - start_time
    print(f"\n总运行时间: {elapsed:.2f} 秒")
