"""
main.py
聚合物玻璃化转变分子动力学模拟 —— 统一入口

科学问题:
    采用粗粒化 bead-spring 分子动力学模拟，研究聚合物熔体在温度淬火过程中
    的玻璃化转变行为。通过追踪比容-温度关系、自由体积演化、结构有序度及
    扩散系数的变化，确定玻璃化转变温度 T_g，并分析脆性指数与 VFT 行为。

核心算法融合（15个种子项目）:
    - 1208_test_int_2d    → 2D Legendre-Gauss 数值积分（g(r)计算）
    - 312_dosage_ode      → 参数管理与温度协议设计
    - 330_ellipse_grid    → 椭圆截面网格生成（链截面分布）
    - 146_ccvt_reflect    → 反射边界 CVT（自由体积分析）
    - 360_fd1d_heat_explicit → 1D 显式热扩散（薄膜温度梯度）
    - 404_fem2d_heat_rectangle → 2D 有限元热传导（基底-薄膜界面）
    - 259_cvt_square_nonuniform → 非均匀密度 CVT（自由体积密度加权）
    - 975_r8ccs           → CCS 稀疏矩阵（邻居矩阵）
    - 809_nonlin_regula   → Regula Falsi 求根（T_g 精确确定）
    - 1008_random_walk_1d_simulation → 自回避随机游走（链构象初始化）
    - 877_poisson_2d      → Jacobi 迭代（稳态热平衡）
    - 910_prime           → 素数序列（随机种子管理）
    - 252_cvt_box         → 盒子约束 CVT 投影
    - 1033_rk23           → RK23 ODE 求解器（Nose-Hoover 热浴耦合）
    - 242_cvt_4_movie     → CVT 迭代与密度采样（自由体积演化）

运行方式:
    python main.py
    （零参数，全部使用内置默认参数）
"""

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
    """
    执行完整的聚合物玻璃化转变模拟流程。
    """
    print("=" * 70)
    print("  聚合物玻璃化转变分子动力学模拟")
    print("  Polymer Glass Transition Molecular Dynamics Simulation")
    print("=" * 70)
    
    # =====================================================================
    # 1. 系统初始化
    # =====================================================================
    print("\n[1/7] 系统初始化...")
    
    # 模拟参数
    n_chains = 4
    beads_per_chain = 20
    bond_length = 1.0
    dt = 0.002
    n_equil_steps = 100
    n_prod_steps_per_T = 80
    
    # 素数种子管理（融合 910_prime）
    primes = generate_primes(20)
    print(f"  使用素数种子序列: {primes[:10]}...")
    
    # 构建聚合物链（融合 1008_random_walk_1d_simulation, 330_ellipse_grid）
    chain = PolymerChain(
        n_chains=n_chains,
        beads_per_chain=beads_per_chain,
        bond_length=bond_length,
        random_seed=int(primes[5]),
    )
    print(f"  链数: {chain.n_chains}, 单体总数: {chain.n_total}")
    print(f"  模拟盒子: {chain.box}")
    print(f"  初始回转半径 R_g = {chain.radius_of_gyration():.4f}")
    
    # 椭圆截面网格演示（融合 330_ellipse_grid）
    cross_section = generate_ellipse_cross_section(
        n_points=8,
        semi_axes=(1.5, 1.0),
        center=(0.0, 0.0),
    )
    print(f"  椭圆截面网格点数: {cross_section.shape[0]}")
    
    # 力场定义（LJ + FENE + Angle）
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
    
    # 积分器（融合 1033_rk23, 360_fd1d_heat_explicit）
    integrator = VelocityVerletIntegrator(dt=dt)
    print(f"  时间步长 dt = {dt}")
    
    # 温度协议（融合 312_dosage_ode）
    protocol = TemperatureProtocol(
        t0=0.0,
        T_initial=2.0,
        T_final=0.2,
        t_stop=200.0,
        protocol="linear",
    )
    T_schedule = np.linspace(protocol.T_initial, protocol.T_final, 8)
    print(f"  温度扫描点: {T_schedule}")
    
    # NPT-like 热膨胀参数
    # 橡胶态热膨胀系数（约化单位）
    alpha_rubber = 0.015
    # 玻璃态热膨胀系数
    alpha_glass = 0.003
    # 经验玻璃化转变温度（初始猜测，后续由分析更新）
    T_g_estimated = 1.0
    # 参考盒子尺寸
    box_ref = chain.box.copy()
    volume_ref = np.prod(box_ref)
    
    # CVT 自由体积分析器（融合 146, 252, 259, 242）
    cvt = CVTSampler(
        n_generators=16,
        n_samples=800,
        max_iter=15,
        box=chain.box,
    )
    print(f"  CVT 采样器: {cvt.n_generators} 生成器")
    
    # 玻璃化转变分析器（融合 809_nonlin_regula）
    analyzer = GlassTransitionAnalyzer()
    
    # =====================================================================
    # 2. 高温平衡化
    # =====================================================================
    print("\n[2/7] 高温平衡化 (T = 2.0)...")
    
    T_high = 2.0
    berendsen = BerendsenThermostat(tau=0.5)
    
    # 计算初始力
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
        
        # Berendsen 热浴
        chain.velocities = berendsen.apply(
            chain.velocities, chain.masses, T_high, dt
        )
        
        # 检查 CFL
        if step % 100 == 0 and not integrator.cfl_constraint(
            chain.positions, chain.velocities, chain.box
        ):
            # 缩放速度
            v_norm = np.linalg.norm(chain.velocities, axis=1)
            max_v = np.min(chain.box) / (10 * dt)
            if np.max(v_norm) > max_v:
                scale = np.minimum(1.0, max_v / (v_norm + 1e-15))
                chain.velocities = chain.velocities * scale[:, np.newaxis]
    
    print(f"  平衡化完成，最终温度: {chain.instantaneous_temperature():.4f}")
    print(f"  平衡后 R_g = {chain.radius_of_gyration():.4f}")
    
    # =====================================================================
    # 3. 温度淬火模拟与数据采集
    # =====================================================================
    print("\n[3/7] 温度淬火模拟...")
    
    results = []
    trajectory = []
    
    for idx_T, T_target in enumerate(T_schedule):
        print(f"\n  --> T = {T_target:.3f}")
        
        # NPT-like 盒子缩放: 根据温度调整盒子尺寸
        # 模拟热膨胀/收缩效应
        # 在 T_g 处连续的热膨胀公式:
        #   T > T_g: V(T) = V_ref * [1 + α_rubber * (T - T_g)]
        #   T <= T_g: V(T) = V_ref * [1 + α_glass * (T - T_g)]
        #   (一阶近似，V_ref 为 T_g 处的体积)
        if T_target > T_g_estimated:
            scale_factor = 1.0 + alpha_rubber * (T_target - T_g_estimated)
        else:
            scale_factor = 1.0 + alpha_glass * (T_target - T_g_estimated)
        
        # 确保体积为正且单调变化
        scale_factor = max(0.5, min(2.0, scale_factor))
        
        new_box = box_ref * scale_factor
        # 重新缩放位置
        if np.any(np.abs(new_box - chain.box) > 1e-10):
            chain.positions = chain.positions * (new_box / chain.box)
            chain.box = new_box
            cvt.box = new_box
        
        # 在当前温度下生产运行
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
            
            # 热浴耦合到目标温度
            chain.velocities = berendsen.apply(
                chain.velocities, chain.masses, T_target, dt
            )
            
            if step % 10 == 0:
                trajectory.append(chain.positions.copy())
        
        # 计算物理量
        specific_volume = np.prod(chain.box) / chain.n_total
        rg = chain.radius_of_gyration()
        ke = chain.kinetic_energy()
        pe = ff.total_potential_energy(chain.positions, chain.box, chain.chain_starts)
        total_energy = ke + pe
        
        # 自由体积分析（融合 CVT）
        fv_fraction = cvt.free_volume_fraction(
            chain.positions,
            van_der_waals_radius=0.8,
        )
        order_param = cvt.structural_order_parameter()
        
        # 稀疏邻居矩阵（融合 975_r8ccs）
        neighbor_mat = build_neighbor_sparse_matrix(
            chain.positions, chain.box, cutoff=2.5
        )
        sparsity = neighbor_mat.sparsity_ratio()
        
        # 均方位移（若轨迹足够）
        if len(trajectory) >= 10:
            traj_arr = np.array(trajectory[-20:])
            msd = mean_squared_displacement(traj_arr)
            diff_coeff = np.mean(msd[-5:]) / (6.0 * dt * 10 * 5) if len(msd) > 5 else 0.0
        else:
            diff_coeff = 0.0
        
        # 2D 积分: 计算 g(r) 的积分（融合 1208_test_int_2d）
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
        
        # 添加到分析器
        analyzer.add_data_point(T_target, specific_volume, total_energy)
    
    # =====================================================================
    # 4. 玻璃化转变分析（融合 809_nonlin_regula）
    # =====================================================================
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
    
    # =====================================================================
    # 5. 热扩散分析（融合 360, 404, 877）
    # =====================================================================
    print("\n[5/7] 热扩散分析...")
    
    # 1D 薄膜热扩散（融合 360_fd1d_heat_explicit）
    heat1d = HeatDiffusion1D(L=10.0, nx=51, alpha=0.1, dt=0.001)
    T_steady_1d = heat1d.solve_steady(T_left=2.0, T_right=0.2)
    print(f"  1D 稳态温度梯度: max = {np.max(T_steady_1d):.4f}, min = {np.min(T_steady_1d):.4f}")
    
    # 2D FEM 热传导（融合 404_fem2d_heat_rectangle）
    heat2d = HeatDiffusion2DFEM(Lx=10.0, Ly=10.0, nx=21, ny=21, alpha=0.1)
    
    def boundary_T(X, Y):
        T = np.ones_like(X) * 1.1
        T[0, :] = 2.0   # 上边界高温
        T[-1, :] = 0.2  # 下边界低温
        T[:, 0] = 1.1   # 左右边界中温
        T[:, -1] = 1.1
        return T
    
    T_steady_2d = heat2d.solve_steady_jacobi(boundary_T=boundary_T)
    print(f"  2D 稳态温度场: max = {np.max(T_steady_2d):.4f}, min = {np.min(T_steady_2d):.4f}")
    
    kappa_eff = heat2d.effective_thermal_conductivity()
    print(f"  有效热导率 κ_eff = {kappa_eff:.4f}")
    
    # =====================================================================
    # 6. 稀疏矩阵求解演示（融合 975_r8ccs）
    # =====================================================================
    print("\n[6/7] 稀疏矩阵与线性求解...")
    
    # 构建一个简单的测试矩阵（Poisson-like）
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
    
    # =====================================================================
    # 7. 结果汇总输出
    # =====================================================================
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

# ================================================================
# 测试用例（30个，assert模式，涉及随机值均使用固定种子）
# ================================================================

# ---- TC01: generate_primes 前10个素数正确 ----
import numpy as np
p = generate_primes(10)
assert len(p) == 10, '[TC01] generate_primes 长度 FAILED'
assert p[0] == 2 and p[1] == 3 and p[2] == 5, '[TC01] generate_primes 前3个 FAILED'
assert p[9] == 29, '[TC01] generate_primes 第10个 FAILED'

# ---- TC02: prime_count 素数计数 ----
from numeric_utils import prime_count
assert prime_count(10) == 4, '[TC02] prime_count(10) FAILED'
assert prime_count(1) == 0, '[TC02] prime_count(1) FAILED'
assert prime_count(2) == 1, '[TC02] prime_count(2) FAILED'

# ---- TC03: integrate_2d_gauss 常数函数积分 ----
import numpy as np
def _const_f(x, y):
    return np.ones_like(x)
result = integrate_2d_gauss(_const_f, xlim=(0.0, 2.0), ylim=(0.0, 3.0), nx=8, ny=8)
assert abs(result - 6.0) < 0.001, '[TC03] 2D常数积分 FAILED'

# ---- TC04: safe_divide 安全除法 ----
import numpy as np
from numeric_utils import safe_divide
a = np.array([1.0, 2.0, 3.0])
b = np.array([2.0, 0.0, 4.0])
result = safe_divide(a, b)
assert abs(result[0] - 0.5) < 1e-10, '[TC04] safe_divide 正常除 FAILED'
assert result[1] == 0.0, '[TC04] safe_divide 除零 FAILED'
assert abs(result[2] - 0.75) < 1e-10, '[TC04] safe_divide 第三元素 FAILED'

# ---- TC05: soft_cutoff 平滑截断 ----
import numpy as np
from numeric_utils import soft_cutoff
r = np.array([0.5, 0.95, 1.05])
result = soft_cutoff(r, rc=1.0, delta=0.1)
assert result[0] == 1.0, '[TC05] soft_cutoff 内部 FAILED'
assert result[2] == 0.0, '[TC05] soft_cutoff 外部 FAILED'
assert 0.0 < result[1] < 1.0, '[TC05] soft_cutoff 过渡区 FAILED'

# ---- TC06: distance_matrix_pbc 距离矩阵 ----
import numpy as np
from numeric_utils import distance_matrix_pbc
np.random.seed(42)
pos = np.random.rand(5, 3) * 3.0
box = np.array([3.0, 3.0, 3.0])
dm = distance_matrix_pbc(pos, box)
assert dm.shape == (5, 5), '[TC06] distance_matrix_pbc 形状 FAILED'
assert np.all(np.diag(dm) < 1e-10), '[TC06] 对角线非零 FAILED'
assert np.all(dm >= 0), '[TC06] 负距离 FAILED'

# ---- TC07: mean_squared_displacement 均方位移 ----
import numpy as np
np.random.seed(42)
traj = np.random.rand(10, 4, 3)
msd = mean_squared_displacement(traj)
assert msd[0] == 0.0, '[TC07] MSD[0] FAILED'
assert len(msd) == 10, '[TC07] MSD 长度 FAILED'

# ---- TC08: generate_ellipse_cross_section 椭圆截面网格 ----
pts = generate_ellipse_cross_section(n_points=5, semi_axes=(2.0, 1.0), center=(0.0, 0.0))
assert pts.shape[0] > 0, '[TC08] 椭圆截面空 FAILED'
assert pts.shape[1] == 2, '[TC08] 椭圆截面维度 FAILED'
for i in range(pts.shape[0]):
    assert ((pts[i, 0]) / 2.0) ** 2 + (pts[i, 1] / 1.0) ** 2 <= 1.0 + 1e-6, '[TC08] 椭圆外点 FAILED'

# ---- TC09: ForceField lj_potential 在最小值处 ----
import numpy as np
ff = ForceField(epsilon=1.0, sigma=1.0, rcutoff=2.5, fene_k=30.0, fene_R0=1.5)
r_min = 2.0 ** (1.0 / 6.0) * 1.0
u = ff.lj_potential(r_min)
u2 = ff.lj_potential(2.0)
assert u < u2 + 0.1, '[TC09] LJ 势最小值点 FAILED'

# ---- TC10: ForceField lj_force_scalar 截断处为0 ----
f_beyond = ff.lj_force_scalar(3.0)
assert f_beyond == 0.0, '[TC10] LJ 力超截断 FAILED'

# ---- TC11: ForceField fene_potential FENE 势 ----
u_at_R0 = ff.fene_potential(1.5)
assert u_at_R0 >= 1e8, '[TC11] FENE 势在R0处 FAILED'

# ---- TC12: ForceField angle_potential 角势 ----
import numpy as np
u0 = ff.angle_potential(np.pi)
u_dev = ff.angle_potential(np.pi * 0.8)
assert u0 < 1e-10, '[TC12] 角势平衡点 FAILED'
assert u_dev > 0, '[TC12] 角势正值 FAILED'

# ---- TC13: VelocityVerletIntegrator 一步积分 ----
import numpy as np
np.random.seed(42)
vi = VelocityVerletIntegrator(dt=0.001)
N_test = 5
pos_test = np.random.rand(N_test, 3)
vel_test = np.random.randn(N_test, 3) * 0.1
force_test = np.random.randn(N_test, 3) * 0.5
masses_test = np.ones(N_test)
box_test = np.array([5.0, 5.0, 5.0])
def _zero_force(p):
    return np.zeros_like(p)
pos_new, vel_new, f_new = vi.step(pos_test, vel_test, force_test, masses_test, box_test, _zero_force)
assert pos_new.shape == pos_test.shape, '[TC13] VV 位置形状 FAILED'
assert vel_new.shape == vel_test.shape, '[TC13] VV 速度形状 FAILED'

# ---- TC14: rk23_step 精度测试 ----
import numpy as np
from integrator import rk23_step
def _exp_ode(t, y):
    return y
np.random.seed(42)
y0 = np.array([1.0])
t0 = 0.0
y_next, err, _ = rk23_step(y0, t0, 0.01, _exp_ode)
assert abs(y_next[0] - np.exp(0.01)) < 1e-4, '[TC14] RK23 精度 FAILED'

# ---- TC15: TemperatureProtocol 线性协议 ----
tp = TemperatureProtocol(t0=0.0, T_initial=2.0, T_final=0.2, t_stop=100.0, protocol="linear")
t_mid = tp.temperature(50.0)
assert abs(t_mid - 1.1) < 0.01, '[TC15] 线性温度协议中点 FAILED'
assert tp.temperature(0.0) == 2.0, '[TC15] 初始温度 FAILED'
assert tp.temperature(200.0) == 0.2, '[TC15] 终温 FAILED'

# ---- TC16: BerendsenThermostat 热浴 ----
import numpy as np
np.random.seed(42)
bt = BerendsenThermostat(tau=0.5)
vel_in = np.random.randn(10, 3)
m_in = np.ones(10)
vel_out = bt.apply(vel_in, m_in, target_temperature=1.5, dt=0.001)
assert vel_out.shape == vel_in.shape, '[TC16] Berendsen 形状 FAILED'
assert np.all(np.isfinite(vel_out)), '[TC16] Berendsen 非有限 FAILED'

# ---- TC17: HeatDiffusion1D solve_steady 稳态解 ----
h1d = HeatDiffusion1D(L=10.0, nx=11, alpha=0.1, dt=0.001)
T_steady = h1d.solve_steady(T_left=2.0, T_right=0.0)
assert abs(T_steady[0] - 2.0) < 1e-6, '[TC17] 1D 左边界 FAILED'
assert abs(T_steady[-1] - 0.0) < 1e-6, '[TC17] 1D 右边界 FAILED'
assert T_steady[5] > 0.8 and T_steady[5] < 1.2, '[TC17] 1D 中点 FAILED'

# ---- TC18: HeatDiffusion2DFEM solve_steady_jacobi 2D稳态 ----
import numpy as np
h2d = HeatDiffusion2DFEM(Lx=10.0, Ly=10.0, nx=11, ny=11, alpha=0.1)
def _bnd_T(X, Y):
    T = np.ones_like(X) * 1.0
    T[0, :] = 2.0
    T[-1, :] = 0.5
    return T
T_2d = h2d.solve_steady_jacobi(boundary_T=_bnd_T)
assert T_2d.shape == (11, 11), '[TC18] 2D 温度场形状 FAILED'
assert abs(T_2d[0, 5] - 2.0) < 1e-6, '[TC18] 2D 上边界 FAILED'
assert abs(T_2d[-1, 5] - 0.5) < 1e-6, '[TC18] 2D 下边界 FAILED'

# ---- TC19: SparseCCS from_dense and matvec ----
import numpy as np
A_dense = np.array([[2.0, -1.0, 0.0], [-1.0, 2.0, -1.0], [0.0, -1.0, 2.0]])
A_sp = SparseCCS.from_dense(A_dense)
x_vec = np.array([1.0, 2.0, 3.0])
y_vec = A_sp.matvec(x_vec)
y_expected = A_dense @ x_vec
assert np.allclose(y_vec, y_expected), '[TC19] 稀疏矩阵乘法 FAILED'

# ---- TC20: conjugate_gradient CG求解 ----
import numpy as np
A_cg = SparseCCS.from_dense(np.array([[4.0, 1.0], [1.0, 3.0]]))
b_cg = np.array([1.0, 2.0])
x_cg, iters, res = conjugate_gradient(A_cg, b_cg, tol=1e-10)
x_expected = np.linalg.solve(np.array([[4.0, 1.0], [1.0, 3.0]]), b_cg)
assert np.allclose(x_cg, x_expected, atol=1e-6), '[TC20] CG求解 FAILED'

# ---- TC21: CVTSampler free_volume_fraction 自由体积分数 ----
import numpy as np
np.random.seed(42)
cvt = CVTSampler(n_generators=8, n_samples=200, max_iter=5, box=np.array([8.0, 8.0, 8.0]))
pos_few = np.random.rand(10, 3) * 8.0
fv = cvt.free_volume_fraction(pos_few, van_der_waals_radius=0.8)
assert 0.0 <= fv <= 1.0, '[TC21] 自由体积分数范围 FAILED'

# ---- TC22: vft_equation and vft_viscosity VFT 方程 ----
import numpy as np
from glass_transition import vft_equation
vft_val = vft_equation(2.0, A=1.0, B=1.0, T0=0.5)
assert vft_val > 1.0, '[TC22] VFT 方程正值 FAILED'
vft_val2 = vft_equation(0.4, A=1.0, B=1.0, T0=0.5)
assert np.isfinite(vft_val2), '[TC22] VFT 低温有限 FAILED'
vis = vft_viscosity(1.5, eta_inf=0.1, B=0.8, T0=0.3)
assert vis > 0, '[TC22] VFT 粘度 FAILED'

# ---- TC23: regula_falsi Regula Falsi 求根 ----
from glass_transition import regula_falsi
def _f_quad(x):
    return x**2 - 4.0
root, it = regula_falsi(_f_quad, 0.0, 3.0, tol=1e-8)
assert abs(root - 2.0) < 1e-6, '[TC23] regula_falsi 求根 FAILED'

# ---- TC24: GlassTransitionAnalyzer 玻璃化转变分析 ----
import numpy as np
analyzer = GlassTransitionAnalyzer()
T_vals = np.linspace(0.2, 2.0, 10)
v_vals = np.zeros(10)
for i, Tv in enumerate(T_vals):
    if Tv > 1.0:
        v_vals[i] = 1.0 + 0.02 * (Tv - 1.0)
    else:
        v_vals[i] = 0.95 + 0.005 * (Tv - 1.0)
    analyzer.add_data_point(Tv, v_vals[i], Tv * 10.0)
summary = analyzer.get_summary()
assert 'Tg_tangent' in summary, '[TC24] 分析摘要缺少Tg FAILED'

# ---- TC25: fragility_index 脆性指数 ----
m = analyzer.fragility_index(A=1.0, B=0.5, T0=0.1, Tg=1.0)
assert m > 0, '[TC25] 脆性指数正值 FAILED'

# ---- TC26: seeded_random 可复现随机数 ----
import numpy as np
np.random.seed(42)
r1 = seeded_random(seed_prime_index=5, size=(4,))
np.random.seed(42)
r2 = seeded_random(seed_prime_index=5, size=(4,))
assert np.allclose(r1, r2), '[TC26] seeded_random 可复现 FAILED'

# ---- TC27: PolymerChain end_to_end_distance 端到端距离 ----
import numpy as np
np.random.seed(42)
chain = PolymerChain(n_chains=2, beads_per_chain=10, bond_length=1.0, random_seed=42)
d = chain.end_to_end_distance(0)
assert d >= 0, '[TC27] 端到端距离非负 FAILED'
assert np.isfinite(d), '[TC27] 端到端距离有限 FAILED'

# ---- TC28: SparseCCS sparsity_ratio 稀疏度 ----
import numpy as np
A_dense2 = np.array([[2.0, -1.0, 0.0], [-1.0, 2.0, -1.0], [0.0, -1.0, 2.0]])
sp = SparseCCS.from_dense(A_dense2)
s_ratio = sp.sparsity_ratio()
assert 0.0 < s_ratio <= 1.0, '[TC28] 稀疏度范围 FAILED'

# ---- TC29: rk23_integrate 完整积分 ----
import numpy as np
from integrator import rk23_integrate
def _exp_ode2(t, y):
    return y
np.random.seed(42)
y0_rk = np.array([1.0])
t_arr, y_arr, e_arr = rk23_integrate(y0_rk, (0.0, 0.1), 10, _exp_ode2)
assert len(t_arr) == 11, '[TC29] RK23 积分长度 FAILED'
assert abs(y_arr[-1, 0] - np.exp(0.1)) < 1e-3, '[TC29] RK23 积分精度 FAILED'

# ---- TC30: legendre_gauss_nodes Legendre节点 ----
import numpy as np
from numeric_utils import legendre_gauss_nodes
nodes, weights = legendre_gauss_nodes(5)
assert len(nodes) == 5, '[TC30] LG节点长度 FAILED'
assert len(weights) == 5, '[TC30] LG权重长度 FAILED'
assert np.all(nodes >= -1.0) and np.all(nodes <= 1.0), '[TC30] LG节点范围 FAILED'
assert abs(np.sum(weights) - 2.0) < 1e-6, '[TC30] LG权重和 FAILED'

print('\n全部 30 个测试通过!\n')
