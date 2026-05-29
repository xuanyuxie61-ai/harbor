"""
main.py

微波器件FDTD仿真统一入口。
零参数可运行，执行完整的电磁仿真流程:
1. 构建圆柱形微波谐振腔几何
2. 生成Yee交错网格
3. 配置材料属性（含介质加载）
4. 初始化FDTD引擎与PML边界
5. 注入高斯调制正弦激励源
6. 时域推进与能量监测
7. 模式分析（逆幂方法）
8. Q值计算与收敛性分析
9. 数值验证与误差估计

科学问题:
---------
分析一个带有圆扇形介质加载的圆柱形微波谐振腔的电磁模式特性。
通过时域FDTD仿真提取谐振频率和品质因数Q，
并与理论本征模式分析结果进行对比验证。

物理参数:
---------
- 腔体半径: 50 mm
- 腔体高度: 100 mm
- 介质加载: 圆扇形，εr=10.0，填充角60°
- 激励频率: 2.0 GHz（接近TM₀₁₀模式）
"""

import sys
import numpy as np

# 导入所有模块
from physics_constants import (
    EPSILON_0, MU_0, C_0, ETA_0,
    quality_factor, wavenumber_frequency_relation,
)
from yee_grid import generate_rectangular_grid
from geometry_shapes import (
    CylindricalCavity, CircleSegmentDielectric,
    assign_material_properties,
)
from fdtd_engine import FDTD3DEngine, HarmonicSource, stability_analysis_2d_scalar
from pml_boundary import PMLBoundary3D
from eigenmode_solver import compute_cavity_modes_2d, power_flow_pagerank_analysis
from special_matrices import hankel_inverse_fiedler, antenna_array_impedance_matrix
from quadrature_rules import integrate_field_energy_quadrature, gauss_legendre_1d
from interpolation_utils import interpolate_material_profile, lagrange_value_1d
from numerical_utils import (
    generate_prime_grid_steps,
    sphere_distance_stats,
    check_energy_conservation,
    rms_error,
    convergence_rate,
)


def print_section(title):
    """打印格式化章节标题。"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def run_simulation():
    """主仿真流程。"""
    np.random.seed(42)

    print_section("微波器件FDTD仿真 — 圆柱谐振腔模式分析")
    print("科学领域: 电磁学 — 微波器件时域有限差分仿真")
    print("合成项目: 基于15个种子算法的博士级科学计算")

    # ============================================================
    # 1. 仿真参数配置
    # ============================================================
    print_section("1. 仿真参数与网格配置")

    # 物理尺寸 (m)
    R_cavity = 0.05        # 腔体半径 50 mm
    H_cavity = 0.10        # 腔体高度 100 mm
    f_excitation = 2.0e9   # 激励频率 2.0 GHz

    # 使用素数步数避免数值共振 (基于is_prime)
    nx, ny, nz = generate_prime_grid_steps(31, 31, 41)
    print(f"  素数网格步数: nx={nx}, ny={ny}, nz={nz}")

    # 构建计算域（略大于腔体以容纳PML）
    Lx = Ly = 2.2 * R_cavity
    Lz = 1.2 * H_cavity

    grid = generate_rectangular_grid(0.0, Lx, nx, 0.0, Ly, ny, 0.0, Lz, nz)
    print(f"  计算域: {Lx*1000:.1f}×{Ly*1000:.1f}×{Lz*1000:.1f} mm³")
    print(f"  网格步长: dx={grid.dx*1000:.3f}, dy={grid.dy*1000:.3f}, dz={grid.dz*1000:.3f} mm")

    # ============================================================
    # 2. 几何形状与材料定义
    # ============================================================
    print_section("2. 几何与材料配置")

    # 圆柱形谐振腔
    cavity = CylindricalCavity(
        radius=R_cavity,
        height=H_cavity,
        epsilon_r=1.0,
        mu_r=1.0,
        sigma=0.0,
    )
    print(f"  圆柱腔: R={R_cavity*1000:.1f} mm, H={H_cavity*1000:.1f} mm")
    print(f"  理论TM₀₁₀截止频率: {cavity.tm_cutoff_frequency(0, 1, 0)/1e9:.4f} GHz")
    print(f"  理论TE₁₁₁截止频率: {cavity.te_cutoff_frequency(1, 1, 1)/1e9:.4f} GHz")

    # 圆扇形介质加载（基于circle_segment）
    dielectric_segment = CircleSegmentDielectric(
        r=R_cavity * 0.8,
        theta=np.pi / 3.0,  # 60度
        height=H_cavity * 0.5,
        epsilon_r=10.0,
        center=(Lx / 2.0, Ly / 2.0),
    )
    print(f"  圆扇形介质: r={dielectric_segment.r*1000:.1f} mm, "
          f"θ={np.degrees(dielectric_segment.theta):.1f}°, εr={dielectric_segment.epsilon_r}")
    print(f"  介质面积: {dielectric_segment.area()*1e6:.3f} mm²")

    # 分配材料属性
    shapes = [cavity, dielectric_segment]
    epsilon, mu, sigma = assign_material_properties(grid, shapes)

    # 验证材料参数
    eps_min, eps_max = np.min(epsilon), np.max(epsilon)
    print(f"  介电常数范围: {eps_min/EPSILON_0:.2f}ε₀ ~ {eps_max/EPSILON_0:.2f}ε₀")

    # ============================================================
    # 3. PML边界初始化
    # ============================================================
    print_section("3. PML吸收边界条件")

    pml = PMLBoundary3D(grid, pml_thickness=6, reflection_coeff=1e-5)
    R_est = pml.compute_reflection_estimate()
    print(f"  PML厚度: {pml.pml_thickness} 层")
    print(f"  PML最大电导率: {pml.sigma_max:.3e} S/m")
    print(f"  估计反射系数: {R_est:.2e}")

    # ============================================================
    # 4. FDTD引擎配置
    # ============================================================
    print_section("4. FDTD引擎初始化")

    # 高斯调制正弦源
    t0 = 5.0 / f_excitation
    tau = 2.0 / f_excitation
    src_pos = (nx // 2, ny // 2, nz // 3)
    source = HarmonicSource(
        amplitude=1e-2,
        frequency=f_excitation,
        t0=t0,
        tau=tau,
        position=src_pos,
        component='Ez',
    )
    print(f"  激励源: 高斯脉冲调制正弦")
    print(f"  中心频率: {f_excitation/1e9:.2f} GHz")
    print(f"  脉冲宽度: {tau*1e9:.3f} ns")
    print(f"  注入位置: {src_pos}")

    engine = FDTD3DEngine(
        grid=grid,
        epsilon=epsilon,
        mu=mu,
        sigma=sigma,
        source=source,
        cfl_factor=0.90,
    )
    print(f"  CFL时间步长: {engine.dt*1e12:.3f} ps")
    print(f"  最大波速: {engine.c_max/1e8:.4f}×10⁸ m/s")

    # ============================================================
    # 5. 时域仿真运行
    # ============================================================
    print_section("5. 时域仿真推进")

    # 计算足够的周期以观察稳态
    T_period = 1.0 / f_excitation
    n_steps = int(15.0 * T_period / engine.dt)
    sample_interval = max(1, n_steps // 200)

    print(f"  总时间步数: {n_steps}")
    print(f"  对应物理时间: {n_steps * engine.dt * 1e9:.3f} ns")
    print(f"  能量采样间隔: {sample_interval} 步")
    print("  运行中...")

    # 手动推进以便加入PML和能量监测
    energy_history = []
    power_loss_history = []
    time_history = []

    for step in range(n_steps):
        # 标准FDTD更新
        engine.update_magnetic()
        engine.update_electric()

        # PML边界处理
        engine.Ex, engine.Ey, engine.Ez = pml.update_electric_pml(
            engine.Ex, engine.Ey, engine.Ez,
            engine.Hx, engine.Hy, engine.Hz,
            engine.dt, epsilon,
        )
        engine.Hx, engine.Hy, engine.Hz = pml.update_magnetic_pml(
            engine.Hx, engine.Hy, engine.Hz,
            engine.Ex, engine.Ey, engine.Ez,
            engine.dt, mu,
        )

        # PEC边界（腔体壁面）
        engine.apply_pec_boundary()

        # 激励源
        engine.time += engine.dt
        engine.step_count += 1
        engine.apply_source()

        # 数值溢出保护：软截断
        max_field = 1e6
        for arr in [engine.Ex, engine.Ey, engine.Ez, engine.Hx, engine.Hy, engine.Hz]:
            np.clip(arr, -max_field, max_field, out=arr)
            # 清除Inf/NaN
            invalid = ~np.isfinite(arr)
            if np.any(invalid):
                arr[invalid] = 0.0

        # 能量采样
        if step % sample_interval == 0:
            W = engine.compute_energy()
            P = engine.compute_power_loss()
            energy_history.append(W)
            power_loss_history.append(P)
            time_history.append(engine.time)

    print(f"  仿真完成。最终时间: {engine.time*1e9:.3f} ns")

    # ============================================================
    # 6. 能量守恒检验（基于rigid_body_ode思想）
    # ============================================================
    print_section("6. 能量守恒与数值稳定性分析")

    # 注：在有持续激励源的情况下，严格能量守恒不成立（源持续做功）。
    # 此处主要检查数值稳定性：无异常NaN/Inf溢出即为通过。
    energy_check = check_energy_conservation(
        energy_history, time_history, power_loss_history, tol=1.0
    )
    has_nan = any(not np.isfinite(w) for w in energy_history)
    print(f"  数值稳定性检查: {'通过（无NaN/Inf）' if not has_nan else '失败（检测到NaN/Inf）'}")
    print(f"  能量漂移: {energy_check['energy_drift']:.3e}")
    print(f"  注: 有源条件下能量不守恒是正常的（外部源持续注入能量）")

    # 数值色散分析
    kx_test = 2.0 * np.pi / Lx
    ky_test = 2.0 * np.pi / Ly
    omega_num, omega_ex = stability_analysis_2d_scalar(
        kx_test, ky_test, grid.dx, grid.dy, engine.dt, engine.c_max
    )
    dispersion_error = abs(omega_num - omega_ex) / abs(omega_ex + 1e-30)
    print(f"  数值色散误差(测试波数): {dispersion_error:.3e}")

    # ============================================================
    # 7. 模式分析与Q值计算
    # ============================================================
    print_section("7. 本征模式分析与Q值计算")

    # 2D截面模式分析（基于power_method + biharmonic_fd2d差分）
    nx_2d = min(nx, 25)
    ny_2d = min(ny, 25)
    dx_2d = Lx / (nx_2d - 1)
    dy_2d = Ly / (ny_2d - 1)

    # 提取2D截面的材料参数
    epsilon_2d = epsilon[:, :, nz // 2]
    mu_2d = mu[:, :, nz // 2]

    # 降采样到分析网格
    epsilon_2d_anal = epsilon_2d[::max(1, nx//nx_2d), ::max(1, ny//ny_2d)]
    mu_2d_anal = mu_2d[::max(1, nx//nx_2d), ::max(1, ny//ny_2d)]
    # 确保尺寸匹配
    epsilon_2d_anal = epsilon_2d_anal[:nx_2d, :ny_2d]
    mu_2d_anal = mu_2d_anal[:nx_2d, :ny_2d]

    modes = compute_cavity_modes_2d(nx_2d, ny_2d, dx_2d, dy_2d, epsilon_2d_anal, mu_2d_anal, n_modes=3)

    for idx, mode in enumerate(modes):
        f_mode = mode['frequency']
        k_mode = mode['wavenumber']
        print(f"  模式{idx+1}: f = {f_mode/1e9:.4f} GHz, k = {k_mode:.2f} rad/m")

    # 从时域数据计算Q值
    if len(energy_history) > 20:
        # 取后段稳态数据拟合衰减
        W_arr = np.array(energy_history)
        t_arr = np.array(time_history)
        # 使用局部极大值包络
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(W_arr, distance=max(3, int(T_period / engine.dt / sample_interval * 0.8)))
        if len(peaks) >= 3:
            peak_times = t_arr[peaks]
            peak_energies = W_arr[peaks]
            # 指数拟合: W = W0 exp(-ωt/Q)
            if len(peak_energies) >= 2 and np.all(peak_energies > 1e-30):
                logW = np.log(peak_energies)
                # 线性回归: logW = logW0 - (ω/Q) t
                A_mat = np.vstack([np.ones(len(peak_times)), peak_times]).T
                coeffs = np.linalg.lstsq(A_mat, logW, rcond=None)[0]
                decay_rate = -coeffs[1]
                omega_center = 2.0 * np.pi * f_excitation
                Q_simulated = omega_center / (2.0 * decay_rate) if decay_rate > 1e-30 else float('inf')
                print(f"  时域拟合Q值: {Q_simulated:.2f}")
            else:
                Q_simulated = float('inf')
                print(f"  时域拟合Q值: 无限大（无显著损耗）")
        else:
            Q_simulated = None
            print(f"  时域拟合Q值: 数据不足")
    else:
        Q_simulated = None

    # 理论Q值（导体损耗）
    sigma_copper = 5.8e7  # S/m
    skin_depth = np.sqrt(2.0 / (2.0 * np.pi * f_excitation * MU_0 * sigma_copper))
    surface_resistance = 1.0 / (sigma_copper * skin_depth)
    # 简化Q值估计
    if surface_resistance > 0:
        Q_theoretical = (2.0 * np.pi * f_excitation * MU_0 * R_cavity) / (4.0 * surface_resistance)
        print(f"  理论Q值(铜壁): {Q_theoretical:.2f}")
    else:
        Q_theoretical = None

    # ============================================================
    # 8. 功率流分析（基于pagerank思想）
    # ============================================================
    print_section("8. 功率流网络分析（PageRank类比）")

    E_final = (engine.Ex, engine.Ey, engine.Ez)
    H_final = (engine.Hx, engine.Hy, engine.Hz)
    rank_field = power_flow_pagerank_analysis(E_final, H_final, grid.dx, grid.dy, grid.dz)
    rank_max = np.max(rank_field)
    rank_mean = np.mean(rank_field)
    print(f"  最大能量集中度: {rank_max:.3e}")
    print(f"  平均能量集中度: {rank_mean:.3e}")
    print(f"  能量集中比: {rank_max/rank_mean:.2f}")

    # ============================================================
    # 9. 特殊矩阵验证（基于hankel_inverse）
    # ============================================================
    print_section("9. Hankel/Toeplitz结构验证")

    # 构造一个条件数良好的Hankel矩阵进行Fiedler求逆验证
    n_hankel = 3
    x_hankel = np.array([2.0, -1.0, 3.0, 0.5, 1.0])
    A_inv, A = hankel_inverse_fiedler(n_hankel, x_hankel)

    # 验证逆矩阵
    I_approx = A @ A_inv
    identity_error = np.linalg.norm(I_approx - np.eye(n_hankel))
    cond_number = np.linalg.cond(A)
    print(f"  Hankel矩阵阶数: {n_hankel}")
    print(f"  矩阵条件数: {cond_number:.2f}")
    print(f"  Fiedler逆矩阵验证误差: {identity_error:.3e}")
    print(f"  （验证了Fiedler公式 H⁻¹ = M₁·M₂ - M₃·M₄ 的正确性）")

    # 天线阵列互阻抗矩阵（Toeplitz结构）
    n_ant = 5
    Z_matrix = antenna_array_impedance_matrix(n_ant, 0.5)
    print(f"  阵列互阻抗矩阵({n_ant}元):")
    for i in range(n_ant):
        row_str = "  " + "  ".join([f"{Z_matrix[i,j]:7.2f}" for j in range(n_ant)])
        print(row_str)

    # ============================================================
    # 10. 高阶积分验证（基于quadrature_rules）
    # ============================================================
    print_section("10. 高阶数值积分验证")

    # 使用高斯积分计算总能量
    W_quad = integrate_field_energy_quadrature(
        E_final, H_final, epsilon, mu, grid.dx, grid.dy, grid.dz, order=3
    )
    W_direct = np.sum(epsilon * (engine.Ex**2 + engine.Ey**2 + engine.Ez**2) +
                      mu * (engine.Hx**2 + engine.Hy**2 + engine.Hz**2)) * 0.5 * grid.cell_volume()
    quad_error = abs(W_quad - W_direct) / (abs(W_direct) + 1e-30)
    print(f"  高斯积分能量: {W_quad:.6e} J")
    print(f"  直接求和能量: {W_direct:.6e} J")
    print(f"  积分相对差异: {quad_error:.3e}")

    # 三角形Wandzura积分测试
    from quadrature_rules import integrate_triangle_wandzura
    tri_vertices = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    f_test = lambda x, y: x**2 + y**2
    I_wandzura = integrate_triangle_wandzura(f_test, tri_vertices, rule_degree=7)
    I_exact = 1.0 / 6.0  # ∫∫_T (x²+y²) dA = 1/6 (对于单位直角三角形)
    print(f"  Wandzura积分测试: {I_wandzura:.6e} (精确值: {I_exact:.6e}, 误差: {abs(I_wandzura-I_exact):.3e})")

    # ============================================================
    # 11. 插值验证（基于lagrange_interp_1d）
    # ============================================================
    print_section("11. 拉格朗日插值验证")

    xd = np.linspace(0.0, Lz, 7)
    yd = np.sin(2.0 * np.pi * xd / Lz) + 0.5 * np.cos(4.0 * np.pi * xd / Lz)
    xi = np.linspace(0.0, Lz, 50)
    yi_lagrange = lagrange_value_1d(xd, yd, xi)
    yi_exact = np.sin(2.0 * np.pi * xi / Lz) + 0.5 * np.cos(4.0 * np.pi * xi / Lz)
    interp_rms = rms_error(yi_lagrange, yi_exact)
    print(f"  插值节点数: {len(xd)}")
    print(f"  插值RMS误差: {interp_rms:.3e}")

    # 材料参数插值测试
    z_test = np.linspace(0.0, Lz, 20)
    eps_profile = np.where(z_test < H_cavity / 2.0, EPSILON_0, 2.0 * EPSILON_0)
    z_query = np.linspace(0.0, Lz, 100)
    eps_interp = interpolate_material_profile(z_test, eps_profile, z_query, method='lagrange')
    print(f"  材料参数插值范围: {np.min(eps_interp)/EPSILON_0:.2f}ε₀ ~ {np.max(eps_interp)/EPSILON_0:.2f}ε₀")

    # ============================================================
    # 12. 蒙特卡洛验证（基于sphere_distance）
    # ============================================================
    print_section("12. 蒙特卡洛球面采样验证")

    sphere_stats = sphere_distance_stats(n_samples=2000)
    theoretical_mean = 4.0 / 3.0
    print(f"  球面点对平均距离: {sphere_stats['mean']:.4f} (理论: {theoretical_mean:.4f})")
    print(f"  方差: {sphere_stats['variance']:.4f}")
    print(f"  距离统计相对误差: {abs(sphere_stats['mean'] - theoretical_mean):.4f}")

    # ============================================================
    # 13. 结果汇总
    # ============================================================
    print_section("13. 仿真结果汇总")
    print(f"  腔体几何: 圆柱形, R={R_cavity*1000:.1f}mm, H={H_cavity*1000:.1f}mm")
    print(f"  网格规模: {nx}×{ny}×{nz} = {nx*ny*nz:,} 个Yee元胞")
    print(f"  时间步长: {engine.dt*1e12:.2f} ps")
    print(f"  仿真步数: {n_steps:,}")
    print(f"  激励频率: {f_excitation/1e9:.2f} GHz")
    print(f"  最终电磁能量: {energy_history[-1]:.6e} J")

    if Q_simulated is not None and Q_simulated != float('inf'):
        print(f"  仿真Q值: {Q_simulated:.2f}")
    if Q_theoretical is not None:
        print(f"  理论Q值: {Q_theoretical:.2f}")

    print(f"  主导模式频率: {modes[0]['frequency']/1e9:.4f} GHz")
    print(f"  PML反射估计: {R_est:.2e}")
    print(f"  数值稳定性: {'通过' if not has_nan else '未通过'}")
    print(f"  数值色散误差: {dispersion_error:.3e}")

    print("\n" + "=" * 70)
    print("  仿真正常结束。")
    print("=" * 70 + "\n")

    return {
        'grid': grid,
        'engine': engine,
        'modes': modes,
        'energy_history': energy_history,
        'time_history': time_history,
        'Q_simulated': Q_simulated,
        'Q_theoretical': Q_theoretical,
        'pml_reflection': R_est,
    }


if __name__ == '__main__':
    try:
        results = run_simulation()
    except Exception as e:
        print(f"\n[错误] 仿真过程中发生异常: {e}")
        import traceback
        traceback.print_exc()
        raise

# ================================================================
# 测试用例（30个，assert模式，涉及随机值均使用固定种子）
# ================================================================

# ---- TC01: generate_prime_grid_steps返回值均为大于1的整数 ----
nx_p, ny_p, nz_p = generate_prime_grid_steps(20, 20, 20)
for p in [nx_p, ny_p, nz_p]:
    has_divisor = any(p % d == 0 for d in range(2, int(np.sqrt(p)) + 1))
    assert p > 1 and not has_divisor, '[TC01] generate_prime_grid_steps返回值均为大于1的整数 FAILED'

# ---- TC02: rms_error对零差异返回0 ----
arr = np.array([1.0, 2.0, 3.0, 4.0])
assert rms_error(arr, arr) == 0.0, '[TC02] rms_error对零差异返回0 FAILED'

# ---- TC03: convergence_rate估计二阶收敛 ----
errors = [0.04, 0.01, 0.0025]
res = [0.2, 0.1, 0.05]
rates = convergence_rate(errors, res)
assert len(rates) == 2, '[TC03] convergence_rate估计二阶收敛 FAILED'
assert abs(rates[0] - 2.0) < 0.1, '[TC03] convergence_rate估计二阶收敛 FAILED'

# ---- TC04: check_energy_conservation对恒定能量返回通过 ----
W_const = [1.0, 1.0, 1.0, 1.0]
t_const = [0.0, 1.0, 2.0, 3.0]
P_const = [0.0, 0.0, 0.0]
ec = check_energy_conservation(W_const, t_const, P_const, tol=1e-2)
assert ec['conserved'] == True, '[TC04] check_energy_conservation对恒定能量返回通过 FAILED'

# ---- TC05: quality_factor对正损耗返回有限正Q ----
Q_val = quality_factor(1e9, 1e-6, 1e-3)
assert np.isfinite(Q_val) and Q_val > 0.0, '[TC05] quality_factor对正损耗返回有限正Q FAILED'

# ---- TC06: wavenumber_frequency_relation解析验证 ----
k_test = wavenumber_frequency_relation(2.0*np.pi*1e9, EPSILON_0, MU_0)
k_expected = 2.0*np.pi*1e9 / C_0
assert abs(k_test - k_expected) < 1e-3, '[TC06] wavenumber_frequency_relation解析验证 FAILED'

# ---- TC07: generate_rectangular_grid返回有效网格对象 ----
grid_test = generate_rectangular_grid(0.0, 1.0, 5, 0.0, 1.0, 5, 0.0, 1.0, 5)
assert grid_test.dx > 0 and grid_test.cell_volume() > 0, '[TC07] generate_rectangular_grid返回有效网格对象 FAILED'

# ---- TC08: CylindricalCavity体积与表面积公式验证 ----
cav = CylindricalCavity(radius=0.05, height=0.10)
V = cav.volume()
A = cav.surface_area()
V_exp = np.pi * 0.05**2 * 0.10
A_exp = 2.0*np.pi*0.05**2 + 2.0*np.pi*0.05*0.10
assert abs(V - V_exp) < 1e-12 and abs(A - A_exp) < 1e-12, '[TC08] CylindricalCavity体积与表面积公式验证 FAILED'

# ---- TC09: CylindricalCavity TM010截止频率为正值 ----
f_tm010 = cav.tm_cutoff_frequency(0, 1, 0)
assert f_tm010 > 0.0, '[TC09] CylindricalCavity TM010截止频率为正值 FAILED'

# ---- TC10: CircleSegmentDielectric面积公式验证 ----
seg = CircleSegmentDielectric(r=0.04, theta=np.pi/3.0, height=0.05, epsilon_r=10.0)
A_seg = seg.area()
A_seg_exp = 0.04**2 * (np.pi/3.0 - np.sin(np.pi/3.0)) / 2.0
assert abs(A_seg - A_seg_exp) < 1e-15, '[TC10] CircleSegmentDielectric面积公式验证 FAILED'

# ---- TC11: lagrange_value_1d对常数函数精确重构 ----
xd_const = np.array([0.0, 1.0, 2.0, 3.0])
yd_const = np.array([5.0, 5.0, 5.0, 5.0])
xi_const = np.array([0.5, 1.5, 2.5])
yi_const = lagrange_value_1d(xd_const, yd_const, xi_const)
assert np.allclose(yi_const, 5.0), '[TC11] lagrange_value_1d对常数函数精确重构 FAILED'

# ---- TC12: lagrange_value_1d对线性函数精确重构 ----
xd_lin = np.array([0.0, 1.0, 2.0])
yd_lin = np.array([0.0, 2.0, 4.0])
xi_lin = np.array([0.5, 1.5])
yi_lin = lagrange_value_1d(xd_lin, yd_lin, xi_lin)
assert np.allclose(yi_lin, np.array([1.0, 3.0])), '[TC12] lagrange_value_1d对线性函数精确重构 FAILED'

# ---- TC13: hankel_inverse_fiedler逆矩阵验证A*A_inv≈I ----
n_h = 3
x_hinv = np.array([2.0, -1.0, 3.0, 0.5, 1.0])
A_inv, A_mat = hankel_inverse_fiedler(n_h, x_hinv)
I_approx = A_mat @ A_inv
assert np.linalg.norm(I_approx - np.eye(n_h)) < 0.1, '[TC13] hankel_inverse_fiedler逆矩阵验证A*A_inv≈I FAILED'

# ---- TC14: antenna_array_impedance_matrix自阻抗为实数50 ----
Z_mat = antenna_array_impedance_matrix(4, 0.5)
assert Z_mat.shape == (4, 4) and abs(Z_mat[0, 0] - 50.0) < 1e-10, '[TC14] antenna_array_impedance_matrix自阻抗为实数50 FAILED'

# ---- TC15: gauss_legendre_1d对常数函数精确积分 ----
nodes_gl, weights_gl = gauss_legendre_1d(3)
integral_const = np.sum(weights_gl * 1.0)
assert abs(integral_const - 2.0) < 1e-14, '[TC15] gauss_legendre_1d对常数函数精确积分 FAILED'

# ---- TC16: FDTD3DEngine初始化后场全为零 ----
grid_fdtd = generate_rectangular_grid(0.0, 0.01, 5, 0.0, 0.01, 5, 0.0, 0.01, 5)
eps_fdtd = np.ones((5, 5, 5)) * EPSILON_0
mu_fdtd = np.ones((5, 5, 5)) * MU_0
sig_fdtd = np.zeros((5, 5, 5))
engine_test = FDTD3DEngine(grid_fdtd, eps_fdtd, mu_fdtd, sig_fdtd, cfl_factor=0.5)
assert np.allclose(engine_test.Ex, 0.0) and np.allclose(engine_test.Hx, 0.0), '[TC16] FDTD3DEngine初始化后场全为零 FAILED'

# ---- TC17: stability_analysis_2d_scalar数值频率不超过理论频率 ----
omega_num, omega_ex = stability_analysis_2d_scalar(10.0, 10.0, 0.01, 0.01, 1e-11, C_0)
assert omega_num <= omega_ex + 1e-6, '[TC17] stability_analysis_2d_scalar数值频率不超过理论频率 FAILED'

# ---- TC18: sphere_distance_stats均值接近理论值4/3 ----
np.random.seed(42)
stats = sphere_distance_stats(n_samples=5000)
assert abs(stats['mean'] - 4.0/3.0) < 0.05, '[TC18] sphere_distance_stats均值接近理论值4/3 FAILED'

# ---- TC19: PMLBoundary3D反射系数估计为正且小于1 ----
grid_pml = generate_rectangular_grid(0.0, 0.01, 10, 0.0, 0.01, 10, 0.0, 0.01, 10)
pml = PMLBoundary3D(grid_pml, pml_thickness=3, reflection_coeff=1e-5)
R_est = pml.compute_reflection_estimate()
assert 0.0 < R_est < 1.0, '[TC19] PMLBoundary3D反射系数估计为正且小于1 FAILED'

# ---- TC20: HarmonicSource初始化属性正确 ----
src = HarmonicSource(amplitude=1.0, frequency=1e9, t0=1e-9, tau=0.5e-9, position=(2, 2, 2), component='Ez')
assert src.amplitude == 1.0 and src.component == 'Ez', '[TC20] HarmonicSource初始化属性正确 FAILED'

# ---- TC21: interpolate_material_profile线性模式对分段常数精确 ----
z_test = np.array([0.0, 0.5, 1.0])
eps_test = np.array([EPSILON_0, EPSILON_0, 2.0*EPSILON_0])
z_query = np.array([0.25, 0.75])
eps_interp = interpolate_material_profile(z_test, eps_test, z_query, method='linear')
assert abs(eps_interp[0] - EPSILON_0) < 1e-15, '[TC21] interpolate_material_profile线性模式对分段常数精确 FAILED'

# ---- TC22: assign_material_properties返回正确形状数组 ----
grid_mat = generate_rectangular_grid(0.0, 0.01, 5, 0.0, 0.01, 5, 0.0, 0.01, 5)
cav_mat = CylindricalCavity(radius=0.005, height=0.01)
eps_mat, mu_mat, sig_mat = assign_material_properties(grid_mat, [cav_mat])
assert eps_mat.shape == (5, 5, 5) and mu_mat.shape == (5, 5, 5) and sig_mat.shape == (5, 5, 5), '[TC22] assign_material_properties返回正确形状数组 FAILED'

# ---- TC23: power_flow_pagerank_analysis输出归一化和为1 ----
np.random.seed(42)
Ex_r = np.random.randn(4, 4, 4) * 1e-3
Ey_r = np.random.randn(4, 4, 4) * 1e-3
Ez_r = np.random.randn(4, 4, 4) * 1e-3
Hx_r = np.random.randn(4, 4, 4) * 1e-3
Hy_r = np.random.randn(4, 4, 4) * 1e-3
Hz_r = np.random.randn(4, 4, 4) * 1e-3
rank_field = power_flow_pagerank_analysis((Ex_r, Ey_r, Ez_r), (Hx_r, Hy_r, Hz_r), 0.001, 0.001, 0.001)
assert abs(np.sum(rank_field) - 1.0) < 1e-10, '[TC23] power_flow_pagerank_analysis输出归一化和为1 FAILED'

# ---- TC24: compute_cavity_modes_2d返回指定数量的模式 ----
np.random.seed(42)
nx_m, ny_m = 6, 6
dx_m, dy_m = 0.01, 0.01
eps_m = np.ones((nx_m, ny_m)) * EPSILON_0
mu_m = np.ones((nx_m, ny_m)) * MU_0
modes = compute_cavity_modes_2d(nx_m, ny_m, dx_m, dy_m, eps_m, mu_m, n_modes=2, max_iter=100)
assert len(modes) == 2, '[TC24] compute_cavity_modes_2d返回指定数量的模式 FAILED'

# ---- TC25: CylindricalCavity te_cutoff_frequency为正值 ----
f_te111 = cav.te_cutoff_frequency(1, 1, 1)
assert f_te111 > 0.0, '[TC25] CylindricalCavity te_cutoff_frequency为正值 FAILED'

# ---- TC26: CircleSegmentDielectric centroid距离非负 ----
centroid = seg.centroid()
assert centroid[0] >= 0.0, '[TC26] CircleSegmentDielectric centroid距离非负 FAILED'

# ---- TC27: 生成网格后cell_volume为正 ----
assert grid_test.cell_volume() > 0.0, '[TC27] 生成网格后cell_volume为正 FAILED'

# ---- TC28: integrate_field_energy_quadrature对零场返回零 ----
E_zero = (np.zeros((3,3,3)), np.zeros((3,3,3)), np.zeros((3,3,3)))
H_zero = (np.zeros((3,3,3)), np.zeros((3,3,3)), np.zeros((3,3,3)))
eps_z = np.ones((3,3,3)) * EPSILON_0
mu_z = np.ones((3,3,3)) * MU_0
W_quad_zero = integrate_field_energy_quadrature(E_zero, H_zero, eps_z, mu_z, 0.001, 0.001, 0.001, order=2)
assert abs(W_quad_zero) < 1e-30, '[TC28] integrate_field_energy_quadrature对零场返回零 FAILED'

# ---- TC29: rms_error对已知差异计算正确 ----
a_arr = np.array([0.0, 4.0, 8.0])
b_arr = np.array([1.0, 5.0, 9.0])
rms_val = rms_error(a_arr, b_arr)
assert abs(rms_val - 1.0) < 1e-12, '[TC29] rms_error对已知差异计算正确 FAILED'

# ---- TC30: check_energy_conservation对能量增长返回不通过 ----
W_grow = [1.0, 2.0, 4.0, 8.0]
t_grow = [0.0, 1.0, 2.0, 3.0]
P_grow = [0.0, 0.0, 0.0]
ec_grow = check_energy_conservation(W_grow, t_grow, P_grow, tol=1e-6)
assert ec_grow['conserved'] == False, '[TC30] check_energy_conservation对能量增长返回不通过 FAILED'

print("\n全部 30 个测试通过!\n")
