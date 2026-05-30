
import sys
import numpy as np


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
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def run_simulation():
    np.random.seed(42)

    print_section("微波器件FDTD仿真 — 圆柱谐振腔模式分析")
    print("科学领域: 电磁学 — 微波器件时域有限差分仿真")
    print("合成项目: 基于15个种子算法的博士级科学计算")




    print_section("1. 仿真参数与网格配置")


    R_cavity = 0.05
    H_cavity = 0.10
    f_excitation = 2.0e9


    nx, ny, nz = generate_prime_grid_steps(31, 31, 41)
    print(f"  素数网格步数: nx={nx}, ny={ny}, nz={nz}")


    Lx = Ly = 2.2 * R_cavity
    Lz = 1.2 * H_cavity

    grid = generate_rectangular_grid(0.0, Lx, nx, 0.0, Ly, ny, 0.0, Lz, nz)
    print(f"  计算域: {Lx*1000:.1f}×{Ly*1000:.1f}×{Lz*1000:.1f} mm³")
    print(f"  网格步长: dx={grid.dx*1000:.3f}, dy={grid.dy*1000:.3f}, dz={grid.dz*1000:.3f} mm")




    print_section("2. 几何与材料配置")


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


    dielectric_segment = CircleSegmentDielectric(
        r=R_cavity * 0.8,
        theta=np.pi / 3.0,
        height=H_cavity * 0.5,
        epsilon_r=10.0,
        center=(Lx / 2.0, Ly / 2.0),
    )
    print(f"  圆扇形介质: r={dielectric_segment.r*1000:.1f} mm, "
          f"θ={np.degrees(dielectric_segment.theta):.1f}°, εr={dielectric_segment.epsilon_r}")
    print(f"  介质面积: {dielectric_segment.area()*1e6:.3f} mm²")


    shapes = [cavity, dielectric_segment]
    epsilon, mu, sigma = assign_material_properties(grid, shapes)


    eps_min, eps_max = np.min(epsilon), np.max(epsilon)
    print(f"  介电常数范围: {eps_min/EPSILON_0:.2f}ε₀ ~ {eps_max/EPSILON_0:.2f}ε₀")




    print_section("3. PML吸收边界条件")

    pml = PMLBoundary3D(grid, pml_thickness=6, reflection_coeff=1e-5)
    R_est = pml.compute_reflection_estimate()
    print(f"  PML厚度: {pml.pml_thickness} 层")
    print(f"  PML最大电导率: {pml.sigma_max:.3e} S/m")
    print(f"  估计反射系数: {R_est:.2e}")




    print_section("4. FDTD引擎初始化")


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




    print_section("5. 时域仿真推进")


    T_period = 1.0 / f_excitation
    n_steps = int(15.0 * T_period / engine.dt)
    sample_interval = max(1, n_steps // 200)

    print(f"  总时间步数: {n_steps}")
    print(f"  对应物理时间: {n_steps * engine.dt * 1e9:.3f} ns")
    print(f"  能量采样间隔: {sample_interval} 步")
    print("  运行中...")


    energy_history = []
    power_loss_history = []
    time_history = []

    for step in range(n_steps):









        raise NotImplementedError("Hole 3: 请实现FDTD时域推进主循环")

    print(f"  仿真完成。最终时间: {engine.time*1e9:.3f} ns")




    print_section("6. 能量守恒与数值稳定性分析")



    energy_check = check_energy_conservation(
        energy_history, time_history, power_loss_history, tol=1.0
    )
    has_nan = any(not np.isfinite(w) for w in energy_history)
    print(f"  数值稳定性检查: {'通过（无NaN/Inf）' if not has_nan else '失败（检测到NaN/Inf）'}")
    print(f"  能量漂移: {energy_check['energy_drift']:.3e}")
    print(f"  注: 有源条件下能量不守恒是正常的（外部源持续注入能量）")


    kx_test = 2.0 * np.pi / Lx
    ky_test = 2.0 * np.pi / Ly
    omega_num, omega_ex = stability_analysis_2d_scalar(
        kx_test, ky_test, grid.dx, grid.dy, engine.dt, engine.c_max
    )
    dispersion_error = abs(omega_num - omega_ex) / abs(omega_ex + 1e-30)
    print(f"  数值色散误差(测试波数): {dispersion_error:.3e}")




    print_section("7. 本征模式分析与Q值计算")


    nx_2d = min(nx, 25)
    ny_2d = min(ny, 25)
    dx_2d = Lx / (nx_2d - 1)
    dy_2d = Ly / (ny_2d - 1)


    epsilon_2d = epsilon[:, :, nz // 2]
    mu_2d = mu[:, :, nz // 2]


    epsilon_2d_anal = epsilon_2d[::max(1, nx//nx_2d), ::max(1, ny//ny_2d)]
    mu_2d_anal = mu_2d[::max(1, nx//nx_2d), ::max(1, ny//ny_2d)]

    epsilon_2d_anal = epsilon_2d_anal[:nx_2d, :ny_2d]
    mu_2d_anal = mu_2d_anal[:nx_2d, :ny_2d]

    modes = compute_cavity_modes_2d(nx_2d, ny_2d, dx_2d, dy_2d, epsilon_2d_anal, mu_2d_anal, n_modes=3)

    for idx, mode in enumerate(modes):
        f_mode = mode['frequency']
        k_mode = mode['wavenumber']
        print(f"  模式{idx+1}: f = {f_mode/1e9:.4f} GHz, k = {k_mode:.2f} rad/m")


    if len(energy_history) > 20:

        W_arr = np.array(energy_history)
        t_arr = np.array(time_history)

        from scipy.signal import find_peaks
        peaks, _ = find_peaks(W_arr, distance=max(3, int(T_period / engine.dt / sample_interval * 0.8)))
        if len(peaks) >= 3:
            peak_times = t_arr[peaks]
            peak_energies = W_arr[peaks]

            if len(peak_energies) >= 2 and np.all(peak_energies > 1e-30):
                logW = np.log(peak_energies)

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


    sigma_copper = 5.8e7
    skin_depth = np.sqrt(2.0 / (2.0 * np.pi * f_excitation * MU_0 * sigma_copper))
    surface_resistance = 1.0 / (sigma_copper * skin_depth)

    if surface_resistance > 0:
        Q_theoretical = (2.0 * np.pi * f_excitation * MU_0 * R_cavity) / (4.0 * surface_resistance)
        print(f"  理论Q值(铜壁): {Q_theoretical:.2f}")
    else:
        Q_theoretical = None




    print_section("8. 功率流网络分析（PageRank类比）")

    E_final = (engine.Ex, engine.Ey, engine.Ez)
    H_final = (engine.Hx, engine.Hy, engine.Hz)
    rank_field = power_flow_pagerank_analysis(E_final, H_final, grid.dx, grid.dy, grid.dz)
    rank_max = np.max(rank_field)
    rank_mean = np.mean(rank_field)
    print(f"  最大能量集中度: {rank_max:.3e}")
    print(f"  平均能量集中度: {rank_mean:.3e}")
    print(f"  能量集中比: {rank_max/rank_mean:.2f}")




    print_section("9. Hankel/Toeplitz结构验证")


    n_hankel = 3
    x_hankel = np.array([2.0, -1.0, 3.0, 0.5, 1.0])
    A_inv, A = hankel_inverse_fiedler(n_hankel, x_hankel)


    I_approx = A @ A_inv
    identity_error = np.linalg.norm(I_approx - np.eye(n_hankel))
    cond_number = np.linalg.cond(A)
    print(f"  Hankel矩阵阶数: {n_hankel}")
    print(f"  矩阵条件数: {cond_number:.2f}")
    print(f"  Fiedler逆矩阵验证误差: {identity_error:.3e}")
    print(f"  （验证了Fiedler公式 H⁻¹ = M₁·M₂ - M₃·M₄ 的正确性）")


    n_ant = 5
    Z_matrix = antenna_array_impedance_matrix(n_ant, 0.5)
    print(f"  阵列互阻抗矩阵({n_ant}元):")
    for i in range(n_ant):
        row_str = "  " + "  ".join([f"{Z_matrix[i,j]:7.2f}" for j in range(n_ant)])
        print(row_str)




    print_section("10. 高阶数值积分验证")


    W_quad = integrate_field_energy_quadrature(
        E_final, H_final, epsilon, mu, grid.dx, grid.dy, grid.dz, order=3
    )
    W_direct = np.sum(epsilon * (engine.Ex**2 + engine.Ey**2 + engine.Ez**2) +
                      mu * (engine.Hx**2 + engine.Hy**2 + engine.Hz**2)) * 0.5 * grid.cell_volume()
    quad_error = abs(W_quad - W_direct) / (abs(W_direct) + 1e-30)
    print(f"  高斯积分能量: {W_quad:.6e} J")
    print(f"  直接求和能量: {W_direct:.6e} J")
    print(f"  积分相对差异: {quad_error:.3e}")


    from quadrature_rules import integrate_triangle_wandzura
    tri_vertices = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    f_test = lambda x, y: x**2 + y**2
    I_wandzura = integrate_triangle_wandzura(f_test, tri_vertices, rule_degree=7)
    I_exact = 1.0 / 6.0
    print(f"  Wandzura积分测试: {I_wandzura:.6e} (精确值: {I_exact:.6e}, 误差: {abs(I_wandzura-I_exact):.3e})")




    print_section("11. 拉格朗日插值验证")

    xd = np.linspace(0.0, Lz, 7)
    yd = np.sin(2.0 * np.pi * xd / Lz) + 0.5 * np.cos(4.0 * np.pi * xd / Lz)
    xi = np.linspace(0.0, Lz, 50)
    yi_lagrange = lagrange_value_1d(xd, yd, xi)
    yi_exact = np.sin(2.0 * np.pi * xi / Lz) + 0.5 * np.cos(4.0 * np.pi * xi / Lz)
    interp_rms = rms_error(yi_lagrange, yi_exact)
    print(f"  插值节点数: {len(xd)}")
    print(f"  插值RMS误差: {interp_rms:.3e}")


    z_test = np.linspace(0.0, Lz, 20)
    eps_profile = np.where(z_test < H_cavity / 2.0, EPSILON_0, 2.0 * EPSILON_0)
    z_query = np.linspace(0.0, Lz, 100)
    eps_interp = interpolate_material_profile(z_test, eps_profile, z_query, method='lagrange')
    print(f"  材料参数插值范围: {np.min(eps_interp)/EPSILON_0:.2f}ε₀ ~ {np.max(eps_interp)/EPSILON_0:.2f}ε₀")




    print_section("12. 蒙特卡洛球面采样验证")

    sphere_stats = sphere_distance_stats(n_samples=2000)
    theoretical_mean = 4.0 / 3.0
    print(f"  球面点对平均距离: {sphere_stats['mean']:.4f} (理论: {theoretical_mean:.4f})")
    print(f"  方差: {sphere_stats['variance']:.4f}")
    print(f"  距离统计相对误差: {abs(sphere_stats['mean'] - theoretical_mean):.4f}")




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
        sys.exit(0)
    except Exception as e:
        print(f"\n[错误] 仿真过程中发生异常: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
