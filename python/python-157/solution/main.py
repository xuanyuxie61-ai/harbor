"""
main.py
燃烧科学：爆轰波结构与传播 — 统一入口

本项目基于15个科研代码项目的核心算法，融合构建面向
爆轰波结构与传播问题的博士级科学计算系统。

运行方式:
    python main.py
无需任何参数，自动执行完整计算流程。
"""
import numpy as np
import time

# ============================================================
# 模块导入
# ============================================================
from combustion_utils import (
    cj_detonation_velocity, von_neumann_spike_conditions,
    sound_speed_from_prho, check_positive
)
from reaction_kinetics import ReactiveState, chemical_source_term
from znd_structure import ZNDSolver
from euler_reactive_solver import ReactiveEulerSolver, SparseCRS
from sparse_grid_chemistry import SparseGridChemistry
from thermal_quadrature import integrate_thermal_source, average_temperature_profile
from adaptive_mesh import AdaptiveDetonationMesh, triangle_area, basis_t3
from monte_carlo_ignition import (
    sample_ellipse, ignition_probability_monte_carlo, critical_kernel_escape_time
)
from cj_condition_optimizer import CJConditionSolver, zero_brent, local_min_brent
from reaction_network_graph import build_hydrogen_oxygen_network
from stability_analysis import DetonationStability


def section_header(title):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def main():
    t_start = time.time()
    np.random.seed(42)

    # ============================================================
    # 全局物理参数
    # ============================================================
    gamma = 1.4
    Q = 2.5e6          # J/kg
    Ea = 8.314e4       # J/mol
    A_pre = 1.0e8      # 1/s
    rho0 = 1.225       # kg/m^3
    p0 = 101325.0      # Pa
    T0 = 300.0         # K
    W_mol = 0.029      # kg/mol

    print("=" * 70)
    print("  燃烧科学：爆轰波结构与传播 — 博士级科学计算系统")
    print("=" * 70)
    print(f"\n物理参数:")
    print(f"  比热比 gamma = {gamma}")
    print(f"  单位质量释热 Q = {Q:.3e} J/kg")
    print(f"  活化能 Ea = {Ea:.3e} J/mol")
    print(f"  指前因子 A = {A_pre:.3e} 1/s")
    print(f"  未燃密度 rho0 = {rho0:.3f} kg/m^3")
    print(f"  未燃压强 p0 = {p0:.1f} Pa")
    print(f"  未燃温度 T0 = {T0:.1f} K")

    # ============================================================
    # 1. CJ 爆轰速度与条件优化
    # 融合来源：836_opt_quadratic, 695_local_min_rc, 1432_zero_rc
    # ============================================================
    section_header("1. CJ/DCJ 爆轰条件优化求解")

    cj_solver = CJConditionSolver(gamma=gamma, Q=Q, p0=p0, rho0=rho0)
    D_cj_exact = cj_solver.exact_cj_velocity()
    print(f"\n解析 CJ 速度: D_CJ = {D_cj_exact:.3f} m/s")

    # 用 Brent 根查找验证 CJ 条件
    def rayleigh_hugoniot_residual(D):
        v0 = 1.0 / rho0
        vs = np.linspace(0.15 * v0, 0.85 * v0, 200)
        diffs = []
        for v in vs:
            try:
                pH = cj_solver.hugoniot_pressure(v)
                pR = cj_solver.rayleigh_line(v, D)
                diffs.append(abs(pH - pR))
            except Exception:
                continue
        return min(diffs) if diffs else np.inf

    try:
        D_cj_opt, _ = local_min_brent(rayleigh_hugoniot_residual,
                                       D_cj_exact * 0.8, D_cj_exact * 1.2,
                                       tol=1.0e-6)
        print(f"优化 CJ 速度: D_CJ = {D_cj_opt:.3f} m/s")
    except Exception as e:
        print(f"优化过程遇到边界问题，使用解析值: {e}")
        D_cj_opt = D_cj_exact

    # Von Neumann 尖峰状态
    p_vn, rho_vn, T_vn, M = von_neumann_spike_conditions(D_cj_exact, gamma, p0, rho0)
    print(f"\nVon Neumann 尖峰状态:")
    print(f"  压强 p_vN = {p_vn:.3e} Pa")
    print(f"  密度 rho_vN = {rho_vn:.3f} kg/m^3")
    print(f"  温度 T_vN = {T_vn:.1f} K")
    print(f"  激波马赫数 M = {M:.3f}")

    # ============================================================
    # 2. ZND 爆轰结构一维求解
    # 融合来源：861_pendulum_nonlinear_ode, 315_double_well_ode
    # ============================================================
    section_header("2. ZND 爆轰结构一维求解")

    znd = ZNDSolver(gamma=gamma, Q=Q, A=A_pre, Ea=Ea,
                    rho0=rho0, p0=p0, T0=T0, W_mol=W_mol)
    xi, sol = znd.solve(D=D_cj_exact, ximax=5.0e-4, npts=2000)

    rho_profile = sol[:, 0]
    u_profile = sol[:, 1]
    p_profile = sol[:, 2]
    lambda_profile = sol[:, 3]

    L_ind = znd.induction_length(xi, sol, threshold=0.95)
    L_half = znd.half_reaction_length(xi, sol)

    print(f"\nZND 结构特征长度:")
    print(f"  诱导区长度 (95%反应) L_ind = {L_ind:.3e} m")
    print(f"  半反应长度 L_1/2 = {L_half:.3e} m")
    print(f"  最终反应进度 lambda_end = {lambda_profile[-1]:.6f}")
    print(f"  最终密度 rho_end = {rho_profile[-1]:.4f} kg/m^3")
    print(f"  最终压强 p_end = {p_profile[-1]:.3e} Pa")

    # ============================================================
    # 3. 二维反应 Euler 方程数值模拟
    # 融合来源：978_r8crs
    # ============================================================
    section_header("3. 二维可压缩反应 Euler 方程数值模拟")

    nx, ny = 60, 20
    dx = 1.0e-4
    dy = 2.0e-4
    euler = ReactiveEulerSolver(nx, ny, dx, dy,
                                 gamma=gamma, Q=Q, A=A_pre, Ea=Ea)
    euler.initialize_cj_planar_wave(D_cj_exact, rho0, p0,
                                     lambda0=0.0, width_factor=3.0)

    t_final = 2.0e-7
    print(f"\n网格: {nx} x {ny}, dx={dx:.2e} m, dy={dy:.2e} m")
    print(f"推进时间: t_final = {t_final:.2e} s")
    t_end, n_step = euler.advance(t_final, cfl=0.2, n_print=20)
    print(f"完成: {n_step} 步, 实际到达时间 t = {t_end:.2e} s")

    # 提取结果统计
    U_final = euler.U
    lambda_field = np.zeros((nx, ny))
    T_field = np.zeros((nx, ny))
    rho_field = np.zeros((nx, ny))
    for i in range(nx):
        for j in range(ny):
            state = ReactiveState.from_conservative(U_final[i, j], gamma, Q)
            lambda_field[i, j] = state.lambda_var
            T_field[i, j] = state.temperature(gamma, Q, W_mol)
            rho_field[i, j] = state.rho

    T_avg = average_temperature_profile(T_field, dx, dy)
    print(f"\n模拟结果统计:")
    print(f"  平均温度 T_avg = {T_avg:.1f} K")
    print(f"  最大温度 T_max = {np.max(T_field):.1f} K")
    print(f"  最小密度 rho_min = {np.min(rho_field):.4f} kg/m^3")
    print(f"  平均反应进度 lambda_avg = {np.mean(lambda_field):.4f}")

    # ============================================================
    # 4. 稀疏网格高维化学流形
    # 融合来源：1133_spinterp
    # ============================================================
    section_header("4. 稀疏网格高维化学流形插值")

    def dummy_rate_func(y):
        r"""
        示例高维反应速率函数:
            k_eff = A * exp(-Ea/(R*T_eff)) * phi_eff * x_r_eff
        其中 y[0]~T, y[1]~p, y[2]~phi, y[3]~x_r
        """
        T_norm = 0.5 * (y[0] + 1.0)  # 映射到 [0,1]
        p_norm = 0.5 * (y[1] + 1.0)
        phi_norm = 0.5 * (y[2] + 1.0)
        xr_norm = 0.5 * (y[3] + 1.0)

        T_eff = 300.0 + T_norm * 2700.0
        phi_eff = np.exp(-0.5 * ((phi_norm * 2.5 - 1.0) / 0.3) ** 2)
        xr_eff = 1.0 - xr_norm * 0.5
        rate = A_pre * np.exp(-Ea / (8.314 * max(T_eff, 100.0))) * phi_eff * xr_eff
        return rate

    sg = SparseGridChemistry(max_level=3, dim=4)
    sg.build(dummy_rate_func)
    test_points = np.array([
        [0.0, 0.0, 0.0, 0.0],
        [0.5, 0.5, 0.5, 0.5],
        [-0.5, -0.5, -0.5, -0.5],
        [0.8, -0.3, 0.2, -0.6]
    ])
    print("\n稀疏网格插值评估（4维空间）:")
    for p in test_points:
        approx = sg.evaluate(p)
        exact = dummy_rate_func(p)
        err = abs(approx - exact) / max(abs(exact), 1.0e-12)
        print(f"  点 {p}: 插值={approx:.4e}, 精确={exact:.4e}, 相对误差={err:.4e}")

    # ============================================================
    # 5. 热力学高精度求积
    # 融合来源：1151_square_symq_rule
    # ============================================================
    section_header("5. 热力学积分高精度对称求积")

    q_total = integrate_thermal_source(lambda_field, T_field, rho_field,
                                        dx, dy, degree=5,
                                        A=A_pre, Ea=Ea, Q=Q)
    print(f"\n反应区总释热率: q_dot_total = {q_total:.3e} W/m")

    # 验证求积精度：积分常数函数
    def const_one(x, y):
        return np.ones_like(x)

    from thermal_quadrature import integrate_square
    I_const = integrate_square(const_one, degree=5)
    print(f"求积精度验证: ∫∫_[-1,1]^2 1 dx dy = {I_const:.6f} (理论=4.0)")

    # ============================================================
    # 6. 自适应网格生成
    # 融合来源：373_fem_basis_t3_display, 261_cvt_square_uniform
    # ============================================================
    section_header("6. 爆轰波前自适应网格生成")

    mesh = AdaptiveDetonationMesh(x_min=0.0, x_max=1.0e-3,
                                   y_min=-0.5e-3, y_max=0.5e-3,
                                   n_base=200, wave_x=0.5e-3,
                                   wave_width=0.05e-3)
    nodes, elements = mesh.generate(cvt_samples=1000, cvt_iter=10)
    quality = mesh.element_quality(nodes, elements)
    print(f"\n自适应网格统计:")
    print(f"  节点数: {len(nodes)}")
    print(f"  三角形单元数: {len(elements)}")
    print(f"  平均网格质量: {quality:.4f}")

    # 验证 T3 基函数
    if len(elements) > 0:
        elem = elements[0]
        t = nodes[elem].T
        p_test = np.mean(nodes[elem], axis=0)
        phi0, dphi_dx, dphi_dy = basis_t3(t, 0, p_test)
        print(f"\nT3 基函数验证 (在单元形心处):")
        print(f"  phi_0 = {phi0:.4f} (理论≈1/3)")
        print(f"  dphi_0/dx = {dphi_dx:.4e}")
        print(f"  dphi_0/dy = {dphi_dy:.4e}")

    # ============================================================
    # 7. 蒙特卡洛点火概率
    # 融合来源：331_ellipse_monte_carlo, 711_mandelbrot_area,
    #           1092_snakes_and_ladders_simulation
    # ============================================================
    section_header("7. 蒙特卡洛点火概率与临界核分析")

    # 椭圆采样（模拟局部热点区域）
    A_ellipse = np.array([[4.0, 1.0], [1.0, 3.0]])
    ellipse_samples = sample_ellipse(500, A_ellipse, r=0.01)
    print(f"\n椭圆内采样: {ellipse_samples.shape[0]} 个点")
    print(f"  样本均值: ({np.mean(ellipse_samples[:,0]):.4e}, {np.mean(ellipse_samples[:,1]):.4e})")
    print(f"  样本协方差特征值: {np.linalg.eigvals(np.cov(ellipse_samples.T))}")

    # 点火概率
    mean_prob, std_prob, batch_probs = ignition_probability_monte_carlo(
        n_samples=2000, T_mean=1200.0, T_std=200.0,
        p_mean=5.0e5, p_std=1.0e5,
        phi_mean=1.0, phi_std=0.2,
        Ea=Ea, A=A_pre, T_ign=1500.0, n_batches=5
    )
    print(f"\n点火概率蒙特卡洛评估 (5批次, 每批400样本):")
    print(f"  平均点火概率: {mean_prob:.4f}")
    print(f"  批次标准差: {std_prob:.4f}")
    print(f"  各批次概率: {[f'{p:.4f}' for p in batch_probs]}")

    # 临界核逃逸时间
    area_frac, avg_esc, _ = critical_kernel_escape_time(
        n_grid=40, it_max=100, D_wave=D_cj_exact,
        gamma=gamma, Q=Q, rho0=rho0, p0=p0
    )
    print(f"\n临界核逃逸分析 (40x40 网格, 100 迭代):")
    print(f"  成功点火区域占比: {area_frac:.4f}")
    print(f"  平均逃逸时间: {avg_esc:.1f} 迭代步")

    # ============================================================
    # 8. 反应网络图分析
    # 融合来源：489_grf_display
    # ============================================================
    section_header("8. H2-O2 燃烧反应网络图分析")

    net = build_hydrogen_oxygen_network()
    stats = net.network_statistics()
    print(f"\n反应网络统计:")
    print(f"  物种数: {stats['n_nodes']}")
    print(f"  反应边数: {stats['n_edges']}")
    print(f"  平均度数: {stats['avg_degree']:.2f}")
    print(f"  图密度: {stats['density']:.4f}")
    print(f"  平均聚类系数: {stats['avg_clustering']:.4f}")

    path = net.bfs_shortest_path("H2", "H2O")
    print(f"\n最短路径 H2 → H2O: {' -> '.join(path)}")

    cycles = net.find_cycles(max_length=5)
    print(f"\n检测到的反应循环 (长度≤5): {len(cycles)} 个")
    for c in cycles[:5]:
        names = [net.node_names[i] for i in c]
        print(f"  {' -> '.join(names)}")

    # ============================================================
    # 9. 线性稳定性分析
    # 融合来源：315_double_well_ode, 861_pendulum_nonlinear_ode
    # ============================================================
    section_header("9. 爆轰波线性稳定性分析")

    stability = DetonationStability(xi, sol, gamma=gamma, Q=Q)
    evals, evecs = stability.eigenvalue_analysis()
    unstable = stability.instability_modes()

    print(f"\n稳定性矩阵特征值 (前4个):")
    for k in range(min(4, len(evals))):
        ev = evals[k]
        print(f"  sigma_{k} = {ev.real:+.4e} {ev.imag:+.4e}j")

    if unstable:
        print(f"\n检测到 {len(unstable)} 个不稳定模态:")
        for m in unstable[:3]:
            print(f"  增长率 alpha={m['growth_rate']:.4e} 1/s, "
                  f"频率 f={m['frequency']:.4e} Hz")
    else:
        print("\n未发现线性不稳定模态（当前近似下）。")

    f_puls = stability.pulsation_frequency_estimate()
    print(f"\n爆轰头振荡频率估计: f_puls = {f_puls:.4e} Hz")

    # ============================================================
    # 10. 稀疏矩阵验证
    # 融合来源：978_r8crs
    # ============================================================
    section_header("10. 稀疏矩阵 CRS 格式验证")

    m, n = 5, 5
    row_ptr = np.array([0, 2, 4, 6, 8, 10])
    col_idx = np.array([0, 1, 1, 2, 2, 3, 3, 4, 0, 4])
    vals = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    sp = SparseCRS(m, n, row_ptr, col_idx, vals)

    x_test = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    y_test = sp.multiply(x_test)
    yt_test = sp.multiply_transpose(x_test)
    print(f"\nCRS 矩阵-向量乘法验证:")
    print(f"  A * [1,1,1,1,1]^T = {y_test}")
    print(f"  A^T * [1,1,1,1,1]^T = {yt_test}")

    # ============================================================
    # 总结
    # ============================================================
    t_elapsed = time.time() - t_start
    section_header("计算完成")
    print(f"\n总运行时间: {t_elapsed:.2f} 秒")
    print("所有模块运行完毕，无报错。")
    print("=" * 70)


if __name__ == "__main__":
    main()
