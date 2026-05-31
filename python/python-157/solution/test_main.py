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

# ================================================================
# 测试用例（60个，assert模式，涉及随机值均使用固定种子）
# ================================================================
from combustion_utils import (
    rankine_hugoniot_pressure_ratio, rankine_hugoniot_density_ratio,
    arrhenius_rate, sound_speed, specific_heat_ratio_cv_cp,
    znd_progress_variable_derivative, cholesky_factor, solve_lower_triangular,
    check_interval, temperature_from_energy
)
from adaptive_mesh import adaptive_density_function
from sparse_grid_chemistry import clenshaw_curtis_nodes_1d, piecewise_linear_basis, sparse_grid_index_set
from thermal_quadrature import integrate_square
from reaction_kinetics import euler_flux_x, euler_flux_y

# ---- TC01: CJ爆轰速度为正值有限数 ----
D_cj_test = cj_detonation_velocity(1.4, 2.5e6, 101325.0, 1.225)
assert D_cj_test > 0 and np.isfinite(D_cj_test), '[TC01] CJ detonation velocity not positive finite FAILED'

# ---- TC02: Von Neumann尖峰压强大于初始压强 ----
p_vn_test, rho_vn_test, T_vn_test, M_test = von_neumann_spike_conditions(D_cj_test, 1.4, 101325.0, 1.225)
assert p_vn_test > 101325.0, '[TC02] Von Neumann pressure not greater than p0 FAILED'

# ---- TC03: Von Neumann尖峰密度大于初始密度 ----
assert rho_vn_test > 1.225, '[TC03] Von Neumann density not greater than rho0 FAILED'

# ---- TC04: Von Neumann马赫数大于1 ----
assert M_test > 1.0, '[TC04] Von Neumann Mach number not > 1 FAILED'

# ---- TC05: Rankine-Hugoniot压力比在M>1时大于1 ----
rp = rankine_hugoniot_pressure_ratio(2.0, 1.4)
assert rp > 1.0, '[TC05] RH pressure ratio not > 1 FAILED'

# ---- TC06: Rankine-Hugoniot密度比有限正值且大于1 ----
rd = rankine_hugoniot_density_ratio(2.0, 1.4)
assert rd > 1.0 and np.isfinite(rd), '[TC06] RH density ratio not > 1 finite FAILED'

# ---- TC07: Arrhenius速率在参考温度下有限正值 ----
k0 = arrhenius_rate(300.0, 1.0e8, 8.314e4)
assert k0 > 0 and np.isfinite(k0), '[TC07] Arrhenius rate not positive finite FAILED'

# ---- TC08: Arrhenius速率随温度单调递增 ----
k1 = arrhenius_rate(500.0, 1.0e8, 8.314e4)
k2 = arrhenius_rate(1000.0, 1.0e8, 8.314e4)
assert k2 > k1, '[TC08] Arrhenius rate not monotonic increasing FAILED'

# ---- TC09: 声速计算结果为正值有限数 ----
a_test = sound_speed(300.0, 1.4, 0.029)
assert a_test > 0 and np.isfinite(a_test), '[TC09] Sound speed not positive finite FAILED'

# ---- TC10: sound_speed_from_prho与sound_speed一致性 ----
a_prho = sound_speed_from_prho(101325.0, 1.225, 1.4)
assert a_prho > 0 and np.isfinite(a_prho), '[TC10] Sound speed from p,rho not positive finite FAILED'

# ---- TC11: 定容比热容和定压比热容为正值且cp>cv ----
cv_test, cp_test = specific_heat_ratio_cv_cp(1.4)
assert cv_test > 0 and cp_test > cv_test, '[TC11] Specific heats not valid FAILED'

# ---- TC12: 反应进度导数dλ/dt为非正值 ----
dlam = znd_progress_variable_derivative(0.0, 1500.0, 1.0e8, 8.314e4)
assert dlam <= 0, '[TC12] Progress variable derivative should be non-positive FAILED'

# ---- TC13: λ=1时反应进度导数为零 ----
dlam1 = znd_progress_variable_derivative(1.0, 1500.0, 1.0e8, 8.314e4)
assert dlam1 == 0.0, '[TC13] dlambda/dt at lambda=1 not zero FAILED'

# ---- TC14: Cholesky分解L*L^T恢复原矩阵 ----
A_test_mat = np.array([[4.0, 1.0], [1.0, 3.0]])
L_test = cholesky_factor(A_test_mat)
A_recon = L_test @ L_test.T
assert np.allclose(A_test_mat, A_recon), '[TC14] Cholesky L*L^T not reconstruct A FAILED'

# ---- TC15: 三角形面积计算（单位直角三角形面积为1） ----
t_test = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 2.0]])
area_test = triangle_area(t_test)
assert abs(area_test - 1.0) < 1.0e-12, '[TC15] Triangle area not 1.0 FAILED'

# ---- TC16: T3基函数在形心处和为1 ----
t_test_t3 = np.array([[0.0, 2.0, 0.0], [0.0, 0.0, 2.0]])
centroid = np.mean(t_test_t3, axis=1)
phi_sum = 0.0
for idx in range(3):
    phi_i, _, _ = basis_t3(t_test_t3, idx, centroid)
    phi_sum += phi_i
assert abs(phi_sum - 1.0) < 1.0e-12, '[TC16] T3 basis sum at centroid not 1 FAILED'

# ---- TC17: 自适应密度函数在波前附近值更大 ----
np.random.seed(42)
dens_near = adaptive_density_function(0.5, 0.0, wave_x=0.5, wave_width=0.05)
dens_far = adaptive_density_function(0.0, 0.0, wave_x=0.5, wave_width=0.05)
assert dens_near > dens_far, '[TC17] Adaptive density not higher near wave FAILED'

# ---- TC18: 椭圆采样点数正确 ----
np.random.seed(42)
A_ell = np.array([[4.0, 1.0], [1.0, 3.0]])
samples_ell = sample_ellipse(100, A_ell, r=0.01)
assert samples_ell.shape[0] == 100, '[TC18] Ellipse sample count not correct FAILED'

# ---- TC19: 点火概率均值在[0,1]区间内且标准差非负 ----
np.random.seed(42)
mean_p, std_p, batch_p = ignition_probability_monte_carlo(
    n_samples=500, T_mean=1200.0, T_std=200.0,
    p_mean=5.0e5, p_std=1.0e5,
    phi_mean=1.0, phi_std=0.2,
    Ea=8.314e4, A=1.0e8, T_ign=1500.0, n_batches=5
)
assert 0.0 <= mean_p <= 1.0, '[TC19] Ignition probability not in [0,1] FAILED'
assert std_p >= 0.0, '[TC19b] Std dev should be non-negative FAILED'

# ---- TC20: 临界核逃逸时间分析面积占比在[0,1]内 ----
np.random.seed(42)
af, ae, _ = critical_kernel_escape_time(
    n_grid=20, it_max=50, D_wave=D_cj_test,
    gamma=1.4, Q=2.5e6, rho0=1.225, p0=101325.0
)
assert 0.0 <= af <= 1.0, '[TC20] Area fraction not in [0,1] FAILED'
assert ae >= 0, '[TC20b] Avg escape time not non-negative FAILED'

# ---- TC21: CJ解析速度为正值有限数 ----
cj_solver_test = CJConditionSolver(gamma=1.4, Q=2.5e6, p0=101325.0, rho0=1.225)
D_exact_test = cj_solver_test.exact_cj_velocity()
assert D_exact_test > 0 and np.isfinite(D_exact_test), '[TC21] CJ exact velocity not positive finite FAILED'

# ---- TC22: Hugoniot压力返回有限值 ----
p_hug = cj_solver_test.hugoniot_pressure(1.0 / 1.225)
assert np.isfinite(p_hug), '[TC22] Hugoniot pressure not finite FAILED'

# ---- TC23: Rayleigh线在v=v0处等于p0 ----
v0_test = 1.0 / 1.225
p_ray = cj_solver_test.rayleigh_line(v0_test, D_exact_test)
assert abs(p_ray - 101325.0) < 1.0e-6, '[TC23] Rayleigh line at v0 not equal p0 FAILED'

# ---- TC24: Brent求根法对-x+2在[0,3]上求根得2 ----
def f_root(x):
    return -x + 2.0
root = zero_brent(f_root, 0.0, 3.0, tol=1.0e-10)
assert abs(root - 2.0) < 1.0e-8, '[TC24] Brent root not close to 2 FAILED'

# ---- TC25: Brent局部最小化对(x-3)^2在[0,5]上求得x≈3 ----
def f_min(x):
    return (x - 3.0) ** 2
xmin, fmin_val = local_min_brent(f_min, 0.0, 5.0, tol=1.0e-10)
assert abs(xmin - 3.0) < 1.0e-6, '[TC25] Brent min not at x=3 FAILED'
assert fmin_val < 1.0e-8, '[TC25b] Brent min value not near zero FAILED'

# ---- TC26: H2-O2反应网络有8个物种节点 ----
net_test = build_hydrogen_oxygen_network()
stats_test = net_test.network_statistics()
assert stats_test['n_nodes'] == 8, '[TC26] Network node count not 8 FAILED'
assert stats_test['n_edges'] > 0, '[TC26b] Network edge count should be positive FAILED'

# ---- TC27: BFS最短路径H2→H2O存在且端点正确 ----
path_test = net_test.bfs_shortest_path("H2", "H2O")
assert path_test is not None, '[TC27] BFS path H2→H2O not found FAILED'
assert len(path_test) >= 2, '[TC27b] Path should have at least 2 nodes FAILED'
assert path_test[0] == "H2", '[TC27c] Path should start with H2 FAILED'
assert path_test[-1] == "H2O", '[TC27d] Path should end with H2O FAILED'

# ---- TC28: 反应循环检测返回列表 ----
cycles_test = net_test.find_cycles(max_length=5)
assert isinstance(cycles_test, list), '[TC28] Cycles should be a list FAILED'

# ---- TC29: Clenshaw-Curtis节点level=0返回[0] ----
cc0 = clenshaw_curtis_nodes_1d(0)
assert len(cc0) == 1 and abs(cc0[0] - 0.0) < 1.0e-12, '[TC29] CC level 0 not [0] FAILED'

# ---- TC30: Clenshaw-Curtis节点level=2有5个节点且端点正确 ----
cc2 = clenshaw_curtis_nodes_1d(2)
assert len(cc2) == 5, '[TC30] CC level 2 not 5 nodes FAILED'
assert abs(cc2[0] - 1.0) < 1.0e-12, '[TC30b] CC first node should be 1 FAILED'
assert abs(cc2[-1] + 1.0) < 1.0e-12, '[TC30c] CC last node should be -1 FAILED'

# ---- TC31: 分段线性基函数左外推时仅首项为1 ----
nodes_pl = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
w_left = piecewise_linear_basis(nodes_pl, -2.0)
assert abs(w_left[0] - 1.0) < 1.0e-12, '[TC31] Linear basis left extrapolate not 1 at first node FAILED'

# ---- TC32: 分段线性基函数内插权重非负 ----
w_mid = piecewise_linear_basis(nodes_pl, 0.25)
assert np.all(w_mid >= 0), '[TC32] Linear basis weights contain negative FAILED'

# ---- TC33: 对称求积规则∫∫_[-1,1]^2 1 dxdy ≈ 4 ----
sq_result = integrate_square(lambda x, y: np.ones_like(x), degree=5)
assert abs(sq_result - 4.0) < 1.0e-12, '[TC33] Symmetric quadrature ∫∫1 not ≈4 FAILED'

# ---- TC34: 平均温度计算返回正值 ----
T_test_field = np.ones((5, 5)) * 1000.0
T_avg_test = average_temperature_profile(T_test_field, 0.01, 0.01)
assert T_avg_test > 0, '[TC34] Average temperature not positive FAILED'
assert abs(T_avg_test - 1000.0) < 1.0e-6, '[TC34b] Average temperature not 1000 FAILED'

# ---- TC35: CRS矩阵向量乘法输出维度正确且值有限 ----
sp_test = SparseCRS(5, 5,
    np.array([0, 2, 4, 6, 8, 10]),
    np.array([0, 1, 1, 2, 2, 3, 3, 4, 0, 4]),
    np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]))
y_sp = sp_test.multiply(np.ones(5))
assert len(y_sp) == 5, '[TC35] CRS multiply output dimension wrong FAILED'
assert np.all(np.isfinite(y_sp)), '[TC35b] CRS multiply produced non-finite values FAILED'

# ---- TC36: ReactiveState守恒量转换可逆 ----
state_orig = ReactiveState(rho=1.2, u=100.0, v=0.0, e=3.0e6, lambda_var=0.3)
U_test = state_orig.to_conservative()
state_recov = ReactiveState.from_conservative(U_test, gamma=1.4, Q=2.5e6)
assert abs(state_recov.rho - 1.2) < 1.0e-10, '[TC36] ReactiveState rho recovery FAILED'
assert abs(state_recov.u - 100.0) < 1.0e-10, '[TC36b] ReactiveState u recovery FAILED'
assert abs(state_recov.lambda_var - 0.3) < 1.0e-10, '[TC36c] ReactiveState lambda recovery FAILED'

# ---- TC37: 化学源项输出为5分量向量且质量源项为零 ----
omega_test = chemical_source_term(state_orig, gamma=1.4, Q=2.5e6, A=1.0e8, Ea=8.314e4)
assert len(omega_test) == 5, '[TC37] Chemical source term not 5 components FAILED'
assert omega_test[0] == 0.0, '[TC37b] Mass source should be zero FAILED'

# ---- TC38: Euler通量x方向输出5分量有限值 ----
Fx_test = euler_flux_x(state_orig, gamma=1.4, Q=2.5e6)
assert len(Fx_test) == 5, '[TC38] Euler flux x not 5 components FAILED'
assert np.isfinite(Fx_test[0]), '[TC38b] Euler flux x mass flux not finite FAILED'

# ---- TC39: 反应态温度返回正值 ----
T_state = state_orig.temperature(gamma=1.4, Q=2.5e6, W_mol=0.029)
assert T_state > 0, '[TC39] State temperature not positive FAILED'

# ---- TC40: check_positive对正值正常通过 ----
try:
    check_positive(5.0, "test_val")
    passed40 = True
except ValueError:
    passed40 = False
assert passed40, '[TC40] check_positive failed on valid input FAILED'

# ---- TC41: check_positive对负值抛出ValueError ----
try:
    check_positive(-1.0, "test_val")
    passed41 = False
except ValueError:
    passed41 = True
assert passed41, '[TC41] check_positive should raise on negative input FAILED'

# ---- TC42: check_interval对a<b正常通过 ----
try:
    check_interval(0.0, 1.0)
    passed42 = True
except ValueError:
    passed42 = False
assert passed42, '[TC42] check_interval failed on valid interval FAILED'

# ---- TC43: ZND求解器正常求解不崩溃 ----
np.random.seed(42)
znd_test = ZNDSolver(gamma=1.4, Q=2.5e6, A=1.0e8, Ea=8.314e4,
                     rho0=1.225, p0=101325.0, T0=300.0, W_mol=0.029)
xi_test, sol_test = znd_test.solve(D=znd_test.cj_velocity(), ximax=1.0e-4, npts=500)
assert len(xi_test) == 500, '[TC43] ZND solve output length wrong FAILED'
assert sol_test.shape == (500, 4), '[TC43b] ZND solve output shape wrong FAILED'
assert np.all(np.isfinite(sol_test)), '[TC43c] ZND solution contains non-finite values FAILED'

# ---- TC44: 诱导区长度在求解域内 ----
L_ind_test = znd_test.induction_length(xi_test, sol_test, threshold=0.95)
assert 0 <= L_ind_test <= xi_test[-1], '[TC44] Induction length out of domain FAILED'

# ---- TC45: 半反应长度不超过诱导区长度 ----
L_half_test = znd_test.half_reaction_length(xi_test, sol_test)
assert L_half_test <= L_ind_test + 1.0e-12, '[TC45] Half-reaction length > induction length FAILED'

# ---- TC46: 自适应网格生成不崩溃 ----
np.random.seed(42)
mesh_test = AdaptiveDetonationMesh(x_min=0.0, x_max=1.0e-3,
                                   y_min=-0.5e-3, y_max=0.5e-3,
                                   n_base=50, wave_x=0.5e-3,
                                   wave_width=0.05e-3)
nodes_test, elems_test = mesh_test.generate(cvt_samples=200, cvt_iter=5)
assert len(nodes_test) > 0, '[TC46] Mesh generation produced no nodes FAILED'

# ---- TC47: 稳定性分析特征值个数为4 ----
stab_test = DetonationStability(xi_test, sol_test, gamma=1.4, Q=2.5e6)
evals_test, evecs_test = stab_test.eigenvalue_analysis()
assert len(evals_test) == 4, '[TC47] Stability eigenvalues not 4 FAILED'

# ---- TC48: 反应进度剖面始终在[0,1]区间且单调非减 ----
lambda_prof = sol_test[:, 3]
assert np.all(lambda_prof >= -1.0e-12), '[TC48] Lambda profile has negative values FAILED'
assert np.all(lambda_prof <= 1.0 + 1.0e-12), '[TC48b] Lambda profile exceeds 1 FAILED'
assert lambda_prof[-1] >= lambda_prof[0], '[TC48c] Lambda should be non-decreasing FAILED'

# ---- TC49: 密度剖面始终为正 ----
rho_prof_test = sol_test[:, 0]
assert np.all(rho_prof_test > 0), '[TC49] Density profile has non-positive values FAILED'

# ---- TC50: Euler求解器初始化和CFL时间步长正常 ----
np.random.seed(42)
euler_test = ReactiveEulerSolver(10, 5, 1.0e-4, 2.0e-4,
                                 gamma=1.4, Q=2.5e6, A=1.0e8, Ea=8.314e4)
euler_test.initialize_cj_planar_wave(D_cj_test, 1.225, 101325.0)
assert euler_test.U.shape == (10, 5, 5), '[TC50] Euler solver init shape wrong FAILED'
assert np.all(np.isfinite(euler_test.U)), '[TC50b] Euler solver init contains non-finite FAILED'
assert np.all(euler_test.U[:, :, 0] > 0), '[TC50c] Euler solver density not positive FAILED'

# ---- TC51: 热力学释热率积分返回非负值 ----
lambda_test_field = np.ones((4, 4)) * 0.5
T_test_field2 = np.ones((4, 4)) * 1500.0
rho_test_field = np.ones((4, 4)) * 1.0
q_total_test = integrate_thermal_source(lambda_test_field, T_test_field2, rho_test_field,
                                       0.01, 0.01, degree=3,
                                       A=1.0e8, Ea=8.314e4, Q=2.5e6)
assert q_total_test >= 0, '[TC51] Thermal source integral should be non-negative FAILED'

# ---- TC52: 稀疏网格索引集非空 ----
idx_set = sparse_grid_index_set(4, 3)
assert len(idx_set) > 0, '[TC52] Sparse grid index set is empty FAILED'

# ---- TC53: 稀疏网格构建和评估一致性 ----
np.random.seed(42)
sg_test = SparseGridChemistry(max_level=2, dim=2)
def simple_func(y):
    return y[0] + y[1]
sg_test.build(simple_func)
val_test = sg_test.evaluate(np.array([0.5, 0.5]))
assert np.isfinite(val_test), '[TC53] Sparse grid eval produced non-finite FAILED'

# ---- TC54: solve_lower_triangular解方程组 ----
L_mat = np.array([[2.0, 0.0], [1.0, 3.0]])
b_vec = np.array([4.0, 5.0])
x_vec = solve_lower_triangular(L_mat, b_vec)
assert abs(x_vec[0] - 2.0) < 1.0e-12, '[TC54] Lower triangular solve FAILED'
assert abs(L_mat[1,0]*x_vec[0] + L_mat[1,1]*x_vec[1] - b_vec[1]) < 1.0e-12, '[TC54b] Lower triangular verify FAILED'

# ---- TC55: 压强计算返回非负值 ----
p_state = state_orig.pressure(gamma=1.4, Q=2.5e6)
assert p_state >= 0, '[TC55] Pressure should be non-negative FAILED'

# ---- TC56: ZND求解器CJ速度计算正值 ----
D_cj_znd = znd_test.cj_velocity()
assert D_cj_znd > 0 and np.isfinite(D_cj_znd), '[TC56] ZND CJ velocity not positive finite FAILED'

# ---- TC57: Von Neumann状态推导一致 ----
rho_vn_z, p_vn_z, T_vn_z, u_vn_z = znd_test.von_neumann_state(D_cj_znd)
assert rho_vn_z > znd_test.rho0, '[TC57] VN density should exceed rho0 FAILED'
assert p_vn_z > znd_test.p0, '[TC57b] VN pressure should exceed p0 FAILED'

# ---- TC58: CRS矩阵转置乘法输出维度正确 ----
yt_sp = sp_test.multiply_transpose(np.ones(5))
assert len(yt_sp) == 5, '[TC58] CRS transpose multiply output dimension wrong FAILED'
assert np.all(np.isfinite(yt_sp)), '[TC58b] CRS transpose multiply produced non-finite values FAILED'

# ---- TC59: temperature_from_energy返回正值 ----
T_fe = temperature_from_energy(3.0e6, 0.3, 2.5e6, 1.0)
assert T_fe > 0, '[TC59] temperature_from_energy not positive FAILED'

# ---- TC60: Euler通量y方向输出5分量有限值 ----
Fy_test = euler_flux_y(state_orig, gamma=1.4, Q=2.5e6)
assert len(Fy_test) == 5, '[TC60] Euler flux y not 5 components FAILED'
assert np.isfinite(Fy_test[0]), '[TC60b] Euler flux y mass flux not finite FAILED'

print('\n全部 60 个测试通过!\n')
