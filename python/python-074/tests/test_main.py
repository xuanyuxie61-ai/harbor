r"""
main.py
=======
圆柱涡激振动（VIV）流固耦合数值模拟的统一入口程序。

运行方式
--------
    python main.py

无需任何命令行参数。程序内部设定典型物理参数，执行完整的：
1. 流场初始化与网格生成
2. 涡量-流函数 NS 方程时间推进
3. 结构动力学响应计算
4. 流固耦合信息传递
5. Jacobi 谱精度后处理
6. CVT 自适应网格质量评估
7. 耦合稳定性分析
8. 误差估计与收敛检验
9. 涡脱落统计概率建模
10. 复杂几何解析与传感器优化

科学问题描述
------------
研究雷诺数 Re = 100 的均匀来流绕过弹性支撑圆柱时诱发的
涡激振动现象。采用涡量-流函数形式的二维不可压 Navier-Stokes 方程
耦合两自由度弹簧-质量-阻尼结构模型，分析其动力响应、升阻力特性、
稳定性边界及统计行为。

物理参数（国际单位制）
----------------------
- 来流速度 U_\infty = 1.0 m/s
- 圆柱直径 D = 0.1 m
- 运动粘性系数 \nu = 0.001 m^2/s  (Re = U*D/\nu = 100)
- 结构质量 m = 1.0 kg
- 刚度 k = 39.478 N/m  (固有频率 f_n = 1.0 Hz)
- 阻尼比 \zeta = 0.01
- 质量比 m^* = 2.0
- 计算域：L_x = 5.0 m, L_y = 2.5 m

核心公式汇总
------------
1. 涡量方程：
   \partial_t \omega + u \partial_x \omega + v \partial_y \omega
   = \nu (\partial_{xx} + \partial_{yy}) \omega

2. 流函数泊松方程：
   \nabla^2 \psi = -\omega

3. 结构运动方程：
   M \ddot{X} + C \dot{X} + K X + K_{nl} X^3 = F_{fluid}

4. 升阻力系数：
   C_L = -\frac{2}{U^2 D} \oint \omega_{wall} \cos\theta \, R d\theta
   C_D =  \frac{2}{U^2 D} \oint \omega_{wall} \sin\theta \, R d\theta

5. 对数范数稳定性判据：
   \mu_p(J) < 0 \Rightarrow 系统渐近稳定

6. 不完全 Beta 相位分布：
   F(\theta) = I_{\theta/(2\pi)}(p, q)
r"""

import numpy as np
import time

# 导入各科学计算模块
from vorticity_transport import VorticityTransportSolver
from poisson_fem import PoissonFEM2D
from structure_dynamics import StructureDynamics
from jacobi_spectral import (
    gauss_jacobi_rule,
    boundary_layer_map,
    integrate_boundary_layer,
    test_jacobi_spectral,
)
from cvt_adaptation import CVT1DPeriodic, CVT2DReflect, wake_density_function
from stability_coupling import analyze_stability
from error_estimator import (
    compute_solution_quality_metrics,
    l1_norm_discrete,
    l2_norm_discrete,
    richardson_error_estimate,
)
from probability_model import (
    estimate_vortex_shedding_frequency,
    fit_beta_parameters,
    phase_pdf,
    phase_cdf,
    incomplete_beta,
)
from geometry_utils import (
    parse_stl_ascii,
    compute_face_normals,
    stla_check,
    sensor_placement_optimization,
    vortex_caustic_map,
    generate_simple_cylinder_stl_lines,
)


def main():
    print("=" * 72)
    print("  圆柱涡激振动流固耦合数值模拟系统")
    print("  Vortex-Induced Vibration Fluid-Structure Interaction Solver")
    print("=" * 72)
    t_start = time.time()

    # =====================================================================
    # 1. 物理与数值参数设定
    # =====================================================================
    U_inf = 1.0
    D_cyl = 0.2
    nu = 0.002
    Re = U_inf * D_cyl / nu

    Lx = 4.0
    Ly = 2.0
    nx = 81
    ny = 41
    dt = 0.004
    n_steps = 500

    mass = 1.0
    stiffness = 39.47841760435743  # (2*pi)^2, f_n = 1.0 Hz
    damping = 2.0 * 0.01 * np.sqrt(mass * stiffness)
    mass_ratio = 2.0
    k_nl = 100.0  # 非线性硬化刚度

    print(f"\n[参数设定]")
    print(f"  雷诺数 Re = {Re:.1f}")
    print(f"  计算域  = {Lx:.1f} x {Ly:.1f} m")
    print(f"  网格    = {nx} x {ny}")
    print(f"  时间步长 dt = {dt:.4f} s, 总步数 = {n_steps}")
    print(f"  结构固有频率 f_n = {np.sqrt(stiffness/mass)/(2*np.pi):.3f} Hz")

    # =====================================================================
    # 2. 初始化流场求解器
    # =====================================================================
    cylinder_params = {
        'cx': Lx * 0.25,
        'cy': Ly * 0.5,
        'r': D_cyl / 2.0,
    }

    fluid_solver = VorticityTransportSolver(
        nx=nx, ny=ny, lx=Lx, ly=Ly,
        nu=nu, u_inf=U_inf, dt=dt,
        cylinder_params=cylinder_params
    )

    poisson_solver = PoissonFEM2D(
        nx=nx, ny=ny, lx=Lx, ly=Ly,
        solid_mask=fluid_solver.solid_mask
    )

    # =====================================================================
    # 3. 初始化结构动力学求解器
    # =====================================================================
    structure = StructureDynamics(
        mass=mass,
        damping=damping,
        stiffness=stiffness,
        mass_ratio=mass_ratio,
        u_inf=U_inf,
        diameter=D_cyl,
        fn=1.0,
        nonlinear_params={'k_nl': k_nl},
        time_integrator='cn_rk2',
    )
    structure.set_time_step(dt)

    # =====================================================================
    # 4. 初始化存储数组与初始扰动
    # =====================================================================
    time_history = np.zeros(n_steps)
    cl_history = np.zeros(n_steps)
    cd_history = np.zeros(n_steps)
    disp_history = np.zeros((n_steps, 2))
    vel_history = np.zeros((n_steps, 2))

    # 添加初始扰动以触发涡脱落（打破对称性）
    # 在圆柱表面附近注入反对称涡量对（上侧正、下侧负）
    for j in range(1, ny - 1):
        for i in range(1, nx - 1):
            if not fluid_solver.solid_mask[j, i]:
                dx_ = fluid_solver.X[j, i] - cylinder_params['cx']
                dy_ = fluid_solver.Y[j, i] - cylinder_params['cy']
                dist = np.sqrt(dx_ ** 2 + dy_ ** 2)
                if dist > cylinder_params['r'] and dist < 2.0 * cylinder_params['r']:
                    sign = 1.0 if dy_ > 0 else -1.0
                    fluid_solver.omega[j, i] = (
                        sign * 3.0 * U_inf / D_cyl
                        * np.exp(-(dist - cylinder_params['r']) ** 2
                                  / (0.4 * cylinder_params['r']) ** 2)
                    )
    fluid_solver.apply_boundary_conditions()

    # =====================================================================
    # 5. 主时间步进循环
    # =====================================================================
    print("\n[时间推进开始]")
    for step in range(n_steps):
        t = step * dt
        time_history[step] = t

        # --- 5a. 流体步：推进涡量方程 ---
        fluid_solver.time_step(step)

        # --- 5b. 求解流函数泊松方程 ---
        dirichlet_mask = np.zeros((ny, nx), dtype=bool)
        dirichlet_values = np.zeros((ny, nx))

        # 入口：\psi = U_inf * y
        dirichlet_mask[:, 0] = True
        dirichlet_values[:, 0] = U_inf * fluid_solver.Y[:, 0]
        # 上下壁面
        dirichlet_mask[0, :] = True
        dirichlet_values[0, :] = 0.0
        dirichlet_mask[ny - 1, :] = True
        dirichlet_values[ny - 1, :] = U_inf * Ly
        # 圆柱表面：\psi = 0（无滑移）
        dirichlet_mask[fluid_solver.solid_mask] = True
        dirichlet_values[fluid_solver.solid_mask] = 0.0

        psi_new = poisson_solver.solve(
            rhs_field=fluid_solver.omega,
            dirichlet_mask=dirichlet_mask,
            dirichlet_values=dirichlet_values,
        )
        fluid_solver.psi = psi_new

        # 基于更新后的 psi 重新计算壁面涡量（Thom 公式）
        fluid_solver.apply_boundary_conditions()

        # --- 5c. 重构速度场 ---
        fluid_solver.compute_velocity_from_psi()

        # --- 5d. 计算流体作用力 ---
        c_d, c_l = fluid_solver.compute_force_coefficients()
        cl_history[step] = c_l
        cd_history[step] = c_d

        # 升阻力转换为物理力（归一化后反推）
        rho_f = 1.0
        q_dyn = 0.5 * rho_f * U_inf ** 2 * D_cyl
        f_drag = c_d * q_dyn
        f_lift = c_l * q_dyn
        force_ext = np.array([f_drag, f_lift])

        # --- 5e. 结构步：推进结构运动 ---
        structure.step(force_ext)
        disp_history[step] = structure.get_displacement()
        vel_history[step] = structure.get_velocity()

        # 每 100 步输出状态
        if (step + 1) % 100 == 0:
            print(f"  Step {step+1:4d}/{n_steps}, t={t:.3f}s, "
                  f"C_L={c_l:+.4f}, C_D={c_d:+.4f}, "
                  f"Y_disp={disp_history[step,1]:+.4e}")

    print("[时间推进完成]")

    # =====================================================================
    # 6. Jacobi 谱方法后处理：边界层涡量积分
    # =====================================================================
    print("\n[Jacobi 谱后处理]")
    n_gauss = 16
    alpha_j = 0.5
    beta_j = 0.0
    nodes_gj, weights_gj = gauss_jacobi_rule(n_gauss, alpha_j, beta_j)

    # 在尾流区 x = cx + 2D 处提取涡量剖面
    x_wake = cylinder_params['cx'] + 2.0 * D_cyl
    wake_profile, y_profile = fluid_solver.get_wake_profile(x_wake)
    valid = ~np.isnan(wake_profile)

    if np.sum(valid) > n_gauss:
        # 用 Jacobi-Gauss 节点插值到尾流剖面
        y_valid = y_profile[valid]
        w_valid = wake_profile[valid]
        # 线性插值到 Gauss 节点（映射到 y 范围）
        y_min, y_max = np.min(y_valid), np.max(y_valid)
        y_mapped = 0.5 * (y_max - y_min) * (nodes_gj + 1.0) + y_min
        w_interp = np.interp(y_mapped, y_valid, w_valid, left=0.0, right=0.0)
        wake_enstrophy = np.sum(weights_gj * w_interp ** 2) * 0.5 * (y_max - y_min)
        print(f"  尾流区涡量拟能 (Jacobi-Gauss {n_gauss}点) = {wake_enstrophy:.6e}")
    else:
        wake_enstrophy = 0.0
        print(f"  尾流区有效数据点不足，跳过谱积分。")

    # Jacobi 自检
    test_jacobi_spectral()

    # =====================================================================
    # 7. CVT 自适应网格质量评估
    # =====================================================================
    print("\n[CVT 自适应网格评估]")
    # 1D 周期 CVT：展向节点分布
    cvt1d = CVT1DPeriodic(num_generators=20, domain_length=Ly, it_max=30)
    gen_1d = cvt1d.iterate()
    print(f"  1D 周期 CVT 收敛步数 = {len(cvt1d.energy_history)}, "
          f"最终能量 = {cvt1d.energy_history[-1]:.6e}")

    # 2D 反射 CVT：尾流区节点分布
    density_func = lambda x, y: wake_density_function(
        np.array([[x]]), np.array([[y]]),
        cylinder_params['cx'], cylinder_params['cy'], cylinder_params['r'],
        base_density=1.0, peak_density=8.0
    )[0, 0]
    cvt2d = CVT2DReflect(
        num_generators=50,
        bounds=((cylinder_params['cx'], Lx), (0.0, Ly)),
        density_func=density_func,
        it_max=20,
        sample_num=5000,
    )
    gen_2d = cvt2d.iterate()
    print(f"  2D 反射 CVT 收敛步数 = {len(cvt2d.energy_history)}, "
          f"最终能量 = {cvt2d.energy_history[-1]:.6e}")

    # =====================================================================
    # 8. 耦合稳定性分析
    # =====================================================================
    print("\n[耦合稳定性分析]")
    stab_report = analyze_stability(
        nx=20, nu=nu, mass=mass, damping=damping,
        stiffness=stiffness, rho_f=rho_f, D_cyl=D_cyl,
    )
    print(f"  对数范数 \mu_1(J)  = {stab_report['mu_1']:+.4f}")
    print(f"  对数范数 \mu_2(J)  = {stab_report['mu_2']:+.4f}")
    print(f"  对数范数 \mu_\infty(J) = {stab_report['mu_inf']:+.4f}")
    print(f"  谱横坐标           = {stab_report['spectral_abscissa']:+.4f}")
    print(f"  Gershgorin 界      = [{stab_report['gershgorin_min']:+.4f}, "
          f"{stab_report['gershgorin_max']:+.4f}]")
    print(f"  谱稳定判据         = {'稳定' if stab_report['stable_spectral'] else '可能不稳定'}")

    # =====================================================================
    # 9. 误差估计与解质量评估
    # =====================================================================
    print("\n[误差估计与解质量]")
    metrics = compute_solution_quality_metrics(
        fluid_solver.omega, fluid_solver.psi,
        fluid_solver.u, fluid_solver.v,
        fluid_solver.dx, fluid_solver.dy,
        solid_mask=fluid_solver.solid_mask,
    )
    print(f"  涡量 L2 范数       = {metrics['omega_l2']:.6e}")
    print(f"  流函数 L2 范数     = {metrics['psi_l2']:.6e}")
    print(f"  动能估计           = {metrics['kinetic_energy']:.6e}")
    print(f"  涡量拟能           = {metrics['enstrophy']:.6e}")
    print(f"  散度 RMS           = {metrics['divergence_rms']:.6e}")

    # Richardson 误差估计（用粗网格快照对比）
    # 构造粗网格（2:1）上的参考解作为代理
    omega_coarse = fluid_solver.omega[::2, ::2].copy()
    # 映射回细网格（最近邻）
    omega_coarse_fine = np.zeros_like(fluid_solver.omega)
    for j in range(omega_coarse.shape[0]):
        for i in range(omega_coarse.shape[1]):
            omega_coarse_fine[j * 2, i * 2] = omega_coarse[j, i]
    # 简单填充
    from scipy.ndimage import zoom
    try:
        omega_coarse_fine = zoom(omega_coarse, (fluid_solver.ny / omega_coarse.shape[0],
                                                  fluid_solver.nx / omega_coarse.shape[1]),
                                 order=1)
    except Exception:
        omega_coarse_fine = fluid_solver.omega.copy()  # fallback

    rich_err = richardson_error_estimate(
        fluid_solver.omega, omega_coarse_fine, ratio=2.0, order=2.0
    )
    err_l1 = l1_norm_discrete(rich_err, fluid_solver.dx, fluid_solver.dy)
    err_l2 = l2_norm_discrete(rich_err, fluid_solver.dx, fluid_solver.dy)
    print(f"  Richardson L1 误差 = {err_l1:.6e}")
    print(f"  Richardson L2 误差 = {err_l2:.6e}")

    # =====================================================================
    # 10. 涡脱落统计概率建模
    # =====================================================================
    print("\n[涡脱落统计概率建模]")
    # 截取后半段稳态数据
    start_idx = n_steps // 2
    cl_steady = cl_history[start_idx:]

    f_est, st_est = estimate_vortex_shedding_frequency(cl_steady, dt)
    print(f"  估计涡脱落频率     = {f_est:.4f} Hz")
    print(f"  Strouhal 数        = {st_est:.4f} (归一化)")

    # 升力相位角（用 Hilbert 变换近似）
    analytic_signal = np.fft.ifft(
        np.fft.fft(cl_steady) * (np.arange(len(cl_steady)) >= 0) * 2.0
    )
    phase = np.angle(analytic_signal)
    phase = np.mod(phase, 2.0 * np.pi)

    p_beta, q_beta = fit_beta_parameters(phase)
    print(f"  Beta 分布参数      = p={p_beta:.3f}, q={q_beta:.3f}")

    # 示例 CDF 计算
    theta_test = np.pi
    cdf_val = phase_cdf(theta_test, p_beta, q_beta)
    pdf_val = phase_pdf(theta_test, p_beta, q_beta)
    print(f"  相位 CDF(\pi)       = {cdf_val:.4f}")
    print(f"  相位 PDF(\pi)       = {pdf_val:.4f}")

    # 不完全 Beta 自检
    ib_test = incomplete_beta(0.5, 2.0, 3.0)
    print(f"  不完全 Beta 自检   = I_0.5(2,3) = {ib_test:.6f} (参考 ~0.6875)")

    # =====================================================================
    # 11. 几何处理与传感器优化
    # =====================================================================
    print("\n[几何处理与传感器优化]")
    # STL 解析测试
    stl_lines = generate_simple_cylinder_stl_lines(diameter=D_cyl, num_facets=24)
    verts, norms = parse_stl_ascii(stl_lines)
    if verts.size > 0:
        check_code = stla_check(verts, norms)
        print(f"  STL 解析面片数     = {len(verts)}")
        print(f"  STL 一致性检查     = {'通过' if check_code == 0 else f'错误码 {check_code}'}")
        recomputed_norms = compute_face_normals(verts)
        norm_diff = np.max(np.linalg.norm(recomputed_norms - norms, axis=1))
        print(f"  法向量重构最大偏差 = {norm_diff:.3e}")

    # 传感器优化布置
    candidates = [(j, i) for j in range(1, ny - 1) for i in range(1, nx - 1)
                  if not fluid_solver.solid_mask[j, i]]
    selected, achieved = sensor_placement_optimization(
        candidates, fluid_solver.omega, budget_num=15, influence_radius=2.0 * D_cyl
    )
    print(f"  候选测点数         = {len(candidates)}")
    print(f"  选中测点数         = {np.sum(selected)}")
    print(f"  总信息权重         = {achieved:.4f}")

    # 涡核 caustic 拓扑分析
    caustic_conn, wnum = vortex_caustic_map(
        n_points=100, m_ratio=7,
        cylinder_center=(cylinder_params['cx'], cylinder_params['cy']),
        radius_scale=D_cyl * 3.0,
    )
    print(f"  Caustic 拓扑环绕数 = {wnum}")

    # =====================================================================
    # 12. 结果汇总
    # =====================================================================
    t_elapsed = time.time() - t_start
    print("\n" + "=" * 72)
    print("  模拟完成")
    print(f"  总耗时 = {t_elapsed:.3f} s")
    print(f"  最终横向振幅 = {np.max(np.abs(disp_history[:,1])):.4e} m")
    print(f"  最终流向振幅 = {np.max(np.abs(disp_history[:,0])):.4e} m")
    print(f"  升力系数 RMS = {np.sqrt(np.mean(cl_steady**2)):.4f}")
    print("=" * 72)

    return {
        'time_history': time_history,
        'cl_history': cl_history,
        'cd_history': cd_history,
        'disp_history': disp_history,
        'vel_history': vel_history,
        'stab_report': stab_report,
        'metrics': metrics,
    }


if __name__ == "__main__":
    main()

# ================================================================
# 测试用例（33个，assert模式，涉及随机值均使用固定种子）
# ================================================================
# ---- TC01: Jacobi多项式 P0 恒为1 ----
from jacobi_spectral import jacobi_polynomial
x_test = np.array([0.0, 0.5, -0.5, 0.9])
P = jacobi_polynomial(x_test, 0, 0.5, 0.5)
assert np.allclose(P[:, 0], 1.0), '[TC01] Jacobi多项式 P0 恒为1 FAILED'

# ---- TC02: Gauss-Jacobi 节点在 [-1,1] 内且权重为正 ----
nodes, weights = gauss_jacobi_rule(8, 0.5, -0.3)
assert np.all(nodes >= -1.0) and np.all(nodes <= 1.0), '[TC02] Gauss-Jacobi 节点越界 FAILED'
assert np.all(weights > 0), '[TC02] Gauss-Jacobi 权重非正 FAILED'

# ---- TC03: Gauss-Jacobi 权重和等于零阶矩 ----
from scipy.special import gamma as scipy_gamma
alpha, beta = 0.5, -0.3
nodes, weights = gauss_jacobi_rule(8, alpha, beta)
zemu = (2.0 ** (alpha + beta + 1.0)) * scipy_gamma(alpha + 1.0) * scipy_gamma(beta + 1.0) / scipy_gamma(alpha + beta + 2.0)
assert np.abs(np.sum(weights) - zemu) < 1e-12, '[TC03] Gauss-Jacobi 权重和 FAILED'

# ---- TC04: imtqlx 对角化单位矩阵 ----
from jacobi_spectral import imtqlx
d = np.array([1.0, 1.0, 1.0])
e = np.array([0.0, 0.0, 0.0])
z = np.array([1.0, 0.0, 0.0])
d_out, z_out = imtqlx(3, d, e, z)
assert np.allclose(d_out, 1.0), '[TC04] imtqlx 单位矩阵特征值 FAILED'
assert np.allclose(z_out, z), '[TC04] imtqlx 单位矩阵变换向量 FAILED'

# ---- TC05: boundary_layer_map 映射正确性 ----
eta = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
delta = 1.0
xi, dy_dxi = boundary_layer_map(eta, delta)
expected_xi = 2.0 * np.sqrt(eta / delta) - 1.0
assert np.allclose(xi, expected_xi), '[TC05] boundary_layer_map xi FAILED'

# ---- TC06: 谱微分矩阵对 f(x)=x 导数为1 ----
from jacobi_spectral import spectral_differentiation_matrix
x_nodes = np.array([-1.0, 0.0, 1.0])
D = spectral_differentiation_matrix(x_nodes)
df = D @ x_nodes
assert np.allclose(df, 1.0), '[TC06] 谱微分矩阵线性导数 FAILED'

# ---- TC07: L1范数一维常数场 ----
field = np.ones(10)
dx = 0.1
l1 = l1_norm_discrete(field, dx)
assert np.abs(l1 - 1.0) < 1e-12, '[TC07] L1范数一维常数场 FAILED'

# ---- TC08: L2范数二维常数场 ----
field = np.ones((5, 5))
dx, dy = 0.1, 0.2
l2 = l2_norm_discrete(field, dx, dy)
expected = np.sqrt(5 * 5 * 0.1 * 0.2)
assert np.abs(l2 - expected) < 1e-12, '[TC08] L2范数二维常数场 FAILED'

# ---- TC09: Richardson误差估计已知值 ----
u_fine = np.ones((4, 4)) * 2.0
u_coarse = np.ones((4, 4)) * 1.0
err = richardson_error_estimate(u_fine, u_coarse, ratio=2.0, order=2.0)
assert np.allclose(err, 1.0 / 3.0), '[TC09] Richardson误差估计已知值 FAILED'

# ---- TC10: 解质量指标输出结构与散度为零 ----
nx, ny = 11, 6
omega = np.zeros((ny, nx))
psi = np.zeros((ny, nx))
u = np.ones((ny, nx))
v = np.zeros((ny, nx))
dx, dy = 0.1, 0.1
metrics = compute_solution_quality_metrics(omega, psi, u, v, dx, dy)
assert 'omega_l2' in metrics, '[TC10] 解质量指标缺少omega_l2 FAILED'
assert 'kinetic_energy' in metrics, '[TC10] 解质量指标缺少kinetic_energy FAILED'
assert metrics['divergence_rms'] == 0.0, '[TC10] 解质量指标散度非零 FAILED'

# ---- TC11: STL解析面片数正确 ----
stl_lines = generate_simple_cylinder_stl_lines(diameter=1.0, num_facets=12)
verts, norms = parse_stl_ascii(stl_lines)
assert verts.shape[0] == 12, '[TC11] STL解析面片数 FAILED'
assert norms.shape[0] == 12, '[TC11] STL解析法向量数 FAILED'

# ---- TC12: STL一致性检查通过 ----
check = stla_check(verts, norms)
assert check == 0, '[TC12] STL一致性检查 FAILED'

# ---- TC13: 法向量重构精度 ----
recomputed = compute_face_normals(verts)
diff = np.max(np.linalg.norm(recomputed - norms, axis=1))
assert diff < 1e-10, '[TC13] 法向量重构精度 FAILED'

# ---- TC14: 子集和swap启发式不超限 ----
from geometry_utils import subset_sum_swap
weights = np.array([5.0, 3.0, 2.0, 1.0])
budget = 5.0
selected, achieved = subset_sum_swap(weights, budget)
assert achieved <= budget + 1e-12, '[TC14] 子集和swap超预算 FAILED'
assert np.sum(selected) >= 1, '[TC14] 子集和swap未选中 FAILED'

# ---- TC15: caustic拓扑连接数与环绕数 ----
conn, wnum = vortex_caustic_map(n_points=100, m_ratio=7, cylinder_center=(0.0, 0.0), radius_scale=1.0)
from math import gcd
expected_wnum = 100 // gcd(100, 7)
assert wnum == expected_wnum, '[TC15] caustic环绕数 FAILED'
assert len(conn) == 100, '[TC15] caustic连接数 FAILED'

# ---- TC16: 等值线提取返回线段列表 ----
from geometry_utils import extract_iso_line_segments
field = np.array([[0.0, 1.0], [1.0, 0.0]])
x_c = np.array([0.0, 1.0])
y_c = np.array([0.0, 1.0])
segs, adj = extract_iso_line_segments(field, 0.5, x_c, y_c)
assert isinstance(segs, list), '[TC16] 等值线提取返回值类型 FAILED'
assert isinstance(adj, dict), '[TC16] 等值线提取邻接表类型 FAILED'

# ---- TC17: alogam 与 scipy gammaln 精度 ----
from probability_model import alogam
from scipy.special import gammaln
for x in [1.0, 2.0, 5.0, 10.0]:
    val, flag = alogam(x)
    assert flag == 0, '[TC17] alogam 错误标志 FAILED'
    assert np.abs(val - gammaln(x)) < 1e-8, '[TC17] alogam 精度 FAILED'

# ---- TC18: log_beta 与 scipy beta 精度 ----
from probability_model import log_beta
from scipy.special import beta as sp_beta
p, q = 2.0, 3.0
lb = log_beta(p, q)
assert np.abs(lb - np.log(sp_beta(p, q))) < 1e-10, '[TC18] log_beta 精度 FAILED'

# ---- TC19: 不完全Beta边界值 ----
assert abs(incomplete_beta(0.0, 2.0, 3.0) - 0.0) < 1e-14, '[TC19] 不完全Beta x=0 FAILED'
assert abs(incomplete_beta(1.0, 2.0, 3.0) - 1.0) < 1e-14, '[TC19] 不完全Beta x=1 FAILED'

# ---- TC20: 不完全Beta已知值 I_0.5(2,3) ≈ 0.6875 ----
ib = incomplete_beta(0.5, 2.0, 3.0)
assert abs(ib - 0.6875) < 1e-4, '[TC20] 不完全Beta已知值 FAILED'

# ---- TC21: 相位PDF积分为1 ----
p, q = 2.0, 2.0
n_theta = 1000
thetas = np.linspace(0.001, 2*np.pi - 0.001, n_theta)
pdf_vals = np.array([phase_pdf(t, p, q) for t in thetas])
integral = np.trapezoid(pdf_vals, thetas)
assert abs(integral - 1.0) < 0.05, '[TC21] 相位PDF积分 FAILED'

# ---- TC22: 对数范数对角矩阵 ----
from stability_coupling import log_norm
A = np.diag([-1.0, -2.0, -3.0])
mu_1 = log_norm(A, 1)
mu_2 = log_norm(A, 2)
mu_inf = log_norm(A, np.inf)
assert abs(mu_1 - (-1.0)) < 1e-12, '[TC22] 对数范数 mu_1 FAILED'
assert abs(mu_2 - (-1.0)) < 1e-12, '[TC22] 对数范数 mu_2 FAILED'
assert abs(mu_inf - (-1.0)) < 1e-12, '[TC22] 对数范数 mu_inf FAILED'

# ---- TC23: Gershgorin界包围真实特征值 ----
from stability_coupling import gershgorin_bounds
A = np.array([[2.0, 1.0], [1.0, 2.0]])
lam_min, lam_max = gershgorin_bounds(A)
eigvals = np.linalg.eigvals(A)
assert lam_min <= np.min(eigvals), '[TC23] Gershgorin下界 FAILED'
assert lam_max >= np.max(eigvals), '[TC23] Gershgorin上界 FAILED'

# ---- TC24: 稳定性分析报告结构完整 ----
report = analyze_stability(nx=10, nu=0.01, mass=1.0, damping=0.1, stiffness=10.0)
assert 'mu_1' in report, '[TC24] 稳定性分析缺少mu_1 FAILED'
assert 'mu_2' in report, '[TC24] 稳定性分析缺少mu_2 FAILED'
assert 'mu_inf' in report, '[TC24] 稳定性分析缺少mu_inf FAILED'
assert 'spectral_abscissa' in report, '[TC24] 稳定性分析缺少spectral_abscissa FAILED'
assert report['stable_spectral'] in (True, False), '[TC24] 稳定性分析stable_spectral类型 FAILED'

# ---- TC25: 结构动力学初始状态为零 ----
sd = StructureDynamics(mass=1.0, damping=0.1, stiffness=10.0, mass_ratio=2.0, u_inf=1.0, diameter=0.1, fn=1.0)
sd.set_time_step(0.001)
disp0 = sd.get_displacement()
vel0 = sd.get_velocity()
assert np.allclose(disp0, 0.0), '[TC25] 结构动力学初始位移 FAILED'
assert np.allclose(vel0, 0.0), '[TC25] 结构动力学初始速度 FAILED'

# ---- TC26: 结构动力学CN-RK2步进后状态有限 ----
sd.step(np.array([0.0, 0.0]))
disp1 = sd.get_displacement()
vel1 = sd.get_velocity()
assert np.all(np.isfinite(disp1)), '[TC26] 结构动力学步进位移有限 FAILED'
assert np.all(np.isfinite(vel1)), '[TC26] 结构动力学步进速度有限 FAILED'

# ---- TC27: 涡量输运求解器初始化形状正确 ----
solver = VorticityTransportSolver(nx=21, ny=11, lx=2.0, ly=1.0, nu=0.01, u_inf=1.0, dt=0.001)
assert solver.omega.shape == (11, 21), '[TC27] 涡量输运求解器初始化形状 FAILED'
assert np.sum(solver.solid_mask) > 0, '[TC27] 涡量输运求解器固体掩码 FAILED'

# ---- TC28: 尾流密度函数返回值在合理范围 ----
X = np.array([[1.0, 2.0], [3.0, 4.0]])
Y = np.array([[0.5, 0.5], [0.5, 0.5]])
rho = wake_density_function(X, Y, cylinder_x=1.0, cylinder_y=0.5, r_cyl=0.1, base_density=1.0, peak_density=10.0)
assert np.all(rho >= 1.0), '[TC28] 尾流密度低于基值 FAILED'
assert np.all(rho <= 30.0), '[TC28] 尾流密度超出上限 FAILED'

# ---- TC29: 一维周期CVT收敛且生成元数量正确 ----
np.random.seed(42)
cvt1d = CVT1DPeriodic(num_generators=10, domain_length=1.0, it_max=50)
gen = cvt1d.iterate()
assert len(gen) == 10, '[TC29] 一维CVT生成元数量 FAILED'
assert np.all(gen >= 0) and np.all(gen < 1.0), '[TC29] 一维CVT生成元范围 FAILED'
assert len(cvt1d.energy_history) > 0, '[TC29] 一维CVT能量历史 FAILED'

# ---- TC30: 涡脱落频率估计正弦信号 ----
np.random.seed(42)
dt = 0.01
t = np.arange(0, 10, dt)
f_true = 1.5
cl = np.sin(2.0 * np.pi * f_true * t)
f_est, st_est = estimate_vortex_shedding_frequency(cl, dt)
assert abs(f_est - f_true) < 0.1, '[TC30] 涡脱落频率估计 FAILED'

# ---- TC31: 泊松FEM求解形状与有限性 ----
fem = PoissonFEM2D(nx=11, ny=6, lx=1.0, ly=0.5)
rhs = np.zeros((6, 11))
mask = np.zeros((6, 11), dtype=bool)
vals = np.zeros((6, 11))
mask[:, 0] = True
vals[:, 0] = 0.0
mask[:, -1] = True
vals[:, -1] = 0.0
mask[0, :] = True
vals[0, :] = 0.0
mask[-1, :] = True
vals[-1, :] = 0.0
psi = fem.solve(rhs, mask, vals)
assert psi.shape == (6, 11), '[TC31] 泊松FEM求解形状 FAILED'
assert np.all(np.isfinite(psi)), '[TC31] 泊松FEM求解含非有限值 FAILED'

# ---- TC32: Beta参数拟合对称分布 ----
np.random.seed(42)
x_samples = np.random.beta(2.0, 2.0, size=20000)
phase_samples = x_samples * 2.0 * np.pi
p_est, q_est = fit_beta_parameters(phase_samples)
assert abs(p_est - 2.0) < 0.5, '[TC32] Beta参数拟合 p 估计 FAILED'
assert abs(q_est - 2.0) < 0.5, '[TC32] Beta参数拟合 q 估计 FAILED'

# ---- TC33: 传感器优化布置输出类型 ----
np.random.seed(42)
field = np.random.rand(6, 11)
candidates = [(j, i) for j in range(1, 5) for i in range(1, 10)]
selected, achieved = sensor_placement_optimization(candidates, field, budget_num=5, influence_radius=0.5)
assert isinstance(selected, np.ndarray), '[TC33] 传感器优化输出类型 FAILED'
assert len(selected) == len(candidates), '[TC33] 传感器优化选中长度 FAILED'
assert achieved >= 0, '[TC33] 传感器优化权重非负 FAILED'

print('\n全部 33 个测试通过!\n')
