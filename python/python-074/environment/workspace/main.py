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

        # TODO: Hole 2 — 请将升阻力系数转换为物理力并推进结构运动
        # 要求：
        #   1. 由来流速度 U_inf、圆柱直径 D_cyl、流体密度 rho_f=1.0 计算动压 q_dyn
        #   2. 将系数 c_d, c_l 转换为物理力 f_drag, f_lift
        #   3. 组装为二维力向量 force_ext = [f_drag, f_lift]
        #   4. 调用 structure.step(force_ext) 推进结构动力学
        #   5. 记录当前步位移和速度到 disp_history, vel_history
        # 注意：此处的力传递格式必须与 structure_dynamics.py 中 step_cn_rk2 的期望一致。
        raise NotImplementedError("Hole 2: 升阻力到物理力的转换与结构步调用尚未实现")

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
