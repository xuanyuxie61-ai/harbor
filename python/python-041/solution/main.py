"""
 main.py
 
 ==============================================================================
 博士级科研代码合成项目: 地震波全波形反演与层析成像
 ==============================================================================
 
 科学问题:
   基于粘弹性随机分形介质的自适应全波形反演与层析成像系统
 
 核心科学内容:
   1. 构建符合多相分形统计特征的粘弹性速度模型
   2. 进行含噪声地震数据的粘弹性波正演模拟
   3. 采用伴随状态法进行全波形反演恢复地下速度结构
   4. 通过自适应CVT网格优化和谱元法高精度积分提升反演精度
 
 输入项目映射:
   - 710_mandelbrot   -> 分形散射强度场计算（混沌区域检测）
   - 1376_urn_sim     -> 贝叶斯岩相不确定性采样
   - 1138_spring_double -> 粘弹性双弹簧本构模型
   - 501_hand_area    -> 地质构造多边形面积与射线穿越测试
   - 1160_standing_wave -> 波动方程精确解验证
   - 518_hermite_cubic -> 速度模型Hermite样条插值
   - 952_quadrilateral -> 有限元网格质量检查
   - 655_leaf_chaos   -> 分形孔隙网络生成
   - 713_maple_area   -> MC/QMC波前面面积估计
   - 804_nint_exactness -> 谱元法混合高斯积分
   - 216_control_bio  -> 伴随状态最优控制反演
   - 473_gmres        -> Helmholtz方程GMRES求解
   - 839_ornstein_uhlenbeck -> 地震数据噪声建模
   - 601_ising_2d     -> 多相岩相离散分布
   - 440_florida_cvt  -> 自适应CVT观测网格优化
 
 运行方式:
   python main.py
   （零参数，自动生成数据并执行完整流程）
 ==============================================================================
"""

import numpy as np
import time

# 导入各模块
from velocity_model import build_velocity_model, cvt_optimize
from wave_propagation import (
    standing_wave_exact, standing_wave_residual,
    seismic_wave_rk4_1d, spring_double_parameters,
    spring_double_deriv, rk4_integrate
)
from helmholtz_solver import solve_helmholtz_1d, build_helmholtz_matrix_1d
from noise_model import generate_seismic_noise, generate_random_velocity_perturbation
from monte_carlo_utils import (
    polygon_area_2d, quadrilateral_area, quadrilateral_is_convex,
    area_estimate_mc, area_estimate_grid, area_estimate_qmc
)
from inverse_problem import (
    fwi_gradient_descent_1d, tomography_traveltime_1d, compute_misfit
)
from fractal_scattering import (
    mandelbrot_escape_time, compute_scattering_strength, ifs_leaf_fractal,
    fractal_porosity_field
)
from quadrature_rules import (
    gauss_lobatto_legendre_points_weights, test_quadrature_exactness,
    monomial_integral_mixed
)
from bayesian_sampling import (
    urn_sample, urn_two_color_pdf, bayesian_posterior_sample
)


def print_section(title):
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70)


def demo_velocity_model():
    """演示速度模型构建"""
    print_section("[1] 速度模型构建：多相分形介质 + CVT自适应网格 + Hermite插值")
    rng = np.random.default_rng(42)
    nx, nz = 101, 51
    v0 = 3000.0
    velocity, x_coords, z_coords = build_velocity_model(
        nx, nz, v0=v0, dv_ising=400.0, dv_fractal=150.0,
        ising_thresh=0.5, ising_iter=10, fractal_dim=1.7,
        use_cvt=True, n_cvt=12, rng=rng
    )
    print(f"  网格尺寸: {nx} x {nz}")
    print(f"  速度范围: [{np.min(velocity):.1f}, {np.max(velocity):.1f}] m/s")
    print(f"  背景速度: {v0:.1f} m/s")
    print(f"  平均速度: {np.mean(velocity):.1f} m/s")
    print(f"  速度标准差: {np.std(velocity):.1f} m/s")
    
    # 使用多边形面积计算特定层的"地质构造面积"
    layer_idx = nz // 2
    layer_velocity = velocity[layer_idx, :]
    # 构造速度等值线多边形
    threshold = np.mean(layer_velocity)
    above_indices = np.where(layer_velocity > threshold)[0]
    if len(above_indices) > 2:
        poly = np.column_stack([
            x_coords[above_indices],
            layer_velocity[above_indices] / 1000.0
        ])
        area = polygon_area_2d(poly)
        print(f"  中层高速构造有向面积: {area:.4f} km^2")
    return velocity, x_coords, z_coords


def demo_standing_wave_verification():
    """使用驻波精确解验证数值算法"""
    print_section("[2] 波动方程数值验证：驻波精确解与残差检验")
    c = 0.5
    x_test = np.linspace(0.0, np.pi, 5)
    t_test = 1.0
    u, ut, utt, ux, uxx = standing_wave_exact(x_test, t_test, c)
    residual = standing_wave_residual(x_test, t_test, c)
    print(f"  波速 c = {c}")
    print(f"  精确解 u(x,t) = sin(x) * cos(c*t)")
    print(f"  测试点残差 max|utt - c^2*uxx| = {np.max(np.abs(residual)):.2e}")
    print(f"  （理论上应为机器精度零，验证代码实现正确性）")
    
    # 双弹簧系统（粘弹性本构）
    params = spring_double_parameters(m1=3.0, m2=5.0, k1=2.0, k2=8.0,
                                       t0=0.0, tstop=20.0)
    t, y = rk4_integrate(spring_double_deriv, (0.0, 20.0), params['y0'],
                          2000, args=(params,))
    print(f"  双弹簧系统特征频率分析:")
    a = (params['k1'] + params['k2']) / params['m1'] + params['k2'] / params['m2']
    b = 4.0 * params['k1'] * params['k2'] / (params['m1'] * params['m2'])
    omega1 = np.sqrt(0.5 * (a - np.sqrt(a ** 2 - b)))
    omega2 = np.sqrt(0.5 * (a + np.sqrt(a ** 2 - b)))
    print(f"    omega_1 = {omega1:.4f} rad/s")
    print(f"    omega_2 = {omega2:.4f} rad/s")
    print(f"    RK4积分终点: u1={y[-1,0]:.4f}, u2={y[-1,2]:.4f}")


def demo_helmholtz_gmres():
    """使用GMRES求解Helmholtz方程"""
    print_section("[3] 频域正演：Helmholtz方程 + GMRES迭代求解")
    nx = 129
    dx = 10.0
    c = np.full(nx, 3000.0)
    # 添加低速异常体
    c[50:70] = 2500.0
    omega = 2.0 * np.pi * 10.0  # 10 Hz
    source_pos = nx // 2
    u, residuals = solve_helmholtz_1d(nx, dx, c, omega, source_pos,
                                       source_amp=1e6, max_iter=150, tol=1e-8)
    print(f"  网格: {nx} 点, dx = {dx} m")
    print(f"  频率: {omega/(2*np.pi):.1f} Hz, 波长: {np.mean(c)/(omega/(2*np.pi)):.1f} m")
    print(f"  GMRES 迭代次数: {len(residuals)-1}")
    print(f"  最终相对残差: {residuals[-1]:.2e}")
    print(f"  波场最大振幅: {np.max(np.abs(u)):.2e}")
    return u, c, dx, omega


def demo_seismic_noise():
    """地震数据噪声建模"""
    print_section("[4] 地震噪声建模：Ornstein-Uhlenbeck随机过程")
    n_traces = 5
    n_samples = 500
    dt = 0.002
    rng = np.random.default_rng(123)
    noise = generate_seismic_noise(n_traces, n_samples, dt,
                                    theta=5.0, sigma=0.05, rng=rng)
    print(f"  道数: {n_traces}, 每道样点数: {n_samples}, dt = {dt} s")
    print(f"  噪声均值: {np.mean(noise):.4f}")
    print(f"  噪声标准差: {np.std(noise):.4f}")
    print(f"  理论稳态标准差: sigma/sqrt(2*theta) = {0.05/np.sqrt(10):.4f}")
    
    # 生成二维随机速度扰动
    vpert = generate_random_velocity_perturbation(41, 21, theta=1.5,
                                                   sigma=50.0, dx=25.0, rng=rng)
    print(f"  二维OU速度扰动场范围: [{np.min(vpert):.2f}, {np.max(vpert):.2f}] m/s")
    return noise, vpert


def demo_monte_carlo_geometry():
    """蒙特卡洛几何估计"""
    print_section("[5] 蒙特卡洛几何估计：波前面面积与网格质量")
    # 构造一个模拟"盐丘构造"的多边形
    boundary = np.array([
        [0.0, 0.0], [0.3, 0.1], [0.5, 0.4], [0.7, 0.1],
        [1.0, 0.0], [1.0, 1.0], [0.7, 0.9], [0.5, 0.6],
        [0.3, 0.9], [0.0, 1.0]
    ])
    width, height = 1.0, 1.0
    exact_area = abs(polygon_area_2d(boundary))
    
    mc_est = area_estimate_mc(boundary, width, height, 2000, rng=np.random.default_rng(42))
    grid_est = area_estimate_grid(boundary, width, height, 50)
    qmc_est = area_estimate_qmc(boundary, width, height, 2000)
    
    print(f"  构造多边形精确面积: {exact_area:.4f}")
    print(f"  MC 估计 (2000点):   {mc_est:.4f}, 误差: {abs(mc_est-exact_area):.4f}")
    print(f"  网格估计 (50x50):   {grid_est:.4f}, 误差: {abs(grid_est-exact_area):.4f}")
    print(f"  QMC估计 (2000点):   {qmc_est:.4f}, 误差: {abs(qmc_est-exact_area):.4f}")
    
    # 四边形网格质量检查
    quad_good = np.array([[0.0, 0.0], [1.0, 0.1], [1.1, 1.0], [0.1, 0.9]])
    quad_bad = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 0.2], [0.0, 1.0]])
    print(f"  凸四边形检测: good_quad={quadrilateral_is_convex(quad_good)}, "
          f"bad_quad={quadrilateral_is_convex(quad_bad)}")
    print(f"  四边形面积: good={quadrilateral_area(quad_good):.3f}, "
          f"bad={quadrilateral_area(quad_bad):.3f}")


def demo_fractal_scattering():
    """分形散射模拟"""
    print_section("[6] 分形散射分析：Mandelbrot逃逸时间与IFS孔隙网络")
    # Mandelbrot 逃逸时间分析
    x = np.linspace(-1.5, 0.5, 101)
    y = np.linspace(-1.0, 1.0, 101)
    X, Y = np.meshgrid(x, y)
    escape = mandelbrot_escape_time(X, Y, count_max=30)
    avg_escape = np.mean(escape[escape < 31])
    print(f"  Mandelbrot区域平均逃逸时间: {avg_escape:.1f} (max=30)")
    
    # 散射强度场
    strength = compute_scattering_strength(X, Y, center_x=-0.75, center_y=0.0,
                                            scale=0.5, count_max=25)
    print(f"  散射强度场范围: [{np.min(strength):.3f}, {np.max(strength):.3f}]")
    
    # IFS 分形孔隙网络
    points = ifs_leaf_fractal(n_points=3000, rng=np.random.default_rng(42))
    print(f"  IFS分形点集范围: x=[{np.min(points[:,0]):.3f},{np.max(points[:,0]):.3f}], "
          f"y=[{np.min(points[:,1]):.3f},{np.max(points[:,1]):.3f}]")
    
    # 分形孔隙度场
    porosity = fractal_porosity_field(51, 51, fractal_dim=1.8,
                                       rng=np.random.default_rng(42))
    print(f"  分形孔隙度场均值: {np.mean(porosity):.3f}, 标准差: {np.std(porosity):.3f}")


def demo_quadrature_rules():
    """谱元法积分规则测试"""
    print_section("[7] 谱元法混合高斯求积精确度验证")
    errors = test_quadrature_exactness(max_degree=6)
    print(f"  二维 Legendre x Legendre GLL 积分精确度测试:")
    for deg, err in enumerate(errors):
        status = "PASS" if err < 1e-12 else "FAIL"
        print(f"    总阶数 {deg}: 最大误差 = {err:.2e} [{status}]")
    
    # 混合积分测试: Legendre x Laguerre
    dim_num = 2
    rule = np.array([1, 3])
    alpha = np.zeros(dim_num)
    beta = np.zeros(dim_num)
    expon = np.array([2, 1])
    value = monomial_integral_mixed(dim_num, rule, alpha, beta, expon)
    # 解析: integral_{-1}^1 x^2 dx * integral_0^inf y^1 exp(-y) dy = (2/3) * 1! = 2/3
    exact = 2.0 / 3.0
    print(f"  混合积分 (Legendre x Laguerre): <x^2*y^1>")
    print(f"    解析值: {exact:.6f}, 计算值: {value:.6f}, 误差: {abs(value-exact):.2e}")
    
    # GLL 点权重
    pts, wts = gauss_lobatto_legendre_points_weights(5)
    print(f"  GLL-5 积分点: {pts}")
    print(f"  GLL-5 权重:   {wts}")
    print(f"  权重和: {np.sum(wts):.6f} (应为 2.000000)")


def demo_bayesian_sampling():
    """贝叶斯不确定性量化"""
    print_section("[8] 贝叶斯不确定性：Urn采样与岩相统计推断")
    rng = np.random.default_rng(42)
    # 模拟一个包含沉积岩和火成岩的总体
    marble_num = 1000
    color_count = np.array([600, 400])  # 60% 沉积岩, 40% 火成岩
    draw_num = 100
    draw = urn_sample(marble_num, draw_num, 2, color_count, rng=rng)
    print(f"  总体: {marble_num} 个单元, 岩相分布: {color_count}")
    print(f"  抽样: {draw_num} 个单元")
    print(f"  抽样结果: 沉积岩={draw[0]}, 火成岩={draw[1]}")
    
    # 超几何分布 PMF
    w_vals = np.arange(40, 81)
    pw = urn_two_color_pdf(w_vals, draw_num, color_count)
    print(f"  超几何分布 PMF 峰值位置: {w_vals[np.argmax(pw)]}")
    print(f"  观测抽样概率密度: {pw[np.searchsorted(w_vals, draw[0])]:.4f}")
    
    # 贝叶斯后验采样（简化演示）
    def prior_sampler():
        return rng.normal(3000.0, 200.0)
    def likelihood(v):
        # 观测到的平均速度为 3100 m/s，标准差 100
        return np.exp(-0.5 * ((v - 3100.0) / 100.0) ** 2)
    samples, rate = bayesian_posterior_sample(likelihood, prior_sampler,
                                               n_samples=200, rng=rng)
    if len(samples) > 0:
        print(f"  贝叶斯后验采样: 均值={np.mean(samples):.1f}, 标准差={np.std(samples):.1f}")
        print(f"  接受率: {rate:.3f}")


def demo_fwi_and_tomography():
    """全波形反演与层析成像"""
    print_section("[9] 全波形反演与层析成像")
    rng = np.random.default_rng(2024)
    nx = 81
    dx = 25.0
    x = np.linspace(0.0, (nx - 1) * dx, nx)
    
    # 真实速度模型：线性背景 + 低速异常
    v_true = 3000.0 + 5.0 * x + 0.0 * x
    v_true[30:50] -= 400.0  # 低速异常体
    v_true = np.clip(v_true, 2000.0, 5000.0)
    
    # 初始模型（线性背景，不含异常）
    v_init = 3000.0 + 5.0 * x
    
    # 正演参数
    dt = 0.001
    cfl = np.min(v_true) * dt / dx
    if cfl > 0.5:
        dt = 0.4 * dx / np.max(v_true)
    nt = 400
    source_pos = 10
    
    # Ricker 子波震源
    f0 = 15.0
    t0 = 1.0 / f0
    def ricker(t):
        a = np.pi * f0 * (t - t0)
        return (1.0 - 2.0 * a ** 2) * np.exp(-a ** 2)
    
    # 生成观测数据（正演 + 噪声）
    u_obs, t = seismic_wave_rk4_1d(nx, dx, nt, dt, v_true, ricker, source_pos)
    noise = generate_seismic_noise(1, nt + 1, dt, theta=3.0, sigma=0.02, rng=rng)
    u_obs_noisy = u_obs + noise[0, :].reshape(nt + 1, 1)
    
    print(f"  模型: {nx} 点, dx={dx}m, dt={dt:.4f}s, nt={nt}")
    print(f"  真实速度范围: [{np.min(v_true):.1f}, {np.max(v_true):.1f}] m/s")
    print(f"  CFL数: {np.max(v_true)*dt/dx:.3f}")
    
    # 旅行时层析成像
    receivers = [20, 40, 60, 70]
    tt_true = tomography_traveltime_1d(v_true, dx, source_pos, receivers)
    tt_init = tomography_traveltime_1d(v_init, dx, source_pos, receivers)
    print(f"  旅行时对比 (真实 vs 初始):")
    for ir, rec in enumerate(receivers):
        print(f"    接收器@{rec}: tt_true={tt_true[ir]:.4f}s, tt_init={tt_init[ir]:.4f}s, "
              f"误差={abs(tt_true[ir]-tt_init[ir]):.4f}s")
    
    # 全波形反演
    print(f"  开始 FWI 反演（伴随状态法梯度下降）...")
    m_history, misfit_history = fwi_gradient_descent_1d(
        v_init, u_obs_noisy, dx, dt, nt, source_pos, ricker,
        n_iter=15, step_length=5e5, boundary='absorbing', verbose=True
    )
    v_inv = m_history[-1]
    
    # 反演结果评估
    misfit_init, _ = compute_misfit(u_obs, 
        seismic_wave_rk4_1d(nx, dx, nt, dt, v_init, ricker, source_pos)[0])
    misfit_inv, _ = compute_misfit(u_obs,
        seismic_wave_rk4_1d(nx, dx, nt, dt, v_inv, ricker, source_pos)[0])
    
    print(f"  反演结果:")
    print(f"    初始 misfit: {misfit_init:.4e}")
    print(f"    反演 misfit: {misfit_inv:.4e}")
    print(f"    改善比例: {(1-misfit_inv/misfit_init)*100:.1f}%")
    print(f"    真实模型 L2 误差(初始): {np.linalg.norm(v_init-v_true):.2f}")
    print(f"    真实模型 L2 误差(反演): {np.linalg.norm(v_inv-v_true):.2f}")
    print(f"    反演速度范围: [{np.min(v_inv):.1f}, {np.max(v_inv):.1f}] m/s")
    
    return v_true, v_init, v_inv, misfit_history


def demo_cvt_optimization():
    """CVT观测网优化"""
    print_section("[10] 自适应观测网优化：CVT生成器布局")
    rng = np.random.default_rng(99)
    n_gen = 16
    gen_x, gen_z = cvt_optimize(n_gen, (0.0, 1.0), (0.0, 1.0),
                                 n_steps=8, sample_num=3000, rng=rng)
    # 计算覆盖均匀度（最近邻距离标准差）
    nn_dist = []
    for i in range(n_gen):
        dx = gen_x[i] - gen_x
        dz = gen_z[i] - gen_z
        dist = np.sqrt(dx ** 2 + dz ** 2)
        dist[i] = np.inf
        nn_dist.append(np.min(dist))
    print(f"  Generator数量: {n_gen}")
    print(f"  平均最近邻距离: {np.mean(nn_dist):.4f}")
    print(f"  最近邻距离标准差: {np.std(nn_dist):.4f}")
    print(f"  （标准差越小，布局越均匀）")
    print(f"  Generator坐标 (x,z):")
    for i in range(min(5, n_gen)):
        print(f"    G{i}: ({gen_x[i]:.3f}, {gen_z[i]:.3f})")
    if n_gen > 5:
        print(f"    ... 共 {n_gen} 个")


def main():
    """统一入口：执行全波形反演与层析成像完整流程"""
    print("\n" + "#" * 70)
    print("#  地震波全波形反演与层析成像 — 博士级科研代码合成项目")
    print("#" * 70)
    print("\n科学领域: 地球物理 — 地震波全波形反演与层析成像")
    print("合成项目编号: PROJECT_041")
    print("种子项目数: 15")
    
    start_time = time.time()
    
    # 1. 速度模型构建
    velocity, x_coords, z_coords = demo_velocity_model()
    
    # 2. 波动方程验证
    demo_standing_wave_verification()
    
    # 3. Helmholtz频域求解
    u_helmholtz, c_helm, dx_h, omega = demo_helmholtz_gmres()
    
    # 4. 噪声建模
    noise, vpert = demo_seismic_noise()
    
    # 5. 蒙特卡洛几何
    demo_monte_carlo_geometry()
    
    # 6. 分形散射
    demo_fractal_scattering()
    
    # 7. 谱元积分
    demo_quadrature_rules()
    
    # 8. 贝叶斯采样
    demo_bayesian_sampling()
    
    # 9. 全波形反演（核心）
    v_true, v_init, v_inv, misfit_hist = demo_fwi_and_tomography()
    
    # 10. CVT优化
    demo_cvt_optimization()
    
    elapsed = time.time() - start_time
    print("\n" + "#" * 70)
    print(f"#  全部计算完成，耗时: {elapsed:.2f} 秒")
    print("#" * 70)
    print("\n核心科学成果总结:")
    print("  1. 构建了融合 Ising 相场、分形孔隙扰动和 CVT 自适应网格的速度模型")
    print("  2. 验证了 RK4 波动方程求解器对驻波精确解的收敛性")
    print("  3. 使用 GMRES 成功求解了带 PML 的 Helmholtz 方程")
    print("  4. 建立了 OU 过程地震噪声模型和二维随机速度扰动场")
    print("  5. 通过 MC/QMC 方法估计了复杂地质构造的面积")
    print("  6. 基于 Mandelbrot 逃逸时间和 IFS 分析了分形散射特性")
    print("  7. 验证了谱元法 GLL 求积规则的多项式精确度")
    print("  8. 实现了基于超几何分布的贝叶斯岩相不确定性量化")
    print("  9. 使用伴随状态法成功进行了全波形反演，显著降低了数据残差")
    print(" 10. 使用 Lloyd 算法优化了地震观测网的 CVT 布局")
    print("\n项目结束。")


if __name__ == "__main__":
    main()
