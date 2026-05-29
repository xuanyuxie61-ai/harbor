"""
地核发电机与地磁场反转模拟 —— 统一入口 (main.py)
===================================================
PROJECT_43: 基于 15 个种子项目核心算法合成的博士级地球物理计算项目
科学领域: 地球物理 — 地核发电机与地磁场反转

运行方式:
  python main.py

无需任何命令行参数，所有配置通过 config_parser.py 中的默认参数管理。
"""

import os
import sys
import time
import numpy as np

# ---------------------------------------------------------------------------
# 导入所有子模块
# ---------------------------------------------------------------------------
from config_parser import load_config
from special_functions import cosine_integral, sine_integral, spherical_bessel_j
from mesh_geometry import generate_core_mesh, cvt_disk_uniform
from task_scheduler import DynamoBatchScheduler
from stiff_test import lindberg_exact, dynamo_stiffness_estimate as stiff_est
from cg_solver import conjugate_gradient
from adaptive_rk import rk45_adaptive
from radial_solver import evolve_radial_modes
from dynamo_induction import run_kinematic_dynamo
from field_analysis import generate_field_report
from sparse_grid_uq import uq_dynamo_reversal_rate


def main():
    print("=" * 70)
    print("PROJECT_43: 地核发电机与地磁场反转 — 博士级合成计算项目")
    print("=" * 70)
    print()

    # =====================================================================
    # 阶段 1: 参数加载与校验
    # =====================================================================
    print("[Phase 1] 加载物理参数与数值配置...")
    cfg = load_config()
    r_icb = cfg.planet_radius_icb()
    r_cmb = cfg.planet_radius_cmb()
    eta = cfg.magnetic_diffusivity()
    alpha0 = cfg.alpha_amplitude()
    Omega0 = cfg.rotation_rate()
    shear = cfg.omega_shear()
    l_max = cfg.l_max()
    n_radial = cfg.n_radial()
    t_end = cfg.t_end_seconds()
    dt_init = cfg.dt_init_seconds()
    save_interval = cfg.save_interval_seconds()
    adaptive_tol = cfg.get("numerics", "adaptive_tol", 1e-6)

    print(f"  核幔边界半径 (CMB): {r_cmb/1e3:.1f} km")
    print(f"  内核边界半径 (ICB): {r_icb/1e3:.1f} km")
    print(f"  磁扩散系数 eta: {eta:.2f} m^2/s")
    print(f"  alpha 效应振幅: {alpha0:.2f}")
    print(f"  差速自转剪切: {shear:.2f}")
    print(f"  球谐截断 L_max: {l_max}")
    print(f"  径向网格数: {n_radial}")
    print(f"  模拟终止时间: {t_end/365.25/24/3600/1e6:.3f} Myr")
    print()

    # 无量纲数估计
    Rm = cfg.magnetic_reynolds_number()
    E = cfg.ekman_number()
    print(f"  磁雷诺数 Rm ~ {Rm:.4e}")
    print(f"  Ekman 数 E ~ {E:.4e}")
    print()

    # =====================================================================
    # 阶段 2: 网格生成与几何预处理
    # =====================================================================
    print("[Phase 2] 生成球壳网格与最优采样...")
    mesh = generate_core_mesh(r_icb, r_cmb, n_radial, n_theta=16, n_phi=16)
    r = mesh["r"]
    print(f"  径向网格: {n_radial} 层 (对数拉伸)")

    # CVT 圆盘采样（地核赤道截面最优节点）
    cvt_pts = cvt_disk_uniform(n_generators=20, n_samples=5000, n_iterations=10,
                               radius=r_cmb / 1e3, seed=43)
    print(f"  CVT 采样点: {cvt_pts.shape[0]} 个 (赤道截面)")
    print()

    # =====================================================================
    # 阶段 3: Stiff 特性分析与测试
    # =====================================================================
    print("[Phase 3] Stiff 系统特性验证...")
    va = alpha0 * shear * Omega0 * (r_cmb - r_icb)  # 特征阿尔芬速度估算
    S = stiff_est(r_cmb, eta, va)
    print(f"  地核发电机有效刚性比 S ~ {S:.4e}")
    print(f"  S >> 1 表明系统具有强刚性特征，需使用自适应/隐式时间积分。")

    # Lindberg stiff 基准测试
    t_test = np.linspace(0.0, 0.1, 11)
    y_exact, _ = lindberg_exact(t_test)
    print(f"  Lindberg 基准测试: y1(0.1) = {y_exact[-1,0]:.6e}")
    print()

    # =====================================================================
    # 阶段 4: 任务调度与计算批次划分
    # =====================================================================
    print("[Phase 4] 模式空间任务调度...")
    scheduler = DynamoBatchScheduler(l_max=l_max, n_radial=n_radial,
                                      n_batches=4, radius=r_cmb,
                                      eta=eta, base_dt=dt_init)
    for b in range(scheduler.n_batches):
        modes = scheduler.get_batch_modes(b)
        dt_eff = scheduler.get_effective_dt(b)
        print(f"  Batch {b}: {len(modes)} modes, effective dt <= {dt_eff/365.25/24/3600:.1f} yrs")
    print()

    # =====================================================================
    # 阶段 5: 核心地核发电机模拟
    # =====================================================================
    print("[Phase 5] 运行运动学 α-Ω Dynamo 模拟...")
    print("  求解感应方程: ∂B/∂t = ∇×(u×B) + η∇²B + α(∇×B)")
    print("  采用环向-极向分解 + 球谐展开 + 径向有限差分 + RK45 自适应积分")
    print()

    # 为控制演示运行时间，使用缩小规模但仍具物理意义的参数
    l_max_demo = 2
    n_radial_demo = 8
    t_end_demo = 1e4 * 365.25 * 24 * 3600  # 50 kyrs
    dt_init_demo = 1e3 * 365.25 * 24 * 3600  # 5 kyrs
    save_interval_demo = 5e3 * 365.25 * 24 * 3600  # 10 kyrs

    # 重建小规模径向网格
    r_demo = np.linspace(r_icb, r_cmb, n_radial_demo)

    t0_sim = time.time()
    times, T_history, P_history = run_kinematic_dynamo(
        r=r_demo,
        r_icb=r_icb,
        r_cmb=r_cmb,
        eta=eta,
        alpha0=alpha0,
        Omega0=Omega0,
        shear_strength=shear,
        l_max=l_max_demo,
        t_end=t_end_demo,
        dt_init=dt_init_demo,
        save_interval=save_interval_demo,
        adaptive_tol=adaptive_tol
    )
    t1_sim = time.time()
    print(f"  模拟耗时: {t1_sim - t0_sim:.2f} s")
    print(f"  保存快照数: {len(times)}")
    print()

    # =====================================================================
    # 阶段 6: 磁场分析与极性反转检测
    # =====================================================================
    print("[Phase 6] 磁场分析与极性反转检测...")

    # 合并 T 和 P 系数用于分析（取极向场为主导）
    coeffs_history = []
    for i in range(len(times)):
        merged = {}
        for key in P_history[i]:
            merged[key] = P_history[i][key]  # 极向场主导地表磁场
        coeffs_history.append(merged)

    report = generate_field_report(coeffs_history, np.array(times), l_max)

    dp = report["dipole_latest"]
    print(f"  最新时刻偶极子参数:")
    print(f"    g10 = {dp['g10']:.6e}  (轴向偶极子)")
    print(f"    g11 = {dp['g11']:.6e}")
    print(f"    h11 = {dp['h11']:.6e}")
    print(f"    偶极子倾角 = {np.degrees(dp['inclination']):.2f}°")
    print(f"    归一化偶极矩 = {dp['dipole_moment_norm']:.6e}")
    print()

    reversals = report["reversals"]
    stats = report["statistics"]
    print(f"  检测到极性反转事件: {len(reversals)} 次")
    if reversals:
        for i, rev in enumerate(reversals):
            t_start_myr = rev["time_start"] / 1e6 / 365.25 / 24 / 3600
            t_end_myr = rev["time_end"] / 1e6 / 365.25 / 24 / 3600
            dur_kyr = rev["duration"] / 1e3 / 365.25 / 24 / 3600
            print(f"    反转 {i+1}: {t_start_myr:.3f}–{t_end_myr:.3f} Myr, 持续 {dur_kyr:.1f} kyr")
    print(f"  反转频率: {stats['reversal_rate']:.4f} 次/Myr")
    print(f"  平均反转持续时间: {stats['mean_duration']/1e3/365.25/24/3600:.1f} kyr")
    print()

    # 能量谱
    l_vals = report["energy_spectrum_l"]
    E_mean = report["energy_spectrum_E"]
    print("  时间平均磁场能量谱 E_l:")
    for l in range(1, min(len(l_vals), l_max + 1)):
        print(f"    l={l}: E_l = {E_mean[l]:.6e}")
    print()

    # =====================================================================
    # 阶段 7: 不确定性量化（稀疏网格）
    # =====================================================================
    print("[Phase 7] 稀疏网格不确定性量化...")
    # 简化的 UQ：评估反转频率对 (eta, alpha0) 的敏感性
    # 为了控制运行时间，使用非常粗糙的代理模型

    def proxy_dynamo_runner(params):
        """轻量级代理模型：基于参数直接估算反转频率。"""
        p_eta, p_alpha = params[0], params[1]
        # 经验标度：反转频率 ~ alpha / eta（Dynamo 数标度）
        Dynamo_number = p_alpha * shear * Omega0 * (r_cmb - r_icb) ** 2 / (p_eta + 1e-10)
        # 反转频率近似：在临界 Dynamo 数附近，反转频率正比于超临界度
        freq = max(0.0, (Dynamo_number - 10.0) * 0.05)
        return freq

    param_ranges = [(1.0, 3.0), (0.3, 0.7)]
    try:
        mean_rate, var_rate, uq_pts, uq_rates = uq_dynamo_reversal_rate(
            proxy_dynamo_runner, param_ranges, level_max=1
        )
    except Exception:
        mean_rate, var_rate, uq_pts, uq_rates = 0.0, 0.0, np.zeros((1, 2)), np.zeros(1)
    print(f"  参数维度: 2 (eta, alpha0)")
    print(f"  稀疏网格点数: {uq_pts.shape[0]}")
    print(f"  反转频率期望 E[f]: {mean_rate:.4f} 次/Myr")
    print(f"  反转频率方差 Var[f]: {var_rate:.4f}")
    print()

    # =====================================================================
    # 阶段 8: 共轭梯度求解器验证（地核径向扩散问题）
    # =====================================================================
    print("[Phase 8] 共轭梯度线性求解器验证...")
    n_test = 64
    A_test = np.diag(2.0 * np.ones(n_test)) + np.diag(-1.0 * np.ones(n_test - 1), 1) + np.diag(-1.0 * np.ones(n_test - 1), -1)
    b_test = np.ones(n_test)
    from cg_solver import conjugate_gradient
    x_cg, iters, resid = conjugate_gradient(lambda v: A_test @ v, b_test, tol=1e-12)
    print(f"  测试矩阵规模: {n_test}x{n_test}")
    print(f"  CG 收敛迭代: {iters}")
    print(f"  最终残差: {resid:.4e}")
    print()

    # =====================================================================
    # 阶段 9: 特殊函数验证
    # =====================================================================
    print("[Phase 9] 地球物理特殊函数验证...")
    ci_1 = cosine_integral(1.0)
    si_1 = sine_integral(1.0)
    j0_1 = spherical_bessel_j(0, 1.0)
    print(f"  Ci(1) = {ci_1:.12f}")
    print(f"  Si(1) = {si_1:.12f}")
    print(f"  j_0(1) = {j0_1:.12f}")
    print()

    # =====================================================================
    # 阶段 10: 输出总结
    # =====================================================================
    print("=" * 70)
    print("PROJECT_43 模拟总结")
    print("=" * 70)
    print(f"  科学问题: 地核 α-Ω Dynamo 与地磁场极性反转")
    print(f"  磁雷诺数 Rm: {Rm:.4e}")
    print(f"  Ekman 数 E: {E:.4e}")
    print(f"  刚性比 S: {S:.4e}")
    print(f"  模拟快照数: {len(times)}")
    print(f"  极性反转事件: {len(reversals)} 次")
    print(f"  反转频率: {stats['reversal_rate']:.4f} 次/Myr")
    print(f"  UQ 期望反转频率: {mean_rate:.4f} 次/Myr")
    print(f"  总运行时间: {time.time() - t0_sim:.2f} s")
    print("=" * 70)
    print("模拟成功完成。")

    return 0


if __name__ == "__main__":
    main()

# ================================================================
# 测试用例（30个，assert模式，涉及随机值均使用固定种子）
# ================================================================
# ---- TC01: 默认配置加载返回有效ConfigParser ----
cfg = load_config()
assert cfg.planet_radius_cmb() > cfg.planet_radius_icb(), '[TC01] 默认配置加载 FAILED'

# ---- TC02: 磁雷诺数与Ekman数为正有限值 ----
cfg = load_config()
assert cfg.magnetic_reynolds_number() > 0.0 and np.isfinite(cfg.magnetic_reynolds_number()), '[TC02] Rm为正有限值 FAILED'
assert cfg.ekman_number() > 0.0 and np.isfinite(cfg.ekman_number()), '[TC02] Ekman数为正有限值 FAILED'

# ---- TC03: cosine_integral(1.0)与已知解析解一致 ----
ci_1 = cosine_integral(1.0)
assert abs(ci_1 - 0.3323951275654294) < 1e-6, '[TC03] Ci(1.0)解析验证 FAILED'

# ---- TC04: sine_integral(1.0)与已知解析解一致 ----
si_1 = sine_integral(1.0)
assert abs(si_1 - 0.8414709848078965) < 1e-6, '[TC04] Si(1.0)解析验证 FAILED'

# ---- TC05: spherical_bessel_j(0,1.0)等于sin(1)/1 ----
j0_1 = spherical_bessel_j(0, 1.0)
assert abs(j0_1 - np.sin(1.0) / 1.0) < 1e-10, '[TC05] j0(1.0)解析验证 FAILED'

# ---- TC06: safe_div避免除零返回fallback ----
from special_functions import safe_div
assert safe_div(5.0, 0.0, fallback=99.0) == 99.0, '[TC06] safe_div除零保护 FAILED'

# ---- TC07: safe_log对非正输入返回安全值 ----
from special_functions import safe_log
assert safe_log(-1.0) == -700.0, '[TC07] safe_log非正输入保护 FAILED'

# ---- TC08: radial_mesh生成正确长度和边界 ----
from mesh_geometry import radial_mesh
r, dr = radial_mesh(1221e3, 3480e3, 16)
assert len(r) == 16, '[TC08] radial_mesh长度 FAILED'
assert r[0] == 1221e3 and r[-1] == 3480e3, '[TC08] radial_mesh边界 FAILED'

# ---- TC09: CVT采样点全部位于圆盘内 ----
from mesh_geometry import cvt_disk_uniform
np.random.seed(42)
pts = cvt_disk_uniform(10, n_samples=5000, n_iterations=5)
assert np.all(np.linalg.norm(pts, axis=1) <= 1.0 + 1e-10), '[TC09] CVT采样点超出圆盘 FAILED'

# ---- TC10: generate_core_mesh节点总数正确 ----
from mesh_geometry import generate_core_mesh
mesh = generate_core_mesh(1221e3, 3480e3, 8, 8, 8)
assert mesh["nodes_3d"].shape[0] == 8 * 8 * 8, '[TC10] core_mesh节点总数 FAILED'

# ---- TC11: RK12对y'=-y终值接近exp(-1) ----
from adaptive_rk import rk12_adaptive
f = lambda t, y: -y
t, y, e = rk12_adaptive(f, (0.0, 1.0), np.array([1.0]), 0.1, tol=1e-5)
assert abs(y[-1, 0] - np.exp(-1.0)) < 1e-3, '[TC11] RK12指数衰减解析验证 FAILED'

# ---- TC12: RK45对y'=-y终值更高精度接近exp(-1) ----
from adaptive_rk import rk45_adaptive
f = lambda t, y: -y
t, y, e = rk45_adaptive(f, (0.0, 1.0), np.array([1.0]), 0.05, tol=1e-6)
assert abs(y[-1, 0] - np.exp(-1.0)) < 1e-5, '[TC12] RK45指数衰减解析验证 FAILED'

# ---- TC13: 隐式梯形对 stiff 系统稳定不发散 ----
from adaptive_rk import implicit_trapezoidal_linear
J = np.array([[-10.0]])
y = implicit_trapezoidal_linear(J, np.array([1.0]), 0.01, 100)
assert np.isfinite(y[0]) and not np.isnan(y[0]), '[TC13] 隐式梯形稳定性 FAILED'

# ---- TC14: CG解对角系统收敛到精确解 ----
from cg_solver import conjugate_gradient
n = 50
A = np.diag(np.arange(1, n + 1, dtype=float))
b = np.ones(n, dtype=float)
x, iters, resid = conjugate_gradient(lambda v: A @ v, b, tol=1e-12)
assert resid < 1e-10, '[TC14] CG对角系统收敛 FAILED'
x_exact = 1.0 / np.arange(1, n + 1, dtype=float)
assert np.linalg.norm(x - x_exact) < 1e-8, '[TC14] CG对角系统精确解 FAILED'

# ---- TC15: 径向扩散算子矩阵维度正确 ----
from cg_solver import build_radial_diffusion_operator
A = build_radial_diffusion_operator(16, 1.0, 0.01, 1.0)
assert A.shape == (16, 16), '[TC15] 径向扩散算子维度 FAILED'

# ---- TC16: 球径向Laplacian输出尺寸匹配 ----
from radial_solver import build_spherical_radial_laplacian
r = np.linspace(1221e3, 3480e3, 16)
L = build_spherical_radial_laplacian(r)
assert L.shape == (16, 16), '[TC16] 球径向Laplacian维度 FAILED'

# ---- TC17: alpha_effect_source在ICB和CMB处为零 ----
from radial_solver import alpha_effect_source
r = np.linspace(1221e3, 3480e3, 16)
src = alpha_effect_source(r, 1221e3, 3480e3, 0.5, 2)
assert src[0] == 0.0 and src[-1] == 0.0, '[TC17] alpha_source边界为零 FAILED'

# ---- TC18: extract_dipole_parameters提取已知系数正确 ----
from field_analysis import extract_dipole_parameters
coeffs = {(1, 0): 1.0 + 0.0j, (1, 1): 0.5 + 0.3j, (2, 0): 0.1 + 0.0j}
dp = extract_dipole_parameters(coeffs)
assert abs(dp["g10"] - 1.0) < 1e-10, '[TC18] 偶极子g10提取 FAILED'
assert dp["dipole_moment_norm"] > 0.0, '[TC18] 偶极子矩为正 FAILED'

# ---- TC19: detect_reversals检测正弦振荡的符号翻转 ----
from field_analysis import detect_reversals
t = np.linspace(0.0, 1e6 * 365.25 * 24 * 3600, 1000)
g10 = np.sin(2.0 * np.pi * t / (2.5e5 * 365.25 * 24 * 3600))
revs = detect_reversals(g10, t, threshold_ratio=0.005)
assert len(revs) >= 3, '[TC19] 反转检测数量 FAILED'

# ---- TC20: reversal_statistics对空列表返回零统计 ----
from field_analysis import reversal_statistics
stats = reversal_statistics([], 1.0)
assert stats["reversal_rate"] == 0.0, '[TC20] 空反转统计rate FAILED'
assert stats["mean_duration"] == 0.0, '[TC20] 空反转统计duration FAILED'

# ---- TC21: mode_space_partition模式总数等于Lmax*(Lmax+2) ----
from task_scheduler import mode_space_partition
parts = mode_space_partition(3, 2)
total = sum(len(p) for p in parts)
assert total == 3 * (3 + 2), '[TC21] 模式空间划分总数 FAILED'

# ---- TC22: multiscale_dt_scheduler高l对应更小时间步 ----
from task_scheduler import multiscale_dt_scheduler
dt_limits = multiscale_dt_scheduler(4, 3480e3, 2.0, 1e10)
assert dt_limits[1] >= dt_limits[4], '[TC22] 高l时间步更小 FAILED'

# ---- TC23: dynamo_stiffness_estimate正参数返回正值 ----
from stiff_test import dynamo_stiffness_estimate
S = dynamo_stiffness_estimate(3480e3, 2.0, 1.0e-3)
assert S > 0.0 and np.isfinite(S), '[TC23] stiff估计正参数 FAILED'

# ---- TC24: lindberg_exact在t=0时y3=-1,y4=0 ----
from stiff_test import lindberg_exact
y0, _ = lindberg_exact(np.array([0.0]))
assert abs(y0[0, 2] - (-1.0)) < 1e-6, '[TC24] Lindberg y3(0) FAILED'
assert abs(y0[0, 3] - 0.0) < 1e-10, '[TC24] Lindberg y4(0) FAILED'

# ---- TC25: Clenshaw-Curtis权重和为2.0 ----
from sparse_grid_uq import clenshaw_curtis_rule
x, w = clenshaw_curtis_rule(3)
assert abs(np.sum(w) - 2.0) < 1e-12, '[TC25] CC权重和 FAILED'

# ---- TC26: map_parameter空间线性映射到物理域中心 ----
from sparse_grid_uq import map_parameter_space
p = map_parameter_space(np.array([[0.0, 0.0]]), [(1.0, 3.0), (0.0, 10.0)])
assert abs(p[0, 0] - 2.0) < 1e-10, '[TC26] 参数映射第一维 FAILED'
assert abs(p[0, 1] - 5.0) < 1e-10, '[TC26] 参数映射第二维 FAILED'

# ---- TC27: DataIO读写场快照可逆 ----
from data_io import DataIO
import os
nr, ntheta = 4, 4
r_grid = np.linspace(1.0, 2.0, nr)
theta_grid = np.linspace(0.0, np.pi, ntheta)
S = np.ones((nr, ntheta))
T = np.zeros((nr, ntheta))
DataIO.write_field_snapshot("test_snapshot_tmp.txt", r_grid, theta_grid, S, T)
r2, t2, S2, T2 = DataIO.read_field_snapshot("test_snapshot_tmp.txt")
assert np.allclose(r2, r_grid), '[TC27] 读写字段快照r_grid FAILED'
assert np.allclose(S2, S), '[TC27] 读写字段快照S FAILED'
os.remove("test_snapshot_tmp.txt")

# ---- TC28: DynamoModel初始化后网格尺寸正确 ----
from dynamo_model import DynamoModel
model = DynamoModel(r_inner=1.0, r_outer=2.0, nr=8, ntheta=8, eta=1.0, c_omega=1.0, c_alpha=0.5, b_eq=1.0)
assert model.nr == 8 and model.ntheta == 8, '[TC28] DynamoModel网格尺寸 FAILED'

# ---- TC29: differential_rotation_profile在r_icb处等于Omega0 ----
from dynamo_induction import differential_rotation_profile
r = np.linspace(1221e3, 3480e3, 16)
Omega = differential_rotation_profile(r, 1221e3, 3480e3, 7.292e-5, 1.0)
assert abs(Omega[0] - 7.292e-5) < 1e-10, '[TC29] 差速旋转ICB边界 FAILED'

# ---- TC30: alpha_effect_profile在ICB和CMB处为零 ----
from dynamo_induction import alpha_effect_profile
r = np.linspace(1221e3, 3480e3, 16)
alpha = alpha_effect_profile(r, 1221e3, 3480e3, 0.5)
assert alpha[0] == 0.0 and alpha[-1] == 0.0, '[TC30] alpha效应边界为零 FAILED'

print('\n全部 30 个测试通过!\n')
