
import os
import sys
import time
import numpy as np




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


    Rm = cfg.magnetic_reynolds_number()
    E = cfg.ekman_number()
    print(f"  磁雷诺数 Rm ~ {Rm:.4e}")
    print(f"  Ekman 数 E ~ {E:.4e}")
    print()




    print("[Phase 2] 生成球壳网格与最优采样...")
    mesh = generate_core_mesh(r_icb, r_cmb, n_radial, n_theta=16, n_phi=16)
    r = mesh["r"]
    print(f"  径向网格: {n_radial} 层 (对数拉伸)")


    cvt_pts = cvt_disk_uniform(n_generators=20, n_samples=5000, n_iterations=10,
                               radius=r_cmb / 1e3, seed=43)
    print(f"  CVT 采样点: {cvt_pts.shape[0]} 个 (赤道截面)")
    print()




    print("[Phase 3] Stiff 系统特性验证...")
    va = alpha0 * shear * Omega0 * (r_cmb - r_icb)
    S = stiff_est(r_cmb, eta, va)
    print(f"  地核发电机有效刚性比 S ~ {S:.4e}")
    print(f"  S >> 1 表明系统具有强刚性特征，需使用自适应/隐式时间积分。")


    t_test = np.linspace(0.0, 0.1, 11)
    y_exact, _ = lindberg_exact(t_test)
    print(f"  Lindberg 基准测试: y1(0.1) = {y_exact[-1,0]:.6e}")
    print()




    print("[Phase 4] 模式空间任务调度...")
    scheduler = DynamoBatchScheduler(l_max=l_max, n_radial=n_radial,
                                      n_batches=4, radius=r_cmb,
                                      eta=eta, base_dt=dt_init)
    for b in range(scheduler.n_batches):
        modes = scheduler.get_batch_modes(b)
        dt_eff = scheduler.get_effective_dt(b)
        print(f"  Batch {b}: {len(modes)} modes, effective dt <= {dt_eff/365.25/24/3600:.1f} yrs")
    print()




    print("[Phase 5] 运行运动学 α-Ω Dynamo 模拟...")
    print("  求解感应方程: ∂B/∂t = ∇×(u×B) + η∇²B + α(∇×B)")
    print("  采用环向-极向分解 + 球谐展开 + 径向有限差分 + RK45 自适应积分")
    print()


    l_max_demo = 4
    n_radial_demo = 16
    t_end_demo = 5e4 * 365.25 * 24 * 3600
    dt_init_demo = 5e3 * 365.25 * 24 * 3600
    save_interval_demo = 1e4 * 365.25 * 24 * 3600


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




    print("[Phase 6] 磁场分析与极性反转检测...")


    coeffs_history = []
    for i in range(len(times)):
        merged = {}
        for key in P_history[i]:
            merged[key] = P_history[i][key]
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


    l_vals = report["energy_spectrum_l"]
    E_mean = report["energy_spectrum_E"]
    print("  时间平均磁场能量谱 E_l:")
    for l in range(1, min(len(l_vals), l_max + 1)):
        print(f"    l={l}: E_l = {E_mean[l]:.6e}")
    print()




    print("[Phase 7] 稀疏网格不确定性量化...")



    def proxy_dynamo_runner(params):






        raise NotImplementedError("Hole_4: Dynamo 代理模型待实现")
        return 0.0

    param_ranges = [(1.0, 3.0), (0.3, 0.7)]
    mean_rate, var_rate, uq_pts, uq_rates = uq_dynamo_reversal_rate(
        proxy_dynamo_runner, param_ranges, level_max=3
    )
    print(f"  参数维度: 2 (eta, alpha0)")
    print(f"  稀疏网格点数: {uq_pts.shape[0]}")
    print(f"  反转频率期望 E[f]: {mean_rate:.4f} 次/Myr")
    print(f"  反转频率方差 Var[f]: {var_rate:.4f}")
    print()




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




    print("[Phase 9] 地球物理特殊函数验证...")
    ci_1 = cosine_integral(1.0)
    si_1 = sine_integral(1.0)
    j0_1 = spherical_bessel_j(0, 1.0)
    print(f"  Ci(1) = {ci_1:.12f}")
    print(f"  Si(1) = {si_1:.12f}")
    print(f"  j_0(1) = {j0_1:.12f}")
    print()




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
    sys.exit(main())
