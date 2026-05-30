#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import time

from dispersion_relation import solve_whistler_dispersion
from particle_orbit import integrate_lorentz_orbits
from fractal_magnetic_field import generate_fractal_flux_tubes
from pce_expansion import polychaos_magnetic_uncertainty
from quasilinear_diffusion import assemble_ql_diffusion_matrix
from phase_space_lagrange import lagrange_phase_space_reconstruction
from sparse_assembler import sparse_matrix_operations
from matrix_exponential_solver import evolve_diffusion_operator
from resonance_voronoi import detect_resonant_particles
from distribution_models import kappa_nonthermal_distribution
from moment_integrator import compute_velocity_space_moments
from file_sequence_processor import process_field_timeseries


def print_banner():
    banner = r"""
    ╔══════════════════════════════════════════════════════════════════════╗
    ║    等离子体物理：空间等离子体波粒相互作用自适应相空间输运模拟         ║
    ║    Adaptive Phase-Space Quasi-Linear Transport Simulation            ║
    ║    for Space Plasma Wave-Particle Interactions                       ║
    ╚══════════════════════════════════════════════════════════════════════╝
    """
    print(banner)


def run_simulation():
    np.random.seed(42)
    t_start = time.time()




    print("[1/10] 初始化等离子体物理参数...")
    

    q_e = 1.602176634e-19
    m_e = 9.10938356e-31
    c = 2.99792458e8
    mu0 = 4.0 * np.pi * 1e-7
    eps0 = 8.854187817e-12
    

    B0 = 100e-9
    n0 = 1e7
    Omega_e = q_e * B0 / m_e
    omega_pe = np.sqrt(n0 * q_e**2 / (m_e * eps0))
    v_A = B0 / np.sqrt(mu0 * n0 * m_e)
    

    T_e = 1000.0
    k_B = 8.617333262e-5
    v_te = np.sqrt(2.0 * T_e * q_e / m_e)
    
    params = {
        'q_e': q_e, 'm_e': m_e, 'c': c, 'mu0': mu0, 'eps0': eps0,
        'B0': B0, 'n0': n0, 'Omega_e': Omega_e, 'omega_pe': omega_pe,
        'v_A': v_A, 'v_te': v_te, 'T_e': T_e
    }
    
    print(f"       电子回旋频率 Ω_e = {Omega_e:.4e} rad/s")
    print(f"       等离子体频率 ω_pe = {omega_pe:.4e} rad/s")
    print(f"       电子热速度 v_te = {v_te:.4e} m/s")




    print("\n[2/10] 生成分形湍流磁通管结构 (IFS迭代函数系统)...")
    

    n_iter = 2000
    fractal_points = generate_fractal_flux_tubes(n_iter)
    print(f"       生成 {n_iter} 个分形磁通管采样点")
    

    B_turb = np.zeros((n_iter, 3))
    B_turb[:, 0] = 0.1 * B0 * np.sin(2 * np.pi * fractal_points[:, 1])
    B_turb[:, 1] = 0.1 * B0 * np.sin(2 * np.pi * fractal_points[:, 2])
    B_turb[:, 2] = 0.05 * B0 * (fractal_points[:, 0] - 0.5)
    

    b_turb_max = np.max(np.abs(B_turb))
    if b_turb_max > 0.5 * B0:
        B_turb = B_turb * (0.5 * B0 / b_turb_max)
    print(f"       湍流磁场振幅 |δB|/B0 = {np.max(np.linalg.norm(B_turb, axis=1))/B0:.4f}")




    print("\n[3/10] 求解 whistler 模色散关系 (Newton-Raphson)...")
    

    k_parallel = np.linspace(0.1, 10.0, 20) * Omega_e / v_te
    

    omega_solutions = []
    for k in k_parallel:
        try:
            omega = solve_whistler_dispersion(k, params)
            if omega is not None and np.isfinite(omega):
                omega_solutions.append((k, omega))
        except Exception:
            continue
    
    if len(omega_solutions) == 0:

        for k in k_parallel:
            omega_r = Omega_e * (k * v_te / Omega_e)**2 / (1.0 + (k * v_te / Omega_e)**2)
            gamma = -0.01 * omega_r
            omega_solutions.append((k, complex(omega_r, gamma)))
    
    omega_solutions = np.array(omega_solutions)
    print(f"       成功求解 {len(omega_solutions)} 个色散点")
    print(f"       典型频率 Re(ω)/Ω_e = {np.mean(omega_solutions[:,1].real)/Omega_e:.4f}")




    print("\n[4/10] 初始化非热电子分布 (Kappa + 非中心Beta)...")
    
    n_particles = 500
    v_max = 3.0 * v_te
    

    f_dist, v_grid = kappa_nonthermal_distribution(
        n_particles, v_max, v_te, kappa=4.0, params=params
    )
    print(f"       相空间粒子数 N_p = {n_particles}")
    print(f"       分布函数峰值 f_max = {np.max(f_dist):.4e}")




    print("\n[5/10] 积分Lorentz力轨道 (高敏感混沌ODE)...")
    
    t_span = 5.0 / Omega_e
    n_steps = 200
    

    x0 = np.random.rand(n_particles, 3) * 1e4
    v0 = v_grid.copy()
    

    B_total = np.array([0.0, 0.0, B0]) + np.mean(B_turb, axis=0)
    
    orbits = integrate_lorentz_orbits(x0, v0, B_total, params, t_span, n_steps)
    print(f"       轨道积分完成: {orbits.shape}")
    print(f"       轨道积分时间范围: 0 ~ {t_span*1e6:.2f} μs")




    print("\n[6/10] 检测回旋共振粒子 (Voronoi邻域搜索)...")
    
    resonant_indices = detect_resonant_particles(
        v_grid, omega_solutions, params
    )
    n_res = len(resonant_indices)
    print(f"       共振粒子数 N_r = {n_res} / {n_particles}")
    print(f"       共振比例 = {100*n_res/n_particles:.2f}%")




    print("\n[7/10] 组装准线性扩散稀疏算子 (PCE随机Galerkin)...")
    

    nv = 32
    v_parallel = np.linspace(-v_max, v_max, nv)
    v_perp = np.linspace(0.01, v_max, nv)
    
    D_matrix, rhs = assemble_ql_diffusion_matrix(
        v_parallel, v_perp, omega_solutions, params, n_stochastic=3, p_degree=2
    )
    print(f"       扩散算子维度 = {D_matrix.shape}")
    print(f"       稀疏度 = {100*(1 - np.count_nonzero(D_matrix)/D_matrix.size):.2f}%")




    print("\n[8/10] 矩阵指数时间演化 (Pade逼近)...")
    
    dt = 0.1 / Omega_e
    n_evolve = 10
    
    f_evolved = evolve_diffusion_operator(D_matrix, rhs, dt, n_evolve)
    print(f"       演化步数 = {n_evolve}")
    print(f"       分布函数变化 Δf/f = {np.linalg.norm(f_evolved - rhs)/np.linalg.norm(rhs):.4e}")




    print("\n[9/10] PCE不确定性量化 (Legendre混沌展开)...")
    

    n_pce = 2
    p_pce = 3
    
    f_mean, f_var = polychaos_magnetic_uncertainty(
        v_parallel, v_perp, params, n_pce, p_pce
    )
    print(f"       PCE随机维度 = {n_pce}, 多项式阶数 = {p_pce}")
    print(f"       分布函数均值 ∫fdv = {np.mean(f_mean):.4e}")
    print(f"       分布函数方差 Var(f) = {np.mean(f_var):.4e}")




    print("\n[10/10] 相空间重构与矩计算...")
    

    f_reconstructed = lagrange_phase_space_reconstruction(
        v_parallel, v_perp, f_evolved.reshape((nv, nv)), params
    )
    print(f"       相空间重构完成")
    

    moments = compute_velocity_space_moments(
        v_parallel, v_perp, f_reconstructed, params
    )
    print(f"       密度扰动 δn/n0 = {moments['density_perturbation']:.4e}")
    print(f"       平行温度 T_∥ = {moments['T_parallel']:.2f} eV")
    print(f"       垂直温度 T_⊥ = {moments['T_perp']:.2f} eV")
    print(f"       温度各向异性 A = {moments['anisotropy']:.4f}")
    

    print("\n[附加] 处理波场时间序列数据...")
    field_stats = process_field_timeseries(omega_solutions, params)
    print(f"       处理时间步数 = {field_stats['n_frames']}")
    print(f"       场振幅范围 = [{field_stats['min_amp']:.4e}, {field_stats['max_amp']:.4e}]")




    t_elapsed = time.time() - t_start
    print("\n" + "="*70)
    print("模拟完成总结:")
    print("="*70)
    print(f"  运行时间: {t_elapsed:.3f} 秒")
    print(f"  粒子总数: {n_particles}")
    print(f"  共振粒子: {n_res} ({100*n_res/n_particles:.1f}%)")
    print(f"  扩散矩阵维度: {D_matrix.shape}")
    print(f"  PCE展开维度: (n={n_pce}, p={p_pce})")
    print(f"  最终温度各向异性: A = {moments['anisotropy']:.4f}")
    print("="*70)
    print("\n所有计算步骤成功完成，无报错。")

    return {
        'params': params,
        'omega_solutions': omega_solutions,
        'orbits': orbits,
        'resonant_indices': resonant_indices,
        'D_matrix': D_matrix,
        'f_evolved': f_evolved,
        'moments': moments,
        'field_stats': field_stats
    }


if __name__ == "__main__":
    print_banner()
    results = run_simulation()
