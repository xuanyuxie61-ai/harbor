#!/usr/bin/env python3
"""
================================================================================
中尺度涡旋动力学与能量逆串级综合模拟系统
Mesoscale Eddy Dynamics and Inverse Energy Cascade Synthesis
================================================================================

本程序融合15个种子科研项目的核心算法，构建了一个面向海洋科学前沿问题的
博士级综合计算系统。核心科学问题为：

    β平面准地转单层模型中的中尺度涡旋生成、演化、能量逆串级、
    涡旋追踪与参数不确定性量化。

物理控制方程（扰动位涡形式）：
    ∂q/∂t + J(ψ, q) + β·∂ψ/∂x = ν∇⁴ψ − r∇²ψ + F_backscatter
    q = ∇²ψ − (1/Ld²) ψ

种子项目融合：
    939 (谱积分) → 能量谱诊断的高精度积分
    1040 (CMRG随机数) → 次网格随机反散射参数化
    604 (Jacobi本征值) → 线性化QG正规模稳定性分析
    026 (Cholesky) → 隐式Helmholtz方程求解
    265+308 (CVT+DistMesh) → 自适应非结构网格生成
    325 (边界检测) → 涡旋边界与分割
    911 (素因数分解) → FFT维度优化
    444 (动态规划) → 多涡旋最优追踪
    427 (Fibonacci螺旋) → Lagrange粒子初始化
    570+508 (ICE I/O + HB→MM) → 网格与稀疏矩阵I/O
    1109 (稀疏网格) → 涡旋扩散系数不确定性量化
    454 (Gauss+Hermite) → 高斯涡旋初始化与谱滤波
    886 (多边形积分) → 有限体积精确通量计算
================================================================================
"""

import os
import numpy as np

# ------------------------------------------------------------------------------
# 1. FFT维度优化 (seed 911_prime_factors)
# ------------------------------------------------------------------------------
from fft_utils import optimize_qg_grid, prime_factors

Nx_raw, Ny_raw = 64, 64
Nx, Ny = optimize_qg_grid(Nx_raw, Ny_raw)
print(f"[FFT优化] 原始网格: {Nx_raw}x{Ny_raw} → 优化后: {Nx}x{Ny}")
print(f"[FFT优化] Nx素因数分解: {prime_factors(Nx)}")
print(f"[FFT优化] Ny素因数分解: {prime_factors(Ny)}")

# ------------------------------------------------------------------------------
# 物理参数
# ------------------------------------------------------------------------------
Lx = 2.0 * np.pi * 5.0e5   # ~1000 km × 1000 km 域
Ly = 2.0 * np.pi * 5.0e5
beta = 2.0e-11             # [s⁻¹·m⁻¹]
Ld = 2.5e4                 # Rossby变形半径 [m]
nu = 5.0e3                 # 双调和粘性 [m⁴/s]
r_drag = 1.0e-7            # 线性Ekman拖曳 [s⁻¹]
dt = 1800.0                # 时间步长 [s] (30 min)
n_steps = 20               # 总积分步数

print(f"\n[物理参数] Lx={Lx/1e3:.1f} km, Ly={Ly/1e3:.1f} km, β={beta:.2e}")
print(f"[物理参数] Ld={Ld/1e3:.1f} km, ν={nu:.2e}, r={r_drag:.2e}")

# ------------------------------------------------------------------------------
# 2. 准地转动力学求解器 (核心模块)
# ------------------------------------------------------------------------------
from qg_dynamics import QGBetaPlaneSolver

solver = QGBetaPlaneSolver(Nx=Nx, Ny=Ny, Lx=Lx, Ly=Ly,
                           beta=beta, Ld=Ld, nu=nu, r=r_drag, dt=dt)

# ------------------------------------------------------------------------------
# 3. 高斯涡旋初始化 (seed 454_gaussian)
# ------------------------------------------------------------------------------
from gaussian_vortex import initialize_gaussian_vortex_2d, hermite_spectral_filter

# 初始化三个高斯涡旋 (使用物理合理的振幅)
psi_total = np.zeros((Nx, Ny), dtype=np.float64)
zeta_total = np.zeros((Nx, Ny), dtype=np.float64)

eddy_centers = [(0.3*Lx, 0.5*Ly), (0.7*Lx, 0.4*Ly), (0.5*Lx, 0.7*Ly)]
eddy_amplitudes = [2.0e3, -1.5e3, 1.2e3]    # [m²/s]
eddy_sigmas = [3.5e4, 2.8e4, 2.2e4]         # [m]
eddy_signs = [1.0, -1.0, 1.0]

for (x0, y0), A, sig, sgn in zip(eddy_centers, eddy_amplitudes, eddy_sigmas, eddy_signs):
    psi_eddy, zeta_eddy = initialize_gaussian_vortex_2d(
        Nx, Ny, Lx, Ly, x0, y0, A, sig, vorticity_sign=sgn
    )
    psi_total += psi_eddy
    zeta_total += zeta_eddy

# TODO: Initialize solver.q from solver.psi using the Helmholtz relation.
# In spectral QG dynamics: q = ∇²ψ - (1/Ld²) ψ
# In spectral space: q̂ = -(k² + 1/Ld²) ψ̂ = self.helmholtz * ψ̂
# Convert ψ to spectral, apply Helmholtz operator, then back to physical space.
# Also reset the Adams-Bashforth RHS history.
raise NotImplementedError("main: solver.q initialization from psi not implemented.")

print(f"\n[初始化] 已放置 {len(eddy_centers)} 个高斯涡旋")
print(f"[初始化] 初始总能量 E₀ = {solver.compute_energy():.4e} J/m²")
print(f"[初始化] 初始总涡度 Z₀ = {solver.compute_enstrophy():.4e} s⁻²")

# ------------------------------------------------------------------------------
# 4. 随机反散射参数化 (seed 1040_rnglib)
# ------------------------------------------------------------------------------
from stochastic_backscatter import StochasticBackscatter

stoch = StochasticBackscatter(Nx, Ny, Lx, Ly, epsilon=5.0e-10, k_c=None, seed=42)

# ------------------------------------------------------------------------------
# 5. 隐式Helmholtz求解器 (seed 026_asa007)
# ------------------------------------------------------------------------------
from implicit_solver import ImplicitHelmholtzSolver

implicit_solver = ImplicitHelmholtzSolver(Nx, Ny, solver.dx, solver.dy, dt, nu, r_drag)
print(f"\n[隐式求解器] 稀疏矩阵维度: {Nx*Ny}x{Nx*Ny}")

# ------------------------------------------------------------------------------
# 6. 谱积分规则 (seed 939_quad_fast_rule)
# ------------------------------------------------------------------------------
from spectral_quadrature import integrate_radial_energy_spectrum

# ------------------------------------------------------------------------------
# 7. 涡旋边界检测 (seed 325_edge)
# ------------------------------------------------------------------------------
from front_detector import EddyFrontDetector, shepp_logan_ocean_tracer

detector = EddyFrontDetector(threshold_ratio=0.25)

# ------------------------------------------------------------------------------
# 8. Lagrange粒子初始化 (seed 427_fibonacci_spiral)
# ------------------------------------------------------------------------------
from particle_dynamics import seed_particles_in_eddy, advect_particles_rk4

n_particles = 200
px, py = seed_particles_in_eddy(n_particles, eddy_centers[0][0], eddy_centers[0][1],
                                 eddy_sigmas[0], Lx, Ly)
print(f"\n[Lagrange粒子] 已播种 {n_particles} 个粒子")

# ------------------------------------------------------------------------------
# 9. 正规模稳定性分析 (seed 604_jacobi_eigenvalue)
# ------------------------------------------------------------------------------
from normal_mode_analysis import qg_normal_mode_stability, compute_growth_rate_spectrum

def jet_profile(y):
    """背景纬向流: 双曲正切喷流"""
    U0 = 0.15  # m/s
    W = 5.0e4  # m
    yc = Ly / 2.0
    return U0 * np.tanh((y - yc) / W)

Ny_stab = 32
c_modes, phi_modes = qg_normal_mode_stability(Ny_stab, Ly, jet_profile, beta, Ld,
                                               k_zonal=2.0*np.pi/Lx * 2.0)
ci_max = np.max(np.imag(c_modes))
print(f"\n[稳定性分析] 最大不稳定增长率 ci_max = {ci_max:.4e} m/s")

k_vals = np.linspace(2.0*np.pi/Lx, 2.0*np.pi/Lx * 8.0, 8)
_, sigma_max = compute_growth_rate_spectrum(Ny_stab, Ly, jet_profile, beta, Ld, k_vals)
print(f"[稳定性分析] 最大增长速率 σ_max = {np.max(sigma_max):.4e} s⁻¹")

# ------------------------------------------------------------------------------
# 10. 自适应网格生成 (seed 265_cvtp_1d + 308_distmesh)
# ------------------------------------------------------------------------------
from adaptive_mesh import AdaptiveOceanMesh

# 使用缩小的演示域以控制计算量
mesh_gen = AdaptiveOceanMesh(bbox=(0.0, 10.0, 0.0, 10.0))
fd_rect = lambda p: mesh_gen.drectangle(p, 0.0, 10.0, 0.0, 10.0)
fh_uniform = lambda p: np.ones(len(p)) * 0.25
p_mesh, t_mesh = mesh_gen.generate_mesh(fd_rect, fh_uniform, h0=0.25, max_iter=10)
quality = mesh_gen.compute_mesh_quality(p_mesh, t_mesh)
print(f"\n[自适应网格] 节点数: {len(p_mesh)}, 单元数: {len(t_mesh)}")
print(f"[自适应网格] 最小质量: {np.min(quality):.4f}, 平均质量: {np.mean(quality):.4f}")

# ------------------------------------------------------------------------------
# 11. 稀疏网格不确定性量化 (seed 1109_sparse_grid_total_poly)
# ------------------------------------------------------------------------------
from sparse_grid_uq import SparseGridUQ, total_degree_size

uq = SparseGridUQ(dim=2, level=2)
mean_uq, var_uq, nodes_uq, values_uq = uq.propagate_expectation(
    lambda xi: np.exp(-0.5 * (xi[0]**2 + xi[1]**2))  # 代理模型
)
print(f"\n[稀疏网格UQ] 节点数: {len(nodes_uq)}")
print(f"[稀疏网格UQ] 期望: {mean_uq:.6f}, 方差: {var_uq:.6f}")
print(f"[稀疏网格UQ] 总多项式空间维度 (degree=3): {total_degree_size(2, 3)}")

# ------------------------------------------------------------------------------
# 12. 多边形精确积分 (seed 886_polygon_integrals)
# ------------------------------------------------------------------------------
from polygon_flux import polygon_area, polygon_centroid, polygon_ellipse_parameters

# 构造一个示例涡旋边界多边形
theta_poly = np.linspace(0, 2*np.pi, 20, endpoint=False)
xv = eddy_centers[0][0] + eddy_sigmas[0] * np.cos(theta_poly)
yv = eddy_centers[0][1] + 0.7 * eddy_sigmas[0] * np.sin(theta_poly)
area_exact = polygon_area(xv, yv)
cx, cy = polygon_centroid(xv, yv)
a_ell, b_ell, theta_ell = polygon_ellipse_parameters(xv, yv)
print(f"\n[多边形积分] 涡旋面积: {area_exact/1e6:.2f} km²")
print(f"[多边形积分] 质心: ({cx/1e3:.1f}, {cy/1e3:.1f}) km")
print(f"[多边形积分] 等效椭圆: a={a_ell/1e3:.1f} km, b={b_ell/1e3:.1f} km, θ={theta_ell:.3f} rad")

# ------------------------------------------------------------------------------
# 13. 涡旋追踪 (seed 444_football_dynamic)
# ------------------------------------------------------------------------------
from eddy_tracker import EddyTracker

tracker = EddyTracker(w_pos=1.0, w_area=0.5, w_vort=0.3, w_overlap=2.0)

# ------------------------------------------------------------------------------
# 14. 网格与矩阵I/O (seed 570_ice_io + 508_hb_to_mm)
# ------------------------------------------------------------------------------
from matrix_io import write_unstructured_mesh, build_sparse_laplacian_unstructured
import tempfile

# 保存网格
mesh_file = os.path.join(tempfile.gettempdir(), "ocean_mesh_demo.txt")
write_unstructured_mesh(mesh_file, p_mesh * 1e5, t_mesh)
print(f"\n[矩阵I/O] 网格已写入: {mesh_file}")

# 构建并保存稀疏Laplacian
L_sparse = build_sparse_laplacian_unstructured(p_mesh, t_mesh)
print(f"[矩阵I/O] 稀疏Laplacian非零元数: {L_sparse.nnz}")

# ------------------------------------------------------------------------------
# 主时间积分循环
# ------------------------------------------------------------------------------
print("\n" + "="*60)
print("开始QG动力学时间积分...")
print("="*60)

energy_history = []
enstrophy_history = []
eddy_snapshots = []
u_fields = []
v_fields = []

for step in range(n_steps):
    # 生成随机强迫
    F_stoch = stoch.generate_forcing(dt)

    # 时间步进
    solver.step(stochastic_forcing=F_stoch)

    # 能量与涡度诊断
    E = solver.compute_energy()
    Z = solver.compute_enstrophy()
    energy_history.append(E)
    enstrophy_history.append(Z)

    # 速度场存储 (用于粒子平流)
    u, v = solver.get_velocity()
    u_fields.append(u)
    v_fields.append(v)

    # 涡旋检测与追踪
    zeta = solver.compute_vorticity()
    labels, n_eddy, stats = detector.segment_eddies(zeta, solver.dx, solver.dy, zeta)
    eddy_snapshots.append(stats)

    if step % 5 == 0 or step == n_steps - 1:
        print(f"  Step {step:3d}: E={E:.4e}  Z={Z:.4e}  N_eddy={n_eddy}")

print("="*60)
print("积分完成。")

# ------------------------------------------------------------------------------
# 后处理与追踪
# ------------------------------------------------------------------------------
# 涡旋追踪
if len(eddy_snapshots) >= 2:
    trajectories = tracker.track(eddy_snapshots)
    track_stats = tracker.compute_lifetime_statistics(trajectories, eddy_snapshots)
    print(f"\n[涡旋追踪] 检测到的轨迹数: {track_stats['n_tracks']}")
    print(f"[涡旋追踪] 平均寿命: {track_stats['mean_lifetime_steps']:.2f} 步")
    print(f"[涡旋追踪] 平均传播速度: {track_stats['mean_speed']:.4e} m/s")

# 能量谱诊断
k_bins, T_radial, Pi = solver.compute_energy_flux_spectrum()
# 径向积分能谱: 将2D谱场投影到径向壳层
E_k_2d = 0.5 * solver.ksq * np.abs(solver._physical_to_spectral(solver.psi))**2
E_k_radial = np.zeros_like(k_bins)
for i in range(len(k_bins) - 1):
    mask = ((solver.ksq >= k_bins[i]**2) & (solver.ksq < k_bins[i+1]**2))
    if np.any(mask):
        E_k_radial[i] = np.sum(E_k_2d[mask])

from spectral_quadrature import integrate_radial_energy_spectrum
energy_low = integrate_radial_energy_spectrum(k_bins[:len(k_bins)//2],
                                               E_k_radial[:len(k_bins)//2],
                                               rule='gauss_legendre', n_quad=32)
energy_high = integrate_radial_energy_spectrum(k_bins[len(k_bins)//2:],
                                                E_k_radial[len(k_bins)//2:],
                                                rule='clenshaw_curtis', n_quad=32)
print(f"\n[能量谱诊断] 低波数带能量: {energy_low:.4e}")
print(f"[能量谱诊断] 高波数带能量: {energy_high:.4e}")
print(f"[能量谱诊断] 逆串级通量 Π(k_c) ≈ {Pi[len(Pi)//2]:.4e}")

# Lagrange粒子平流
if len(u_fields) >= 2:
    x_grid = np.linspace(0, Lx, Nx)
    y_grid = np.linspace(0, Ly, Ny)
    px_adv, py_adv = advect_particles_rk4(px, py, u_fields[-1], v_fields[-1],
                                          x_grid, y_grid, dt)
    # 周期性边界
    px_adv = np.mod(px_adv, Lx)
    py_adv = np.mod(py_adv, Ly)
    disp_mean = np.mean(np.sqrt((px_adv - px)**2 + (py_adv - py)**2))
    print(f"\n[Lagrange粒子] 平均位移: {disp_mean/1e3:.3f} km")

# 隐式求解器验证
rhs_test = np.random.randn(Nx, Ny)
psi_impl = implicit_solver.solve(rhs_test)
residual = np.linalg.norm(rhs_test - (psi_impl + dt * r_drag * psi_impl))
print(f"\n[隐式求解器验证] 测试残差范数: {residual:.4e}")

# 谱滤波验证
psihat_filt = hermite_spectral_filter(psihat, solver.KX, solver.KY,
                                      k_cutoff=2.0*np.pi/Lx * Nx/4.0, order=4)
print(f"[谱滤波] 滤波前后能量比: {np.sum(np.abs(psihat_filt)**2)/np.sum(np.abs(psihat)**2):.4f}")

# ------------------------------------------------------------------------------
# 总结输出
# ------------------------------------------------------------------------------
print("\n" + "="*60)
print("综合模拟系统运行总结")
print("="*60)
print(f"总积分步数: {n_steps}")
print(f"最终总能量: {energy_history[-1]:.4e} J/m²")
print(f"最终总涡度: {enstrophy_history[-1]:.4e} s⁻²")
print(f"能量变化率: {(energy_history[-1]-energy_history[0])/energy_history[0]*100:.2f}%")
print(f"涡度变化率: {(enstrophy_history[-1]-enstrophy_history[0])/enstrophy_history[0]*100:.2f}%")
print(f"自适应网格节点数: {len(p_mesh)}")
print(f"稀疏网格UQ节点数: {len(nodes_uq)}")
print(f"隐式求解器矩阵非零元: {L_sparse.nnz}")
print(f"FFT友好网格: {Nx}x{Ny}")
print(f"稳定性分析最大增长率: {np.max(sigma_max):.4e} s⁻¹")
print("="*60)
print("所有模块运行成功，无报错。")
print("="*60)
