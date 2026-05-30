#!/usr/bin/env python3

import os
import numpy as np




from fft_utils import optimize_qg_grid, prime_factors

Nx_raw, Ny_raw = 64, 64
Nx, Ny = optimize_qg_grid(Nx_raw, Ny_raw)
print(f"[FFT优化] 原始网格: {Nx_raw}x{Ny_raw} → 优化后: {Nx}x{Ny}")
print(f"[FFT优化] Nx素因数分解: {prime_factors(Nx)}")
print(f"[FFT优化] Ny素因数分解: {prime_factors(Ny)}")




Lx = 2.0 * np.pi * 5.0e5
Ly = 2.0 * np.pi * 5.0e5
beta = 2.0e-11
Ld = 2.5e4
nu = 5.0e3
r_drag = 1.0e-7
dt = 1800.0
n_steps = 20

print(f"\n[物理参数] Lx={Lx/1e3:.1f} km, Ly={Ly/1e3:.1f} km, β={beta:.2e}")
print(f"[物理参数] Ld={Ld/1e3:.1f} km, ν={nu:.2e}, r={r_drag:.2e}")




from qg_dynamics import QGBetaPlaneSolver

solver = QGBetaPlaneSolver(Nx=Nx, Ny=Ny, Lx=Lx, Ly=Ly,
                           beta=beta, Ld=Ld, nu=nu, r=r_drag, dt=dt)




from gaussian_vortex import initialize_gaussian_vortex_2d, hermite_spectral_filter


psi_total = np.zeros((Nx, Ny), dtype=np.float64)
zeta_total = np.zeros((Nx, Ny), dtype=np.float64)

eddy_centers = [(0.3*Lx, 0.5*Ly), (0.7*Lx, 0.4*Ly), (0.5*Lx, 0.7*Ly)]
eddy_amplitudes = [2.0e3, -1.5e3, 1.2e3]
eddy_sigmas = [3.5e4, 2.8e4, 2.2e4]
eddy_signs = [1.0, -1.0, 1.0]

for (x0, y0), A, sig, sgn in zip(eddy_centers, eddy_amplitudes, eddy_sigmas, eddy_signs):
    psi_eddy, zeta_eddy = initialize_gaussian_vortex_2d(
        Nx, Ny, Lx, Ly, x0, y0, A, sig, vorticity_sign=sgn
    )
    psi_total += psi_eddy
    zeta_total += zeta_eddy

solver.psi = psi_total
from numpy.fft import fft2, fftshift
psihat = fftshift(fft2(solver.psi))
solver.q = solver._spectral_to_physical(solver.helmholtz * psihat)
solver.rhs_hist = [None, None, None]

print(f"\n[初始化] 已放置 {len(eddy_centers)} 个高斯涡旋")
print(f"[初始化] 初始总能量 E₀ = {solver.compute_energy():.4e} J/m²")
print(f"[初始化] 初始总涡度 Z₀ = {solver.compute_enstrophy():.4e} s⁻²")




from stochastic_backscatter import StochasticBackscatter

stoch = StochasticBackscatter(Nx, Ny, Lx, Ly, epsilon=5.0e-10, k_c=None, seed=42)




from implicit_solver import ImplicitHelmholtzSolver

implicit_solver = ImplicitHelmholtzSolver(Nx, Ny, solver.dx, solver.dy, dt, nu, r_drag)
print(f"\n[隐式求解器] 稀疏矩阵维度: {Nx*Ny}x{Nx*Ny}")




from spectral_quadrature import integrate_radial_energy_spectrum




from front_detector import EddyFrontDetector, shepp_logan_ocean_tracer

detector = EddyFrontDetector(threshold_ratio=0.25)




from particle_dynamics import seed_particles_in_eddy, advect_particles_rk4

n_particles = 200
px, py = seed_particles_in_eddy(n_particles, eddy_centers[0][0], eddy_centers[0][1],
                                 eddy_sigmas[0], Lx, Ly)
print(f"\n[Lagrange粒子] 已播种 {n_particles} 个粒子")




from normal_mode_analysis import qg_normal_mode_stability, compute_growth_rate_spectrum

def jet_profile(y):
    U0 = 0.15
    W = 5.0e4
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




from adaptive_mesh import AdaptiveOceanMesh


mesh_gen = AdaptiveOceanMesh(bbox=(0.0, 10.0, 0.0, 10.0))
fd_rect = lambda p: mesh_gen.drectangle(p, 0.0, 10.0, 0.0, 10.0)
fh_uniform = lambda p: np.ones(len(p)) * 0.25
p_mesh, t_mesh = mesh_gen.generate_mesh(fd_rect, fh_uniform, h0=0.25, max_iter=10)
quality = mesh_gen.compute_mesh_quality(p_mesh, t_mesh)
print(f"\n[自适应网格] 节点数: {len(p_mesh)}, 单元数: {len(t_mesh)}")
print(f"[自适应网格] 最小质量: {np.min(quality):.4f}, 平均质量: {np.mean(quality):.4f}")




from sparse_grid_uq import SparseGridUQ, total_degree_size

uq = SparseGridUQ(dim=2, level=2)
mean_uq, var_uq, nodes_uq, values_uq = uq.propagate_expectation(
    lambda xi: np.exp(-0.5 * (xi[0]**2 + xi[1]**2))
)
print(f"\n[稀疏网格UQ] 节点数: {len(nodes_uq)}")
print(f"[稀疏网格UQ] 期望: {mean_uq:.6f}, 方差: {var_uq:.6f}")
print(f"[稀疏网格UQ] 总多项式空间维度 (degree=3): {total_degree_size(2, 3)}")




from polygon_flux import polygon_area, polygon_centroid, polygon_ellipse_parameters


theta_poly = np.linspace(0, 2*np.pi, 20, endpoint=False)
xv = eddy_centers[0][0] + eddy_sigmas[0] * np.cos(theta_poly)
yv = eddy_centers[0][1] + 0.7 * eddy_sigmas[0] * np.sin(theta_poly)
area_exact = polygon_area(xv, yv)
cx, cy = polygon_centroid(xv, yv)
a_ell, b_ell, theta_ell = polygon_ellipse_parameters(xv, yv)
print(f"\n[多边形积分] 涡旋面积: {area_exact/1e6:.2f} km²")
print(f"[多边形积分] 质心: ({cx/1e3:.1f}, {cy/1e3:.1f}) km")
print(f"[多边形积分] 等效椭圆: a={a_ell/1e3:.1f} km, b={b_ell/1e3:.1f} km, θ={theta_ell:.3f} rad")




from eddy_tracker import EddyTracker

tracker = EddyTracker(w_pos=1.0, w_area=0.5, w_vort=0.3, w_overlap=2.0)




from matrix_io import write_unstructured_mesh, build_sparse_laplacian_unstructured
import tempfile


mesh_file = os.path.join(tempfile.gettempdir(), "ocean_mesh_demo.txt")
write_unstructured_mesh(mesh_file, p_mesh * 1e5, t_mesh)
print(f"\n[矩阵I/O] 网格已写入: {mesh_file}")


L_sparse = build_sparse_laplacian_unstructured(p_mesh, t_mesh)
print(f"[矩阵I/O] 稀疏Laplacian非零元数: {L_sparse.nnz}")




print("\n" + "="*60)
print("开始QG动力学时间积分...")
print("="*60)

energy_history = []
enstrophy_history = []
eddy_snapshots = []
u_fields = []
v_fields = []

for step in range(n_steps):

    F_stoch = stoch.generate_forcing(dt)


    solver.step(stochastic_forcing=F_stoch)


    E = solver.compute_energy()
    Z = solver.compute_enstrophy()
    energy_history.append(E)
    enstrophy_history.append(Z)


    u, v = solver.get_velocity()
    u_fields.append(u)
    v_fields.append(v)


    zeta = solver.compute_vorticity()
    labels, n_eddy, stats = detector.segment_eddies(zeta, solver.dx, solver.dy, zeta)
    eddy_snapshots.append(stats)

    if step % 5 == 0 or step == n_steps - 1:
        print(f"  Step {step:3d}: E={E:.4e}  Z={Z:.4e}  N_eddy={n_eddy}")

print("="*60)
print("积分完成。")





if len(eddy_snapshots) >= 2:
    trajectories = tracker.track(eddy_snapshots)
    track_stats = tracker.compute_lifetime_statistics(trajectories, eddy_snapshots)
    print(f"\n[涡旋追踪] 检测到的轨迹数: {track_stats['n_tracks']}")
    print(f"[涡旋追踪] 平均寿命: {track_stats['mean_lifetime_steps']:.2f} 步")
    print(f"[涡旋追踪] 平均传播速度: {track_stats['mean_speed']:.4e} m/s")


k_bins, T_radial, Pi = solver.compute_energy_flux_spectrum()

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


if len(u_fields) >= 2:
    x_grid = np.linspace(0, Lx, Nx)
    y_grid = np.linspace(0, Ly, Ny)
    px_adv, py_adv = advect_particles_rk4(px, py, u_fields[-1], v_fields[-1],
                                          x_grid, y_grid, dt)

    px_adv = np.mod(px_adv, Lx)
    py_adv = np.mod(py_adv, Ly)
    disp_mean = np.mean(np.sqrt((px_adv - px)**2 + (py_adv - py)**2))
    print(f"\n[Lagrange粒子] 平均位移: {disp_mean/1e3:.3f} km")


rhs_test = np.random.randn(Nx, Ny)
psi_impl = implicit_solver.solve(rhs_test)
residual = np.linalg.norm(rhs_test - (psi_impl + dt * r_drag * psi_impl))
print(f"\n[隐式求解器验证] 测试残差范数: {residual:.4e}")


psihat_filt = hermite_spectral_filter(psihat, solver.KX, solver.KY,
                                      k_cutoff=2.0*np.pi/Lx * Nx/4.0, order=4)
print(f"[谱滤波] 滤波前后能量比: {np.sum(np.abs(psihat_filt)**2)/np.sum(np.abs(psihat)**2):.4f}")




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
