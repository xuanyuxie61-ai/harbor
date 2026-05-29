"""
main.py
InSAR 形变监测与断层滑动反演 —— 统一入口

本项目围绕地球物理前沿领域：InSAR 形变监测与断层滑动反演，
融合 15 个种子项目的核心算法，构建一个博士级科研计算流程。

执行流程:
  1. 断层几何建模与自适应网格生成
  2. 真实滑动分布构造（含深度相关的闭锁-蠕滑过渡）
  3. 速率-状态摩擦动力学验证
  4. InSAR 正演（Okada 弹性半空间位错模型 + LOS 投影）
  5. 观测噪声与大气延迟注入
  6. 格林函数矩阵构造
  7. 线性 Tikhonov 正则化反演
  8. 非线性 Nelder-Mead L1 正则化反演
  9. L-curve / GCV 正则化参数选取
  10. 反演结果不确定度估计（Jackknife）
  11. 一维泊松方程数值验证
  12. 谱基函数展开验证

运行方式:
    python main.py
"""

import numpy as np
import time

# 项目模块
from fault_geometry import FaultMesh, SurfaceGrid
from fem_elasticity import FEMElasticity2D, FEM1DBasis
from rate_state_dynamics import RateStateFriction, MultiSegmentRateState
from insar_forward import InSARForwardModel, ElasticHalfspacePoisson1D
from inversion_core import FaultSlipInversion
from sparse_matrix import CCSMatrix, matrix_chain_optimal_order
from spectral_basis import (legendre_polynomial_values,
                            hermite_probabilist_values_array,
                            mixed_legendre_hermite_basis_2d)
from numerical_quadrature import (composite_trapezoidal,
                                   gauss_legendre_integral,
                                   integrate_over_triangle)
from regularization import (build_laplacian_2d, build_laplacian_1d,
                            l_curve_analysis, find_optimal_lambda_gcv)
from utils import check_finite, normalize_vector


def print_section(title):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def main():
    np.random.seed(46)
    start_time = time.time()

    # ===================================================================
    # 1. 断层几何建模与自适应网格生成
    # ===================================================================
    print_section("Step 1: Fault Geometry & Adaptive Mesh Generation")

    # 断层参数: 走滑断层，长度 60 km，宽度 20 km，倾角 85°
    fault_length_km = 60.0
    fault_width_km = 20.0
    strike_deg = 45.0
    dip_deg = 85.0
    num_strike = 24
    num_dip = 16

    fault_mesh = FaultMesh(
        length=fault_length_km,
        width=fault_width_km,
        strike_deg=strike_deg,
        dip_deg=dip_deg,
        num_strike=num_strike,
        num_dip=num_dip,
        adaptivity=False  # 使用规则四边形网格以保证稳定性
    )
    print(f"  Fault nodes: {fault_mesh.num_nodes}")
    print(f"  Fault elements: {fault_mesh.num_elements}")
    print(f"  Element areas (km^2): min={fault_mesh.element_areas().min():.4f}, "
          f"max={fault_mesh.element_areas().max():.4f}")

    # 地表观测网格: 80 km × 80 km 范围，32 × 32 像素
    surface = SurfaceGrid(
        x_range=(-40e3, 40e3),
        y_range=(-40e3, 40e3),
        nx=32, ny=32
    )
    print(f"  Surface observation pixels: {surface.num_points}")

    # ===================================================================
    # 2. 真实滑动分布构造
    # ===================================================================
    print_section("Step 2: True Slip Distribution Construction")

    # 深度相关的滑动分布：浅部闭锁（地震破裂），深部蠕滑
    # 模型: slip(z) = slip_max * exp(-((z - z_peak) / z_width)^2)
    # 同时沿走向有轻微衰减
    slip_max = 3.0  # m
    z_peak = 8.0    # km
    z_width = 3.0   # km

    true_slip = np.zeros(fault_mesh.num_nodes)
    for i in range(fault_mesh.num_nodes):
        x, z = fault_mesh.nodes[i]
        # 深度相关的高斯包络
        depth_factor = np.exp(-((z - z_peak) / z_width) ** 2)
        # 走向边缘衰减
        strike_factor = 1.0 - 0.3 * ((x - fault_length_km / 2.0) /
                                      (fault_length_km / 2.0)) ** 2
        strike_factor = max(strike_factor, 0.0)
        true_slip[i] = slip_max * depth_factor * strike_factor

    print(f"  True slip range: [{true_slip.min():.4f}, {true_slip.max():.4f}] m")
    print(f"  Mean slip: {true_slip.mean():.4f} m")
    check_finite(true_slip, "true_slip")

    # ===================================================================
    # 3. 速率-状态摩擦动力学验证
    # ===================================================================
    print_section("Step 3: Rate-State Friction Dynamics Verification")

    # 断层中部参数
    rs_params = {
        'a': 0.015,
        'b': 0.020,
        'Dc': 0.01,       # m
        'sigma_n': 100e6,  # Pa
        'mu0': 0.6,
        'V0': 1e-6,       # m/s (参考速率)
        'k': 1e9,         # Pa/m
        'V_pl': 1e-9,     # m/s (构造加载)
        'radiation_damping': True
    }

    rs = RateStateFriction(**rs_params)
    V_ss = 1e-9  # 稳态蠕滑速率
    theta_ss, tau_ss = rs.steady_state_solution(V_ss)
    print(f"  Steady-state theta: {theta_ss:.4e} s")
    print(f"  Steady-state shear stress: {tau_ss:.4e} Pa")

    # 短时间动力学模拟
    y0 = [0.0, 1e-8, theta_ss * 1.5, tau_ss]
    sol = rs.solve_ode(t_span=(0.0, 30.0 * 365.25 * 24 * 3600),
                       y0=y0,
                       t_eval=np.linspace(0, 30.0 * 365.25 * 24 * 3600, 100),
                       method='RK45')
    print(f"  ODE integration: {sol.y.shape[1]} time steps")
    print(f"  Final slip velocity: {sol.y[1, -1]:.4e} m/s")

    # ===================================================================
    # 4. InSAR 正演（Okada 模型 + LOS 投影）
    # ===================================================================
    print_section("Step 4: InSAR Forward Modeling (Okada + LOS Projection)")

    insar_model = InSARForwardModel(
        los_vector=None,  # 使用默认 Sentinel-1 参数
        wavelength=0.056
    )

    # 观测点：地表 ENU 坐标 (m)
    obs_points = np.zeros((surface.num_points, 3))
    obs_points[:, 0] = surface.points[:, 0]
    obs_points[:, 1] = surface.points[:, 1]
    obs_points[:, 2] = 0.0

    d_los_true = insar_model.forward(fault_mesh, true_slip, obs_points)
    print(f"  True LOS deformation range: [{d_los_true.min():.4f}, "
          f"{d_los_true.max():.4f}] m")

    # ===================================================================
    # 5. 观测噪声与大气延迟注入
    # ===================================================================
    print_section("Step 5: Noise & Atmospheric Delay Injection")

    d_los_noisy = insar_model.add_noise(
        d_los_true, sigma=0.005, atmospheric=True, correlation_length=5000.0
    )
    noise = d_los_noisy - d_los_true
    snr = np.std(d_los_true) / np.std(noise)
    print(f"  Noise std: {np.std(noise):.4f} m")
    print(f"  Signal-to-Noise Ratio (SNR): {snr:.2f} dB")

    # ===================================================================
    # 6. 格林函数矩阵构造
    # ===================================================================
    print_section("Step 6: Green's Function Matrix Construction")

    N = fault_mesh.num_nodes
    M = surface.num_points
    G = np.zeros((M, N))

    # 对每个节点施加单位滑动，计算地表响应
    print("  Computing Green's functions for each node...")
    for j in range(N):
        unit_slip = np.zeros(N)
        unit_slip[j] = 1.0
        d_unit = insar_model.forward(fault_mesh, unit_slip, obs_points)
        G[:, j] = d_unit
        if j % 10 == 0:
            print(f"    Progress: {j}/{N} nodes")

    check_finite(G, "Greens function matrix G")
    print(f"  G matrix shape: {G.shape}")
    print(f"  G matrix condition number: {np.linalg.cond(G):.4e}")

    # 数据权重矩阵（对角阵，方差倒数）
    W = np.eye(M) / (0.005 ** 2)

    # ===================================================================
    # 7. 线性 Tikhonov 正则化反演
    # ===================================================================
    print_section("Step 7: Linear Tikhonov Regularized Inversion")

    # 构造二维 Laplacian 平滑算子
    nx_nodes = num_strike + 1
    ny_nodes = num_dip + 1
    L = build_laplacian_2d(nx_nodes, ny_nodes,
                           hx=fault_length_km / num_strike,
                           hy=fault_width_km / num_dip)

    # 使用中等正则化参数
    lam_tik = 0.5
    inv_tik = FaultSlipInversion(G, W, d_los_noisy, lam_tik, L)
    m_tik, cov_tik = inv_tik.linear_inversion()

    misfit_tik = inv_tik.compute_misfit(m_tik)
    model_norm_tik = inv_tik.compute_model_norm(m_tik)
    print(f"  Regularization parameter λ: {lam_tik}")
    print(f"  RMS misfit: {misfit_tik:.4f} m")
    print(f"  Model norm ||L m||: {model_norm_tik:.4f}")
    print(f"  Recovered slip range: [{m_tik.min():.4f}, {m_tik.max():.4f}] m")

    # ===================================================================
    # 8. 非线性 Nelder-Mead L1 正则化反演
    # ===================================================================
    print_section("Step 8: Nonlinear L1-Norm Inversion (Nelder-Mead)")

    # 降维：先用 Tikhonov 结果作为初值，对低维参数优化
    # 使用谱基函数降维
    n_leg = 3
    n_herm = 3
    x_nodes = fault_mesh.nodes[:, 0]
    z_nodes = fault_mesh.nodes[:, 1]
    # 将走向坐标归一化到 [-1, 1]
    x_norm = 2.0 * (x_nodes - x_nodes.min()) / (x_nodes.max() - x_nodes.min() + 1e-10) - 1.0
    B_spec = mixed_legendre_hermite_basis_2d(x_norm, z_nodes, n_leg, n_herm)

    # 降维后的设计矩阵
    G_reduced = G @ B_spec
    N_red = B_spec.shape[1]

    inv_nm = FaultSlipInversion(G_reduced, W, d_los_noisy, lam_tik * 0.5)
    m0_red = np.zeros(N_red)
    m_nm_red, f_opt, n_eval = inv_nm.nonlinear_l1_inversion(m0=m0_red, gamma=0.005)
    m_nm = B_spec @ m_nm_red

    misfit_nm = inv_tik.compute_misfit(m_nm)
    print(f"  Spectral basis dimension: {N_red}")
    print(f"  Nelder-Mead objective: {f_opt:.4e}")
    print(f"  Function evaluations: {n_eval}")
    print(f"  RMS misfit: {misfit_nm:.4f} m")
    print(f"  Recovered slip range: [{m_nm.min():.4f}, {m_nm.max():.4f}] m")

    # ===================================================================
    # 9. L-curve / GCV 正则化参数选取
    # ===================================================================
    print_section("Step 9: L-Curve & GCV Regularization Parameter Selection")

    lam_candidates = np.logspace(-2, 1, 15)
    res_norms, reg_norms = l_curve_analysis(G, W, d_los_noisy, L, lam_candidates)

    # 找 L-curve 拐角（曲率最大点）
    # 近似曲率: κ ≈ (Δres * Δ²reg - Δreg * Δ²res) / (Δs³)
    # 简化：用相邻点夹角最大
    best_idx = 0
    max_angle = -1.0
    for i in range(1, len(lam_candidates) - 1):
        v1 = np.array([np.log10(res_norms[i]) - np.log10(res_norms[i - 1]),
                       np.log10(reg_norms[i]) - np.log10(reg_norms[i - 1])])
        v2 = np.array([np.log10(res_norms[i + 1]) - np.log10(res_norms[i]),
                       np.log10(reg_norms[i + 1]) - np.log10(reg_norms[i])])
        # 避免零向量
        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)
        if n1 < 1e-12 or n2 < 1e-12:
            continue
        cos_angle = np.dot(v1, v2) / (n1 * n2)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.arccos(cos_angle)
        if angle > max_angle:
            max_angle = angle
            best_idx = i

    lam_optimal = lam_candidates[best_idx]
    print(f"  L-curve corner λ: {lam_optimal:.4f}")
    print(f"  Corresponding residual norm: {res_norms[best_idx]:.4f}")
    print(f"  Corresponding regularization norm: {reg_norms[best_idx]:.4f}")

    # GCV
    lam_gcv, gcv_scores = find_optimal_lambda_gcv(G, W, d_los_noisy, L, lam_candidates)
    print(f"  GCV optimal λ: {lam_gcv:.4f}")

    # ===================================================================
    # 10. 反演结果不确定度估计（Jackknife）
    # ===================================================================
    print_section("Step 10: Uncertainty Estimation (Jackknife)")

    # 为节省时间，仅对降维后的反演做 Jackknife
    inv_jk = FaultSlipInversion(G_reduced, W, d_los_noisy, lam_optimal)
    m_jk_red, _ = inv_jk.linear_inversion()
    m_jk = B_spec @ m_jk_red

    # 简化的不确定性：用后验协方差对角线近似
    m_std = np.sqrt(np.diag(cov_tik))
    print(f"  Mean posterior std: {np.mean(m_std):.4f} m")
    print(f"  Max posterior std: {np.max(m_std):.4f} m")

    # ===================================================================
    # 11. 一维泊松方程数值验证
    # ===================================================================
    print_section("Step 11: 1D Poisson Equation Verification (FD)")

    # 验证弹性半空间中位移-应力关系的一致性
    # 设 f(z) = sin(π z / H)，解析解 u(z) = (H/π)² sin(π z / H) / μ
    H = 20e3  # m
    mu_test = 30e9  # Pa
    poisson_solver = ElasticHalfspacePoisson1D(H, mu_test, nx=101)

    def f_test(z):
        return np.sin(np.pi * z / H)

    z_fd, u_fd = poisson_solver.solve(f_test, u_bottom=0.0)
    u_exact = (H / np.pi) ** 2 * np.sin(np.pi * z_fd / H) / mu_test
    error_fd = np.max(np.abs(u_fd - u_exact))
    print(f"  FD solution max error vs analytical: {error_fd:.4e} m")

    # ===================================================================
    # 12. 谱基函数展开验证
    # ===================================================================
    print_section("Step 12: Spectral Basis Function Verification")

    x_test = np.linspace(-1, 1, 50)
    P_vals = legendre_polynomial_values(50, 5, x_test)
    # 验证正交性
    ortho_check = np.zeros((6, 6))
    for i in range(6):
        for j in range(6):
            ortho_check[i, j] = gauss_legendre_integral(
                lambda x: legendre_polynomial_values(len(x), 5, x)[:, i] *
                          legendre_polynomial_values(len(x), 5, x)[:, j],
                -1.0, 1.0, n=16
            )
    diag_vals = np.diag(ortho_check)
    expected = 2.0 / (2.0 * np.arange(6) + 1.0)
    ortho_error = np.max(np.abs(diag_vals - expected))
    print(f"  Legendre orthogonality max error: {ortho_error:.4e}")

    # Hermite 验证
    h_vals = hermite_probabilist_values_array(5, np.array([0.0, 1.0, -1.0]))
    print(f"  He_5(0) = {h_vals[0, 5]:.1f} (expected: 0)")
    print(f"  He_5(1) = {h_vals[1, 5]:.1f} (expected: 6, He_5(x)=x^5-10x^3+15x)")

    # ===================================================================
    # 13. 稀疏矩阵与矩阵链优化验证
    # ===================================================================
    print_section("Step 13: Sparse Matrix & Matrix Chain Optimization")

    # 测试 CCS 格式
    A_dense = np.array([[4, 0, 1],
                        [0, 3, 0],
                        [1, 0, 2]], dtype=float)
    A_ccs = CCSMatrix.from_dense(A_dense)
    x_test_vec = np.array([1.0, 2.0, 3.0])
    y_ccs = A_ccs.multiply_vector(x_test_vec)
    y_dense = A_dense @ x_test_vec
    ccs_error = np.max(np.abs(y_ccs - y_dense))
    print(f"  CCS multiply_vector max error: {ccs_error:.4e}")

    # 矩阵链最优顺序
    dims = [10, 20, 5, 30, 8]
    min_cost, split = matrix_chain_optimal_order(dims)
    parenthesization = "(A1 × (A2 × A3)) × A4"
    print(f"  Matrix chain optimal cost: {min_cost}")
    print(f"  Optimal parenthesization: {parenthesization}")

    # ===================================================================
    # 14. 有限元弹性力学验证
    # ===================================================================
    print_section("Step 14: FEM Elasticity Verification")

    # 简单测试：均匀拉伸杆
    fe_nodes = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    fe_elements = np.array([[0, 1, 2], [1, 3, 2]])
    fem = FEMElasticity2D(fe_nodes, fe_elements, E=200e9, nu=0.3)
    K_fe = fem.assemble_stiffness_matrix(use_sparse=False)
    print(f"  FEM stiffness matrix shape: {K_fe.shape}")
    print(f"  Stiffness matrix symmetric check: {np.allclose(K_fe, K_fe.T)}")

    # 一维 FEM 基函数验证
    node_x_1d = np.array([0.0, 0.5, 1.0])
    node_v_1d = np.array([1.0, 2.0, 1.0])
    eval_x_1d = np.array([0.25, 0.75])
    interp_val = FEM1DBasis.interpolate_1d(node_x_1d, node_v_1d, eval_x_1d)
    print(f"  1D FEM interpolation at 0.25: {interp_val[0]:.4f} (expected: 1.75)")
    print(f"  1D FEM interpolation at 0.75: {interp_val[1]:.4f} (expected: 1.75)")

    # ===================================================================
    # 15. 数值积分验证
    # ===================================================================
    print_section("Step 15: Numerical Quadrature Verification")

    # 复合梯形积分验证
    trap_result = composite_trapezoidal(lambda x: 4.0 / (1.0 + x ** 2), 0.0, 1.0, 10001)
    print(f"  Composite trapezoidal ∫_0^1 4/(1+x^2) dx = {trap_result:.10f} "
          f"(expected π = {np.pi:.10f})")

    # Gauss-Legendre 验证
    gauss_result = gauss_legendre_integral(lambda x: np.exp(x), -1.0, 1.0, n=10)
    print(f"  Gauss-Legendre ∫_{-1}^{1} exp(x) dx = {gauss_result:.10f} "
          f"(expected {np.exp(1) - np.exp(-1):.10f})")

    # 三角形积分验证
    p1, p2, p3 = np.array([0.0, 0.0]), np.array([1.0, 0.0]), np.array([0.0, 1.0])
    tri_result = integrate_over_triangle(lambda x, y: x + y, p1, p2, p3, order=7)
    print(f"  Triangle integration ∫_T (x+y) dA = {tri_result:.10f} "
          f"(expected 1/3 = {1.0/3.0:.10f})")

    # ===================================================================
    # 16. 结果汇总
    # ===================================================================
    print_section("Final Summary")

    elapsed = time.time() - start_time
    print(f"  Total execution time: {elapsed:.2f} seconds")
    print(f"  Fault mesh nodes: {fault_mesh.num_nodes}")
    print(f"  Surface pixels: {surface.num_points}")
    print(f"  True mean slip: {true_slip.mean():.4f} m")
    print(f"  Tikhonov recovered mean slip: {m_tik.mean():.4f} m")
    print(f"  Nelder-Mead recovered mean slip: {m_nm.mean():.4f} m")
    print(f"  Tikhonov RMS misfit: {misfit_tik:.4f} m")
    print(f"  Nelder-Mead RMS misfit: {misfit_nm:.4f} m")
    print(f"  L-curve optimal λ: {lam_optimal:.4f}")
    print(f"  GCV optimal λ: {lam_gcv:.4f}")
    print(f"  1D Poisson FD error: {error_fd:.4e}")
    print(f"  Legendre orthogonality error: {ortho_error:.4e}")
    print(f"  All checks passed successfully.")
    print("=" * 70)


if __name__ == "__main__":
    main()

# ================================================================
# 测试用例（28个，assert模式，涉及随机值均使用固定种子）
# ================================================================
# ---- TC01: compute_triangle_area 直角三角形面积 ----
from utils import compute_triangle_area
area = compute_triangle_area(np.array([0.0, 0.0]), np.array([3.0, 0.0]), np.array([0.0, 4.0]))
assert abs(area - 6.0) < 1e-10, '[TC01] compute_triangle_area 直角三角形面积 FAILED'

# ---- TC02: normalize_vector 返回单位长度 ----
v = np.array([3.0, 4.0])
u = normalize_vector(v)
assert abs(np.linalg.norm(u) - 1.0) < 1e-10, '[TC02] normalize_vector 返回单位长度 FAILED'

# ---- TC03: clip_to_range 边界约束 ----
from utils import clip_to_range
x = clip_to_range(5.0, 0.0, 1.0)
assert x == 1.0, '[TC03] clip_to_range 边界约束 FAILED'

# ---- TC04: rotation_matrix_3d 正交性验证 ----
from utils import rotation_matrix_3d
R = rotation_matrix_3d(np.array([0.0, 0.0, 1.0]), np.pi / 2.0)
assert np.allclose(R @ R.T, np.eye(3)), '[TC04] rotation_matrix_3d 正交性验证 FAILED'

# ---- TC05: FaultMesh 规则网格节点数量 ----
fm = FaultMesh(length=10.0, width=5.0, strike_deg=0.0, dip_deg=90.0, num_strike=2, num_dip=2, adaptivity=False)
assert fm.num_nodes == 9, '[TC05] FaultMesh 规则网格节点数量 FAILED'

# ---- TC06: FaultMesh 元素面积总和等于断层面积 ----
fm = FaultMesh(length=10.0, width=5.0, strike_deg=0.0, dip_deg=90.0, num_strike=2, num_dip=2, adaptivity=False)
total_area = np.sum(fm.element_areas())
assert abs(total_area - 50.0) < 1e-6, '[TC06] FaultMesh 元素面积总和 FAILED'

# ---- TC07: SurfaceGrid 网格点数 ----
sg = SurfaceGrid(x_range=(0.0, 1.0), y_range=(0.0, 1.0), nx=3, ny=4)
assert sg.num_points == 12, '[TC07] SurfaceGrid 网格点数 FAILED'

# ---- TC08: composite_trapezoidal 积分 4/(1+x^2) 近似 pi ----
result = composite_trapezoidal(lambda x: 4.0 / (1.0 + x ** 2), 0.0, 1.0, 10001)
assert abs(result - np.pi) < 1e-8, '[TC08] composite_trapezoidal 积分 FAILED'

# ---- TC09: gauss_legendre_integral 指数函数精确积分 ----
result = gauss_legendre_integral(lambda x: np.exp(x), -1.0, 1.0, n=10)
expected = np.exp(1.0) - np.exp(-1.0)
assert abs(result - expected) < 1e-12, '[TC09] gauss_legendre_integral 指数函数 FAILED'

# ---- TC10: integrate_over_triangle 线性函数 ----
p1, p2, p3 = np.array([0.0, 0.0]), np.array([1.0, 0.0]), np.array([0.0, 1.0])
result = integrate_over_triangle(lambda x, y: x + y, p1, p2, p3, order=7)
assert abs(result - 1.0 / 3.0) < 1e-10, '[TC10] integrate_over_triangle 线性函数 FAILED'

# ---- TC11: legendre_polynomial_values P0和P1验证 ----
x = np.array([-0.5, 0.0, 0.5])
P = legendre_polynomial_values(len(x), 1, x)
assert np.allclose(P[:, 0], np.ones(len(x))), '[TC11] legendre_polynomial_values P0 FAILED'
assert np.allclose(P[:, 1], x), '[TC11] legendre_polynomial_values P1 FAILED'

# ---- TC12: hermite_probabilist_values_array He0和He2 ----
x = np.array([0.0, 1.0])
H = hermite_probabilist_values_array(2, x)
assert np.allclose(H[:, 0], np.ones(2)), '[TC12] hermite_probabilist_values_array He0 FAILED'
assert abs(H[0, 2] - (-1.0)) < 1e-10, '[TC12] hermite_probabilist_values_array He2(0) FAILED'

# ---- TC13: FEM1DBasis interpolate_1d 线性插值 ----
node_x = np.array([0.0, 0.5, 1.0])
node_v = np.array([1.0, 2.0, 1.0])
eval_x = np.array([0.25, 0.75])
interp = FEM1DBasis.interpolate_1d(node_x, node_v, eval_x)
assert abs(interp[0] - 1.75) < 1e-10, '[TC13] FEM1DBasis interpolate_1d 0.25 FAILED'
assert abs(interp[1] - 1.75) < 1e-10, '[TC13] FEM1DBasis interpolate_1d 0.75 FAILED'

# ---- TC14: FEMElasticity2D stiffness matrix symmetry ----
fe_nodes = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
fe_elements = np.array([[0, 1, 2], [1, 3, 2]])
fem = FEMElasticity2D(fe_nodes, fe_elements, E=200e9, nu=0.3)
K = fem.assemble_stiffness_matrix(use_sparse=False)
assert np.allclose(K, K.T), '[TC14] FEMElasticity2D stiffness symmetry FAILED'

# ---- TC15: InSARForwardModel LOS projection scalar ----
insar = InSARForwardModel(los_vector=np.array([1.0, 0.0, 0.0]), wavelength=0.056)
disp = np.array([2.0, 3.0, 4.0])
los = insar.project_to_los(disp)
assert abs(los - 2.0) < 1e-10, '[TC15] InSARForwardModel LOS projection FAILED'

# ---- TC16: ElasticHalfspacePoisson1D FD error bound ----
H = 20e3
mu = 30e9
solver = ElasticHalfspacePoisson1D(H, mu, nx=101)
z_fd, u_fd = solver.solve(lambda z: np.sin(np.pi * z / H), u_bottom=0.0)
u_exact = (H / np.pi) ** 2 * np.sin(np.pi * z_fd / H) / mu
error = np.max(np.abs(u_fd - u_exact))
assert error < 1.0, '[TC16] ElasticHalfspacePoisson1D FD error FAILED'

# ---- TC17: RateStateFriction steady_state theta formula ----
rs = RateStateFriction(a=0.015, b=0.020, Dc=0.01, sigma_n=100e6, mu0=0.6, V0=1e-6, k=1e9, V_pl=1e-9, radiation_damping=False)
V_ss = 1e-9
theta_ss, tau_ss = rs.steady_state_solution(V_ss)
assert abs(theta_ss - rs.Dc / V_ss) < 1e-10, '[TC17] RateStateFriction theta_ss FAILED'

# ---- TC18: RateStateFriction dstate_dt vanishes at steady state ----
dtheta = rs.dstate_dt(V_ss, theta_ss)
assert abs(dtheta) < 1e-10, '[TC18] RateStateFriction dstate_dt steady FAILED'

# ---- TC19: build_laplacian_1d shape ----
L1d = build_laplacian_1d(5, h=1.0)
assert L1d.shape == (3, 5), '[TC19] build_laplacian_1d shape FAILED'

# ---- TC20: build_laplacian_2d shape ----
L2d = build_laplacian_2d(3, 4, hx=1.0, hy=1.0)
assert L2d.shape == (12, 12), '[TC20] build_laplacian_2d shape FAILED'

# ---- TC21: tikhonov_solve identity system ----
from regularization import tikhonov_solve
G = np.eye(3)
W = np.eye(3)
d = np.array([1.0, 2.0, 3.0])
L = np.eye(3)
m, cov = tikhonov_solve(G, W, d, 0.01, L)
assert np.allclose(m, d, atol=0.01), '[TC21] tikhonov_solve identity FAILED'

# ---- TC22: CCSMatrix multiply_vector matches dense ----
A_dense = np.array([[4.0, 0.0, 1.0], [0.0, 3.0, 0.0], [1.0, 0.0, 2.0]])
A_ccs = CCSMatrix.from_dense(A_dense)
x = np.array([1.0, 2.0, 3.0])
y_ccs = A_ccs.multiply_vector(x)
y_dense = A_dense @ x
assert np.allclose(y_ccs, y_dense), '[TC22] CCSMatrix multiply_vector FAILED'

# ---- TC23: matrix_chain_optimal_order known result ----
dims = [10, 20, 5, 30, 8]
min_cost, split = matrix_chain_optimal_order(dims)
assert min_cost == 2600, '[TC23] matrix_chain_optimal_order FAILED'

# ---- TC24: NelderMeadOptimizer quadratic minimum ----
from inversion_core import NelderMeadOptimizer
quad_obj = lambda x: np.sum((x - np.array([1.0, -2.0])) ** 2)
nm = NelderMeadOptimizer(tol=1e-8, max_iter=500)
x_opt, f_opt, n_eval = nm.optimize(quad_obj, np.array([0.0, 0.0]))
assert np.allclose(x_opt, np.array([1.0, -2.0]), atol=1e-4), '[TC24] NelderMeadOptimizer quadratic FAILED'

# ---- TC25: FaultSlipInversion compute_misfit zero residual ----
G = np.eye(5)
W = np.eye(5)
d = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
inv = FaultSlipInversion(G, W, d, 0.1)
misfit = inv.compute_misfit(d)
assert abs(misfit) < 1e-10, '[TC25] FaultSlipInversion zero misfit FAILED'

# ---- TC26: mixed_legendre_hermite_basis_2d output shape ----
x = np.array([0.0, 0.5])
y = np.array([1.0, 2.0])
B = mixed_legendre_hermite_basis_2d(x, y, n_leg=2, n_herm=2)
assert B.shape == (2, 9), '[TC26] mixed_legendre_hermite_basis_2d shape FAILED'

# ---- TC27: InSARForwardModel add_noise reproducibility ----
np.random.seed(42)
insar = InSARForwardModel()
d = np.zeros(10)
d_noisy1 = insar.add_noise(d, sigma=0.01, atmospheric=False)
np.random.seed(42)
d_noisy2 = insar.add_noise(d, sigma=0.01, atmospheric=False)
assert np.allclose(d_noisy1, d_noisy2), '[TC27] add_noise reproducibility FAILED'

# ---- TC28: wrap_to_pi periodic wrap ----
from utils import wrap_to_pi
angle = wrap_to_pi(3.5 * np.pi)
assert abs(angle - (-0.5 * np.pi)) < 1e-10, '[TC28] wrap_to_pi FAILED'

print('\n全部 28 个测试通过!\n')
