#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main.py
博士级合成项目 179 统一入口
============================
科学领域：计算数学 — 低秩矩阵近似与张量分解
合成问题：参数化反应-扩散-形态演化系统的高维响应面低秩张量近似

运行方式：
    python main.py
无需任何命令行参数，全自动完成从参数采样、PDE 求解、张量构建到 TT 分解压缩的完整流程。
"""

import numpy as np
import os
import sys

# ---------------------------------------------------------------------------
# 导入合成项目各模块
# ---------------------------------------------------------------------------
from system_utils import initialize_project, EPS
from tensor_io import write_tensor_mm, write_symmetric_tensor_mm
from domain_generator import (
    hand_outline_polygon, hand_ellipse_fourier_approx,
    universal_egg_half_profile, chebyshev_nodes_1d
)
from fem_discretization import assemble_fem_matrices_1d, fem_l2_norm
from reaction_kinetics import twoway_exact_solution, parametric_reaction_source
from tridiagonal_solver import (
    r83_dif2, r83_cg, r83_cr_fa, r83_cr_sl, r83_jac_sl, r83_gs_sl, r83_res
)
from randomized_sketching import (
    randomized_svd, hilbert_matrix, low_rank_test_matrix, adaptive_rank_threshold,
    annulus_sample
)
from quadrature_integrator import (
    grid_integrate_1d, monte_carlo_integrate, qmc_integrate,
    estimate_tensor_frobenius_norm_mc
)
from rank_analysis import (
    rref_rank, rref_columns, tensor_multilinear_ranks,
    estimate_tensor_train_ranks, build_hankel_tensor_from_sequence,
    collatz_polynomial_sequence, hankel_matrix_from_sequence
)
from nmf_initializer import nmf_init_coltable, nmf_init_random, nonnegative_projection
from tensor_train_decomposition import tt_svd, tt_als_approximate, tt_frobenius_norm, tt_inner_product
from pde_solver import solve_reaction_diffusion, build_solution_tensor, compress_and_analyze


def main():
    # ========================================================================
    # 0. 系统初始化
    # ========================================================================
    logger = initialize_project()

    # ========================================================================
    # 1. 域生成与低秩形状分析（原 502_hand_data + 093_bird_egg）
    # ========================================================================
    logger.info("【步骤 1】生成参数化计算域与低秩形状分析")
    n_shape = 100
    hand_xy, hand_pcs = hand_ellipse_fourier_approx(n_shape)
    logger.info(f"手部轮廓 SVD 主成分 shape={hand_pcs.shape}")

    # 通用蛋形参数化域
    L_egg = 2.0
    x_egg = chebyshev_nodes_1d(-L_egg / 2, L_egg / 2, 32)
    r_egg = universal_egg_half_profile(B=1.0, L=L_egg, w=0.1, D=0.6, x=x_egg)
    logger.info(f"蛋形域 Chebyshev 节点数={len(x_egg)}, max_radius={r_egg.max():.4f}")

    # ========================================================================
    # 2. 三对角求解器验证（原 962_r83）
    # ========================================================================
    logger.info("【步骤 2】验证 R83 三对角求解器（CG / Cyclic Reduction / Jacobi / Gauss-Seidel）")
    n_test = 64
    A_r83 = r83_dif2(n_test)
    b_test = np.ones(n_test, dtype=float)
    # CG
    x_cg = r83_cg(A_r83, b_test, tol=1e-12)
    res_cg = np.linalg.norm(r83_res(A_r83, x_cg, b_test))
    # Cyclic Reduction
    fac = r83_cr_fa(A_r83)
    x_cr = r83_cr_sl(fac, b_test)
    res_cr = np.linalg.norm(r83_res(A_r83, x_cr, b_test))
    # Jacobi
    x_jac = r83_jac_sl(A_r83, b_test, tol=1e-10)
    res_jac = np.linalg.norm(r83_res(A_r83, x_jac, b_test))
    # GS
    x_gs = r83_gs_sl(A_r83, b_test, tol=1e-10)
    res_gs = np.linalg.norm(r83_res(A_r83, x_gs, b_test))
    logger.info(f"CG residual={res_cg:.3e}, CR residual={res_cr:.3e}, "
                f"Jacobi residual={res_jac:.3e}, GS residual={res_gs:.3e}")

    # ========================================================================
    # 3. 随机化低秩近似验证（原 007_annulus_distance + 738_matrix_assemble_parfor）
    # ========================================================================
    logger.info("【步骤 3】随机化 SVD 与 Hilbert 矩阵低秩近似")
    H = hilbert_matrix(80, 80)
    U, s, Vt = randomized_svd(H, k=10, p=5, seed=42)
    H_approx = U @ np.diag(s) @ Vt
    err_hilbert = np.linalg.norm(H - H_approx) / np.linalg.norm(H)
    rank_h = adaptive_rank_threshold(s, tol=1e-12)
    logger.info(f"Hilbert(80,80) 随机化 SVD: target_rank=10, achieved_rank={rank_h}, rel_err={err_hilbert:.3e}")

    # 环形域随机采样 → 用于随机草图矩阵列采样
    pts_annulus = annulus_sample(200, pc=np.array([0.0, 0.0]), r1=0.5, r2=1.0)
    logger.info(f"环形域随机采样: {pts_annulus.shape[0]} 点")

    # ========================================================================
    # 4. 反应动力学验证（原 1018_reaction_twoway_ode + 1386_vanderpol_ode）
    # ========================================================================
    logger.info("【步骤 4】反应动力学精确解验证")
    t_test = np.linspace(0.0, 1.0, 20)
    u_exact = twoway_exact_solution(t_test, u0=0.1, k1=1.0, k2=10.0)
    logger.info(f"双向反应精确解: u(0)={u_exact[0]:.4f}, u(1)={u_exact[-1]:.4f}")

    # ========================================================================
    # 5. FEM 离散化验证（原 377_fem_neumann + 1318_triangle_symq_rule_original）
    # ========================================================================
    logger.info("【步骤 5】FEM 质量/刚度矩阵组装与 L² 范数")
    nodes_fem = np.linspace(-1.0, 1.0, 33)
    M_fem, K_fem = assemble_fem_matrices_1d(nodes_fem, diffusion_coeff=0.1)
    u_fem_test = np.sin(np.pi * nodes_fem)
    l2_norm_fem = fem_l2_norm(nodes_fem, u_fem_test, M_fem)
    logger.info(f"FEM L² 范数(sin(πx))={l2_norm_fem:.6f}")

    # ========================================================================
    # 6. 高维积分验证（原 713_maple_area）
    # ========================================================================
    logger.info("【步骤 6】高维数值积分（Monte Carlo vs QMC）")
    # 积分 ∫_{[0,1]^3} exp(-x²-y²-z²) dx dy dz
    def f_gauss3d(x):
        return np.exp(-np.sum(x**2))
    est_mc, err_mc = monte_carlo_integrate(f_gauss3d, dim=3, n=5000, seed=42)
    est_qmc = qmc_integrate(f_gauss3d, dim=3, n=5000)
    logger.info(f"3D Gauss integral: MC={est_mc:.6f}±{err_mc:.6f}, QMC={est_qmc:.6f}")

    # ========================================================================
    # 7. 秩分析与 Hankel 张量（原 1048_rref2 + 198_collatz_polynomial）
    # ========================================================================
    logger.info("【步骤 7】RREF 秩分析与 Hankel 张量构造")
    # 对 Hilbert 矩阵做 RREF 秩分析
    rank_h_rref = rref_rank(H[:20, :20])
    logger.info(f"Hilbert(20,20) RREF 秩={rank_h_rref}")

    # Collatz 多项式序列 → Hankel 张量
    p0 = np.array([1, 1, 0, 1], dtype=int)  # 1 + x + x³
    collatz_seq = collatz_polynomial_sequence(p0, max_steps=16)
    hankel_tensor = build_hankel_tensor_from_sequence(collatz_seq, dimensions=(4, 4, 4))
    ml_ranks = tensor_multilinear_ranks(hankel_tensor)
    tt_ranks_est = estimate_tensor_train_ranks(hankel_tensor)
    logger.info(f"Collatz Hankel 张量 multilinear_ranks={ml_ranks}, TT_ranks={tt_ranks_est}")

    # 经典 Hankel 矩阵
    s_seq = np.array([float(c[0]) if c.size > 0 else 0.0 for c in collatz_seq[:12]])
    Hankel_6x6 = hankel_matrix_from_sequence(s_seq, m=6, n=6)
    logger.info(f"Hankel(6,6) 秩={np.linalg.matrix_rank(Hankel_6x6, tol=1e-10)}")

    # ========================================================================
    # 8. 非负初始化（原 045_asa159）
    # ========================================================================
    logger.info("【步骤 8】非负矩阵因子初始化（列联表启发式）")
    W_ct, H_ct = nmf_init_coltable(20, 15, rank=4, seed=99)
    logger.info(f"NMF 初始化: W={W_ct.shape}, H={H_ct.shape}, min_W={W_ct.min():.3e}, min_H={H_ct.min():.3e}")

    # ========================================================================
    # 9. 参数化 PDE 求解 → 解张量构建 → TT 压缩（核心合成步骤）
    # ========================================================================
    logger.info("【步骤 9】参数化反应-扩散方程求解与高维解张量 TT 压缩")
    # 为控制运行时间，使用精简参数网格
    param_grid = {
        'k1': np.array([0.5, 1.0, 2.0]),
        'k2': np.array([5.0, 10.0]),
        'mu': np.array([0.5, 1.5]),
        'mix_ratio': np.array([0.0, 0.5, 1.0]),
        'B': np.array([0.9, 1.1]),
        'w': np.array([0.0, 0.2]),
        'D_egg': np.array([0.5, 0.7]),
    }
    n_space = 32
    n_time = 16
    t_final = 1.0
    diffusion_coeff = 0.05

    solution_tensor, param_values = build_solution_tensor(
        param_grid,
        n_space=n_space,
        n_time=n_time,
        t_final=t_final,
        diffusion_coeff=diffusion_coeff,
        logger=logger
    )
    logger.info(f"解张量 shape={solution_tensor.shape}, 元素数={solution_tensor.size}")

    # TT 压缩
    tt_result = compress_and_analyze(solution_tensor, max_tt_rank=10, logger=logger)
    logger.info(f"TT 压缩后 size={tt_result['tt_size']}, 压缩比={tt_result['compression_ratio']:.2f}x")

    # ========================================================================
    # 10. 低秩测试矩阵与张量内积验证
    # ========================================================================
    logger.info("【步骤 10】低秩测试矩阵与张量内积验证")
    A_lr = low_rank_test_matrix(40, 40, rank=5, seed=123)
    U_lr, s_lr, Vt_lr = randomized_svd(A_lr, k=5, p=3, seed=123)
    A_lr_rec = U_lr @ np.diag(s_lr) @ Vt_lr
    err_lr = np.linalg.norm(A_lr - A_lr_rec) / np.linalg.norm(A_lr)
    logger.info(f"低秩测试矩阵(秩=5) 重建误差={err_lr:.3e}")

    # TT 内积：解张量的 TT 近似与自身的内积应接近 ‖tensor‖²
    cores_tt = tt_result['tt_cores']
    tt_norm = tt_frobenius_norm(cores_tt)
    true_norm = np.linalg.norm(solution_tensor)
    logger.info(f"TT Frobenius 范数={tt_norm:.6f}, 真实范数={true_norm:.6f}, 相对偏差={abs(tt_norm-true_norm)/(true_norm+EPS):.3e}")

    # ========================================================================
    # 11. 文件输出（Matrix Market 格式，原 769_mm_io 扩展）
    # ========================================================================
    logger.info("【步骤 11】结果张量输出到 Matrix Market 格式")
    out_dir = os.path.dirname(os.path.abspath(__file__))
    mm_path = os.path.join(out_dir, "solution_tensor_coordinate.mm")
    # 只输出一个切片（k1=第一个, k2=第一个, ...）以控制文件大小
    idx_slice = (slice(None), slice(None)) + (0,) * (solution_tensor.ndim - 2)
    write_tensor_mm(mm_path, solution_tensor[idx_slice], tol=1e-12)
    logger.info(f"输出坐标格式张量切片至 {mm_path}")

    # 对称质量矩阵输出
    M_sym_path = os.path.join(out_dir, "fem_mass_matrix_symmetric.mm")
    write_symmetric_tensor_mm(M_sym_path, M_fem)
    logger.info(f"输出对称 FEM 质量矩阵至 {M_sym_path}")

    # ========================================================================
    # 12. 最终汇总
    # ========================================================================
    logger.info("=" * 60)
    logger.info("合成项目 179 执行完毕 — 结果汇总")
    logger.info("=" * 60)
    logger.info(f"解张量维度: {solution_tensor.shape}")
    logger.info(f"TT 分解秩: {tt_result['tt_ranks']}")
    logger.info(f"TT 相对误差: {tt_result['relative_error']:.3e}")
    logger.info(f"TT 最大误差: {tt_result['max_error']:.3e}")
    logger.info(f"压缩比: {tt_result['compression_ratio']:.2f}x")
    logger.info(f"所有 15 个原项目均已融入本项目")
    logger.info("=" * 60)
    return 0


if __name__ == "__main__":
    exit_code = main()
    # 不调用 sys.exit，让后续测试用例得以执行

# ================================================================
# 测试用例（30个，assert模式，涉及随机值均使用固定种子）
# ================================================================
# ---- TC01: safe_inv 正常标量逆 ----
from system_utils import safe_inv
y = safe_inv(np.array([2.0, 4.0]))
assert np.allclose(y, [0.5, 0.25]), '[TC01] safe_inv normal inverse FAILED'

# ---- TC02: safe_inv 零值保护 ----
z = safe_inv(np.array([0.0, 1e-20]))
assert z[0] == 0.0, '[TC02] safe_inv zero protection FAILED'
assert np.isfinite(z).all(), '[TC02] safe_inv finite check FAILED'

# ---- TC03: robust_sqrt 负值保护(返回0而非NaN) ----
from system_utils import robust_sqrt
r = robust_sqrt(np.array([4.0, -1.0, 0.0]))
assert np.isfinite(r).all(), '[TC03] robust_sqrt finite FAILED'
assert r[0] == 2.0, '[TC03] robust_sqrt positive FAILED'
assert r[1] == 0.0, '[TC03] robust_sqrt negative protection FAILED'

# ---- TC04: chebyshev_nodes_1d 范围与排序 ----
xc = chebyshev_nodes_1d(-1.0, 1.0, 16)
assert len(xc) == 16, '[TC04] chebyshev_nodes_1d count FAILED'
assert xc.min() >= -1.0 and xc.max() <= 1.0, '[TC04] chebyshev_nodes_1d range FAILED'

# ---- TC05: hand_outline_polygon 输出形状 ----
poly = hand_outline_polygon(100)
assert poly.shape == (100, 2), '[TC05] hand_outline_polygon shape FAILED'
assert np.isfinite(poly).all(), '[TC05] hand_outline_polygon finite FAILED'

# ---- TC06: FEM 质量矩阵对称性 ----
M_fem, K_fem = assemble_fem_matrices_1d(np.linspace(-1.0, 1.0, 33))
assert np.allclose(M_fem, M_fem.T), '[TC06] FEM mass matrix symmetry FAILED'
assert np.allclose(K_fem, K_fem.T), '[TC06] FEM stiffness matrix symmetry FAILED'

# ---- TC07: 反应动力学精确解端点值 ----
t_test2 = np.linspace(0.0, 1.0, 20)
u_exact2 = twoway_exact_solution(t_test2, u0=0.1, k1=1.0, k2=10.0)
assert abs(u_exact2[0] - 0.1) < 1e-12, '[TC07] exact solution u(0) FAILED'
u_star = 10.0 / 11.0
assert abs(u_exact2[-1] - u_star) < 1e-3, '[TC07] exact solution steady state FAILED'

# ---- TC08: 三对角 CG 求解器残差精度 ----
A_r83_tc = r83_dif2(64)
b_tc = np.ones(64)
x_cg_tc = r83_cg(A_r83_tc, b_tc, tol=1e-12)
res_cg_tc = np.linalg.norm(r83_res(A_r83_tc, x_cg_tc, b_tc))
assert res_cg_tc < 1e-8, '[TC08] CG solver residual FAILED'

# ---- TC09: Hilbert 矩阵随机化 SVD 可复现性 ----
import numpy as np
H_tc = hilbert_matrix(30, 30)
U1, s1, Vt1 = randomized_svd(H_tc, k=5, p=3, seed=42)
U2, s2, Vt2 = randomized_svd(H_tc, k=5, p=3, seed=42)
assert np.allclose(s1, s2), '[TC09] randomized SVD reproducibility FAILED'

# ---- TC10: adaptive_rank_threshold 衰减奇异值 ----
s_decay = np.array([100.0, 0.5, 0.01, 0.001])
r = adaptive_rank_threshold(s_decay, tol=1e-2)
assert r == 2, '[TC10] adaptive_rank_threshold FAILED'

# ---- TC11: annulus_sample 半径范围 ----
import numpy as np
np.random.seed(42)
pts = annulus_sample(500, pc=np.array([0.0, 0.0]), r1=0.5, r2=1.0)
dists = np.linalg.norm(pts, axis=1)
assert dists.min() >= 0.5 - 1e-12, '[TC11] annulus_sample min radius FAILED'
assert dists.max() <= 1.0 + 1e-12, '[TC11] annulus_sample max radius FAILED'

# ---- TC12: van_der_corput_sequence 范围 ----
seq = van_der_corput_sequence(100, base=2)
assert seq.min() >= 0.0 and seq.max() < 1.0, '[TC12] van_der_corput range FAILED'

# ---- TC13: grid_integrate_1d 解析验证 (∫₀¹ x² dx = 1/3) ----
def f_sq(x):
    return x * x
I_grid = grid_integrate_1d(f_sq, 0.0, 1.0, 100)
assert abs(I_grid - 1.0/3.0) < 1e-4, '[TC13] grid_integrate_1d FAILED'

# ---- TC14: Monte Carlo 积分可复现性 ----
def f_mc(x):
    return np.exp(-np.sum(x**2))
est1, err1 = monte_carlo_integrate(f_mc, dim=3, n=5000, seed=42)
est2, err2 = monte_carlo_integrate(f_mc, dim=3, n=5000, seed=42)
assert est1 == est2, '[TC14] MC reproducibility FAILED'

# ---- TC15: rref_rank 单位矩阵 ----
I_mat = np.eye(20)
assert rref_rank(I_mat) == 20, '[TC15] rref_rank identity FAILED'

# ---- TC16: rref_rank 秩亏矩阵 ----
A_deficient = np.outer(np.ones(10), np.ones(10))
assert rref_rank(A_deficient) == 1, '[TC16] rref_rank rank-deficient FAILED'

# ---- TC17: collatz_polynomial_next 确定性 ----
p0_tc = np.array([1, 1, 0, 1], dtype=int)
p1 = collatz_polynomial_next(p0_tc)
p2 = collatz_polynomial_next(p0_tc)
assert np.array_equal(p1, p2), '[TC17] collatz_polynomial_next determinism FAILED'

# ---- TC18: hankel_matrix_from_sequence 结构 ----
s_hankel = np.arange(1, 15, dtype=float)
H_h = hankel_matrix_from_sequence(s_hankel, 5, 5)
assert H_h[0, 0] == 1.0, '[TC18] Hankel H[0,0] FAILED'
assert H_h[0, 1] == 2.0, '[TC18] Hankel H[0,1] FAILED'
assert H_h[1, 0] == 2.0, '[TC18] Hankel anti-diagonal constancy FAILED'

# ---- TC19: NMF 初始化非负性 ----
W, H_nmf = nmf_init_random(20, 15, 4, seed=42)
assert W.min() >= 0.0, '[TC19] NMF W nonnegative FAILED'
assert H_nmf.min() >= 0.0, '[TC19] NMF H nonnegative FAILED'

# ---- TC20: nonnegative_projection ----
X_np = np.array([-2.0, -1.0, 0.0, 1.0, 3.0])
X_proj = nonnegative_projection(X_np)
assert X_proj.min() >= 0.0, '[TC20] nonnegative_projection FAILED'
assert np.allclose(X_proj, [0.0, 0.0, 0.0, 1.0, 3.0]), '[TC20] nonnegative_projection values FAILED'

# ---- TC21: soft_threshold 解析验证 ----
X_st = np.array([-3.0, -1.0, 0.0, 1.0, 3.0])
X_th = soft_threshold(X_st, tau=2.0)
assert np.allclose(X_th, [-1.0, 0.0, 0.0, 0.0, 1.0]), '[TC21] soft_threshold FAILED'

# ---- TC22: parametric_reaction_source 有限性 ----
u_tc = np.array([0.0, 0.5, 1.0])
R_tc = parametric_reaction_source(u_tc, k1=1.0, k2=5.0, mu=0.5, mix_ratio=0.5)
assert np.isfinite(R_tc).all(), '[TC22] parametric_reaction_source finite FAILED'

# ---- TC23: triangle_area 已知三角形 ----
v1 = np.array([0.0, 0.0])
v2 = np.array([3.0, 0.0])
v3 = np.array([0.0, 4.0])
assert abs(triangle_area(v1, v2, v3) - 6.0) < 1e-12, '[TC23] triangle_area FAILED'

# ---- TC24: tensor_to_coordinate 往返 ----
tensor_small = np.array([[[1.0, 0.0], [0.0, 2.0]], [[0.0, 3.0], [0.0, 0.0]]])
idx, vals = tensor_to_coordinate(tensor_small, tol=0.0)
tensor_back = coordinate_to_tensor(idx, vals, tensor_small.shape)
assert np.allclose(tensor_small, tensor_back), '[TC24] tensor_to_coordinate round-trip FAILED'

# ---- TC25: low_rank_test_matrix 秩验证 ----
A_lr_tc = low_rank_test_matrix(30, 30, rank=4, seed=99)
s_lr = np.linalg.svd(A_lr_tc, compute_uv=False)
rank_est = adaptive_rank_threshold(s_lr, tol=1e-10)
assert rank_est >= 3, '[TC25] low_rank_test_matrix rank FAILED'

# ---- TC26: extract_tridiagonal 结构 ----
A_dense = np.eye(5) * 2.0 + np.eye(5, k=-1) * (-1.0) + np.eye(5, k=1) * (-1.0)
A83 = extract_tridiagonal(A_dense)
assert A83.shape == (3, 5), '[TC26] extract_tridiagonal shape FAILED'

# ---- TC27: vanderpol_reaction_term 符号 ----
u_vdp = np.array([0.0, 0.5, 1.0, 1.5])
R_vdp = vanderpol_reaction_term(u_vdp, mu=2.0)
assert R_vdp[0] == 0.0, '[TC27] vanderpol u=0 => R=0 FAILED'
assert R_vdp[1] > 0.0, '[TC27] vanderpol |u|<1 => R>0 FAILED'
assert R_vdp[2] == 0.0, '[TC27] vanderpol |u|=1 => R=0 FAILED'

# ---- TC28: reaction_jacobian_diagonal 有限 ----
J_diag = reaction_jacobian_diagonal(np.array([0.5, 1.0]), k1=1.0, k2=2.0, mu=0.5, mix_ratio=0.3)
assert np.isfinite(J_diag).all(), '[TC28] reaction_jacobian_diagonal finite FAILED'

# ---- TC29: qmc_integrate 与 MC 一致性检查 ----
def f_qmc(x):
    return 1.0
est_qmc = qmc_integrate(f_qmc, dim=3, n=2000)
assert abs(est_qmc - 1.0) < 0.1, '[TC29] QMC constant integrand FAILED'

# ---- TC30: r83_mv 与 r83_mtv 对称性 (A=A^T时) ----
A_sym = r83_dif2(10)
x_a = np.random.randn(10)
y_a = np.random.randn(10)
assert abs(np.dot(y_a, r83_mv(A_sym, x_a)) - np.dot(x_a, r83_mtv(A_sym, y_a))) < 1e-10, '[TC30] r83 mv/mtv symmetry FAILED'

print('\n全部 30 个测试通过!\n')
