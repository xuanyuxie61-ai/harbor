#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import os
import sys




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



    logger = initialize_project()




    logger.info("【步骤 1】生成参数化计算域与低秩形状分析")
    n_shape = 100
    hand_xy, hand_pcs = hand_ellipse_fourier_approx(n_shape)
    logger.info(f"手部轮廓 SVD 主成分 shape={hand_pcs.shape}")


    L_egg = 2.0
    x_egg = chebyshev_nodes_1d(-L_egg / 2, L_egg / 2, 32)
    r_egg = universal_egg_half_profile(B=1.0, L=L_egg, w=0.1, D=0.6, x=x_egg)
    logger.info(f"蛋形域 Chebyshev 节点数={len(x_egg)}, max_radius={r_egg.max():.4f}")




    logger.info("【步骤 2】验证 R83 三对角求解器（CG / Cyclic Reduction / Jacobi / Gauss-Seidel）")
    n_test = 64
    A_r83 = r83_dif2(n_test)
    b_test = np.ones(n_test, dtype=float)

    x_cg = r83_cg(A_r83, b_test, tol=1e-12)
    res_cg = np.linalg.norm(r83_res(A_r83, x_cg, b_test))

    fac = r83_cr_fa(A_r83)
    x_cr = r83_cr_sl(fac, b_test)
    res_cr = np.linalg.norm(r83_res(A_r83, x_cr, b_test))

    x_jac = r83_jac_sl(A_r83, b_test, tol=1e-10)
    res_jac = np.linalg.norm(r83_res(A_r83, x_jac, b_test))

    x_gs = r83_gs_sl(A_r83, b_test, tol=1e-10)
    res_gs = np.linalg.norm(r83_res(A_r83, x_gs, b_test))
    logger.info(f"CG residual={res_cg:.3e}, CR residual={res_cr:.3e}, "
                f"Jacobi residual={res_jac:.3e}, GS residual={res_gs:.3e}")




    logger.info("【步骤 3】随机化 SVD 与 Hilbert 矩阵低秩近似")
    H = hilbert_matrix(80, 80)
    U, s, Vt = randomized_svd(H, k=10, p=5, seed=42)
    H_approx = U @ np.diag(s) @ Vt
    err_hilbert = np.linalg.norm(H - H_approx) / np.linalg.norm(H)
    rank_h = adaptive_rank_threshold(s, tol=1e-12)
    logger.info(f"Hilbert(80,80) 随机化 SVD: target_rank=10, achieved_rank={rank_h}, rel_err={err_hilbert:.3e}")


    pts_annulus = annulus_sample(200, pc=np.array([0.0, 0.0]), r1=0.5, r2=1.0)
    logger.info(f"环形域随机采样: {pts_annulus.shape[0]} 点")




    logger.info("【步骤 4】反应动力学精确解验证")
    t_test = np.linspace(0.0, 1.0, 20)
    u_exact = twoway_exact_solution(t_test, u0=0.1, k1=1.0, k2=10.0)
    logger.info(f"双向反应精确解: u(0)={u_exact[0]:.4f}, u(1)={u_exact[-1]:.4f}")




    logger.info("【步骤 5】FEM 质量/刚度矩阵组装与 L² 范数")
    nodes_fem = np.linspace(-1.0, 1.0, 33)
    M_fem, K_fem = assemble_fem_matrices_1d(nodes_fem, diffusion_coeff=0.1)
    u_fem_test = np.sin(np.pi * nodes_fem)
    l2_norm_fem = fem_l2_norm(nodes_fem, u_fem_test, M_fem)
    logger.info(f"FEM L² 范数(sin(πx))={l2_norm_fem:.6f}")




    logger.info("【步骤 6】高维数值积分（Monte Carlo vs QMC）")

    def f_gauss3d(x):
        return np.exp(-np.sum(x**2))
    est_mc, err_mc = monte_carlo_integrate(f_gauss3d, dim=3, n=5000, seed=42)
    est_qmc = qmc_integrate(f_gauss3d, dim=3, n=5000)
    logger.info(f"3D Gauss integral: MC={est_mc:.6f}±{err_mc:.6f}, QMC={est_qmc:.6f}")




    logger.info("【步骤 7】RREF 秩分析与 Hankel 张量构造")

    rank_h_rref = rref_rank(H[:20, :20])
    logger.info(f"Hilbert(20,20) RREF 秩={rank_h_rref}")


    p0 = np.array([1, 1, 0, 1], dtype=int)
    collatz_seq = collatz_polynomial_sequence(p0, max_steps=16)
    hankel_tensor = build_hankel_tensor_from_sequence(collatz_seq, dimensions=(4, 4, 4))
    ml_ranks = tensor_multilinear_ranks(hankel_tensor)
    tt_ranks_est = estimate_tensor_train_ranks(hankel_tensor)
    logger.info(f"Collatz Hankel 张量 multilinear_ranks={ml_ranks}, TT_ranks={tt_ranks_est}")


    s_seq = np.array([float(c[0]) if c.size > 0 else 0.0 for c in collatz_seq[:12]])
    Hankel_6x6 = hankel_matrix_from_sequence(s_seq, m=6, n=6)
    logger.info(f"Hankel(6,6) 秩={np.linalg.matrix_rank(Hankel_6x6, tol=1e-10)}")




    logger.info("【步骤 8】非负矩阵因子初始化（列联表启发式）")
    W_ct, H_ct = nmf_init_coltable(20, 15, rank=4, seed=99)
    logger.info(f"NMF 初始化: W={W_ct.shape}, H={H_ct.shape}, min_W={W_ct.min():.3e}, min_H={H_ct.min():.3e}")




    logger.info("【步骤 9】参数化反应-扩散方程求解与高维解张量 TT 压缩")

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


    tt_result = compress_and_analyze(solution_tensor, max_tt_rank=10, logger=logger)
    logger.info(f"TT 压缩后 size={tt_result['tt_size']}, 压缩比={tt_result['compression_ratio']:.2f}x")




    logger.info("【步骤 10】低秩测试矩阵与张量内积验证")
    A_lr = low_rank_test_matrix(40, 40, rank=5, seed=123)
    U_lr, s_lr, Vt_lr = randomized_svd(A_lr, k=5, p=3, seed=123)
    A_lr_rec = U_lr @ np.diag(s_lr) @ Vt_lr
    err_lr = np.linalg.norm(A_lr - A_lr_rec) / np.linalg.norm(A_lr)
    logger.info(f"低秩测试矩阵(秩=5) 重建误差={err_lr:.3e}")


    cores_tt = tt_result['tt_cores']
    tt_norm = tt_frobenius_norm(cores_tt)
    true_norm = np.linalg.norm(solution_tensor)
    logger.info(f"TT Frobenius 范数={tt_norm:.6f}, 真实范数={true_norm:.6f}, 相对偏差={abs(tt_norm-true_norm)/(true_norm+EPS):.3e}")




    logger.info("【步骤 11】结果张量输出到 Matrix Market 格式")
    out_dir = os.path.dirname(os.path.abspath(__file__))
    mm_path = os.path.join(out_dir, "solution_tensor_coordinate.mm")

    idx_slice = (slice(None), slice(None)) + (0,) * (solution_tensor.ndim - 2)
    write_tensor_mm(mm_path, solution_tensor[idx_slice], tol=1e-12)
    logger.info(f"输出坐标格式张量切片至 {mm_path}")


    M_sym_path = os.path.join(out_dir, "fem_mass_matrix_symmetric.mm")
    write_symmetric_tensor_mm(M_sym_path, M_fem)
    logger.info(f"输出对称 FEM 质量矩阵至 {M_sym_path}")




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
    sys.exit(main())
