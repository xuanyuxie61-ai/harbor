# -*- coding: utf-8 -*-

import numpy as np
import sys
import os


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from shell_geometry import CylindricalShellGeometry
from mesh_triangulation import ShellTriMesh
from shell_fem_element import ShellMaterial, ShellFEModel
from sparse_matrix_io import coo_bandwidth, matrix_profile_reduction, coo_to_mm
from linear_buckling import LinearBucklingAnalyzer, bessel_zeros_vector
from nonlinear_solver import NewtonRaphsonSolver, PseudoTimeSolver
from arc_length_tracker import ArcLengthTracker
from defect_generator import DefectGenerator
from stability_analysis import StabilityAnalyzer


def main():
    print("=" * 72)
    print("圆柱壳屈曲与后屈曲路径跟踪 — 博士级科研计算程序")
    print("=" * 72)




    print("\n[1] 定义圆柱壳几何与材料参数")
    R = 0.25
    L = 0.50
    t = 0.001
    E = 70.0e9
    nu = 0.33
    rho = 2700.0

    geom = CylindricalShellGeometry(R, L, t)
    mat = ShellMaterial(E, nu, rho)
    print(f"    半径 R = {R:.4f} m, 长度 L = {L:.4f} m, 厚度 t = {t:.4f} m")
    print(f"    长径比 L/R = {geom.aspect_ratio():.2f}")
    print(f"    Batdorf 参数 Z = {geom.batdorf_parameter(E, nu):.2f}")
    print(f"    材料: E = {E/1e9:.1f} GPa, ν = {nu:.2f}")




    print("\n[2] 生成三角网格并评估质量")
    n_theta = 16
    n_x = 12
    mesh = ShellTriMesh(n_theta, n_x, geom)
    print(f"    节点数: {mesh.n_nodes}, 单元数: {mesh.n_elem}")
    alpha_min, alpha_ave, alpha_area = mesh.alpha_measure()
    print(f"    网格质量 α_min = {alpha_min:.4f}, α_ave = {alpha_ave:.4f}, α_area = {alpha_area:.4f}")
    if alpha_min < 0.3:
        print("    警告: 网格质量偏低，执行 Delaunay 翻转优化")
        n_flip = mesh.delaunay_flip(max_iter=5)
        print(f"    翻转次数: {n_flip}")
        alpha_min, alpha_ave, alpha_area = mesh.alpha_measure()
        print(f"    优化后 α_min = {alpha_min:.4f}")




    print("\n[3] 组装有限元模型")
    fem = ShellFEModel(mesh, mat)
    K_lin = fem.assemble_linear_stiffness()
    print(f"    总自由度: {fem.n_dof}")
    ml, mu, bw = coo_bandwidth(K_lin)
    print(f"    刚度矩阵带宽: ML={ml}, MU={mu}, M={bw}")




    print("\n[4] 线性屈曲特征值分析")
    buckling = LinearBucklingAnalyzer(geom, mat)
    Ncr_classical = buckling.analytical_buckling_load()
    print(f"    经典解析屈曲载荷 N_cr = {Ncr_classical/1e6:.4f} MN/m")

    N_min, m_opt, n_opt, modes = buckling.buckling_modes_discrete(m_max=8, n_max=12)
    print(f"    离散搜索最小屈曲载荷 N_min = {N_min/1e6:.4f} MN/m")
    print(f"    最优模态: 轴向半波数 m = {m_opt}, 环向波数 n = {n_opt}")


    zeros = buckling.bessel_verification(n_circumferential=n_opt, n_zeros=5)
    print(f"    Bessel J_{n_opt} 前5个零点: " + ", ".join([f"{z:.4f}" for z in zeros]))




    print("\n[5] 定义轴向压缩载荷")

    f_ext = np.zeros(fem.n_dof)

    bottom, top = mesh.get_boundary_nodes()
    area_per_top_node = (2.0 * np.pi * R * t) / max(len(top), 1)
    for nid in top:
        f_ext[nid * 3 + 2] = -Ncr_classical * area_per_top_node
    f_ext_norm = np.linalg.norm(f_ext)
    print(f"    参考外载荷范数: {f_ext_norm:.4e} N")




    print("\n[6] 线性静力分析")
    from scipy.sparse.linalg import spsolve

    fixed_dofs = []
    for nid in bottom:
        fixed_dofs.extend([nid * 3 + 0, nid * 3 + 1, nid * 3 + 2])
    for nid in top:
        fixed_dofs.extend([nid * 3 + 0, nid * 3 + 1])
    if len(bottom) > 0:
        fixed_dofs.append(bottom[0] * 3 + 2)
    fixed_dofs = np.unique(fixed_dofs)
    free_dofs = np.setdiff1d(np.arange(fem.n_dof), fixed_dofs)

    K0_ff = K_lin[free_dofs][:, free_dofs]
    try:
        u_lin_f = spsolve(K0_ff, f_ext[free_dofs])
    except Exception:
        u_lin_f = spsolve(K0_ff + 1e-8 * np.eye(len(free_dofs), format='csr'), f_ext[free_dofs])
    u_lin = np.zeros(fem.n_dof)
    u_lin[free_dofs] = u_lin_f
    max_w_lin = np.max(np.abs(u_lin[2::3])) if fem.n_dof >= 3 else 0.0
    print(f"    线性最大法向位移: {max_w_lin:.6e} m")




    print("\n[7] 非线性 Newton-Raphson 求解 (λ = 0.5)")
    nr_solver = NewtonRaphsonSolver(max_iter=30, tol_force=1e-5, tol_disp=1e-7)
    res_nr = nr_solver.solve(fem, f_ext, lambda_load=0.5, u0=u_lin * 0.5)
    print(f"    收敛: {res_nr['converged']}, 迭代次数: {res_nr['iterations']}")
    print(f"    残余范数: {res_nr['residual_norm']:.4e}")
    print(f"    势能: {res_nr['energy']:.4e} J")




    print("\n[8] 弧长法后屈曲路径跟踪")
    tracker = ArcLengthTracker(initial_arc_length=0.01, min_arc_length=1e-5,
                               max_arc_length=0.3, adaptivity=0.5)
    path_result = tracker.track_path(fem, f_ext, n_steps=15, lambda_max=1.5)
    path = path_result['path']
    print(f"    成功跟踪 {len(path)-1} 个载荷步")
    for i, p in enumerate(path):
        marker = ""
        if i > 0:
            for bp in path_result['bifurcation_points']:
                if bp['step'] == i:
                    marker = "  <-- 分岔点检测"
        print(f"    Step {i}: λ = {p['lambda']:.4f}, w_max = {p['max_disp']:.4e}, "
              f"min_eig = {p['min_eig']:.4e}{marker}")

    if path_result['bifurcation_points']:
        print(f"    检测到 {len(path_result['bifurcation_points'])} 个分岔/极限点")


    ch_ind = tracker.chirikov_stability_indicator(path)
    if len(ch_ind) > 0:
        n_chaotic = int(np.sum(ch_ind))
        print(f"    Chirikov 混沌指标: {n_chaotic}/{len(ch_ind)} 步处于混沌区域")




    print("\n[9] 后屈曲稳定性分析")
    stab = StabilityAnalyzer(fem)


    if len(path) > 1:
        u_final = path[-1]['disp']
        eig_res = stab.tangent_stiffness_eigenvalues(u_final, k=3)
        print(f"    最小特征值: {eig_res['min_eig']:.4e}")
        print(f"    稳定状态: {eig_res['stable']}")


    lyap = stab.lyapunov_exponent_discrete(path)
    print(f"    离散 Lyapunov 指数: {lyap:.4e}")
    if lyap > 0.01:
        print("    提示: Lyapunov 指数为正，路径呈现局部指数发散特征")


    bif_class = stab.koiter_bifurcation_class(path)
    print(f"    Koiter 分岔类型: {bif_class}")

    energy_bar = stab.energy_barrier(path)
    print(f"    能量势垒估计: {energy_bar:.4f}")




    print("\n[10] 几何缺陷生成与敏感性分析")
    defect_gen = DefectGenerator(geom, seed=86)


    single_defect = defect_gen.single_mode_defect(m=m_opt, n=n_opt,
                                                   amplitude=0.01 * t)
    stats_single = defect_gen.defect_statistics(mesh, single_defect)
    print(f"    单模态缺陷: δ_max/t = {stats_single['defect_to_thickness']:.4f}, "
          f"RMS = {stats_single['rms_defect']:.4e} m")


    double_c_defect = defect_gen.double_c_defect(n1=n_opt, n2=n_opt + 2,
                                                  amplitude=0.01 * t)
    stats_dc = defect_gen.defect_statistics(mesh, double_c_defect)
    print(f"    双C型缺陷: δ_max/t = {stats_dc['defect_to_thickness']:.4f}, "
          f"RMS = {stats_dc['rms_defect']:.4e} m")


    mc_defect = defect_gen.monte_carlo_multi_mode(n_modes=8, amplitude_ratio=0.01)
    stats_mc = defect_gen.defect_statistics(mesh, mc_defect)
    print(f"    MC多模态缺陷: δ_max/t = {stats_mc['defect_to_thickness']:.4f}, "
          f"RMS = {stats_mc['rms_defect']:.4e} m")


    reduction = buckling.imperfection_sensitivity_koiter(
        imperfection_amplitude=0.01 * t, imperfection_mode=n_opt)
    print(f"    Koiter 载荷降低因子 (δ/t=0.01): {reduction:.4f}")
    print(f"    预测实际屈曲载荷: {reduction * Ncr_classical / 1e6:.4f} MN/m")




    print("\n[11] 稀疏矩阵格式转换")
    mm_str = coo_to_mm(K_lin.tocoo())
    lines = mm_str.strip().splitlines()
    print(f"    Matrix Market 格式: {len(lines)} 行")
    print(f"    头信息: {lines[0]}")
    print(f"    维度: {lines[1]}")


    perm = matrix_profile_reduction(K_lin.tocoo())
    print(f"    RCM 重排列长度: {len(perm)}")
    print(f"    原始半带宽: {bw}, 预期缩减后: 显著降低")




    print("\n[12] 伪时间动态松弛求解 (λ = 0.3)")
    pt_solver = PseudoTimeSolver(damping_ratio=0.92, dt=0.005, max_steps=400)
    res_pt = pt_solver.solve(fem, f_ext, lambda_load=0.3)
    print(f"    收敛: {res_pt['converged']}, 步数: {res_pt['steps']}")
    print(f"    残余范数: {res_pt['residual_norm']:.4e}")
    if res_pt['converged']:
        max_w_pt = np.max(np.abs(res_pt['disp'][2::3])) if fem.n_dof >= 3 else 0.0
        print(f"    最大法向位移: {max_w_pt:.6e} m")




    print("\n" + "=" * 72)
    print("计算完成。核心结果汇总:")
    print(f"  - 经典屈曲载荷: {Ncr_classical/1e6:.4f} MN/m")
    print(f"  - 最优屈曲模态: (m={m_opt}, n={n_opt})")
    print(f"  - 路径跟踪步数: {len(path)-1}")
    print(f"  - 分岔点检测: {len(path_result['bifurcation_points'])}")
    print(f"  - Koiter 分岔类型: {bif_class}")
    print(f"  - 缺陷敏感性 (δ/t=1%): 载荷降低至 {reduction*100:.1f}%")
    print("=" * 72)

    return 0


if __name__ == "__main__":
    sys.exit(main())
