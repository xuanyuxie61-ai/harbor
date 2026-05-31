# -*- coding: utf-8 -*-
"""
main.py
圆柱壳轴向压缩屈曲与后屈曲路径跟踪 — 统一入口

科学问题:
  研究薄壁圆柱壳在轴向压力下的线性屈曲、非线性后屈曲路径及
  初始几何缺陷敏感性。采用 Donnell-Mushtari-Vlasov 壳理论、
  有限元离散化、Newton-Raphson 迭代与弧长法路径跟踪。

运行方式:
  python main.py
"""

import numpy as np
import sys
import os

# 确保模块路径正确
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

    # =====================================================================
    # 1. 几何与材料参数定义
    # =====================================================================
    print("\n[1] 定义圆柱壳几何与材料参数")
    R = 0.25          # 半径 (m)
    L = 0.50          # 长度 (m)
    t = 0.001         # 厚度 (m)
    E = 70.0e9        # 杨氏模量 (Pa), 铝合金
    nu = 0.33         # 泊松比
    rho = 2700.0      # 密度 (kg/m³)

    geom = CylindricalShellGeometry(R, L, t)
    mat = ShellMaterial(E, nu, rho)
    print(f"    半径 R = {R:.4f} m, 长度 L = {L:.4f} m, 厚度 t = {t:.4f} m")
    print(f"    长径比 L/R = {geom.aspect_ratio():.2f}")
    print(f"    Batdorf 参数 Z = {geom.batdorf_parameter(E, nu):.2f}")
    print(f"    材料: E = {E/1e9:.1f} GPa, ν = {nu:.2f}")

    # =====================================================================
    # 2. 网格生成与质量评估 (469_geompack + 1239_tet_mesh_tet_neighbors)
    # =====================================================================
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

    # =====================================================================
    # 3. 有限元模型组装 (417_fem3d_pack)
    # =====================================================================
    print("\n[3] 组装有限元模型")
    fem = ShellFEModel(mesh, mat)
    K_lin = fem.assemble_linear_stiffness()
    print(f"    总自由度: {fem.n_dof}")
    ml, mu, bw = coo_bandwidth(K_lin)
    print(f"    刚度矩阵带宽: ML={ml}, MU={mu}, M={bw}")

    # =====================================================================
    # 4. 线性屈曲分析 (081_besselzero)
    # =====================================================================
    print("\n[4] 线性屈曲特征值分析")
    buckling = LinearBucklingAnalyzer(geom, mat)
    Ncr_classical = buckling.analytical_buckling_load()
    print(f"    经典解析屈曲载荷 N_cr = {Ncr_classical/1e6:.4f} MN/m")

    N_min, m_opt, n_opt, modes = buckling.buckling_modes_discrete(m_max=8, n_max=12)
    print(f"    离散搜索最小屈曲载荷 N_min = {N_min/1e6:.4f} MN/m")
    print(f"    最优模态: 轴向半波数 m = {m_opt}, 环向波数 n = {n_opt}")

    # Bessel 零点验证
    zeros = buckling.bessel_verification(n_circumferential=n_opt, n_zeros=5)
    print(f"    Bessel J_{n_opt} 前5个零点: " + ", ".join([f"{z:.4f}" for z in zeros]))

    # =====================================================================
    # 5. 外载荷定义 (轴向均匀压力)
    # =====================================================================
    print("\n[5] 定义轴向压缩载荷")
    # 均匀轴向压力分布到节点
    f_ext = np.zeros(fem.n_dof)
    # 仅顶部节点 (x=L) 承受压力
    bottom, top = mesh.get_boundary_nodes()
    area_per_top_node = (2.0 * np.pi * R * t) / max(len(top), 1)
    for nid in top:
        f_ext[nid * 3 + 2] = -Ncr_classical * area_per_top_node  # 法向压力
    f_ext_norm = np.linalg.norm(f_ext)
    print(f"    参考外载荷范数: {f_ext_norm:.4e} N")

    # =====================================================================
    # 6. 线性静力分析 (验证模型)
    # =====================================================================
    print("\n[6] 线性静力分析")
    from scipy.sparse.linalg import spsolve
    # 边界条件: 底部固支, 顶部限制面内位移
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

    # =====================================================================
    # 7. 非线性求解验证 (Newton-Raphson) (1286_trapezoidal + 908_predator_prey_ode)
    # =====================================================================
    print("\n[7] 非线性 Newton-Raphson 求解 (λ = 0.5)")
    nr_solver = NewtonRaphsonSolver(max_iter=30, tol_force=1e-5, tol_disp=1e-7)
    res_nr = nr_solver.solve(fem, f_ext, lambda_load=0.5, u0=u_lin * 0.5)
    print(f"    收敛: {res_nr['converged']}, 迭代次数: {res_nr['iterations']}")
    print(f"    残余范数: {res_nr['residual_norm']:.4e}")
    print(f"    势能: {res_nr['energy']:.4e} J")

    # =====================================================================
    # 8. 弧长法后屈曲路径跟踪 (171_chirikov_iteration + 199_collatz_recursive)
    # =====================================================================
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

    # Chirikov 稳定性指标
    ch_ind = tracker.chirikov_stability_indicator(path)
    if len(ch_ind) > 0:
        n_chaotic = int(np.sum(ch_ind))
        print(f"    Chirikov 混沌指标: {n_chaotic}/{len(ch_ind)} 步处于混沌区域")

    # =====================================================================
    # 9. 稳定性分析 (171_chirikov_iteration)
    # =====================================================================
    print("\n[9] 后屈曲稳定性分析")
    stab = StabilityAnalyzer(fem)

    # 对路径最后一步进行特征值分析
    if len(path) > 1:
        u_final = path[-1]['disp']
        eig_res = stab.tangent_stiffness_eigenvalues(u_final, k=3)
        print(f"    最小特征值: {eig_res['min_eig']:.4e}")
        print(f"    稳定状态: {eig_res['stable']}")

    # Lyapunov 指数
    lyap = stab.lyapunov_exponent_discrete(path)
    print(f"    离散 Lyapunov 指数: {lyap:.4e}")
    if lyap > 0.01:
        print("    提示: Lyapunov 指数为正，路径呈现局部指数发散特征")

    # Koiter 分岔分类
    bif_class = stab.koiter_bifurcation_class(path)
    print(f"    Koiter 分岔类型: {bif_class}")

    energy_bar = stab.energy_barrier(path)
    print(f"    能量势垒估计: {energy_bar:.4f}")

    # =====================================================================
    # 10. 初始缺陷生成与分析 (069_ball_monte_carlo + 314_double_c_data)
    # =====================================================================
    print("\n[10] 几何缺陷生成与敏感性分析")
    defect_gen = DefectGenerator(geom, seed=86)

    # 单模态缺陷
    single_defect = defect_gen.single_mode_defect(m=m_opt, n=n_opt,
                                                   amplitude=0.01 * t)
    stats_single = defect_gen.defect_statistics(mesh, single_defect)
    print(f"    单模态缺陷: δ_max/t = {stats_single['defect_to_thickness']:.4f}, "
          f"RMS = {stats_single['rms_defect']:.4e} m")

    # 双C型缺陷
    double_c_defect = defect_gen.double_c_defect(n1=n_opt, n2=n_opt + 2,
                                                  amplitude=0.01 * t)
    stats_dc = defect_gen.defect_statistics(mesh, double_c_defect)
    print(f"    双C型缺陷: δ_max/t = {stats_dc['defect_to_thickness']:.4f}, "
          f"RMS = {stats_dc['rms_defect']:.4e} m")

    # Monte Carlo 多模态缺陷
    mc_defect = defect_gen.monte_carlo_multi_mode(n_modes=8, amplitude_ratio=0.01)
    stats_mc = defect_gen.defect_statistics(mesh, mc_defect)
    print(f"    MC多模态缺陷: δ_max/t = {stats_mc['defect_to_thickness']:.4f}, "
          f"RMS = {stats_mc['rms_defect']:.4e} m")

    # Koiter 缺陷敏感性
    reduction = buckling.imperfection_sensitivity_koiter(
        imperfection_amplitude=0.01 * t, imperfection_mode=n_opt)
    print(f"    Koiter 载荷降低因子 (δ/t=0.01): {reduction:.4f}")
    print(f"    预测实际屈曲载荷: {reduction * Ncr_classical / 1e6:.4f} MN/m")

    # =====================================================================
    # 11. 稀疏矩阵格式转换与输出 (771_mm_to_msm + 1157_st_to_hb + 1158_st_to_mm)
    # =====================================================================
    print("\n[11] 稀疏矩阵格式转换")
    mm_str = coo_to_mm(K_lin.tocoo())
    lines = mm_str.strip().splitlines()
    print(f"    Matrix Market 格式: {len(lines)} 行")
    print(f"    头信息: {lines[0]}")
    print(f"    维度: {lines[1]}")

    # 矩阵轮廓缩减演示
    perm = matrix_profile_reduction(K_lin.tocoo())
    print(f"    RCM 重排列长度: {len(perm)}")
    print(f"    原始半带宽: {bw}, 预期缩减后: 显著降低")

    # =====================================================================
    # 12. 伪时间步进动态松弛验证 (908_predator_prey_ode)
    # =====================================================================
    print("\n[12] 伪时间动态松弛求解 (λ = 0.3)")
    pt_solver = PseudoTimeSolver(damping_ratio=0.92, dt=0.005, max_steps=400)
    res_pt = pt_solver.solve(fem, f_ext, lambda_load=0.3)
    print(f"    收敛: {res_pt['converged']}, 步数: {res_pt['steps']}")
    print(f"    残余范数: {res_pt['residual_norm']:.4e}")
    if res_pt['converged']:
        max_w_pt = np.max(np.abs(res_pt['disp'][2::3])) if fem.n_dof >= 3 else 0.0
        print(f"    最大法向位移: {max_w_pt:.6e} m")

    # =====================================================================
    # 总结
    # =====================================================================
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
    main()

# ================================================================
# 测试用例（40个，assert模式，涉及随机值均使用固定种子）
# ================================================================

# ---- TC01: CylindricalShellGeometry 参数化表面输出形状为 (2,2,3) ----
geom_test = CylindricalShellGeometry(0.25, 0.50, 0.001)
theta_grid = np.array([[0.0, np.pi/2], [np.pi, 3*np.pi/2]])
x_grid = np.array([[0.0, 0.0], [0.5, 0.5]])
surf = geom_test.parametric_surface(theta_grid, x_grid)
assert surf.shape == (2, 2, 3), '[TC01] 参数化表面输出形状 FAILED'

# ---- TC02: CylindricalShellGeometry 测地距离对称性 ----
geom_test = CylindricalShellGeometry(0.25, 0.50, 0.001)
p1 = np.array([0.25, 0.0, 0.0])
p2 = np.array([0.0, 0.25, 0.5])
d12 = geom_test.geodesic_distance(p1, p2)
d21 = geom_test.geodesic_distance(p2, p1)
assert abs(d12 - d21) < 1e-12, '[TC02] 测地距离对称性 FAILED'

# ---- TC03: CylindricalShellGeometry 第一基本形式解析验证 ----
geom_test = CylindricalShellGeometry(0.25, 0.50, 0.001)
E, F, G = geom_test.first_fundamental_form()
assert abs(E - 0.0625) < 1e-12 and F == 0.0 and G == 1.0, '[TC03] 第一基本形式 FAILED'

# ---- TC04: CylindricalShellGeometry 主曲率解析验证 ----
geom_test = CylindricalShellGeometry(0.25, 0.50, 0.001)
k1, k2 = geom_test.principal_curvatures()
assert abs(k1 - 4.0) < 1e-12 and k2 == 0.0, '[TC04] 主曲率 FAILED'

# ---- TC05: CylindricalShellGeometry 中曲面面积解析验证 ----
geom_test = CylindricalShellGeometry(0.25, 0.50, 0.001)
A = geom_test.surface_area()
assert abs(A - 2.0 * np.pi * 0.25 * 0.50) < 1e-12, '[TC05] 中曲面面积 FAILED'

# ---- TC06: CylindricalShellGeometry Batdorf 参数为正有限值 ----
geom_test = CylindricalShellGeometry(0.25, 0.50, 0.001)
Z = geom_test.batdorf_parameter(70e9, 0.33)
assert Z > 0.0 and np.isfinite(Z), '[TC06] Batdorf 参数范围 FAILED'

# ---- TC07: ShellTriMesh 节点数和单元数正确 ----
geom_test = CylindricalShellGeometry(0.25, 0.50, 0.001)
mesh_test = ShellTriMesh(4, 3, geom_test)
assert mesh_test.n_nodes == 12 and mesh_test.n_elem == 16, '[TC07] 节点数和单元数 FAILED'

# ---- TC08: ShellTriMesh 单元面积均为正 ----
geom_test = CylindricalShellGeometry(0.25, 0.50, 0.001)
mesh_test = ShellTriMesh(4, 3, geom_test)
areas = [mesh_test.element_area(eid) for eid in range(mesh_test.n_elem)]
assert all(a > 0.0 for a in areas), '[TC08] 单元面积均为正 FAILED'

# ---- TC09: ShellTriMesh alpha_measure 范围在 [0,1] ----
geom_test = CylindricalShellGeometry(0.25, 0.50, 0.001)
mesh_test = ShellTriMesh(4, 3, geom_test)
amin, aave, aarea = mesh_test.alpha_measure()
assert 0.0 <= amin <= 1.0 and 0.0 <= aave <= 1.0 and 0.0 <= aarea <= 1.0, '[TC09] alpha_measure 范围 FAILED'

# ---- TC10: ShellTriMesh 边界节点数量 ----
geom_test = CylindricalShellGeometry(0.25, 0.50, 0.001)
mesh_test = ShellTriMesh(4, 3, geom_test)
bottom, top = mesh_test.get_boundary_nodes()
assert len(bottom) == 4 and len(top) == 4, '[TC10] 边界节点数量 FAILED'

# ---- TC11: ShellMaterial 拉伸刚度解析验证 ----
mat_test = ShellMaterial(70e9, 0.33, 2700.0)
C = mat_test.extensional_rigidity(0.001)
expected_C = 70e9 * 0.001 / (1.0 - 0.33**2)
assert abs(C - expected_C) < 1e-3, '[TC11] 拉伸刚度 FAILED'

# ---- TC12: ShellMaterial 弯曲刚度解析验证 ----
mat_test = ShellMaterial(70e9, 0.33, 2700.0)
D = mat_test.bending_rigidity(0.001)
expected_D = 70e9 * 0.001**3 / (12.0 * (1.0 - 0.33**2))
assert abs(D - expected_D) < 1e-9, '[TC12] 弯曲刚度 FAILED'

# ---- TC13: ShellMaterial 膜矩阵对称正定 ----
mat_test = ShellMaterial(70e9, 0.33, 2700.0)
Cm = mat_test.membrane_matrix(0.001)
assert np.allclose(Cm, Cm.T) and np.all(np.linalg.eigvalsh(Cm) > 0), '[TC13] 膜矩阵对称正定 FAILED'

# ---- TC14: ShellFEModel 线性刚度矩阵尺寸正确 ----
geom_test = CylindricalShellGeometry(0.25, 0.50, 0.001)
mesh_test = ShellTriMesh(4, 3, geom_test)
mat_test = ShellMaterial(70e9, 0.33, 2700.0)
fem_test = ShellFEModel(mesh_test, mat_test)
K = fem_test.assemble_linear_stiffness()
assert K.shape == (36, 36), '[TC14] 线性刚度矩阵尺寸 FAILED'

# ---- TC15: ShellFEModel 零位移内力为零 ----
geom_test = CylindricalShellGeometry(0.25, 0.50, 0.001)
mesh_test = ShellTriMesh(4, 3, geom_test)
mat_test = ShellMaterial(70e9, 0.33, 2700.0)
fem_test = ShellFEModel(mesh_test, mat_test)
u0 = np.zeros(fem_test.n_dof)
fint = fem_test.internal_force(u0)
assert np.linalg.norm(fint) < 1e-12, '[TC15] 零位移内力 FAILED'

# ---- TC16: LinearBucklingAnalyzer 解析屈曲载荷解析验证 ----
geom_test = CylindricalShellGeometry(0.25, 0.50, 0.001)
mat_test = ShellMaterial(70e9, 0.33, 2700.0)
buckling = LinearBucklingAnalyzer(geom_test, mat_test)
Ncr = buckling.analytical_buckling_load()
expected_Ncr = 70e9 * 0.001**2 / (0.25 * np.sqrt(3.0 * (1.0 - 0.33**2)))
assert abs(Ncr - expected_Ncr) < 1e-3, '[TC16] 解析屈曲载荷 FAILED'

# ---- TC17: LinearBucklingAnalyzer 离散搜索返回有限值 ----
geom_test = CylindricalShellGeometry(0.25, 0.50, 0.001)
mat_test = ShellMaterial(70e9, 0.33, 2700.0)
buckling = LinearBucklingAnalyzer(geom_test, mat_test)
N_min, m_opt, n_opt, modes = buckling.buckling_modes_discrete(m_max=5, n_max=5)
assert np.isfinite(N_min) and m_opt >= 1 and n_opt >= 0, '[TC17] 离散搜索 FAILED'

# ---- TC18: LinearBucklingAnalyzer Koiter 零缺陷敏感性为 1 ----
geom_test = CylindricalShellGeometry(0.25, 0.50, 0.001)
mat_test = ShellMaterial(70e9, 0.33, 2700.0)
buckling = LinearBucklingAnalyzer(geom_test, mat_test)
ratio = buckling.imperfection_sensitivity_koiter(0.0, 5)
assert abs(ratio - 1.0) < 1e-12, '[TC18] Koiter 零缺陷敏感性 FAILED'

# ---- TC19: bessel_zero_halley J0 第一个零点接近 2.4048 ----
from linear_buckling import bessel_zero_halley
z1 = bessel_zero_halley(0.0, 1, kind=1)
assert abs(z1 - 2.404825557685772) < 1e-10, '[TC19] Bessel J0 第一个零点 FAILED'

# ---- TC20: bessel_zeros_vector 返回单调递增向量 ----
zeros = bessel_zeros_vector(1.0, 4, kind=1)
assert len(zeros) == 4 and all(np.diff(zeros) > 0), '[TC20] Bessel 零点向量 FAILED'

# ---- TC21: DefectGenerator 单模态缺陷边界处为零 ----
geom_test = CylindricalShellGeometry(0.25, 0.50, 0.001)
defect_gen = DefectGenerator(geom_test, seed=42)
func = defect_gen.single_mode_defect(m=1, n=4, amplitude=0.001)
theta = np.array([0.0, np.pi/2, np.pi])
x = np.array([0.0, 0.25, 0.50])
w = func(theta, x)
assert abs(w[0]) < 1e-14, '[TC21] 单模态缺陷边界 FAILED'

# ---- TC22: DefectGenerator 缺陷统计返回正确键 ----
geom_test = CylindricalShellGeometry(0.25, 0.50, 0.001)
mesh_test = ShellTriMesh(4, 3, geom_test)
defect_gen = DefectGenerator(geom_test, seed=42)
func = defect_gen.single_mode_defect(m=1, n=2, amplitude=0.001)
stats = defect_gen.defect_statistics(mesh_test, func)
assert set(stats.keys()) == {'max_defect', 'rms_defect', 'defect_to_thickness', 'mean_defect'}, '[TC22] 缺陷统计键 FAILED'

# ---- TC23: sparse_matrix_io coo_to_mm 输出格式正确 ----
from scipy.sparse import coo_matrix
row = np.array([0, 1, 2])
col = np.array([1, 2, 0])
data = np.array([1.0, 2.0, 3.0])
coo = coo_matrix((data, (row, col)), shape=(3, 3))
mm_str = coo_to_mm(coo)
lines = mm_str.strip().splitlines()
assert lines[0].startswith('%%MatrixMarket') and '3 3 3' in lines[1], '[TC23] MM 格式输出 FAILED'

# ---- TC24: sparse_matrix_io coo_bandwidth 返回正数 ----
from scipy.sparse import coo_matrix
row = np.array([0, 1, 2, 2])
col = np.array([0, 1, 2, 0])
data = np.array([1.0, 2.0, 3.0, 4.0])
coo = coo_matrix((data, (row, col)), shape=(3, 3))
ml, mu, bw = coo_bandwidth(coo)
assert ml >= 0 and mu >= 0 and bw > 0, '[TC24] 带宽计算 FAILED'

# ---- TC25: sparse_matrix_io matrix_profile_reduction 返回排列 ----
from scipy.sparse import coo_matrix
row = np.array([0, 1, 2, 0])
col = np.array([0, 1, 2, 2])
data = np.array([1.0, 2.0, 3.0, 4.0])
coo = coo_matrix((data, (row, col)), shape=(3, 3))
perm = matrix_profile_reduction(coo)
assert len(perm) == 3 and set(perm) == {0, 1, 2}, '[TC25] RCM 排列 FAILED'

# ---- TC26: NewtonRaphsonSolver 零载荷立即收敛 ----
geom_test = CylindricalShellGeometry(0.25, 0.50, 0.001)
mesh_test = ShellTriMesh(4, 3, geom_test)
mat_test = ShellMaterial(70e9, 0.33, 2700.0)
fem_test = ShellFEModel(mesh_test, mat_test)
f_ext = np.zeros(fem_test.n_dof)
nr = NewtonRaphsonSolver(max_iter=5, tol_force=1e-6, tol_disp=1e-8)
res = nr.solve(fem_test, f_ext, lambda_load=0.0)
assert res['converged'] and res['iterations'] <= 1, '[TC26] Newton 零载荷收敛 FAILED'

# ---- TC27: StabilityAnalyzer Koiter 分岔分类对称稳定 ----
stab = StabilityAnalyzer(None)
path = [{'lambda': 0.0, 'max_disp': 0.0}, {'lambda': 0.1, 'max_disp': 1.0}, {'lambda': 0.2, 'max_disp': 2.0}]
cls = stab.koiter_bifurcation_class(path)
assert cls == 'symmetric-stable', '[TC27] Koiter 分岔分类 FAILED'

# ---- TC28: StabilityAnalyzer Lyapunov 短路径返回零 ----
stab = StabilityAnalyzer(None)
path = [{'lambda': 0.0, 'max_disp': 0.0}, {'lambda': 0.1, 'max_disp': 1.0}]
lyap = stab.lyapunov_exponent_discrete(path)
assert lyap == 0.0, '[TC28] Lyapunov 短路径 FAILED'

# ---- TC29: ArcLengthTracker Chirikov 短路径返回空数组 ----
tracker = ArcLengthTracker()
path_short = [{'lambda': 0.0, 'max_disp': 0.0}, {'lambda': 0.1, 'max_disp': 1.0}]
ind = tracker.chirikov_stability_indicator(path_short)
assert len(ind) == 0, '[TC29] Chirikov 短路径 FAILED'

# ---- TC30: ArcLengthTracker _compute_psi 返回非负有限值 ----
tracker = ArcLengthTracker()
psi = tracker._compute_psi(np.array([1.0, 2.0]), 0.5, 10.0)
assert psi >= 0.0 and np.isfinite(psi), '[TC30] psi 非负有限 FAILED'

# ---- TC31: DefectGenerator 蒙特卡洛多模态固定种子可复现 ----
np.random.seed(42)
geom_test = CylindricalShellGeometry(0.25, 0.50, 0.001)
defect_gen1 = DefectGenerator(geom_test, seed=42)
func1 = defect_gen1.monte_carlo_multi_mode(n_modes=3, amplitude_ratio=0.01)
np.random.seed(42)
geom_test2 = CylindricalShellGeometry(0.25, 0.50, 0.001)
defect_gen2 = DefectGenerator(geom_test2, seed=42)
func2 = defect_gen2.monte_carlo_multi_mode(n_modes=3, amplitude_ratio=0.01)
theta = np.array([0.5, 1.0])
x = np.array([0.1, 0.2])
w1 = func1(theta, x)
w2 = func2(theta, x)
assert np.allclose(w1, w2), '[TC31] MC 缺陷可复现性 FAILED'

# ---- TC32: ShellTriMesh 所有单元内角和为 pi ----
geom_test = CylindricalShellGeometry(0.25, 0.50, 0.001)
mesh_test = ShellTriMesh(4, 3, geom_test)
sum_errors = sum(abs(np.sum(mesh_test.element_angles(eid)) - np.pi) for eid in range(mesh_test.n_elem))
assert sum_errors < 1e-10, '[TC32] 单元内角和 FAILED'

# ---- TC33: PseudoTimeSolver 返回结果包含必要键 ----
geom_test = CylindricalShellGeometry(0.25, 0.50, 0.001)
mesh_test = ShellTriMesh(4, 3, geom_test)
mat_test = ShellMaterial(70e9, 0.33, 2700.0)
fem_test = ShellFEModel(mesh_test, mat_test)
f_ext = np.zeros(fem_test.n_dof)
pt = PseudoTimeSolver(damping_ratio=0.9, dt=0.01, max_steps=10)
res = pt.solve(fem_test, f_ext, lambda_load=0.0)
assert set(res.keys()) == {'disp', 'converged', 'steps', 'energy_history', 'residual_norm'}, '[TC33] 伪时间求解器返回键 FAILED'

# ---- TC34: CylindricalShellGeometry 边界排序返回正确长度 ----
geom_test = CylindricalShellGeometry(0.25, 0.50, 0.001)
mesh_test = ShellTriMesh(4, 3, geom_test)
sorted_idx = geom_test.boundary_sort(mesh_test.nodes)
assert len(sorted_idx) == 8, '[TC34] 边界排序长度 FAILED'

# ---- TC35: ShellFEModel 零位移几何刚度对称 ----
geom_test = CylindricalShellGeometry(0.25, 0.50, 0.001)
mesh_test = ShellTriMesh(4, 3, geom_test)
mat_test = ShellMaterial(70e9, 0.33, 2700.0)
fem_test = ShellFEModel(mesh_test, mat_test)
u0 = np.zeros(fem_test.n_dof)
Kg = fem_test.assemble_geometric_stiffness(u0)
Kg_dense = Kg.toarray()
assert np.allclose(Kg_dense, Kg_dense.T), '[TC35] 几何刚度对称性 FAILED'

# ---- TC36: LinearBucklingAnalyzer Bessel 验证返回单调零点 ----
geom_test = CylindricalShellGeometry(0.25, 0.50, 0.001)
mat_test = ShellMaterial(70e9, 0.33, 2700.0)
buckling = LinearBucklingAnalyzer(geom_test, mat_test)
zeros = buckling.bessel_verification(n_circumferential=3, n_zeros=5)
assert len(zeros) == 5 and all(z > 0 for z in zeros), '[TC36] Bessel 验证 FAILED'

# ---- TC37: ShellMaterial 弯曲矩阵对称正定 ----
mat_test = ShellMaterial(70e9, 0.33, 2700.0)
Cb = mat_test.bending_matrix(0.001)
assert np.allclose(Cb, Cb.T) and np.all(np.linalg.eigvalsh(Cb) > 0), '[TC37] 弯曲矩阵对称正定 FAILED'

# ---- TC38: CylindricalShellGeometry 测地距离同一点为零 ----
geom_test = CylindricalShellGeometry(0.25, 0.50, 0.001)
p = np.array([0.25, 0.0, 0.25])
d = geom_test.geodesic_distance(p, p)
assert abs(d) < 1e-12, '[TC38] 测地距离同一点 FAILED'

# ---- TC39: ArcLengthTracker track_path 返回结构正确 ----
geom_test = CylindricalShellGeometry(0.25, 0.50, 0.001)
mesh_test = ShellTriMesh(4, 3, geom_test)
mat_test = ShellMaterial(70e9, 0.33, 2700.0)
fem_test = ShellFEModel(mesh_test, mat_test)
f_ext = np.zeros(fem_test.n_dof)
tracker = ArcLengthTracker(initial_arc_length=0.01, min_arc_length=1e-5, max_arc_length=0.3, adaptivity=0.5)
result = tracker.track_path(fem_test, f_ext, n_steps=2, lambda_max=1.5)
assert set(result.keys()) == {'path', 'bifurcation_points', 'n_steps'}, '[TC39] 路径跟踪返回键 FAILED'

# ---- TC40: StabilityAnalyzer 能量势垒非负 ----
stab = StabilityAnalyzer(None)
path = [{'lambda': 0.0, 'max_disp': 0.0}, {'lambda': 0.2, 'max_disp': 2.0}, {'lambda': 0.1, 'max_disp': 1.0}]
eb = stab.energy_barrier(path)
assert eb >= 0.0, '[TC40] 能量势垒非负 FAILED'

print('\n全部 40 个测试通过!\n')
