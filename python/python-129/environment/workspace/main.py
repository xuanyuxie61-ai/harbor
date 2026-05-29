"""
main.py
血凝级联反应网络多尺度动力学建模的统一入口。

本项目融合15个种子项目的核心算法，构建一个面向
"生物医学：血凝级联反应网络"的博士级计算框架。

运行方式:
    python main.py

无需任何命令行参数，程序将自动执行完整的多尺度模拟流程。
"""

import sys
import numpy as np

# 导入所有子模块
from special_functions import (
    bessel_jx_fractional,
    modified_bessel_clot_profile,
    verify_bessel_accuracy
)
from coagulation_ode import (
    CoagulationNetwork,
    trapezoidal_solve
)
from reaction_diffusion_pde import (
    AnnularDiffusionSolver,
    wound_source_term
)
from network_pagerank import CoagulationNetworkGraph
from monte_carlo_platelet import PlateletMonteCarlo
from quadrature_integrals import (
    HexagonQuadrature,
    TriangleBarycentricQuadrature,
    arrhenius_rate_integral
)
from svd_sensitivity import ParameterSensitivitySVD
from stability_analysis import StabilityAnalyzer
from gray_code_genetics import GrayCodeGenetics
from variomino_clot import VariominoClot
from clustering_phenotypes import SwapClustering, generate_coagulation_phenotypes


def print_section(title):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def run_special_functions_module():
    """模块1: 基于079_besselj的特殊函数验证与clot径向分布"""
    print_section("模块1: 特殊函数与Clot径向分布 (基于079_besselj)")
    err = verify_bessel_accuracy()
    print(f"Bessel Jx 数值精度验证: 最大误差 = {err:.3e}")

    r = np.linspace(0, 50, 100)
    profile = modified_bessel_clot_profile(r, r0=30.0, D_fibrin=2.5e-2, k_poly=1.2e-3)
    print(f"修正Bessel clot分布: r=0 浓度比 = {profile[0]:.4f}, r=r0 = {profile[-1]:.4f}")
    return profile


def run_coagulation_ode_module():
    """模块2: 基于831_ode_trapezoidal与1170_stochastic_heat2d的血凝ODE求解"""
    # TODO: 修复 Hole 3 —— 血凝ODE模块调用与结果解析
    # 需要实现：创建CoagulationNetwork实例、设置12维初始条件向量、
    # 调用trapezoidal_solve求解、解析并打印关键结果，返回(t, y, net)
    pass


def run_reaction_diffusion_module():
    """模块3: 基于1170_stochastic_heat2d与011_annulus_rule的PDE求解"""
    print_section("模块3: 血管截面反应-扩散-源项模型 (基于1170_stochastic_heat2d + 011_annulus_rule)")
    solver = AnnularDiffusionSolver(
        r_inner=1.0, r_outer=50.0, nr=30, ntheta=24,
        D_coef=8.0e-3,
        source_func=lambda r, th, t, c: wound_source_term(r, th, t, c)
    )

    def reaction(c):
        return -0.01 * c / (c + 10.0 + 1e-12)

    c_steady = solver.solve_steady_state(reaction, max_iter=80)
    total = solver.integrate_over_annulus(c_steady)
    print(f"稳态凝血因子总量 (环形积分): {total:.4f}")
    print(f"最大浓度位置: r={solver.r[np.unravel_index(c_steady.argmax(), c_steady.shape)[0]]:.2f} μm")
    return solver, c_steady


def run_network_analysis_module():
    """模块4: 基于845_pagerank2与154_chain_letter_tree的网络分析"""
    print_section("模块4: 血凝级联网络拓扑分析 (基于845_pagerank2 + 154_chain_letter_tree)")
    graph = CoagulationNetworkGraph()
    pr = graph.pagerank()
    ranked = sorted(zip(graph.node_names, pr), key=lambda x: x[1], reverse=True)
    print("PageRank 排名 Top 5:")
    for name, score in ranked[:5]:
        print(f"  {name:20s}: {score:.6f}")

    dist = graph.jaccard_distance_matrix()
    linkage, clusters = graph.hierarchical_clustering(dist)
    print(f"层次聚类合并次数: {len(linkage)}")
    return graph, pr, dist


def run_monte_carlo_module():
    """模块5: 基于534_high_card_simulation的血小板最优停止模拟"""
    print_section("模块5: 血小板粘附最优停止蒙特卡洛 (基于534_high_card_simulation)")
    mc = PlateletMonteCarlo(n_platelets=400, local_iia=25.0, seed=42)
    best_skip, best_score, _ = mc.find_optimal_skip(n_trials=200)
    theoretical = int(round(400 / np.e))
    print(f"蒙特卡洛最优 skip = {best_skip} (理论 N/e = {theoretical})")
    print(f"最优 clot 稳定性评分: {best_score:.4f}")
    return mc


def run_quadrature_module():
    """模块6: 基于527_hexagon_integrals与1313_triangle_quadrature_symmetry的数值积分"""
    print_section("模块6: Clot几何数值积分与Arrhenius速率 (基于527_hexagon_integrals + 1313_triangle_quadrature_symmetry)")
    hex_q = HexagonQuadrature(radius=2.0)
    area_num = hex_q.monomial_integral(0, 0)
    print(f"正六边形面积: 解析 = {hex_q.area:.6f}, 数值 = {area_num:.6f}")
    print(f"Clot孔隙率估算: {hex_q.compute_clot_fiber_volume():.4f}")

    tri_q = TriangleBarycentricQuadrature()
    verts = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    val = tri_q.integrate_on_triangle(lambda x, y: x * y, verts, order=3)
    print(f"参考三角形 ∫xy = {val:.6f} (精确 1/24 = 0.041667)")

    k_a, k_n = arrhenius_rate_integral(T=310.0, A_pre=1e12, Ea_mean=5e4, Ea_sigma=3e3)
    print(f"Arrhenius 速率: 解析 = {k_a:.4e}, 数值 = {k_n:.4e}")
    return hex_q


def run_svd_module():
    """模块7: 基于1190_svd_powers的参数敏感性SVD分析"""
    print_section("模块7: 参数敏感性SVD降维分析 (基于1190_svd_powers)")
    param_names = [
        "k_cat_TF_VIIa_IX", "k_cat_IXa_X", "k_cat_Xa_II",
        "k_inact_ATIII", "k_TFPI_inact", "k_APC_inact_Va",
        "k_polymerization", "k_fibrin_lysis", "k_PLT_act",
        "D_fibrin", "k_clear", "k_plasminogen_act"
    ]
    response_names = ["Peak_IIa", "TFT_50_percent", "Fibrin_yield", "Clot_stability"]

    analyzer = ParameterSensitivitySVD(param_names, response_names)

    def mock_model(p):
        r1 = 100.0 * p[2] / (p[3] + 0.1)
        r2 = 60.0 / (p[0] + 0.01) + 20.0 * p[4]
        r3 = 50.0 * p[6] / (p[7] + 0.1)
        r4 = 80.0 * p[6] - 40.0 * p[7] + 10.0 * p[8]
        return np.array([r1, r2, r3, r4])

    p0 = np.array([1.2, 6.5, 25.0, 0.003, 0.015, 0.05,
                   0.5, 0.008, 0.1, 0.025, 0.001, 0.02])

    S = analyzer.compute_sensitivity_matrix(mock_model, p0, delta_frac=0.01)
    U, sigma, Vt, cum_var = analyzer.svd_decompose(S)
    print(f"前3个奇异值: {sigma[:3]}")
    print(f"前3个主成分累积方差: {cum_var[:3]}")

    directions = analyzer.principal_parameter_directions(Vt, n_components=2)
    for d in directions:
        print(f"主成分 {d['component']} 主导参数: {d['top_params'][0][0]}, {d['top_params'][1][0]}")
    return analyzer, sigma


def run_stability_module(net):
    """模块8: 基于105_boundary_locus2的数值稳定性分析"""
    # TODO: 修复 Hole 4 —— ODE稳定性分析模块调用与结果解析
    # 需要实现：创建StabilityAnalyzer、在典型状态点计算Jacobian、
    # 调用analyze_jacobian_stiffness分析刚性、打印并返回结果
    pass


def run_genetics_module():
    """模块9: 基于485_gray_code_display的基因多态性编码"""
    print_section("模块9: 凝血因子基因Gray码编码 (基于485_gray_code_display)")
    gc = GrayCodeGenetics()
    weights = np.array([3.0, 2.5, 1.0, 2.0, 1.5, 4.0, 3.5, 3.0] + [0.0] * 8)

    patients = [
        ("正常", False, False, False, False, False, False, False, False),
        ("F5 Leiden", True, False, False, False, False, False, False, False),
        ("复合突变", True, True, True, False, False, False, False, False),
    ]
    genotypes = []
    for name, *flags in patients:
        bits = gc.encode_coagulation_genotype(*flags)
        risk = gc.genetic_risk_score(bits, weights)
        genotypes.append(bits)
        print(f"{name:12s}: 风险评分 = {risk:.2f}")

    genotypes = np.array(genotypes)
    dist = gc.binary_distance_matrix(genotypes)
    print(f"患者间Hamming距离矩阵:\n{dist}")
    return gc


def run_variomino_module():
    """模块10: 基于1389_variomino的clot结构分析"""
    print_section("模块10: 纤维蛋白Clot多格结构分析 (基于1389_variomino)")
    P = VariominoClot.generate_random_clot(height=35, width=35, n_fibers=10, seed=42)
    clot = VariominoClot(P)
    Q = clot.condense()
    print(f"原始矩阵: {clot.P.shape}, 凝聚后: {Q.shape}")
    n_pores, _ = clot.count_pore_clusters(threshold=0.15)
    print(f"孔隙连通区域数: {n_pores}")
    ani = clot.compute_anisotropy()
    print(f"结构各向异性指数: {ani:.4f}")
    return clot


def run_clustering_module():
    """模块11: 基于039_asa113的患者表型聚类"""
    print_section("模块11: 血凝表型交换优化聚类 (基于039_asa113)")
    X, true_labels = generate_coagulation_phenotypes(n_samples=120, seed=42)
    X_norm = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-12)
    clusterer = SwapClustering(n_clusters=3, max_iter=80)
    labels, centroids, wcss, n_swaps = clusterer.fit(X_norm)
    print(f"WCSS = {wcss:.4f}, 交换次数 = {n_swaps}")

    from itertools import permutations
    best_acc = 0.0
    for perm in permutations(range(3)):
        mapped = np.array([perm[l] for l in labels])
        acc = np.mean(mapped == true_labels)
        best_acc = max(best_acc, acc)
    print(f"聚类准确率: {best_acc:.1%}")
    return clusterer


def main():
    """
    统一入口：执行所有模块的完整模拟流程。
    """
    print("\n" + "#" * 70)
    print("#  血凝级联反应网络多尺度动力学建模系统")
    print("#  生物医学前沿博士级计算框架")
    print("#" * 70)
    print("\n开始执行多尺度模拟流程...")
    print("包含模块: 特殊函数 | ODE求解 | PDE扩散 | 网络分析 | 蒙特卡洛")
    print("          数值积分 | SVD敏感 | 稳定性 | 基因编码 | 结构分析 | 聚类")

    np.set_printoptions(precision=4, suppress=True)

    # 顺序执行各模块
    try:
        run_special_functions_module()
    except Exception as e:
        print(f"[模块1警告] {e}")

    try:
        t, y, net = run_coagulation_ode_module()
    except Exception as e:
        print(f"[模块2警告] {e}")
        t, y, net = None, None, None

    try:
        run_reaction_diffusion_module()
    except Exception as e:
        print(f"[模块3警告] {e}")

    try:
        run_network_analysis_module()
    except Exception as e:
        print(f"[模块4警告] {e}")

    try:
        run_monte_carlo_module()
    except Exception as e:
        print(f"[模块5警告] {e}")

    try:
        run_quadrature_module()
    except Exception as e:
        print(f"[模块6警告] {e}")

    try:
        run_svd_module()
    except Exception as e:
        print(f"[模块7警告] {e}")

    if net is not None:
        try:
            run_stability_module(net)
        except Exception as e:
            print(f"[模块8警告] {e}")
    else:
        print("[模块8跳过] 网络模型未初始化")

    try:
        run_genetics_module()
    except Exception as e:
        print(f"[模块9警告] {e}")

    try:
        run_variomino_module()
    except Exception as e:
        print(f"[模块10警告] {e}")

    try:
        run_clustering_module()
    except Exception as e:
        print(f"[模块11警告] {e}")

    print("\n" + "#" * 70)
    print("#  模拟流程全部完成")
    print("#" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
