# -*- coding: utf-8 -*-
"""
main.py — 二维材料异质结能带对齐: 高阶有限差分与稳定性分析
==============================================================
统一入口, 零参数运行.

科学问题:
  对于 MoS2/WSe2 等二维材料异质结, 求解有效质量薛定谔方程:
    Hψ = Eψ
    H = -(ℏ²/2) d/dz[1/m*(z) d/dz] + V(z)

  其中 V(z) 包含带偏移、应变、极化等贡献.

  使用高阶有限差分离散化, 分析数值稳定性, 计算:
    1. 子带能级和波函数
    2. 态密度
    3. 隧穿概率 (Feynman-Kac)
    4. 布里渊区积分
    5. 能带插值和有效质量
    6. 自洽 Poisson-Schrödinger

融合种子项目:
  1. 638_lagrange_nd → lagrange_nd_hetband.py (多维 Lagrange 能带面)
  2. 424_feynman_kac_3d → feynman_kac_band_edge.py (F-K 隧穿)
  3. 664_legendre_product_polynomial → legendre_basis_2d.py (Legendre 谱)
  4. 992_r8ri → sparse_hetband.py (稀疏矩阵)
  5. 539_histogram_discrete → dos_histogram.py (直方图 DOS)
  6. 757_mesh2d → adaptive_mesh.py (自适应网格)
  7. 1350_triangulation_refine → adaptive_mesh.py (网格细分)
  8. 542_histogram_pdf_2d_sample → band_sampling.py (2D 采样)
  9. 179_circle_integrals → brillouin_integral.py (圆积分)
  10. 1059_Glyphosate_crystallization → heterostructure_topology.py (拓扑)
  11. 375_fem_basis_t6_display → fem_basis_t6.py (T6 基函数)
  12. 072_barycentric_interp_1d → barycentric_interp.py (重心插值)
  13. 596_interp_trig → trig_interp.py (三角插值)
  14. 754_mesh_display → adaptive_mesh.py (网格操作)
  15. 626_knapsack_random → band_sampling.py (组合采样)
"""

import numpy as np
import sys
import os

# 将当前目录加入路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from material_parameters import (
    MOS2, WSE2, HBN, MOSE2, WS2, GASE,
    anderson_band_offset, lattice_mismatch, tunneling_effective_mass,
    heterostructure_database
)
from hetero_potential import HeteroPotential
from adaptive_mesh import AdaptiveMesh1D, HeteroStructureMesh
from high_order_fd import HighOrderFD, BenDanielDukeFD, PeriodicFD, HBAR, M0, E0
from sparse_hetband import HeteroHamiltonianAssembler, MultiBandHamiltonian
from feynman_kac_band_edge import (
    feynman_kac_tunneling, wkb_tunneling_estimate,
    resonant_tunneling_condition
)
from dos_histogram import DOSCalculator
from brillouin_integral import (
    BrillouinZoneIntegration, circle_monomial_integral, disk_monomial_integral
)
from barycentric_interp import BarycentricInterpolator, BandStructureInterpolator
from trig_interp import TrigonometricInterpolator, PeriodicPotentialExpansion
from lagrange_nd_hetband import LagrangeND, BandSurfaceInterpolator
from legendre_basis_2d import (
    LegendreSpectralBasis, legendre_value, legendre_product_polynomial
)
from fem_basis_t6 import T6Element
from band_sampling import DiscreteCDF2D, BandStructureSampler
from basis_selector import BasisSelector
from heterostructure_topology import (
    HeterostructureTopology, create_typical_heterostructure
)
from poisson_schrodinger import PoissonSchrodingerSolver
from stability_analysis import VonNeumannStability, DispersionAnalysis, ConvergenceStudy


def section_header(title):
    """打印节标题."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def run_material_analysis():
    """
    第 1 部分: 材料参数分析.
    使用 material_parameters.py, heterostructure_topology.py
    """
    section_header("第 1 部分: 材料参数与异质结拓扑")

    # 材料参数
    print("\n[材料参数]")
    for mat in [MOS2, WSE2, MOSE2, WS2, HBN, GASE]:
        eg300 = mat.bandgap_temperature(300.0)
        a_B = mat.bohr_radius_2d() * 1e10  # → Å
        E_bind = mat.exciton_binding_energy()
        print(f"  {mat.name:6s}: Eg(300K)={eg300:.3f}eV, "
              f"a_B={a_B:.2f}Å, E_bind={E_bind*1000:.1f}meV, "
              f"chi={mat.chi:.2f}eV")

    # 异质结带偏移
    print("\n[带偏移 (Anderson 规则)]")
    for name, (m1, m2) in heterostructure_database().items():
        dEc, dEv = anderson_band_offset(m1, m2, T=300.0)
        mismatch = lattice_mismatch(m1, m2)
        m_t = tunneling_effective_mass(m1, m2, 'electron')
        print(f"  {name:15s}: ΔEc={dEc:+.3f}eV, ΔEv={dEv:+.3f}eV, "
              f"失配={mismatch*100:+.2f}%, m_t={m_t:.4f}m0")

    # 异质结拓扑
    print("\n[异质结拓扑 (MoS2/WSe2)]")
    topo = create_typical_heterostructure('MoS2_WSe2')
    info = topo.to_dict()
    print(f"  总层数: {info['n_layers']}")
    print(f"  总厚度: {info['total_thickness']*1e9:.2f} nm")
    for i, layer in enumerate(info['layers']):
        print(f"    层 {i}: {layer['material']} ({layer['thickness']*1e9:.2f}nm, "
              f"{layer['strain_state']})")
    for i, iface in enumerate(info['interfaces']):
        print(f"    界面 {i}: z={iface['z']*1e9:.2f}nm, "
              f"失配={iface['mismatch']*100:.3f}%, {iface['type']}")

    return topo


def run_potential_and_mesh(topo):
    """
    第 2 部分: 势能剖面与自适应网格.
    使用 hetero_potential.py, adaptive_mesh.py
    """
    section_header("第 2 部分: 异质结势能剖面与自适应网格")

    # 创建计算域
    z_min = -20e-9  # -20 nm
    z_max = 20e-9   # +20 nm
    N_init = 200

    # 初始均匀网格
    z_grid = np.linspace(z_min, z_max, N_init)

    # 创建势能
    pot = HeteroPotential(MOS2, WSE2, z_grid, T=300.0, external_field=1e5)
    z_c, Vc = pot.get_conduction_band()
    z_v, Vv = pot.get_valence_band()

    print(f"\n[导带势能]")
    print(f"  范围: [{np.min(Vc):.4f}, {np.max(Vc):.4f}] eV")
    print(f"  势垒高度: {pot.barrier_height('electron'):.4f} eV")
    print(f"  界面电场: {pot.effective_field_at_interface():.2e} V/m")

    print(f"\n[价带势能]")
    print(f"  范围: [{np.min(Vv):.4f}, {np.max(Vv):.4f}] eV")

    # 自适应网格
    print(f"\n[自适应网格生成]")
    mesh = AdaptiveMesh1D(z_min, z_max, N_init,
                          potential_func=lambda z: float(
                              np.interp(z, z_c, Vc)),
                          max_refinement_level=3)
    print(f"  初始网格点: {N_init}")
    print(f"  自适应后: {len(mesh)} 个网格点")
    print(f"  网格质量 Q = {mesh.get_quality_metric():.4f}")

    dz_array = mesh.get_grid_spacing()
    print(f"  最小间距: {np.min(dz_array)*1e12:.2f} pm")
    print(f"  最大间距: {np.max(dz_array)*1e12:.2f} pm")

    return mesh, pot


def run_eigenvalue_problem(mesh, pot):
    """
    第 3 部分: 薛定谔方程求解.
    使用 sparse_hetband.py, high_order_fd.py
    """
    section_header("第 3 部分: 薛定谔方程本征值问题")

    z = mesh.z_nodes
    N = len(z)

    # 有效质量剖面
    z_interface = (z[0] + z[-1]) / 2
    m_star = np.where(z < z_interface, MOS2.me_eff, WSE2.me_eff)

    # 势能剖面 (在自适应网格上插值)
    z_c, Vc = pot.get_conduction_band()
    V_on_mesh = np.interp(z, z_c, Vc)

    # 不同阶数求解
    for order in [2, 4, 6]:
        assembler = HeteroHamiltonianAssembler(z, m_star, V_on_mesh, order=order)
        H_sparse = assembler.assemble()
        H_dense = H_sparse.to_dense()

        # 本征值分解
        eigenvalues = np.linalg.eigvalsh(H_dense)

        # 取最低 5 个
        n_show = min(5, len(eigenvalues))
        print(f"\n[FD{order} 阶: {N} 网格点, {H_sparse.nnz()} 非零元]")
        for i in range(n_show):
            print(f"  E_{i} = {eigenvalues[i]:.6f} eV")

        if order == 4:
            E_subbands = eigenvalues[:n_show]

    # 多带哈密顿量
    print(f"\n[多带 (导带-价带耦合) 分析]")
    z_v, Vv = pot.get_valence_band()
    Vv_on_mesh = np.interp(z, z_v, Vv)
    m_h_profile = np.where(z < z_interface, MOS2.mh_eff, WSE2.mh_eff)

    multi = MultiBandHamiltonian(z, m_star, m_h_profile, V_on_mesh, Vv_on_mesh)
    P_kane = multi.kane_momentum()
    g_cv = multi.coupling_strength()
    print(f"  Kane 动量: {P_kane:.3e} kg·m/s")
    print(f"  耦合强度: {g_cv/E0:.3e} eV·m")

    return E_subbands


def run_stability(mesh, pot):
    """
    第 4 部分: 稳定性与色散分析.
    使用 stability_analysis.py, high_order_fd.py
    """
    section_header("第 4 部分: 数值稳定性与色散分析")

    z = mesh.z_nodes
    dz_avg = (z[-1] - z[0]) / (len(z) - 1)
    z_interface = (z[0] + z[-1]) / 2
    m_star_min = min(MOS2.me_eff, WSE2.me_eff)

    # Von Neumann 稳定性
    vn = VonNeumannStability(order=4)
    dt_max = vn.stability_limit_explicit(m_star_min, dz_avg)
    dt_lf = vn.stability_limit_leapfrog(m_star_min, dz_avg, 1.0)
    print(f"\n[稳定性极限]")
    print(f"  平均网格间距: {dz_avg*1e12:.2f} pm")
    print(f"  最小有效质量: {m_star_min:.4f} m0")
    print(f"  显式 Euler dt_max: {dt_max:.3e} s ({dt_max*1e15:.2f} fs)")
    print(f"  Leapfrog dt_max:   {dt_lf:.3e} s ({dt_lf*1e15:.2f} fs)")

    # 色散分析
    print(f"\n[色散误差分析]")
    disp = DispersionAnalysis(order=4)
    k_nyquist = disp.nyquist_limit(dz_avg)
    k_array = np.linspace(0.01 * k_nyquist, 0.99 * k_nyquist, 50)

    E_exact, E_num = disp.numerical_dispersion(k_array, m_star_min, dz_avg)
    error = disp.dispersion_error(k_array, m_star_min, dz_avg)

    print(f"  Nyquist 波数: {k_nyquist:.3e} 1/m")
    print(f"  最大色散误差: {np.max(error)*100:.2f}%")
    print(f"  k/k_Nyq=0.5 处误差: {np.interp(0.5*k_nyquist, k_array, error)*100:.4f}%")
    print(f"  k/k_Nyq=0.9 处误差: {np.interp(0.9*k_nyquist, k_array, error)*100:.2f}%")

    # 不同阶数比较
    print(f"\n[不同阶数色散误差 (k/k_Nyq = 0.5)]")
    for order in [2, 4, 6]:
        d = DispersionAnalysis(order=order)
        E_e, E_n = d.numerical_dispersion(
            np.array([0.5 * k_nyquist]), m_star_min, dz_avg)
        err = abs(E_n[0] - E_e[0]) / max(abs(E_e[0]), 1e-30)
        print(f"  FD{order:1d}: 误差 = {err*100:.6f}%")

    return dt_max


def run_tunneling(mesh, pot):
    """
    第 5 部分: 隧穿概率 (Feynman-Kac).
    使用 feynman_kac_band_edge.py
    """
    section_header("第 5 部分: 量子隧穿 (Feynman-Kac 路径积分)")

    z_c, Vc = pot.get_conduction_band()
    z_interface = (z_c[0] + z_c[-1]) / 2
    m_t = tunneling_effective_mass(MOS2, WSE2, 'electron')
    V_max = np.max(Vc)
    V_min = np.min(Vc)

    print(f"\n[异质结参数]")
    print(f"  隧穿有效质量: {m_t:.4f} m0")
    print(f"  势垒高度: {V_max - V_min:.4f} eV")

    # WKB 估计
    print(f"\n[WKB 隧穿概率]")
    # 找到经典回转点
    E_test = (V_max + V_min) / 2
    above_barrier = Vc > E_test
    if np.any(above_barrier):
        idx_left = np.where(above_barrier)[0][0]
        idx_right = np.where(above_barrier)[0][-1]
        z_left = z_c[idx_left]
        z_right = z_c[idx_right]

        V_func = lambda z: float(np.interp(z, z_c, Vc))
        T_wkb = wkb_tunneling_estimate(V_func, m_t, E_test, z_left, z_right)
        print(f"  E = {E_test:.4f} eV: T_WKB = {T_wkb:.3e}")

    # Feynman-Kac 蒙特卡洛
    print(f"\n[Feynman-Kac 蒙特卡洛隧穿概率]")
    E_test_values = np.linspace(V_min + 0.1 * (V_max - V_min),
                                 V_max - 0.1 * (V_max - V_min), 3)

    for E_test in E_test_values:
        V_func = lambda z: float(np.interp(z, z_c, Vc))
        T_avg, T_std = feynman_kac_tunneling(
            V_func, m_t, E_test,
            z_start=z_c[len(z_c)//4],
            z_left=z_c[0], z_right=z_c[-1],
            n_paths=200, dt=5e-17, max_steps=1000)
        print(f"  E = {E_test:.4f} eV: T_FK = {T_avg:.3e} ± {T_std:.3e}")

    # 共振隧穿
    barrier_width = abs(z_c[-1] - z_c[0]) * 0.1
    E_res = resonant_tunneling_condition(V_max, V_min, barrier_width, m_t)
    print(f"\n[共振隧穿能级] (势阱宽度 = {barrier_width*1e9:.2f} nm)")
    for i, E in enumerate(E_res[:5]):
        print(f"  E_{i+1} = {E:.4f} eV")


def run_dos_and_sampling(mesh, pot, E_subbands):
    """
    第 6 部分: 态密度与采样.
    使用 dos_histogram.py, band_sampling.py
    """
    section_header("第 6 部分: 态密度与 k 空间采样")

    # DOS
    dos = DOSCalculator(E_subbands, broadening=0.01)

    E_grid = np.linspace(np.min(E_subbands) - 0.1,
                          np.max(E_subbands) + 0.2, 200)
    dos_values = dos.subband_dos(E_grid, MOS2.me_eff)

    print(f"\n[子带态密度]")
    print(f"  能量范围: [{E_grid[0]:.4f}, {E_grid[-1]:.4f}] eV")
    print(f"  DOS 范围: [{np.min(dos_values):.3e}, {np.max(dos_values):.3e}] 1/(eV·m²)")

    # 理想 2D DOS
    g_ideal = dos.ideal_2d_dos(E_grid, MOS2.me_eff)
    print(f"  理想 2D DOS (m*={MOS2.me_eff}): {g_ideal[0]:.3e} 1/(eV·m²)")

    # 直方图 DOS
    dos_hist = dos.histogram_dos(E_grid)
    print(f"  直方图 DOS 最大值: {np.max(dos_hist):.3e}")

    # CDF
    cdf = dos.cdf(E_grid)
    E_50 = np.interp(0.5, cdf, E_grid)
    print(f"  累积 50% 能量: {E_50:.4f} eV")

    # JDOS
    E_v_subbands = E_subbands - MOS2.bandgap_temperature(300) * 0.5
    jdos = dos.joint_dos(E_subbands, E_v_subbands, E_grid,
                         MOS2.me_eff, MOS2.mh_eff)
    print(f"  JDOS 最大值: {np.max(jdos):.3e}")

    # 费米能级
    E_F = dos.fermi_level(300.0, 1e16, MOS2.me_eff, np.min(E_subbands))
    print(f"  费米能级 (n=10^16/m²): {E_F:.4f} eV")

    # 采样
    samples = dos.sample_energies(100)
    print(f"\n[DOS 采样 (100 个)]")
    print(f"  均值: {np.mean(samples):.4f} eV")
    print(f"  标准差: {np.std(samples):.4f} eV")

    return dos


def run_brillouin_zone():
    """
    第 7 部分: 布里渊区积分.
    使用 brillouin_integral.py
    """
    section_header("第 7 部分: 布里渊区积分")

    # 解析圆积分测试
    print(f"\n[单位圆上单积分 (解析验证)]")
    for e1, e2 in [(0, 0), (2, 0), (2, 2), (4, 0), (4, 2)]:
        val = circle_monomial_integral(e1, e2)
        print(f"  ∫ x^{e1} y^{e2} ds = {val:.6f}")

    # 布里渊区积分
    print(f"\n[MoS2 布里渊区积分]")
    bz = BrillouinZoneIntegration(MOS2.a_lattice, method='uniform', n_k=15)

    # 抛物线能带: E(k) = ℏ²k²/(2m*)
    m_kg = MOS2.me_eff * M0

    def E_band(kx, ky):
        k2 = kx**2 + ky**2
        return HBAR**2 * k2 / (2 * m_kg * E0)

    # 平均能量
    E_avg = bz.integrate(E_band)
    print(f"  均匀网格 ({bz.n_k}×{bz.n_k}): <E> = {E_avg:.4f} eV")

    # 特殊点方法
    bz_sp = BrillouinZoneIntegration(MOS2.a_lattice, method='special_points')
    E_avg_sp = bz_sp.integrate(E_band)
    print(f"  特殊点法: <E> = {E_avg_sp:.4f} eV")

    # 有效质量张量
    k0 = np.array([0.0, 0.0])
    M_tensor = bz.effective_mass_tensor(E_band, k0)
    print(f"  有效质量张量 (Γ点):")
    print(f"    [[{M_tensor[0,0]:.4e}, {M_tensor[0,1]:.4e}]")
    print(f"     [{M_tensor[1,0]:.4e}, {M_tensor[1,1]:.4e}]]")


def run_interpolation(z, V_profile, E_subbands):
    """
    第 8 部分: 插值方法.
    使用 barycentric_interp.py, trig_interp.py, lagrange_nd_hetband.py, legendre_basis_2d.py
    """
    section_header("第 8 部分: 多种插值方法")

    # 重心插值
    print(f"\n[重心插值 (势能重建)]")
    n_cheb = min(20, len(z) // 2)
    # Chebyshev 节点
    j = np.arange(n_cheb)
    z_cheb = 0.5 * (z[0] + z[-1]) + 0.5 * (z[-1] - z[0]) * np.cos(
        (2 * j + 1) * np.pi / (2 * n_cheb))
    V_cheb = np.interp(z_cheb, z, V_profile)

    bary = BarycentricInterpolator(z_cheb, V_cheb, 'chebyshev1')
    z_fine = np.linspace(z[0], z[-1], 500)
    V_interp = bary.evaluate(z_fine)
    V_exact = np.interp(z_fine, z, V_profile)
    error_bary = np.max(np.abs(V_interp - V_exact))
    print(f"  Chebyshev 节点数: {n_cheb}")
    print(f"  最大插值误差: {error_bary:.6e} eV")

    # Lebesgue 函数
    leb = bary.lebesgue_function(z_fine)
    print(f"  Lebesgue 函数最大值: {np.max(leb):.2f}")

    # 三角插值
    print(f"\n[三角(傅里叶)插值 (周期势能)]")
    N_trig = 64
    z_trig = np.linspace(z[0], z[-1], N_trig, endpoint=False)
    V_trig = np.interp(z_trig, z, V_profile)

    trig = TrigonometricInterpolator(z_trig, V_trig,
                                      period=z[-1] - z[0])
    z_test = np.array([(z[0] + z[-1]) / 2])
    V_trig_val = trig.evaluate(z_test)
    V_exact_val = np.interp(z_test, z, V_profile)
    print(f"  节点数: {N_trig}")
    print(f"  中点误差: {abs(V_trig_val[0] - V_exact_val[0]):.6e} eV")

    # Fourier 系数
    V_hat = trig.fourier_coefficients()
    n_significant = np.sum(np.abs(V_hat) > 1e-6 * np.max(np.abs(V_hat)))
    print(f"  显著 Fourier 系数: {n_significant}/{N_trig}")

    # 谱导数
    dVdz = trig.derivative_spectral()
    print(f"  谱导数范围: [{np.min(dVdz):.4e}, {np.max(dVdz):.4e}] eV/m")

    # 周期势能展开
    print(f"\n[周期势能 Fourier 展开]")
    ppe = PeriodicPotentialExpansion(z_trig, V_trig, n_harmonics=16)
    G_vecs = ppe.get_reciprocal_vectors()
    print(f"  倒格矢数: {len(G_vecs)}")
    print(f"  最大 |G|: {np.max(np.abs(G_vecs)):.3e} 1/m")

    # Legendre 谱方法
    print(f"\n[Legendre 谱基]")
    leg = LegendreSpectralBasis(15, z[0], z[-1])
    M_mass = leg.build_mass_matrix()
    S_stiff = leg.build_stiffness_matrix()
    print(f"  质量矩阵对角: [{M_mass[0,0]:.4e}, ..., {M_mass[-1,-1]:.4e}]")
    print(f"  刚度矩阵对角: [{S_stiff[0,0]:.4e}, ..., {S_stiff[-1,-1]:.4e}]")

    # Legendre 展开
    V_func = lambda zi: float(np.interp(zi, z, V_profile))
    coeffs = leg.expand_function(V_func)
    print(f"  势能 Legendre 系数 (前 5): {coeffs[:5]}")

    # 多维 Lagrange
    print(f"\n[多维 Lagrange 能带面]")
    # 构造 2D k 空间数据
    kx_pts = np.linspace(-1e9, 1e9, 8)
    ky_pts = np.linspace(-1e9, 1e9, 8)
    KX, KY = np.meshgrid(kx_pts, ky_pts, indexing='ij')
    m_kg = MOS2.me_eff * M0
    E_k = HBAR**2 * (KX**2 + KY**2) / (2 * m_kg * E0)

    bsi = BandSurfaceInterpolator(KX.ravel(), KY.ravel(), E_k.ravel(),
                                   max_degree=3)
    # 评估
    k_test = np.array([0.5e9, 0.5e9])
    E_test = bsi.interp.evaluate(k_test.reshape(1, -1))[0]
    E_exact = HBAR**2 * (k_test[0]**2 + k_test[1]**2) / (2 * m_kg * E0)
    print(f"  测试点 k = ({k_test[0]:.1e}, {k_test[1]:.1e})")
    print(f"  E_Lagrange = {E_test:.6f} eV")
    print(f"  E_exact    = {E_exact:.6f} eV")
    print(f"  误差 = {abs(E_test - E_exact):.3e} eV")


def run_fem_analysis(z, V_profile, m_star_profile):
    """
    第 9 部分: 有限元分析.
    使用 fem_basis_t6.py
    """
    section_header("第 9 部分: T6 二次有限元分析")

    # 创建一个简单的三角形单元
    print(f"\n[T6 二次三角形单元]")
    nodes = np.array([
        [0.0, 0.0],    # 顶点 1
        [1e-9, 0.0],   # 顶点 2
        [0.5e-9, 0.866e-9],  # 顶点 3 (等边)
        [0.5e-9, 0.0],       # 边 1-2 中点
        [0.75e-9, 0.433e-9], # 边 2-3 中点
        [0.25e-9, 0.433e-9], # 边 3-1 中点
    ])

    elem = T6Element(nodes)
    print(f"  单元面积: {elem.area:.3e} m²")

    # 基函数在重心处的值
    xc = np.mean(nodes[:3, 0])
    yc = np.mean(nodes[:3, 1])
    phi = elem.basis_values(xc, yc)
    print(f"  重心处基函数值: {phi}")

    # 刚度矩阵
    K = elem.stiffness_matrix(m_star_inv=1.0 / (MOS2.me_eff * M0))
    print(f"  刚度矩阵条件数: {np.linalg.cond(K):.3e}")

    # 质量矩阵
    M_mat = elem.mass_matrix()
    print(f"  质量矩阵迹: {np.trace(M_mat):.3e}")
    print(f"  质量矩阵条件数: {np.linalg.cond(M_mat):.3e}")

    # 验证: 基函数之和应为 1 (单位分解)
    phi_sum = np.sum(phi)
    print(f"  基函数单位分解验证: Σφ_i = {phi_sum:.6f} (应为 1.0)")


def run_band_sampling_and_selection(z, V_profile, m_star_profile, E_subbands):
    """
    第 10 部分: 能带采样与基函数选择.
    使用 band_sampling.py, basis_selector.py
    """
    section_header("第 10 部分: 能带采样与自适应基选择")

    # 2D 采样
    print(f"\n[2D k 空间采样]")
    kx_1d = np.linspace(-2e9, 2e9, 30)
    ky_1d = np.linspace(-2e9, 2e9, 30)
    KX, KY = np.meshgrid(kx_1d, ky_1d, indexing='ij')
    m_kg = MOS2.me_eff * M0
    E_k = HBAR**2 * (KX**2 + KY**2) / (2 * m_kg * E0)

    cdf2d = DiscreteCDF2D(KX, KY, E_k)
    kx_s, ky_s, E_s = cdf2d.sample(200)
    print(f"  采样 200 个 k 点")
    print(f"  <E> = {np.mean(E_s):.4f} eV, σ(E) = {np.std(E_s):.4f} eV")

    # 重要性采样
    sampler = BandStructureSampler(kx_1d, E_k[:, 0])
    k_imp, E_imp = sampler.importance_sample(
        100, energy_window=(0, 0.1))
    print(f"\n[重要性采样 (E ∈ [0, 0.1] eV)]")
    print(f"  采样数: {len(k_imp)}")
    if len(E_imp) > 0:
        print(f"  <E> = {np.mean(E_imp):.4f} eV")

    # 分层采样
    k_str, E_str = sampler.stratified_sample(5, 20)
    print(f"\n[分层采样 (5层×20)]")
    print(f"  采样数: {len(k_str)}")

    # 基函数选择
    print(f"\n[自适应基函数选择]")
    dz = z[1] - z[0] if len(z) > 1 else 1e-10
    selector = BasisSelector(z, V_profile, m_star_profile)
    recommended = selector.recommend_method()
    print(f"  势能变化率 max|dV/dz|: {selector.max_dVdz:.3e} eV/m")
    print(f"  质量不连续度: {selector.mass_discontinuity:.3f}")
    print(f"  界面锐度: {selector.sharpness:.3f}")
    print(f"  推荐方法: {recommended}")

    for method in ['fd2', 'fd4', 'fd6', 'fd8', 'legendre']:
        err = selector.estimate_error(method)
        print(f"  {method:8s}: 截断误差估计 = {err:.3e}")


def run_self_consistent(z, m_star, V_initial):
    """
    第 11 部分: 自洽 Poisson-Schrödinger.
    使用 poisson_schrodinger.py
    """
    section_header("第 11 部分: 自洽 Poisson-Schrödinger 求解")

    solver = PoissonSchrodingerSolver(
        z, m_star, V_initial, epsilon_r=7.0,
        T=300.0, doping_profile=None)
    solver.max_iter = 10
    solver.tolerance = 1e-4
    solver.n_subbands = 3

    print(f"\n[自洽循环参数]")
    print(f"  最大迭代: {solver.max_iter}")
    print(f"  收敛容差: {solver.tolerance} eV")
    print(f"  混合因子: {solver.mixing_alpha}")
    print(f"  子带数: {solver.n_subbands}")

    energies, wavefunctions = solver.solve_self_consistent()

    print(f"\n[自洽结果]")
    print(f"  收敛: {solver.converged}")
    for i, E in enumerate(energies):
        print(f"  E_{i} = {E:.6f} eV")

    if solver.charge_density is not None:
        print(f"  最大载流子密度: {np.max(solver.charge_density):.3e} 1/m³")

    return energies


def run_convergence_study():
    """
    第 12 部分: 网格收敛性研究.
    使用 stability_analysis.py
    """
    section_header("第 12 部分: 网格收敛性研究")

    z_min, z_max = -10e-9, 10e-9
    V_barrier = 0.5  # eV
    m_star_val = 0.4

    def simple_solver(N):
        """简单势阱求解器."""
        z = np.linspace(z_min, z_max, N)

        # 方势阱
        V = np.where(np.abs(z) < 3e-9, 0.0, V_barrier)
        m_star = np.full(N, m_star_val)

        assembler = HeteroHamiltonianAssembler(z, m_star, V, order=2)
        H = assembler.assemble().to_dense()
        E = np.linalg.eigvalsh(H)
        return E[:3]

    # 先计算参考解 (精细网格)
    E_ref = simple_solver(2000)
    E_ref_0 = E_ref[0] if len(E_ref) > 0 else 0

    print(f"\n[方势阱收敛性 (V0={V_barrier}eV, m*={m_star_val}m0)]")
    print(f"  参考解 (N=2000): E_0 = {E_ref_0:.8f} eV")
    print(f"  {'N':>6s}  {'E_0 (eV)':>12s}  {'E_1 (eV)':>12s}  {'error':>12s}  {'rate':>8s}")

    N_list = [50, 100, 200, 400, 800]
    prev_err = None
    prev_N = None

    for N in N_list:
        E = simple_solver(N)
        E0 = E[0] if len(E) > 0 else 0
        E1 = E[1] if len(E) > 1 else 0
        err = abs(E0 - E_ref_0)

        rate_str = "   -  "
        if prev_err is not None and prev_err > 1e-20 and err > 1e-20:
            rate = np.log(prev_err / err) / np.log(N / prev_N)
            rate_str = f"{rate:8.2f}"

        print(f"  {N:6d}  {E0:12.8f}  {E1:12.8f}  {err:12.3e}  {rate_str}")
        prev_err = err
        prev_N = N

    # Richardson 外推 (使用最后三个)
    E_N1 = simple_solver(200)[0]
    E_N2 = simple_solver(400)[0]
    E_N3 = simple_solver(800)[0]
    r = 2.0  # 加密比
    if abs(E_N1 - E_N2) > 1e-20 and abs(E_N2 - E_N3) > 1e-20:
        p_est = np.log(abs((E_N1 - E_N2) / (E_N2 - E_N3))) / np.log(r)
        E_ext = E_N3 + (E_N3 - E_N2) / (r**p_est - 1)
        print(f"\n  Richardson 外推:")
        print(f"    E_0 ≈ {E_ext:.8f} eV")
        print(f"    估计收敛阶: {p_est:.2f}")


def run_legendre_product_demo():
    """
    演示 Legendre 乘积多项式.
    使用 legendre_basis_2d.py
    """
    print(f"\n[Legendre 乘积多项式验证]")
    # P_2(x)*P_1(y) at (0.5, 0.3)
    val = legendre_product_polynomial((2, 1), np.array([[0.5, 0.3]]))
    P2_05 = legendre_value(2, np.array([0.5]))[0]
    P1_03 = legendre_value(1, np.array([0.3]))[0]
    expected = P2_05 * P1_03
    print(f"  P_2(0.5)*P_1(0.3) = {val[0]:.6f} (期望: {expected:.6f})")
    print(f"  误差: {abs(val[0] - expected):.2e}")


def main():
    """
    主程序入口.

    运行完整的二维材料异质结能带对齐分析流程:
    1. 材料参数分析
    2. 势能剖面与自适应网格
    3. 薛定谔方程求解
    4. 稳定性与色散分析
    5. 量子隧穿 (Feynman-Kac)
    6. 态密度与采样
    7. 布里渊区积分
    8. 多种插值方法
    9. T6 有限元分析
    10. 能带采样与基选择
    11. 自洽 Poisson-Schrödinger
    12. 网格收敛性研究
    """
    print("=" * 60)
    print("  二维材料异质结能带对齐")
    print("  高阶有限差分与稳定性分析")
    print("  (小规模可复现实验)")
    print("=" * 60)

    # 第 1 部分: 材料参数
    topo = run_material_analysis()

    # 第 2 部分: 势能与网格
    mesh, pot = run_potential_and_mesh(topo)

    # 第 3 部分: 本征值
    E_subbands = run_eigenvalue_problem(mesh, pot)

    # 第 4 部分: 稳定性
    dt_max = run_stability(mesh, pot)

    # 第 5 部分: 隧穿
    run_tunneling(mesh, pot)

    # 第 6 部分: DOS
    z_c, Vc = pot.get_conduction_band()
    dos = run_dos_and_sampling(mesh, pot, E_subbands)

    # 第 7 部分: 布里渊区
    run_brillouin_zone()

    # 第 8 部分: 插值
    run_interpolation(mesh.z_nodes, Vc, E_subbands)

    # 第 9 部分: FEM
    z_interface = (mesh.z_nodes[0] + mesh.z_nodes[-1]) / 2
    m_star = np.where(mesh.z_nodes < z_interface, MOS2.me_eff, WSE2.me_eff)
    run_fem_analysis(mesh.z_nodes, Vc, m_star)

    # 第 10 部分: 采样与选择
    run_band_sampling_and_selection(mesh.z_nodes, Vc, m_star, E_subbands)

    # 第 11 部分: 自洽求解
    run_self_consistent(mesh.z_nodes, m_star, Vc)

    # 第 12 部分: 收敛性
    run_convergence_study()

    # Legendre 验证
    run_legendre_product_demo()

    print(f"\n{'='*60}")
    print(f"  计算完成!")
    print(f"  共融合 15 个种子项目的核心算法")
    print(f"  所有模块成功运行")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
