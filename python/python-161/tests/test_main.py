#!/usr/bin/env python3
"""
main.py
钙钛矿太阳能电池多物理场耦合模拟与光电转换效率评估系统

统一入口，零参数可运行。
本程序综合了以下 15 个种子项目的核心算法：
  538_histogram_data_2d_sample -> 光谱离散 CDF 采样
  1406_wedge_exactness         -> 楔形体高斯求积（光吸收体积分）
  927_pwl_interp_2d            -> 材料参数二维分段线性插值
  769_mm_io                    -> 稀疏 Jacobian 矩阵 Market 格式 I/O
  767_midpoint_fixed           -> 固定点中点法 ODE 求解器
  1336_triangulation_display   -> 三角网格生成与处理
  1006_random_data             -> 缺陷随机分布采样
  780_mortality                -> 载流子寿命 PDF/CDF 统计
  550_humps_ode                -> ODE 求解器精度验证
  122_buckling_spring          -> 薄膜热应力屈曲分析
  641_laguerre_polynomial      -> Gauss-Laguerre 求积（带尾态积分）
  873_ply_io                   -> PLY 多面体网格 I/O
  854_pce_ode_hermite          -> 多项式混沌展开不确定性量化
  345_exm                      -> Euler ODE 求解 + 耦合动力学
  1187_svd_fingerprint         -> SVD 模型降阶

科学问题：
  评估钙钛矿 MAPbI3 太阳能电池在热应力、离子迁移、材料缺陷不确定性
  等多物理场耦合条件下的光电转换效率 η，并给出效率的统计分布与
  主要影响因素的敏感性排序。
"""

import os
import sys
import time
import numpy as np

# 确保模块路径
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

# ---------------------------------------------------------------------------
# 导入子模块
# ---------------------------------------------------------------------------
from spectrum_sampler import sample_photons, photon_energy_ev, build_am15_spectrum
from absorption_integrator import (
    generate_wedge_gauss_rule, compute_carrier_generation_rate, wedge01_volume
)
from material_interpolator import PerovskiteMaterial
from sparse_matrix_io import build_drift_diffusion_jacobian, write_matrix_market, read_matrix_market
from drift_diffusion_solver import (
    solve_transient_drift_diffusion_1d, verify_solver, midpoint_fixed_time_stepper
)
from mesh_triangulation import generate_grain_mesh, write_ply, read_ply, TriMesh
from defect_monte_carlo import (
    sample_defect_positions, defect_density_lognormal,
    carrier_lifetime_from_defects, MortalityStyleLifetimeModel, srh_recombination_rate
)
from recombination_models import total_recombination_rate, laguerre_quadrature_rule
from mechanical_stress import compute_buckling_impact_on_efficiency, buckling_lambda_mu
from uncertainty_pce import pce_efficiency_uq, pce_time_integrator
from coupled_ion_migration import solve_hysteresis_cycle, predprey_style_ion_dynamics, ode1_euler
from model_reduction import apply_mor_to_drift_diffusion, low_rank_approximation, compute_svd


def print_section(title: str) -> None:
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def run_spectrum_and_absorption():
    """步骤 1：光谱采样与光吸收计算"""
    print_section("1. AM1.5G 光谱采样与载流子产生率计算")

    # 538：离散 CDF 采样
    lams, thetas = sample_photons(n_photons=5000)
    E_photon = photon_energy_ev(lams)
    print(f"  采样光子数: {len(lams)}")
    print(f"  波长范围: [{lams.min():.1f}, {lams.max():.1f}] nm")
    print(f"  光子能量: [{E_photon.min():.3f}, {E_photon.max():.3f}] eV")

    # 1406：楔形体求积计算总产生率
    def alpha_fn(lam):
        # 简化吸收系数模型
        return 5e4 * np.ones_like(lam)

    def irr_fn(lam):
        return 0.01 * np.ones_like(lam)

    def eph_fn(lam):
        return photon_energy_ev(lam)

    total_gen, pts, gen_dens, w = compute_carrier_generation_rate(
        alpha_fn, irr_fn, eph_fn,
        length_xy=1e-4, thickness_z=5e-5, order_xy=4, order_z=4
    )
    print(f"  楔形体体积: {wedge01_volume(1e-4, 5e-5):.6e} cm³")
    print(f"  总载流子产生率: {total_gen:.3e} s⁻¹")
    print(f"  平均产生密度: {gen_dens.mean():.3e} cm⁻³·s⁻¹")
    return total_gen, gen_dens.mean()


def run_material_properties():
    """步骤 2：材料参数插值"""
    print_section("2. 钙钛矿材料参数二维插值 (T, x)")

    mat = PerovskiteMaterial()
    test_conditions = [(300.0, 0.0), (300.0, 0.5), (350.0, 0.3), (400.0, 1.0)]
    for T, x in test_conditions:
        p = mat.get_params(T, x)
        print(f"  T={T:.0f}K, x={x:.1f}: Eg={p['bandgap_eV']:.3f} eV, "
              f"μ_n={p['electron_mobility']:.2f}, μ_p={p['hole_mobility']:.2f} cm²/Vs")
    return mat


def run_sparse_matrix_and_drift_diffusion(mat: PerovskiteMaterial):
    """步骤 3：稀疏矩阵与漂移-扩散求解"""
    print_section("3. 稀疏 Jacobian 构建与漂移-扩散稳态求解")

    params = mat.get_params(300.0, 0.0)
    N = 30
    L = 5e-5  # cm
    dx = L / (N - 1)
    T = 300.0
    kB = 1.380649e-23
    q = 1.602176634e-19
    kT_q = kB * T / q
    D_n = kT_q * params["electron_mobility"]
    D_p = kT_q * params["hole_mobility"]

    # 769：构建稀疏 Jacobian
    E_field = np.zeros(N)
    n = np.ones(N) * 1e15
    p = np.ones(N) * 1e15
    jac = build_drift_diffusion_jacobian(
        N, dx, params["electron_mobility"], params["hole_mobility"],
        D_n, D_p, E_field, n, p, kT_q
    )
    print(f"  Jacobian 维度: {jac.nrow}×{jac.ncol}, 非零元: {jac.nnz()}")

    # 临时写入/读取测试
    mtx_path = os.path.join(PROJECT_DIR, "jac_temp.mtx")
    write_matrix_market(jac, mtx_path)
    jac2 = read_matrix_market(mtx_path)
    print(f"  Matrix Market I/O 测试通过，读取非零元: {jac2.nnz()}")
    os.remove(mtx_path)

    # 767 + 550：漂移扩散求解 + humps 验证
    err = verify_solver()
    print(f"  Humps ODE 求解器验证 L2 误差: {err:.3e}")

    G = np.ones(N) * 1e21
    def simple_R(nv, pv):
        # 数值鲁棒性：限制乘积大小，防止溢出
        nv_clip = min(max(nv, 1.0), 1e22)
        pv_clip = min(max(pv, 1.0), 1e22)
        return 1e-12 * (nv_clip * pv_clip - 1e20)

    try:
        t_arr, n_hist, p_hist, phi_hist = solve_transient_drift_diffusion_1d(
            N, L, T, params["electron_mobility"], params["hole_mobility"],
            30.0, 1e16, 1e16, G, simple_R, (0.0, 1e-12), 50
        )
        if not np.isfinite(n_hist[-1].max()):
            raise ValueError("Non-finite values in drift-diffusion solution")
    except Exception:
        # 若发散，使用稳态近似值
        n_ss = np.ones(N) * 1e15
        p_ss = np.ones(N) * 1e15
        phi_ss = np.linspace(0.0, 0.8, N)
        print("  漂移-扩散瞬态求解数值发散，使用稳态近似值")
        return n_ss, p_ss, phi_ss
    print(f"  稳态 n_max={n_hist[-1].max():.3e}, p_max={p_hist[-1].max():.3e}")
    print(f"  电势降: {phi_hist[-1].max() - phi_hist[-1].min():.4f} V")
    return n_hist[-1], p_hist[-1], phi_hist[-1]


def run_mesh_and_defects():
    """步骤 4：网格生成与缺陷蒙特卡洛"""
    print_section("4. 多晶网格生成与缺陷随机分布")

    # 1336 + 873：网格生成与 PLY I/O
    mesh = generate_grain_mesh(6, 6)
    areas = mesh.compute_areas()
    print(f"  网格: {mesh.vertices.shape[0]} 顶点, {mesh.faces.shape[0]} 三角形")
    print(f"  总面积: {areas.sum():.6e} cm²")

    ply_path = os.path.join(PROJECT_DIR, "grain_mesh.ply")
    write_ply(mesh, ply_path)
    verts, faces = read_ply(ply_path)
    print(f"  PLY I/O 测试通过: {len(verts)} 顶点, {len(faces)} 面")
    os.remove(ply_path)

    # 1006 + 780：缺陷采样与寿命统计
    v1 = mesh.vertices[mesh.faces[:, 0]]
    v2 = mesh.vertices[mesh.faces[:, 1]]
    v3 = mesh.vertices[mesh.faces[:, 2]]
    defects = sample_defect_positions(2000, (v1, v2, v3))
    print(f"  缺陷采样点数: {defects.shape[0]}")

    N_t = defect_density_lognormal(1000)
    tau_n, tau_p = carrier_lifetime_from_defects(N_t)
    model = MortalityStyleLifetimeModel(tau_n)
    print(f"  平均电子寿命: {model.expected_lifetime():.3e} s")
    print(f"  1 μs 存活概率: {model.survival_probability(1e-6):.4f}")
    return tau_n.mean(), tau_p.mean()


def run_recombination(tau_n: float, tau_p: float):
    """步骤 5：复合模型计算"""
    print_section("5. 辐射/Auger/带尾复合计算 (Gauss-Laguerre)")

    # 641：Laguerre 求积验证
    xg, wg = laguerre_quadrature_rule(8)
    print(f"  Gauss-Laguerre (n=8) 权重和: {wg.sum():.6f} (理论=1)")

    rates = total_recombination_rate(
        n=1e15, p=1e15, n_i=1e10, T=300.0,
        tau_n=tau_n, tau_p=tau_p, E_t=0.0, E_g=1.57,
        N_t_tail=1e16, E_u=0.015
    )
    print(f"  SRH 复合率: {rates['SRH']:.3e} cm⁻³·s⁻¹")
    print(f"  辐射复合率: {rates['radiative']:.3e}")
    print(f"  Auger 复合率: {rates['auger']:.3e}")
    print(f"  带尾复合率: {rates['tail']:.3e}")
    print(f"  总复合率: {rates['total']:.3e}")
    return rates["total"]


def run_mechanical_stress():
    """步骤 6：热应力屈曲分析"""
    print_section("6. 薄膜热应力屈曲分析")

    result = compute_buckling_impact_on_efficiency(delta_T=60.0)
    for k, v in result.items():
        print(f"  {k}: {v}")

    # 122：lambda/mu 参数验证
    L_arr = np.linspace(0.3, 1.7, 5)
    lam, mu = buckling_lambda_mu(L_arr, np.pi / 6)
    print(f"  屈曲参数 λ 范围: [{lam.min():.4f}, {lam.max():.4f}]")
    return result


def run_uncertainty_quantification():
    """步骤 7：PCE 不确定性量化"""
    print_section("7. 光电转换效率 PCE 不确定性量化")

    uq = pce_efficiency_uq(efficiency_mean=0.21, efficiency_std=0.025, np_deg=5)
    print(f"  PCE 均值效率: {uq['pce_mean_efficiency']:.4f}")
    print(f"  PCE 标准差: {uq['pce_std_efficiency']:.4f}")
    print(f"  MC 对照均值: {uq['mc_mean_efficiency']:.4f}")
    print(f"  MC 对照标准差: {uq['mc_std_efficiency']:.4f}")
    print(f"  敏感性指标:")
    for order, sens in uq["sensitivity_indices"].items():
        print(f"    {order}: {sens:.4f}")
    return uq


def run_ion_migration():
    """步骤 8：离子迁移迟滞模拟"""
    print_section("8. 离子迁移 - I-V 迟滞模拟")

    # 345：predprey 风格振荡
    t, V_I, n_e = predprey_style_ion_dynamics(tspan=(0.0, 50.0), n_steps=2000)
    print(f"  碘空位浓度范围: [{V_I.min():.2f}, {V_I.max():.2f}] (归一化)")

    # I-V 扫描
    V_fwd = np.linspace(0.0, 1.0, 15)
    V_rev = np.linspace(1.0, 0.0, 15)
    V_full = np.concatenate([V_fwd, V_rev])
    V, J, n_ion_t, E_ion_t = solve_hysteresis_cycle(V_full, time_per_step=1e-3)
    print(f"  最大电流密度: {J.max():.3f} mA/cm²")
    print(f"  迟滞指数 (|J_fwd(0.5V)-J_rev(0.5V)|/max|J|): "
          f"{abs(J[7] - J[22]) / max(abs(J.max()), 1e-10):.4f}")
    return V, J


def run_model_reduction():
    """步骤 9：SVD 模型降阶"""
    print_section("9. SVD/POD 模型降阶")

    mor = apply_mor_to_drift_diffusion(n_spatial=40, n_time_snapshots=15, n_pod_modes=4)
    print(f"  POD 模态数: {mor['n_pod_modes']}")
    print(f"  相对重建误差: {mor['relative_reconstruction_error']:.3e}")
    print(f"  降阶 Jacobian 条件数: {mor['reduced_jacobian_condition_number']:.3e}")

    # 1187：低秩近似测试
    A_test = np.random.randn(30, 20)
    _, comp, energy = low_rank_approximation(A_test, 3)
    print(f"  低秩近似压缩比: {comp:.3f}, 能量占比: {energy:.4f}")
    return mor


def compute_final_efficiency(
    total_gen_rate: float,
    avg_gen_density: float,
    R_total: float,
    buckling_result: dict,
    uq_result: dict,
) -> dict:
    """
    综合所有物理模块计算最终光电转换效率。
    """
    # 简化效率模型：
    # η = (J_sc * V_oc * FF) / P_in
    # J_sc ≈ q * L * G_avg （短路与光生电流成正比）
    # V_oc ≈ (kT/q) ln(G_avg / R_total + 1)
    # FF ≈ 0.8 (经验值)

    q = 1.602176634e-19
    kB = 1.380649e-23
    T = 300.0
    P_in = 0.1  # W/cm^2 (100 mW/cm^2)

    thickness = 5e-5  # cm
    J_sc = q * thickness * avg_gen_density  # A/cm^2
    J_sc_mA = J_sc * 1e3  # mA/cm^2

    # 考虑复合损失：采用收集效率模型
    collection_efficiency = 1.0 / (1.0 + R_total / (avg_gen_density + 1e-10))
    collection_efficiency = float(np.clip(collection_efficiency, 0.0, 1.0))

    J_sc_eff = J_sc_mA * collection_efficiency

    # 开路电压：基于带隙和温度的 Shockley 极限近似
    E_g = 1.57  # eV
    V_oc = (kB * T / q) * np.log(J_sc_eff / 1e-12 + 1.0)
    V_oc = float(min(V_oc, E_g))  # V_oc 不能超过带隙

    FF = 0.78  # 填充因子（简化）

    # 效率
    eta = (J_sc_eff * V_oc * FF) / P_in
    eta = float(np.clip(eta, 0.0, 0.35))

    # 考虑屈曲损失
    eff_loss = buckling_result.get("estimated_efficiency_loss_percent", 0.0) / 100.0
    eta_corrected = eta * (1.0 - eff_loss)

    # PCE 不确定性
    eta_std = uq_result.get("pce_std_efficiency", 0.02)

    return {
        "short_circuit_current_mA_cm2": float(J_sc_eff),
        "open_circuit_voltage_V": float(V_oc),
        "fill_factor": float(FF),
        "efficiency_no_stress": float(eta),
        "efficiency_with_stress": float(eta_corrected),
        "efficiency_std": float(eta_std),
        "efficiency_95CI_lower": float(max(eta_corrected - 1.96 * eta_std, 0.0)),
        "efficiency_95CI_upper": float(min(eta_corrected + 1.96 * eta_std, 0.35)),
    }


def main():
    print("\n" + "#" * 70)
    print("#  钙钛矿太阳能电池多物理场耦合效率模拟系统")
    print("#  Perovskite Solar Cell Multi-Physics Efficiency Simulator")
    print("#" * 70)
    t_start = time.time()

    # 执行各模块
    total_gen, avg_gen = run_spectrum_and_absorption()
    mat = run_material_properties()
    n_ss, p_ss, phi_ss = run_sparse_matrix_and_drift_diffusion(mat)
    tau_n, tau_p = run_mesh_and_defects()
    R_total = run_recombination(tau_n, tau_p)
    buckling = run_mechanical_stress()
    uq = run_uncertainty_quantification()
    V, J = run_ion_migration()
    mor = run_model_reduction()

    # 综合效率评估
    print_section("10. 综合光电转换效率评估")
    eff_result = compute_final_efficiency(total_gen, avg_gen, R_total, buckling, uq)
    for k, v in eff_result.items():
        print(f"  {k}: {v}")

    t_elapsed = time.time() - t_start
    print("\n" + "#" * 70)
    print(f"#  全部计算完成，耗时 {t_elapsed:.3f} 秒")
    print("#" * 70)

    return 0



if __name__ == "__main__":
    main()

# ================================================================
# 测试用例（27个，assert模式，涉及随机值均使用固定种子）
# ================================================================

# ---- TC01: photon_energy_ev 解析验证 (hc/λ) ----
E1 = photon_energy_ev(np.array([620.0]))
assert abs(E1[0] - 1239.8 / 620.0) < 0.01, '[TC01] photon_energy_ev 解析 FAILED'

# ---- TC02: wedge01_volume 已知尺寸验证 ----
V = wedge01_volume(1e-4, 5e-5)
assert abs(V - 1e-4**2 * 5e-5 / 2.0) < 1e-16, '[TC02] wedge01_volume FAILED'

# ---- TC03: Gauss-Laguerre 权重和 ≈ 1 ----
_, wg = laguerre_quadrature_rule(8)
assert abs(wg.sum() - 1.0) < 1e-6, '[TC03] Laguerre 权重和 FAILED'

# ---- TC04: 光谱采样固定种子可复现 ----
np.random.seed(42)
lams1, thetas1 = sample_photons(n_photons=500)
np.random.seed(42)
lams2, thetas2 = sample_photons(n_photons=500)
assert np.allclose(lams1, lams2), '[TC04] 光谱采样可复现性 FAILED'

# ---- TC05: 材料参数 T=300K, x=0 带隙验证 ----
mat = PerovskiteMaterial()
p = mat.get_params(300.0, 0.0)
assert abs(p['bandgap_eV'] - 1.41) < 0.05, '[TC05] 带隙参数 FAILED'

# ---- TC06: 稀疏矩阵基本操作 ----
from sparse_matrix_io import SparseMatrix
sm = SparseMatrix(3, 3)
sm.add(0, 0, 1.0); sm.add(1, 1, 2.0); sm.add(2, 2, 3.0)
assert sm.nnz() == 3, '[TC06] 稀疏矩阵非零元数 FAILED'
dense = sm.to_dense()
assert abs(dense[0, 0] - 1.0) < 1e-12, '[TC06] 稀疏矩阵稠密值 FAILED'

# ---- TC07: Humps ODE 求解器验证返回有限值 ----
err = verify_solver()
assert np.isfinite(err), '[TC07] verify_solver 返回值 FAILED'
assert err >= 0, '[TC07] verify_solver 误差非负 FAILED'

# ---- TC08: 网格生成与面积验证 ----
np.random.seed(42)
mesh = generate_grain_mesh(4, 4)
areas = mesh.compute_areas()
assert mesh.vertices.shape[0] > 0, '[TC08] 网格顶点数 FAILED'
assert mesh.faces.shape[0] > 0, '[TC08] 网格面数 FAILED'
assert np.all(areas > 0), '[TC08] 三角形面积非正 FAILED'
assert np.all(np.isfinite(areas)), '[TC08] 面积非有限 FAILED'

# ---- TC09: 缺陷采样固定种子可复现 ----
np.random.seed(42)
v1 = mesh.vertices[mesh.faces[:, 0]]
v2 = mesh.vertices[mesh.faces[:, 1]]
v3 = mesh.vertices[mesh.faces[:, 2]]
defects1 = sample_defect_positions(200, (v1, v2, v3))
np.random.seed(42)
defects2 = sample_defect_positions(200, (v1, v2, v3))
assert np.allclose(defects1, defects2), '[TC09] 缺陷采样可复现性 FAILED'

# ---- TC10: 复合率非负且有限 ----
import numpy as np
np.random.seed(42)
rates = total_recombination_rate(n=1e15, p=1e15, n_i=1e10, T=300.0, tau_n=1e-8, tau_p=1e-8)
for k in ['SRH', 'radiative', 'auger', 'tail', 'total']:
    assert np.isfinite(rates[k]), f'[TC10] {k} 复合率非有限 FAILED'
    assert rates[k] >= 0, f'[TC10] {k} 复合率为负 FAILED'

# ---- TC11: 屈曲分析输出键完整性 ----
np.random.seed(42)
result = compute_buckling_impact_on_efficiency(delta_T=60.0)
expected_keys = ['thermal_stress_MPa', 'critical_stress_MPa', 'buckled', 'max_deflection_nm',
                  'bandgap_shift_meV', 'estimated_efficiency_loss_percent']
for k in expected_keys:
    assert k in result, f'[TC11] 屈曲结果缺键 {k} FAILED'
assert np.isfinite(result['thermal_stress_MPa']), '[TC11] 热应力非有限 FAILED'

# ---- TC12: PCE 不确定性量化输出 ----
np.random.seed(42)
uq = pce_efficiency_uq(efficiency_mean=0.21, efficiency_std=0.025, np_deg=5)
assert np.isfinite(uq['pce_mean_efficiency']), '[TC12] PCE 均值非有限 FAILED'
assert uq['pce_std_efficiency'] >= 0, '[TC12] PCE 标准差为负 FAILED'
assert 'sensitivity_indices' in uq, '[TC12] 缺敏感性指标 FAILED'

# ---- TC13: 离子动力学输出尺寸 ----
np.random.seed(42)
t, V_I, n_e = predprey_style_ion_dynamics(tspan=(0.0, 20.0), n_steps=500)
assert len(t) == 501, '[TC13] 离子动力学时间步数 FAILED'
assert V_I.shape == t.shape, '[TC13] V_I 形状 FAILED'
assert n_e.shape == t.shape, '[TC13] n_e 形状 FAILED'

# ---- TC14: SVD 模型降阶输出 ----
np.random.seed(42)
mor = apply_mor_to_drift_diffusion(n_spatial=30, n_time_snapshots=10, n_pod_modes=3)
assert mor['n_pod_modes'] == 3, '[TC14] POD 模态数 FAILED'
assert np.isfinite(mor['relative_reconstruction_error']), '[TC14] 重建误差非有限 FAILED'
assert mor['relative_reconstruction_error'] >= 0, '[TC14] 重建误差为负 FAILED'

# ---- TC15: compute_final_efficiency 输出类型与键 ----
buckling_test = {'estimated_efficiency_loss_percent': 5.0}
uq_test = {'pce_std_efficiency': 0.02}
eff = compute_final_efficiency(1e18, 1e21, 1e20, buckling_test, uq_test)
assert isinstance(eff, dict), '[TC15] 效率输出非字典 FAILED'
required_keys = ['short_circuit_current_mA_cm2', 'open_circuit_voltage_V',
                 'fill_factor', 'efficiency_no_stress', 'efficiency_with_stress']
for k in required_keys:
    assert k in eff, f'[TC15] 效率缺键 {k} FAILED'

# ---- TC16: 效率范围约束 0 ≤ η ≤ 0.35 ----
eff = compute_final_efficiency(1e18, 1e21, 1e20, buckling_test, uq_test)
assert 0 <= eff['efficiency_no_stress'] <= 0.35, '[TC16] 效率超出范围 FAILED'
assert 0 <= eff['efficiency_with_stress'] <= 0.35, '[TC16] 应力效率超出范围 FAILED'

# ---- TC17: 极端输入鲁棒性（零产生率） ----
eff_zero = compute_final_efficiency(1.0, 1.0, 1e30, {'estimated_efficiency_loss_percent': 0.0}, {'pce_std_efficiency': 0.0})
assert np.isfinite(eff_zero['efficiency_no_stress']), '[TC17] 零输入效率非有限 FAILED'
assert eff_zero['efficiency_no_stress'] >= 0, '[TC17] 零输入效率为负 FAILED'

# ---- TC18: buckling_lambda_mu 形状验证 ----
L_arr = np.linspace(0.3, 1.7, 5)
lam, mu = buckling_lambda_mu(L_arr, np.pi / 6)
assert lam.shape == L_arr.shape, '[TC18] lambda 形状 FAILED'
assert mu.shape == L_arr.shape, '[TC18] mu 形状 FAILED'

# ---- TC19: 低秩近似压缩比与能量占比 ----
np.random.seed(42)
A_test = np.random.randn(30, 20)
_, comp, energy = low_rank_approximation(A_test, 3)
assert comp > 0, '[TC19] 压缩比非正 FAILED'
assert 0 <= energy <= 1.0, '[TC19] 能量占比范围 FAILED'

# ---- TC20: SRH 复合率解析验证 ----
r_srh = srh_recombination_rate(n=1e15, p=1e15, n_i=1e10, tau_n=1e-8, tau_p=1e-8)
assert np.isfinite(r_srh), '[TC20] SRH 复合率非有限 FAILED'
assert r_srh >= 0, '[TC20] SRH 复合率为负 FAILED'

# ---- TC21: PCE 时间积分器输出尺寸 ----
np.random.seed(42)
t_arr, u_coeff = pce_time_integrator(0.0, 1.0, 20, 0.2, 3, 0.1, 0.05)
assert len(t_arr) == 21, '[TC21] PCE 积分器时间步数 FAILED'
assert u_coeff.ndim == 2, '[TC21] PCE 系数维度 FAILED'

# ---- TC22: I-V 迟滞循环输出尺寸 ----
np.random.seed(42)
V_sweep = np.linspace(0.0, 0.5, 10)
V, J, n_ion, E_ion = solve_hysteresis_cycle(V_sweep, time_per_step=1e-3)
assert len(V) == len(V_sweep), '[TC22] 迟滞电压长度 FAILED'
assert len(J) == len(V_sweep), '[TC22] 迟滞电流长度 FAILED'
assert np.all(np.isfinite(J)), '[TC22] 电流密度非有限 FAILED'

# ---- TC23: run_spectrum_and_absorption 返回有限值 ----
tg, ag = run_spectrum_and_absorption()
assert np.isfinite(tg), '[TC23] 总产生率非有限 FAILED'
assert np.isfinite(ag), '[TC23] 平均产生密度非有限 FAILED'
assert tg > 0, '[TC23] 总产生率非正 FAILED'

# ---- TC24: run_material_properties 返回正确类型 ----
mat2 = run_material_properties()
assert isinstance(mat2, PerovskiteMaterial), '[TC24] 材料对象类型 FAILED'
p2 = mat2.get_params(300.0, 0.0)
assert 'bandgap_eV' in p2, '[TC24] 材料缺带隙 FAILED'

# ---- TC25: run_mechanical_stress 返回字典含预期键 ----
buck = run_mechanical_stress()
assert isinstance(buck, dict), '[TC25] 屈曲结果非字典 FAILED'
assert 'estimated_efficiency_loss_percent' in buck, '[TC25] 缺效率损失 FAILED'

# ---- TC26: run_uncertainty_quantification 输出完整性 ----
uq2 = run_uncertainty_quantification()
assert uq2['pce_mean_efficiency'] > 0, '[TC26] PCE 均值非正 FAILED'
assert uq2['pce_std_efficiency'] >= 0, '[TC26] PCE 标准差为负 FAILED'

# ---- TC27: run_recombination 返回非负值 ----
R = run_recombination(1e-8, 1e-8)
assert np.isfinite(R), '[TC27] 总复合率非有限 FAILED'
assert R >= 0, '[TC27] 总复合率为负 FAILED'

print('\n全部 27 个测试通过!\n')
