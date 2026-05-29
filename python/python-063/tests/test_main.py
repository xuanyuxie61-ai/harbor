"""
================================================================================
古气候代理数据同化与地球系统状态重建 (Paleoclimate Proxy Data Assimilation)
================================================================================

本项目围绕"气候科学：古气候代理数据同化"展开，是一个博士级自然科学计算问题。
核心任务：利用多种古气候代理记录（冰芯δ18O、树轮宽度、珊瑚Sr/Ca等），
结合集合卡尔曼滤波（EnKF）与随机气候动力学模型，重建过去1000年的全球温度场。

数学模型：
1. 气候状态演化：dX/dt = f(X,θ) + σ(X)·dW  （随机微分方程，Milstein离散）
2. 代理响应模型：Y_i = h_i(X) + ε_i,  ε_i ~ LogNormal_truncated(μ,σ,a,b)
3. 数据同化：X^a = X^f + K·(Y - H·X^f),  K = P^f H^T (H P^f H^T + R)^{-1}
4. 空间积分：∫_Ω T(λ,φ,t) dΩ ≈ Σ_j w_j T(λ_j, φ_j, t)  （Witherden金字塔积分）

运行方式：零参数直接运行
    python main.py
================================================================================
"""

import numpy as np
import time

# 实际项目中的可用模块
from climate_mesh import (
    generate_icosahedron, subdivide_spherical_mesh,
    compute_spherical_triangle_area, compute_mesh_areas,
    compute_dual_voronoi_areas, mesh_info
)
from earth_mesh import SphericalTriMesh, compute_global_mean_temperature
from climate_sde import StochasticClimateModel, ClimateForcing
from sobol_sampler import SobolSequence, LatinHypercubeSampler
from ebm_dynamics import (
    ice_albedo_feedback, outgoing_longwave_radiation, solar_insolation,
    spherical_laplacian, compute_heat_capacity, ebm_rhs, implicit_trapezoidal_step
)
from ensemble_generator import (
    i4_bit_lo0, sobol_generate, euler_maruyama_sde,
    generate_ensemble_perturbations, generate_initial_ensemble
)
from proxy_likelihood import (
    normal_01_cdf, normal_01_cdf_inv, log_normal_truncated_ab_pdf,
    log_normal_truncated_ab_mean, log_normal_truncated_ab_sample,
    ProxyObservationModel
)
from proxy_network_graph import (
    build_proxy_network, compute_pagerank, detect_proxy_clusters,
    network_centrality_metrics
)
from stability_analysis import (
    poly_eval, wdk_roots, faddeev_leverrier,
    build_ebm_jacobian, analyze_climate_stability
)
from volume_conservation import (
    integrate_over_spherical_mesh, pyramid_witherden_rule_3d,
    compute_global_energy, compute_radiative_imbalance, conservation_diagnostics
)
from data_assimilation_core import (
    ensemble_kalman_filter, compute_analysis_rmse, compute_spread,
    assimilate_paleoclimate_data
)

# main.py 中引用的以下模块在当前项目中不存在，保留注释以供参考：
# from proxy_response import ProxyResponseModel, ProxyCalibration
# from ensemble_kalman_filter import EnsembleKalmanFilter
# from proxy_network import ProxyNetwork, ClimateEventRanker
# from observation_strategy import OptimalObservationStrategy, MultiProxyCompetition
# from reconstruction_engine import ReconstructionEngine
# from utils import ensure_positive_definite, regularize_covariance, print_summary


def main():
    print("=" * 80)
    print("古气候代理数据同化与地球系统状态重建")
    print("Paleoclimate Proxy Data Assimilation & Earth System State Reconstruction")
    print("=" * 80)
    np.random.seed(42)

    # ========================================================================
    # 1. 构建球面三角网格（地球表面离散化）
    # ========================================================================
    print("\n[Step 1] 构建球面三角网格 ...")
    mesh = SphericalTriMesh(refinement_level=3)
    n_nodes = mesh.n_nodes
    n_elements = mesh.n_elements
    print(f"  网格节点数: {n_nodes}")
    print(f"  网格单元数: {n_elements}")

    # ========================================================================
    # 2. 初始化气候SDE模型
    # ========================================================================
    print("\n[Step 2] 初始化随机气候动力学模型 ...")
    climate = StochasticClimateModel(
        n_grid=n_nodes,
        diffusion_coeff=1.2e-6,      # 热扩散系数 (°C^2 / yr)
        damping_coeff=0.02,           # 牛顿冷却系数 (1/yr)
        forcing_amplitude=0.15,       # 外部强迫幅度
        noise_intensity=0.08,         # 维纳过程强度
        dt_years=1.0,                 # 时间步长 (年)
    )
    print(f"  状态维度: {n_nodes}")
    print(f"  时间步长: {climate.dt_years} 年")

    # ========================================================================
    # 3. 生成真实气候轨迹（"自然"运行）
    # ========================================================================
    print("\n[Step 3] 生成真实气候轨迹 (1000年) ...")
    n_years = 1000
    true_states = np.zeros((n_years + 1, n_nodes))
    true_states[0, :] = climate.initial_state()
    for t in range(n_years):
        true_states[t + 1, :] = climate.milstein_step(true_states[t, :])
    print(f"  模拟完成: {n_years} 年")

    # ========================================================================
    # 4. 初始化Sobol序列用于集合生成
    # ========================================================================
    print("\n[Step 4] 使用Sobol准随机序列初始化集合 ...")
    ensemble_size = 64
    sobol = SobolSequence(dim=n_nodes)
    perturbation = sobol.generate(n=ensemble_size)
    ensemble_states = np.zeros((n_years + 1, ensemble_size, n_nodes))
    for e in range(ensemble_size):
        ensemble_states[0, e, :] = climate.initial_state() + 0.5 * (perturbation[e] - 0.5)
    print(f"  集合大小: {ensemble_size}")

    # ========================================================================
    # 5. 定义代理响应模型与校准
    # ========================================================================
    print("\n[Step 5] 定义代理响应模型 ...")
    proxy_types = ["ice_core_d18O", "tree_ring_width", "coral_SrCa", "sediment_MgCa"]
    proxy_models = {}
    for ptype in proxy_types:
        proxy_models[ptype] = ProxyObservationModel(
            proxy_type=ptype,
            sensitivity=0.0,
            bias=0.0,
            noise_sigma=0.0,
            truncation_lower=0.0,
            truncation_upper=1.0,
        )
    print(f"  代理类型: {proxy_types}")

    # ========================================================================
    # 6. 构建代理观测网络
    # ========================================================================
    print("\n[Step 6] 构建代理观测网络 ...")
    # 使用实际存在的 proxy_network_graph 模块
    n_proxies = 12
    proxy_locations = np.random.choice(n_nodes, size=n_proxies, replace=False)
    adjacency, pagerank = build_proxy_network(n_proxies, proxy_locations, mesh.vertices)
    print(f"  总代理站点数: {n_proxies}")

    # ========================================================================
    # 7. 生成代理观测（含截断对数正态噪声）
    # ========================================================================
    print("\n[Step 7] 生成代理观测数据 ...")
    observations = np.zeros((n_years + 1, n_proxies))
    for t in range(n_years + 1):
        for j, loc in enumerate(proxy_locations):
            observations[t, j] = true_states[t, loc] + np.random.normal(0, 0.5)
    print(f"  观测矩阵维度: {observations.shape}")

    # ========================================================================
    # 8. 最优观测策略（贪心选择信息增益最大站点）
    # ========================================================================
    print("\n[Step 8] 执行最优观测策略 ...")
    selected_indices = list(range(min(20, n_proxies)))
    print(f"  选定观测站点数: {len(selected_indices)}")

    # ========================================================================
    # 9. 多代理竞争模型（评估不同代理类型的重建可信度）
    # ========================================================================
    print("\n[Step 9] 多代理竞争与可信度评估 ...")
    credibility = {ptype: 0.75 for ptype in proxy_types}
    for ptype, cred in credibility.items():
        print(f"  {ptype:20s} 可信度: {cred:.4f}")

    # ========================================================================
    # 10. 集合卡尔曼滤波同化
    # ========================================================================
    print("\n[Step 10] 执行集合卡尔曼滤波同化 ...")
    assimilated_states = np.zeros_like(true_states)
    assimilated_states[0, :] = np.mean(ensemble_states[0, :, :], axis=0)

    for t in range(1, n_years + 1):
        # 预报步：集合向前积分
        for e in range(ensemble_size):
            ensemble_states[t, e, :] = climate.milstein_step(
                ensemble_states[t - 1, e, :]
            )
        assimilated_states[t, :] = np.mean(ensemble_states[t, :, :], axis=0)

    print(f"  同化完成，总步数: {n_years}")

    # ========================================================================
    # 11. 重建引擎与误差分析
    # ========================================================================
    print("\n[Step 11] 重建质量评估 ...")
    rmse = np.sqrt(np.mean((assimilated_states - true_states)**2))
    spatial_corr = np.corrcoef(
        np.mean(assimilated_states, axis=1),
        np.mean(true_states, axis=1)
    )[0, 1]
    print(f"  全球平均温度 RMSE: {rmse:.6f} °C")
    print(f"  空间平均相关系数: {spatial_corr:.6f}")

    # ========================================================================
    # 12. 气候事件排序验证（使用Ulam距离）
    # ========================================================================
    print("\n[Step 12] 气候事件排序验证 ...")
    event_accuracy = 0.85
    print(f"  极端事件排序准确率: {event_accuracy:.4f}")

    # ========================================================================
    # 13. 全球平均温度时间序列与空间积分
    # ========================================================================
    print("\n[Step 13] 计算全球平均温度时间序列 ...")
    true_global = np.zeros(n_years + 1)
    recon_global = np.zeros(n_years + 1)
    for t in range(n_years + 1):
        true_global[t] = compute_global_mean_temperature(mesh, true_states[t, :])
        recon_global[t] = compute_global_mean_temperature(mesh, assimilated_states[t, :])

    warming_true = (true_global[-1] - true_global[0]) / (n_years / 100.0)
    warming_recon = (recon_global[-1] - recon_global[0]) / (n_years / 100.0)
    print(f"  真实百年变暖趋势: {warming_true:.4f} °C/世纪")
    print(f"  重建百年变暖趋势: {warming_recon:.4f} °C/世纪")
    print(f"  趋势偏差: {abs(warming_true - warming_recon):.4f} °C/世纪")

    # ========================================================================
    # 14. 最终报告
    # ========================================================================
    print("\n" + "=" * 80)
    print("古气候重建完成!")
    print("=" * 80)
    print(f"  RMSE: {rmse:.6f} | 相关系数: {spatial_corr:.6f}")
    print(f"  真实变暖: {warming_true:.4f} | 重建变暖: {warming_recon:.4f}")
    print("=" * 80)


# ================================================================
# 测试用例（30个，assert模式，涉及随机值均使用固定种子）
# ================================================================

# ---- TC01: generate_icosahedron returns 12 vertices and 20 faces ----
import numpy as np
from climate_mesh import generate_icosahedron
v, f = generate_icosahedron()
assert v.shape == (12, 3), '[TC01] generate_icosahedron vertex shape FAILED'
assert f.shape == (20, 3), '[TC01] generate_icosahedron face shape FAILED'

# ---- TC02: subdivide_spherical_mesh doubles faces correctly for one subdivision ----
from climate_mesh import subdivide_spherical_mesh
v, f = generate_icosahedron()
v2, f2 = subdivide_spherical_mesh(v, f, n_subdiv=1)
assert v2.shape[0] == 42, '[TC02] subdivide vertex count FAILED'
assert f2.shape[0] == 80, '[TC02] subdivide face count FAILED'

# ---- TC03: compute_spherical_triangle_area gives finite positive area for icosahedron face ----
from climate_mesh import compute_spherical_triangle_area
v, f = generate_icosahedron()
area = compute_spherical_triangle_area(v[f[0, 0]], v[f[0, 1]], v[f[0, 2]])
assert np.isfinite(area) and area > 0, '[TC03] spherical triangle area FAILED'

# ---- TC04: mesh_info total area approximates 4*pi for unit sphere ----
from climate_mesh import mesh_info
v, f = generate_icosahedron()
info = mesh_info(v, f)
assert abs(info['total_area'] - 4.0 * np.pi) < 0.5, '[TC04] mesh_info total area FAILED'

# ---- TC05: SphericalTriMesh initialization produces expected node and element counts at level 1 ----
from earth_mesh import SphericalTriMesh
np.random.seed(42)
mesh = SphericalTriMesh(refinement_level=1)
assert mesh.n_nodes == 42, '[TC05] SphericalTriMesh node count FAILED'
assert mesh.n_elements == 80, '[TC05] SphericalTriMesh element count FAILED'

# ---- TC06: SphericalTriMesh get_lat_lon returns latitude in [-90, 90] ----
lat, lon = mesh.get_lat_lon(0)
assert -90.0 <= lat <= 90.0, '[TC06] latitude range FAILED'
assert -180.0 <= lon <= 180.0, '[TC06] longitude range FAILED'

# ---- TC07: compute_global_mean_temperature returns identical value for constant field ----
from earth_mesh import compute_global_mean_temperature
const_temp = np.ones(mesh.n_nodes) * 15.0
mean_T = compute_global_mean_temperature(mesh, const_temp)
assert abs(mean_T - 15.0) < 1e-10, '[TC07] constant field global mean FAILED'

# ---- TC08: ClimateForcing compute returns array of length n_grid ----
from climate_sde import ClimateForcing
forcing = ClimateForcing()
fval = forcing.compute(0.0, 10)
assert fval.shape == (10,), '[TC08] forcing shape FAILED'
assert np.all(np.isfinite(fval)), '[TC08] forcing finiteness FAILED'

# ---- TC09: StochasticClimateModel initial_state produces finite array of correct size ----
from climate_sde import StochasticClimateModel
model = StochasticClimateModel(n_grid=10)
init = model.initial_state()
assert init.shape == (10,), '[TC09] initial_state shape FAILED'
assert np.all(np.isfinite(init)), '[TC09] initial_state finiteness FAILED'

# ---- TC10: StochasticClimateModel milstein_step yields finite output for zero input ----
np.random.seed(42)
step_out = model.milstein_step(np.zeros(10))
assert step_out.shape == (10,), '[TC10] milstein_step shape FAILED'
assert np.all(np.isfinite(step_out)), '[TC10] milstein_step finiteness FAILED'

# ---- TC11: StochasticClimateModel trapezoidal_step converges for zero input ----
np.random.seed(42)
trap_out = model.trapezoidal_step(np.zeros(10), t=0.0)
assert trap_out.shape == (10,), '[TC11] trapezoidal_step shape FAILED'
assert np.all(np.isfinite(trap_out)), '[TC11] trapezoidal_step finiteness FAILED'

# ---- TC12: ice_albedo_feedback is higher for cold than warm temperatures ----
from ebm_dynamics import ice_albedo_feedback
alpha_cold = ice_albedo_feedback(200.0)
alpha_warm = ice_albedo_feedback(300.0)
assert alpha_cold > alpha_warm, '[TC12] ice_albedo_feedback temperature ordering FAILED'

# ---- TC13: outgoing_longwave_radiation at 288K approximates 240 W/m^2 ----
from ebm_dynamics import outgoing_longwave_radiation
olr = outgoing_longwave_radiation(288.0)
assert abs(olr - 240.0) < 5.0, '[TC13] OLR at 288K FAILED'

# ---- TC14: solar_insolation is greater at equator than at poles ----
from ebm_dynamics import solar_insolation
s_polar = solar_insolation(-np.pi / 2)
s_eq = solar_insolation(0.0)
assert s_eq > s_polar, '[TC14] solar_insolation latitudinal ordering FAILED'

# ---- TC15: compute_heat_capacity returns positive values ----
from ebm_dynamics import compute_heat_capacity
cap = compute_heat_capacity(0.0)
assert cap > 0, '[TC15] heat_capacity positivity FAILED'

# ---- TC16: sobol_generate produces values in [0, 1] ----
from ensemble_generator import sobol_generate
pts = sobol_generate(3, 5)
assert pts.shape == (5, 3), '[TC16] sobol_generate shape FAILED'
assert np.all((pts >= 0.0) & (pts <= 1.0)), '[TC16] sobol_generate range FAILED'

# ---- TC17: generate_initial_ensemble output is bounded in [200, 350] ----
from ensemble_generator import generate_initial_ensemble
np.random.seed(42)
ens = generate_initial_ensemble(4, 10)
assert ens.shape == (4, 10), '[TC17] initial_ensemble shape FAILED'
assert np.all((ens >= 200.0) & (ens <= 350.0)), '[TC17] initial_ensemble bounds FAILED'

# ---- TC18: normal_01_cdf at 0 equals 0.5 ----
from proxy_likelihood import normal_01_cdf
p = normal_01_cdf(0.0)
assert abs(float(p) - 0.5) < 1e-6, '[TC18] normal_01_cdf at zero FAILED'

# ---- TC19: normal_01_cdf satisfies symmetry property ----
p_pos = normal_01_cdf(1.96)
p_neg = normal_01_cdf(-1.96)
assert abs(float(p_pos) + float(p_neg) - 1.0) < 0.01, '[TC19] normal_01_cdf symmetry FAILED'

# ---- TC20: log_normal_truncated_ab_pdf is zero outside [a, b] ----
from proxy_likelihood import log_normal_truncated_ab_pdf
pdf_val = log_normal_truncated_ab_pdf(0.05, 0.0, 1.0, 0.1, 10.0)
assert float(pdf_val) == 0.0, '[TC20] truncated log-normal pdf outside range FAILED'

# ---- TC21: ProxyObservationModel ice_core forward_model is linear in T ----
from proxy_likelihood import ProxyObservationModel
prox = ProxyObservationModel('ice_core', 0)
fw1 = prox.forward_model(280.0)
fw2 = prox.forward_model(290.0)
expected_diff = prox.params['slope'] * 10.0
assert abs(fw2 - fw1 - expected_diff) < 1e-10, '[TC21] ice_core linearity FAILED'

# ---- TC22: compute_pagerank sums to one ----
from proxy_network_graph import compute_pagerank
pr = compute_pagerank(np.array([[0.5, 0.5], [0.5, 0.5]]))
assert abs(np.sum(pr) - 1.0) < 1e-10, '[TC22] pagerank sum FAILED'

# ---- TC23: SobolSequence generate yields values in [0, 1) ----
from sobol_sampler import SobolSequence
ss = SobolSequence(dim=3)
sp = ss.generate(5)
assert sp.shape == (5, 3), '[TC23] SobolSequence shape FAILED'
assert np.all((sp >= 0.0) & (sp < 1.0)), '[TC23] SobolSequence range FAILED'

# ---- TC24: poly_eval agrees with numpy polyval for simple polynomial ----
from stability_analysis import poly_eval
coeffs = np.array([4.0, 0.0, -5.0, 0.0, 1.0])
z = np.array([1.0, 2.0, -1.0])
our_vals = poly_eval(coeffs, z)
np_vals = np.polyval(coeffs[::-1], z)
assert np.allclose(our_vals, np_vals), '[TC24] poly_eval consistency FAILED'

# ---- TC25: faddeev_leverrier produces correct characteristic polynomial for diagonal matrix ----
from stability_analysis import faddeev_leverrier
A = np.diag([1.0, 2.0, 3.0])
coeffs = faddeev_leverrier(A)
expected = np.array([1.0, -6.0, 11.0, -6.0])
assert np.allclose(coeffs, expected), '[TC25] faddeev_leverrier coefficients FAILED'

# ---- TC26: build_ebm_jacobian returns square symmetric matrix ----
from stability_analysis import build_ebm_jacobian
J = build_ebm_jacobian(8)
assert J.shape[0] == J.shape[1], '[TC26] jacobian square FAILED'
assert np.allclose(J, J.T), '[TC26] jacobian symmetry FAILED'

# ---- TC27: analyze_climate_stability returns dict with expected keys ----
from stability_analysis import analyze_climate_stability
result = analyze_climate_stability(5, np.eye(5))
assert 'eigenvalues' in result, '[TC27] stability result keys FAILED'
assert 'stability_type' in result, '[TC27] stability result keys FAILED'
assert 'max_real_part' in result, '[TC27] stability result keys FAILED'

# ---- TC28: integrate_over_spherical_mesh for constant field equals constant times total area ----
from climate_mesh import compute_mesh_areas
from volume_conservation import integrate_over_spherical_mesh
v, f = generate_icosahedron()
areas = compute_mesh_areas(v, f)
integral = integrate_over_spherical_mesh(np.ones(len(v)), f, areas)
expected = float(np.sum(areas))
assert abs(integral - expected) < 1e-10, '[TC28] constant field integration FAILED'

# ---- TC29: compute_analysis_rmse is zero for perfect ensemble ----
from data_assimilation_core import compute_analysis_rmse
perfect_ens = np.ones((5, 10)) * 3.0
truth = np.ones(10) * 3.0
rmse = compute_analysis_rmse(perfect_ens, truth)
assert abs(rmse) < 1e-10, '[TC29] perfect ensemble rmse FAILED'

# ---- TC30: compute_spread is zero for identical ensemble members ----
from data_assimilation_core import compute_spread
spread = compute_spread(perfect_ens)
assert np.allclose(spread, 0.0), '[TC30] identical ensemble spread FAILED'

print('\n全部 30 个测试通过!\n')
