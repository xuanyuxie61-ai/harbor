"""
main.py

极端天气事件归因分析的统一入口（零参数可运行）。

本项目基于 15 个科研代码项目的核心算法，融合构建了面向
"气候科学：极端天气事件归因分析"的博士级计算框架。

运行方式：
    python main.py

无需任何参数，将自动生成合成气候数据并执行完整的归因分析流程。
"""

import numpy as np
import sys

# 导入所有模块
from sparse_climate_matrix import build_climate_laplacian_sparse
from climate_percolation import run_percolation_attribution
from delaunay_mesh import build_event_mesh
from triangulation_quadrature import integrate_over_triangulation, integrate_nodal_over_triangulation
from spherical_climate_quad import global_radiation_forcing_integral, sphere01_quad_mc
from fast_spectral_quadrature import compute_total_column_water_vapor
from wedge_atmosphere_quad import integrate_over_atmospheric_column
from monte_carlo_ensemble import (
    generate_ensemble_perturbations,
    ensemble_attribution_distance,
    monte_carlo_line_integral,
)
from covariance_pfaffian import extreme_event_pfaffian_correlation, pfaffian_LTL
from climate_interpolation import regrid_field_nearest, dominant_scale_analysis
from energy_cascade_ode import (
    energy_cascade_exact,
    solve_energy_cascade_rk4,
    energy_saturation_time,
)
from regional_aggregation import climate_region_summary
from spectral_analysis import dominant_periods, spectral_coherence, prime_factors


def generate_synthetic_climate_field(m=32, n=32, seed=42):
    """
    生成合成气候异常场。

    模拟一个包含多个极端事件中心的降水异常场：
        φ(x,y) = Σ_i A_i * exp( -[(x-x_i)^2 + (y-y_i)^2] / (2σ_i^2) )
                 + ε(x,y)
    其中 A_i 为极端事件强度，ε 为白噪声。
    """
    rng = np.random.default_rng(seed)
    x = np.linspace(-1, 1, n)
    y = np.linspace(-1, 1, m)
    X, Y = np.meshgrid(x, y)

    field = np.zeros((m, n), dtype=np.float64)
    # 添加 3 个极端事件中心
    centers = [(-0.5, 0.3, 2.5, 0.2), (0.4, -0.4, 2.8, 0.15), (0.0, 0.7, 2.2, 0.25)]
    for cx, cy, amp, sigma in centers:
        field += amp * np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / (2 * sigma ** 2))

    # 添加背景趋势
    field += 0.5 * np.sin(3 * np.pi * X) * np.cos(2 * np.pi * Y)
    # 添加白噪声
    field += 0.2 * rng.standard_normal((m, n))

    # 标准化为 SPI 风格（均值为 0，标准差为 1）
    field = (field - np.mean(field)) / np.std(field)
    return field, X, Y


def main():
    print("=" * 70)
    print("极端天气事件归因分析系统 (CEEAS)")
    print("Climate Extreme Event Attribution System")
    print("=" * 70)

    # ============================================================
    # 1. 生成合成气候数据
    # ============================================================
    print("\n[1/10] 生成合成气候异常场...")
    field, X, Y = generate_synthetic_climate_field(m=40, n=40, seed=65)
    print(f"    场维度: {field.shape}")
    print(f"    场均值: {np.mean(field):.4f}, 标准差: {np.std(field):.4f}")
    print(f"    最大值: {np.max(field):.4f}, 最小值: {np.min(field):.4f}")

    # ============================================================
    # 2. 渗流分析：识别极端事件簇（基于 865_percolation_simulation）
    # ============================================================
    print("\n[2/10] 渗流分析：识别极端事件连通簇...")
    perc_result = run_percolation_attribution(field, threshold=2.0)
    print(f"    占据概率 p = {perc_result['posites']:.4f}")
    print(f"    极端格点数 = {perc_result['nosites']}")
    print(f"    连通分量数 = {len(perc_result['component_sizes'])}")
    print(f"    最大分量大小 = {max(perc_result['component_sizes']) if perc_result['component_sizes'] else 0}")
    print(f"    序参量 P_∞ = {perc_result['p_infinity']:.4f}")
    print(f"    关联长度 ξ = {perc_result['correlation_length']:.4f}")
    print(f"    水平跨越簇 = {perc_result['spanx']}, 垂直跨越簇 = {perc_result['spany']}")

    # ============================================================
    # 3. Delaunay 三角剖分（基于 1330_triangulation）
    # ============================================================
    print("\n[3/10] Delaunay 三角剖分：构建极端事件自适应网格...")
    meshes = build_event_mesh(perc_result["components"], X, Y)
    total_triangles = sum(m["triangles"].shape[1] for m in meshes.values())
    print(f"    检测到 {len(meshes)} 个事件区域的三角网格")
    print(f"    总三角形数 = {total_triangles}")

    # ============================================================
    # 4. 三角网格求积：积分区域物理量（基于 1347_triangulation_quad）
    # ============================================================
    print("\n[4/10] 三角网格求积：计算极端事件区域能量积分...")
    total_energy = 0.0
    total_area = 0.0
    for cid, mesh in meshes.items():
        # 使用节点值积分
        node_vals = np.zeros(mesh["nodes"].shape[1])
        for i in range(mesh["nodes"].shape[1]):
            # 从原始场中插值
            nx_idx = int(np.clip(
                (mesh["nodes"][0, i] - X.min()) / (X.max() - X.min()) * (X.shape[1] - 1),
                0, X.shape[1] - 1
            ))
            ny_idx = int(np.clip(
                (mesh["nodes"][1, i] - Y.min()) / (Y.max() - Y.min()) * (Y.shape[0] - 1),
                0, Y.shape[0] - 1
            ))
            node_vals[i] = field[ny_idx, nx_idx]

        val, area = integrate_nodal_over_triangulation(
            mesh["nodes"], mesh["triangles"], node_vals
        )
        total_energy += val
        total_area += area

    print(f"    极端事件区域总能量积分 = {total_energy:.4f}")
    print(f"    极端事件区域总面积 = {total_area:.4f}")
    if total_area > 1e-14:
        print(f"    区域平均异常强度 = {total_energy / total_area:.4f}")

    # ============================================================
    # 5. 球面求积：全球辐射强迫（基于 1126_sphere_quad）
    # ============================================================
    print("\n[5/10] 球面求积：估算全球平均辐射强迫...")

    def forcing_field(lat, lon):
        # 简化的辐射强迫分布
        return 2.5 + 1.5 * np.sin(2 * lat) * np.cos(3 * lon)

    mean_forcing, node_num = global_radiation_forcing_integral(forcing_field, factor=2)
    print(f"    全球平均辐射强迫 = {mean_forcing:.4f} W/m²")
    print(f"    球面求积节点数 = {node_num}")

    # 蒙特卡洛验证
    def mc_forcing(xyz):
        lat = np.arcsin(np.clip(xyz[2], -1.0, 1.0))
        lon = np.arctan2(xyz[1], xyz[0])
        return forcing_field(lat, lon)

    mc_result = sphere01_quad_mc(mc_forcing, 5000, seed=65)
    mc_mean = mc_result / (4.0 * np.pi)
    print(f"    蒙特卡洛验证（N=5000）= {mc_mean:.4f} W/m²")

    # ============================================================
    # 6. 快速谱求积：大气柱水汽积分（基于 939_quad_fast_rule）
    # ============================================================
    print("\n[6/10] 快速谱求积：计算大气柱水汽含量...")
    z = np.linspace(0, 15000, 50)  # 0-15km，50层
    q = 0.01 * np.exp(-z / 3000.0)  # 比湿随高度指数衰减
    rho = 1.225 * np.exp(-z / 8500.0)  # 空气密度
    tcwv = compute_total_column_water_vapor(z, q, rho)
    print(f"    总水汽含量 TCWV = {tcwv:.4f} kg/m²")

    # ============================================================
    # 7. 楔形区域求积：3D 大气柱能量积分（基于 1407_wedge_felippa_rule）
    # ============================================================
    print("\n[7/10] 楔形区域求积：三维大气柱能量积分...")
    if meshes:
        first_mesh = list(meshes.values())[0]
        tri_v = first_mesh["nodes"][:, first_mesh["triangles"][:, 0]]

        def energy_density(xyz):
            # 简化的能量密度：随高度衰减
            return np.exp(-xyz[2] / 5000.0)

        col_energy = integrate_over_atmospheric_column(
            energy_density, tri_v, z_bottom=0.0, z_top=15000.0,
            line_order=3, triangle_order=7
        )
        print(f"    首个事件区域的 3D 柱能量 = {col_energy:.4f} J/m²")

    # ============================================================
    # 8. 蒙特卡洛集合分析（基于 683_line_monte_carlo + 1177_subset_distance）
    # ============================================================
    print("\n[8/10] 蒙特卡洛集合分析：归因不确定性量化...")
    ensemble = generate_ensemble_perturbations(field, n_members=20,
                                                perturbation_scale=0.15, seed=65)
    # 对每个集合成员进行渗流分析
    ensemble_binary = np.zeros((20,) + field.shape, dtype=np.int64)
    for i in range(20):
        occ = (ensemble[i] > 2.0).astype(np.int64)
        ensemble_binary[i] = occ

    mean_dist, consensus = ensemble_attribution_distance(ensemble_binary,
                                                          threshold_ratio=0.6)
    print(f"    集合成员间平均 Hamming 距离 = {mean_dist:.2f}")
    print(f"    共识区域格点数 = {int(np.sum(consensus))}")

    # 线段蒙特卡洛积分：验证能量级串方程的积分形式
    def energy_rate(x):
        # 验证 ∫_0^1 x^2(1-x) dx = 1/12
        return x ** 2 * (1.0 - x)
    mc_energy = monte_carlo_line_integral(energy_rate, 20000, seed=65)
    print(f"    能量级串方程 MC 积分验证 = {mc_energy:.6f} (精确值=0.083333)")

    # ============================================================
    # 9. Pfaffian 协方差分析（基于 1280_toms923）
    # ============================================================
    print("\n[9/10] Pfaffian 协方差分析：极端事件空间相关模型...")
    if meshes:
        first_mesh = list(meshes.values())[0]
        n_pts = min(first_mesh["nodes"].shape[1], 8)
        nodes = first_mesh["nodes"][:, :n_pts]
        K, pf = extreme_event_pfaffian_correlation(nodes, correlation_length=0.3)
        print(f"    节点数 = {n_pts}")
        print(f"    斜对称协方差矩阵 Pfaffian = {pf:.6e}")
        det_check = np.linalg.det(K) if n_pts % 2 == 0 else 0.0
        if n_pts % 2 == 0:
            print(f"    det(K) = {det_check:.6e}")
            print(f"    |pf|^2 = {pf**2:.6e}")
            if abs(det_check) > 1e-14:
                rel_err = abs(pf ** 2 - det_check) / abs(det_check)
                print(f"    相对误差 = {rel_err:.2e}")

    # ============================================================
    # 10. 能量级串动力学（基于 437_flame_ode）
    # ============================================================
    print("\n[10/10] 能量级串动力学：极端事件发展模拟...")
    t_span = (0.0, 5.0)
    y0 = 0.01
    t_num, y_num = solve_energy_cascade_rk4(t_span, y0, n_steps=1000)
    t_exact = np.linspace(0, 5, 100)
    y_exact = energy_cascade_exact(t_exact, y0)
    tau_sat = energy_saturation_time(y0, epsilon=0.99)
    print(f"    初始扰动 δ = {y0}")
    print(f"    数值解终值 = {y_num[-1]:.4f}")
    print(f"    精确解终值 = {y_exact[-1]:.4f}")
    print(f"    99% 饱和时间 τ_sat = {tau_sat:.4f}")

    # ============================================================
    # 11. 区域统计聚合（基于 118_brc_naive）
    # ============================================================
    print("\n[附加] 区域统计聚合...")
    # 将场分为 4 个象限作为区域
    grid_regions = np.zeros(field.shape, dtype=np.int64)
    m, n = field.shape
    grid_regions[:m // 2, :n // 2] = 1
    grid_regions[:m // 2, n // 2:] = 2
    grid_regions[m // 2:, :n // 2] = 3
    grid_regions[m // 2:, n // 2:] = 4

    summary = climate_region_summary(grid_regions.flatten(), field.flatten())
    for rid, s in summary["stats"].items():
        print(f"    区域 {rid}: 均值={s['mean']:.3f}, 最大值={s['max']:.3f}, "
              f"REI={summary['rei'][rid]:.3f}")

    # ============================================================
    # 12. 谱分析（基于 911_prime_factors）
    # ============================================================
    print("\n[附加] 谱分析：周期识别与 FFT 优化...")
    # 对场中心行进行谱分析
    center_row = field[m // 2, :]
    periods, freqs, power = dominant_periods(center_row, sample_rate=1.0, n_peaks=3)
    print(f"    主导周期: {['%.2f' % p for p in periods]}")

    # 最优 FFT 长度
    opt_len = len(center_row)
    print(f"    当前长度 {opt_len} 的质因数分解: {prime_factors(opt_len)}")

    # 稀疏矩阵验证（基于 992_r8ri）
    print("\n[附加] 稀疏矩阵验证...")
    laplacian = build_climate_laplacian_sparse(5, 5)
    x_test = np.ones(25)
    b_test = laplacian.mv(x_test)
    print(f"    5x5 Laplacian 稀疏矩阵-向量乘法完成")
    print(f"    结果范数 = {np.linalg.norm(b_test):.4f}")

    # ============================================================
    # 最终总结
    # ============================================================
    print("\n" + "=" * 70)
    print("归因分析完成。主要结论：")
    print(f"  - 极端事件占据概率 p = {perc_result['posites']:.4f}")
    print(f"  - 最大连通分量序参量 P_∞ = {perc_result['p_infinity']:.4f}")
    print(f"  - 事件区域总能量积分 = {total_energy:.4f}")
    print(f"  - 全球平均辐射强迫 = {mean_forcing:.4f} W/m²")
    print(f"  - 总水汽含量 TCWV = {tcwv:.4f} kg/m²")
    print(f"  - 集合共识区域占比 = {100*np.sum(consensus)/consensus.size:.1f}%")
    print(f"  - 能量饱和时间 τ_sat = {tau_sat:.4f}")
    print("=" * 70)
    print("所有计算步骤已完成，无报错。")

    return 0


if __name__ == "__main__":
    main()
# ---- TC01: 合成气候场标准化特性 ----
field_test, X_test, Y_test = generate_synthetic_climate_field(m=16, n=16, seed=42)
assert field_test.shape == (16, 16), '[TC01] 合成气候场标准化特性 FAILED'
assert abs(np.mean(field_test)) < 1e-9, '[TC01] 合成气候场标准化特性 FAILED'
assert abs(np.std(field_test) - 1.0) < 1e-9, '[TC01] 合成气候场标准化特性 FAILED'

# ---- TC02: 渗流分析输出结构完整 ----
np.random.seed(42)
field_perc = np.random.randn(20, 20)
perc = run_percolation_attribution(field_perc, threshold=1.5)
assert 'posites' in perc, '[TC02] 渗流分析输出结构完整 FAILED'
assert 0.0 <= perc['posites'] <= 1.0, '[TC02] 渗流分析输出结构完整 FAILED'
assert perc['nosites'] >= 0, '[TC02] 渗流分析输出结构完整 FAILED'

# ---- TC03: Delaunay事件网格构建返回字典 ----
np.random.seed(42)
test_field, test_X, test_Y = generate_synthetic_climate_field(m=20, n=20, seed=42)
test_perc = run_percolation_attribution(test_field, threshold=1.0)
meshes = build_event_mesh(test_perc['components'], test_X, test_Y)
assert isinstance(meshes, dict), '[TC03] Delaunay事件网格构建返回字典 FAILED'

# ---- TC04: 三角网格节点积分常数函数 ----
node_xy = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
tri_node = np.array([[0], [1], [2]], dtype=np.int64)
val, area = integrate_nodal_over_triangulation(node_xy, tri_node, np.array([1.0, 1.0, 1.0]))
assert abs(area - 0.5) < 1e-10, '[TC04] 三角网格节点积分常数函数 FAILED'
assert abs(val - 0.5) < 1e-10, '[TC04] 三角网格节点积分常数函数 FAILED'

# ---- TC05: 球面蒙特卡洛积分常数函数 ----
mc_val = sphere01_quad_mc(lambda xyz: 1.0, 5000, seed=42)
assert abs(mc_val - 4.0*np.pi) < 0.3, '[TC05] 球面蒙特卡洛积分常数函数 FAILED'

# ---- TC06: 全球辐射强迫返回值结构正确 ----
mean_f, nn = global_radiation_forcing_integral(lambda lat, lon: 2.5, factor=2)
assert isinstance(mean_f, (float, np.floating)), '[TC06] 全球辐射强迫返回值结构正确 FAILED'
assert nn > 0, '[TC06] 全球辐射强迫返回值结构正确 FAILED'

# ---- TC07: 大气柱水汽含量为正 ----
z = np.linspace(0, 10000, 20)
q = 0.01 * np.exp(-z / 3000.0)
rho = 1.225 * np.exp(-z / 8500.0)
tcwv = compute_total_column_water_vapor(z, q, rho)
assert tcwv > 0.0, '[TC07] 大气柱水汽含量为正 FAILED'

# ---- TC08: 3D大气柱积分常数函数 ----
tri_v = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
col_en = integrate_over_atmospheric_column(lambda xyz: 1.0, tri_v, z_bottom=0.0, z_top=10.0, line_order=3, triangle_order=7)
assert abs(col_en - 5.0) < 0.1, '[TC08] 3D大气柱积分常数函数 FAILED'

# ---- TC09: 集合扰动可复现性 ----
base = np.ones((5, 5))
ens1 = generate_ensemble_perturbations(base, n_members=3, perturbation_scale=0.1, seed=42)
ens2 = generate_ensemble_perturbations(base, n_members=3, perturbation_scale=0.1, seed=42)
assert np.allclose(ens1, ens2), '[TC09] 集合扰动可复现性 FAILED'

# ---- TC10: 集合归因距离非负 ----
np.random.seed(42)
ens_bin = np.random.randint(0, 2, size=(5, 4, 4))
md, cons = ensemble_attribution_distance(ens_bin, threshold_ratio=0.5)
assert md >= 0.0, '[TC10] 集合归因距离非负 FAILED'
assert cons.shape == (4, 4), '[TC10] 集合归因距离非负 FAILED'

# ---- TC11: 蒙特卡洛线积分解析验证 ----
np.random.seed(42)
mc_int = monte_carlo_line_integral(lambda x: x**2, 100000, seed=42)
assert abs(mc_int - 1.0/3.0) < 0.02, '[TC11] 蒙特卡洛线积分解析验证 FAILED'

# ---- TC12: Pfaffian块对角矩阵返回标量 ----
A_block = np.array([[0,1,0,0],[-1,0,0,0],[0,0,0,1],[0,0,-1,0]], dtype=np.float64)
pf_block = pfaffian_LTL(A_block)
assert isinstance(pf_block, (float, np.floating)), '[TC12] Pfaffian块对角矩阵返回标量 FAILED'

# ---- TC13: Pfaffian奇数维矩阵为0 ----
A_odd = np.array([[0,1,2],[-1,0,3],[-2,-3,0]], dtype=np.float64)
assert abs(pfaffian_LTL(A_odd)) < 1e-10, '[TC13] Pfaffian奇数维矩阵为0 FAILED'

# ---- TC14: Pfaffian相关模型返回矩阵和值 ----
nodes = np.array([[0.0, 1.0, 0.5], [0.0, 0.0, 1.0]])
K, pf_val = extreme_event_pfaffian_correlation(nodes, correlation_length=1.0)
assert K.shape == (3, 3), '[TC14] Pfaffian相关模型返回矩阵和值 FAILED'
assert abs(pf_val) < 10.0, '[TC14] Pfaffian相关模型返回矩阵和值 FAILED'

# ---- TC15: 能量级串精确解单调递增 ----
t_arr = np.linspace(0, 3, 50)
y_exact = energy_cascade_exact(t_arr, y0=0.01)
assert np.all(np.diff(y_exact) >= -1e-10), '[TC15] 能量级串精确解单调递增 FAILED'
assert y_exact[-1] > y_exact[0], '[TC15] 能量级串精确解单调递增 FAILED'

# ---- TC16: RK4数值解接近精确解 ----
t_num, y_num = solve_energy_cascade_rk4((0.0, 2.0), 0.01, n_steps=500)
t_ref = np.linspace(0, 2, 100)
y_ref = energy_cascade_exact(t_ref, 0.01)
y_interp = np.interp(t_ref, t_num, y_num)
max_err = np.max(np.abs(y_ref - y_interp))
assert max_err < 0.05, '[TC16] RK4数值解接近精确解 FAILED'

# ---- TC17: 能量饱和时间合理 ----
tau = energy_saturation_time(0.01, epsilon=0.99)
assert tau > 0.0, '[TC17] 能量饱和时间合理 FAILED'

# ---- TC18: 区域气候摘要统计正确 ----
grid_regions = np.array([1, 1, 2, 2, 2, 3])
grid_anomalies = np.array([10.0, 12.0, 5.0, 6.0, 7.0, 20.0])
summary = climate_region_summary(grid_regions, grid_anomalies)
assert 1 in summary['stats'], '[TC18] 区域气候摘要统计正确 FAILED'
assert abs(summary['stats'][1]['mean'] - 11.0) < 1e-10, '[TC18] 区域气候摘要统计正确 FAILED'

# ---- TC19: 主导周期检测正弦信号 ----
t_sig = np.linspace(0, 10, 256)
signal = np.sin(2 * np.pi * t_sig * 0.5)
periods, freqs, power = dominant_periods(signal, sample_rate=10.0, n_peaks=3)
assert len(periods) > 0, '[TC19] 主导周期检测正弦信号 FAILED'
assert np.max(power) > 0, '[TC19] 主导周期检测正弦信号 FAILED'

# ---- TC20: 质因数分解正确 ----
assert prime_factors(75) == [3, 5, 5], '[TC20] 质因数分解正确 FAILED'
assert prime_factors(13) == [13], '[TC20] 质因数分解正确 FAILED'

# ---- TC21: 稀疏Laplacian矩阵向量乘法 ----
lap = build_climate_laplacian_sparse(3, 3)
x_vec = np.ones(9)
b_vec = lap.mv(x_vec)
assert b_vec.shape == (9,), '[TC21] 稀疏Laplacian矩阵向量乘法 FAILED'
expected_b = np.array([2, 1, 2, 1, 0, 1, 2, 1, 2], dtype=float)
assert np.allclose(b_vec, expected_b), '[TC21] 稀疏Laplacian矩阵向量乘法 FAILED'

# ---- TC22: 最近邻重网格化维度正确 ----
lon_src = np.linspace(0, 10, 5)
lat_src = np.linspace(0, 10, 5)
field_src = np.arange(25).reshape(5, 5).astype(float)
lon_tgt = np.linspace(0, 10, 3)
lat_tgt = np.linspace(0, 10, 3)
field_tgt = regrid_field_nearest(lon_src, lat_src, field_src, lon_tgt, lat_tgt)
assert field_tgt.shape == (3, 3), '[TC22] 最近邻重网格化维度正确 FAILED'

# ---- TC23: 主导尺度分析返回实根数组 ----
np.random.seed(42)
test_1d = np.random.randn(64)
real_roots, coeffs, freqs, power = dominant_scale_analysis(test_1d, max_degree=4)
assert isinstance(real_roots, np.ndarray), '[TC23] 主导尺度分析返回实根数组 FAILED'

# ---- TC24: 三角网格order3积分多项式精确 ----
node_xy_sq = np.array([[0.0, 1.0, 0.0, 1.0], [0.0, 0.0, 1.0, 1.0]])
tri_node_sq = np.array([[0, 0], [1, 3], [3, 2]], dtype=np.int64)
val_sq, area_sq = integrate_over_triangulation(node_xy_sq, tri_node_sq, lambda pts: pts[0,:]**2 + pts[1,:]**2, "order3")
assert abs(area_sq - 1.0) < 1e-10, '[TC24] 三角网格order3积分多项式精确 FAILED'
assert abs(val_sq - 2.0/3.0) < 1e-10, '[TC24] 三角网格order3积分多项式精确 FAILED'

# ---- TC25: 谱相干性范围在0到1之间 ----
np.random.seed(42)
s1 = np.sin(2 * np.pi * np.linspace(0, 10, 128))
s2 = np.sin(2 * np.pi * np.linspace(0, 10, 128))
freqs_coh, coh = spectral_coherence(s1, s2, sample_rate=10.0)
assert np.all(coh >= -1e-10), '[TC25] 谱相干性范围在0到1之间 FAILED'
assert np.all(coh <= 1.0 + 1e-10), '[TC25] 谱相干性范围在0到1之间 FAILED'

# ---- TC26: 渗流占据概率零场为0 ----
zero_field = np.zeros((10, 10))
perc_zero = run_percolation_attribution(zero_field, threshold=0.5)
assert perc_zero['posites'] == 0.0, '[TC26] 渗流占据概率零场为0 FAILED'
assert perc_zero['p_infinity'] == 0.0, '[TC26] 渗流占据概率零场为0 FAILED'

# ---- TC27: 垂直柱积分梯形法则线性函数精确 ----
from fast_spectral_quadrature import integrate_vertical_column
z_col = np.array([0.0, 1.0, 2.0])
v_col = np.array([1.0, 2.0, 3.0])
val_trap = integrate_vertical_column(z_col, v_col, method="trapezoid")
assert abs(val_trap - 4.0) < 1e-10, '[TC27] 垂直柱积分梯形法则线性函数精确 FAILED'

# ---- TC28: 快速Fejer积分多项式精确 ----
from fast_spectral_quadrature import fejer1_integrate_fast
val_fj = fejer1_integrate_fast(lambda x: x**4, 64)
assert abs(val_fj - 2.0/5.0) < 1e-10, '[TC28] 快速Fejer积分多项式精确 FAILED'

# ---- TC29: 连通分量标记和跨越分析 ----
from climate_percolation import components_2d, spanning_analysis
occ_test = np.zeros((5, 5), dtype=np.int64)
occ_test[2, :] = 1
cls_test = components_2d(occ_test)
sx, sy, sizes = spanning_analysis(cls_test, 5, 5)
assert sx == 1, '[TC29] 连通分量标记和跨越分析 FAILED'
assert sy == 0, '[TC29] 连通分量标记和跨越分析 FAILED'
assert sizes == [5], '[TC29] 连通分量标记和跨越分析 FAILED'

# ---- TC30: WDK多项式求根x平方减1 ----
from climate_interpolation import wdk_roots
roots = wdk_roots(np.array([-1.0, 0.0, 1.0]))
assert len(roots) == 2, '[TC30] WDK多项式求根x平方减1 FAILED'
assert np.allclose(np.sort(np.real(roots)), [-1.0, 1.0], atol=1e-8), '[TC30] WDK多项式求根x平方减1 FAILED'

print('\n全部 30 个测试通过!\n')
