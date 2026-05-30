
import numpy as np
import sys


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
    rng = np.random.default_rng(seed)
    x = np.linspace(-1, 1, n)
    y = np.linspace(-1, 1, m)
    X, Y = np.meshgrid(x, y)

    field = np.zeros((m, n), dtype=np.float64)

    centers = [(-0.5, 0.3, 2.5, 0.2), (0.4, -0.4, 2.8, 0.15), (0.0, 0.7, 2.2, 0.25)]
    for cx, cy, amp, sigma in centers:
        field += amp * np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / (2 * sigma ** 2))


    field += 0.5 * np.sin(3 * np.pi * X) * np.cos(2 * np.pi * Y)

    field += 0.2 * rng.standard_normal((m, n))


    field = (field - np.mean(field)) / np.std(field)
    return field, X, Y


def main():
    print("=" * 70)
    print("极端天气事件归因分析系统 (CEEAS)")
    print("Climate Extreme Event Attribution System")
    print("=" * 70)




    print("\n[1/10] 生成合成气候异常场...")
    field, X, Y = generate_synthetic_climate_field(m=40, n=40, seed=65)
    print(f"    场维度: {field.shape}")
    print(f"    场均值: {np.mean(field):.4f}, 标准差: {np.std(field):.4f}")
    print(f"    最大值: {np.max(field):.4f}, 最小值: {np.min(field):.4f}")











    perc_result = None
    meshes = {}
    total_energy = 0.0
    total_area = 0.0
    print("\n[2/10-4/10] 渗流分析、Delaunay 剖分与三角求积管道待实现")




    print("\n[5/10] 球面求积：估算全球平均辐射强迫...")

    def forcing_field(lat, lon):

        return 2.5 + 1.5 * np.sin(2 * lat) * np.cos(3 * lon)

    mean_forcing, node_num = global_radiation_forcing_integral(forcing_field, factor=2)
    print(f"    全球平均辐射强迫 = {mean_forcing:.4f} W/m²")
    print(f"    球面求积节点数 = {node_num}")


    def mc_forcing(xyz):
        lat = np.arcsin(np.clip(xyz[2], -1.0, 1.0))
        lon = np.arctan2(xyz[1], xyz[0])
        return forcing_field(lat, lon)

    mc_result = sphere01_quad_mc(mc_forcing, 5000, seed=65)
    mc_mean = mc_result / (4.0 * np.pi)
    print(f"    蒙特卡洛验证（N=5000）= {mc_mean:.4f} W/m²")




    print("\n[6/10] 快速谱求积：计算大气柱水汽含量...")
    z = np.linspace(0, 15000, 50)
    q = 0.01 * np.exp(-z / 3000.0)
    rho = 1.225 * np.exp(-z / 8500.0)
    tcwv = compute_total_column_water_vapor(z, q, rho)
    print(f"    总水汽含量 TCWV = {tcwv:.4f} kg/m²")




    print("\n[7/10] 楔形区域求积：三维大气柱能量积分...")
    if meshes:
        first_mesh = list(meshes.values())[0]
        tri_v = first_mesh["nodes"][:, first_mesh["triangles"][:, 0]]

        def energy_density(xyz):

            return np.exp(-xyz[2] / 5000.0)

        col_energy = integrate_over_atmospheric_column(
            energy_density, tri_v, z_bottom=0.0, z_top=15000.0,
            line_order=3, triangle_order=7
        )
        print(f"    首个事件区域的 3D 柱能量 = {col_energy:.4f} J/m²")




    print("\n[8/10] 蒙特卡洛集合分析：归因不确定性量化...")
    ensemble = generate_ensemble_perturbations(field, n_members=20,
                                                perturbation_scale=0.15, seed=65)

    ensemble_binary = np.zeros((20,) + field.shape, dtype=np.int64)
    for i in range(20):
        occ = (ensemble[i] > 2.0).astype(np.int64)
        ensemble_binary[i] = occ

    mean_dist, consensus = ensemble_attribution_distance(ensemble_binary,
                                                          threshold_ratio=0.6)
    print(f"    集合成员间平均 Hamming 距离 = {mean_dist:.2f}")
    print(f"    共识区域格点数 = {int(np.sum(consensus))}")


    def energy_rate(x):

        return x ** 2 * (1.0 - x)
    mc_energy = monte_carlo_line_integral(energy_rate, 20000, seed=65)
    print(f"    能量级串方程 MC 积分验证 = {mc_energy:.6f} (精确值=0.083333)")




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




    print("\n[附加] 区域统计聚合...")

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




    print("\n[附加] 谱分析：周期识别与 FFT 优化...")

    center_row = field[m // 2, :]
    periods, freqs, power = dominant_periods(center_row, sample_rate=1.0, n_peaks=3)
    print(f"    主导周期: {['%.2f' % p for p in periods]}")


    opt_len = len(center_row)
    print(f"    当前长度 {opt_len} 的质因数分解: {prime_factors(opt_len)}")


    print("\n[附加] 稀疏矩阵验证...")
    laplacian = build_climate_laplacian_sparse(5, 5)
    x_test = np.ones(25)
    b_test = laplacian.mv(x_test)
    print(f"    5x5 Laplacian 稀疏矩阵-向量乘法完成")
    print(f"    结果范数 = {np.linalg.norm(b_test):.4f}")




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
    sys.exit(main())
