"""
main.py

强关联电子系统 Hubbard 模型的博士级合成计算项目入口。

科学问题:
    研究二维三角晶格 Hubbard 模型在有限温度下的 Mott 转变、
    双占据数动力学、以及谱函数性质。

方法体系:
    1. 精确对角化 (ED) —— 小团簇基准
    2. 行列式量子蒙特卡洛 (DQMC) —— 有限温度统计
    3. Matsubara 格林函数与 Dyson 方程
    4. 解析延拓 (Padé, MaxEnt)
    5. 布里渊区积分 (四面体法, Lebedev 球面)
    6. 实时非平衡动力学 (周期驱动, Velocity-Verlet)
    7. 无序系统采样 (Anderson 型截断正态)
    8. 自洽迭代稳定性分析

零参数运行，自动执行完整计算流程并输出结果。
"""

import numpy as np
import sys
import time

from lattice_geometry import TriangularLattice, hex_grid_in_brillouin_zone, trinity_triangle_tiling_brillouin_zone
from hubbard_hamiltonian import build_hubbard_hamiltonian, exact_diagonalization, thermal_average, compute_ground_state_properties
from dqmc_engine import DQMCConfig, run_dqmc, build_kinetic_matrix
from matsubara_green import build_matsubara_green, dyson_equation, newton_divided_differences, evaluate_divided_difference
from brillouin_zone import compute_dos_tetrahedron, lebedev_sphere_grid, brillouin_zone_area
from dynamics_evolution import doublon_dynamics_hubbard, sawtooth_wave, sawtooth_drive_matrix, driven_hubbard_evolution
from disorder_config import disordered_hubbard_parameters, thermal_spin_configuration, square_surface_sample
from convergence_tools import self_consistent_iteration, iteration_complexity_index, collatz_stopping_time
from spectral_function import pade_spectral_function, maxent_spectral_function, spectral_moments


def print_banner():
    print("=" * 70)
    print("  强关联电子系统 Hubbard 模型 — 多方法合成计算框架")
    print("  Strongly Correlated Electron Systems: Hubbard Model")
    print("  Synthesis of 15 Seed Projects into a PhD-Level Framework")
    print("=" * 70)
    print()


def run_exact_diagonalization():
    """模块 1: 精确对角化 (2×2 方格, 4 格点)。"""
    print("[模块 1] 精确对角化 (Exact Diagonalization)")
    print("-" * 50)
    nsites = 4
    neighbors = [[1, 2], [0, 3], [0, 3], [1, 2]]  # 2x2 方格
    t = 1.0
    U_values = [0.0, 2.0, 4.0, 8.0]
    for U in U_values:
        props = compute_ground_state_properties(nsites, neighbors, t, U, mu=0.0)
        print(f"  U={U:.1f}: E0={props['E0']:.4f},  "
              f"D={props['double_occupancy']:.4f},  "
              f"N={props['n_total']:.2f},  Gap={props['energy_gap']:.4f}")
    print()


def run_dqmc_module():
    """模块 2: 行列式量子蒙特卡洛 (4 格点, β=2)。"""
    print("[模块 2] 行列式量子蒙特卡洛 (DQMC)")
    print("-" * 50)
    nsites = 4
    neighbors = [[1, 2], [0, 3], [0, 3], [1, 2]]
    cfg = DQMCConfig(nsites=nsites, beta=2.0, U=4.0, t=1.0, dtau=0.1)
    result = run_dqmc(cfg, neighbors, n_warmup=50, n_measure=100)
    print(f"  参数: U={cfg.U}, β={cfg.beta}, dtau={cfg.dtau}")
    print(f"  双占据数: {result['double_occupancy']:.4f} ± {result['double_occupancy_err']:.4f}")
    print(f"  动能:     {result['kinetic_energy']:.4f} ± {result['kinetic_energy_err']:.4f}")
    print()


def run_matsubara_selfenergy():
    """模块 3: Matsubara 格林函数与分差插值。"""
    print("[模块 3] Matsubara 格林函数与自能插值")
    print("-" * 50)
    beta = 2.0
    n_max = 10
    omega_n = np.array([(2 * n + 1) * np.pi / beta for n in range(n_max)])
    # 假设一个简单的自能: Σ(iω) = U^2 / (4 iω)
    U = 4.0
    Sigma_iw = U ** 2 / (4.0 * 1j * omega_n)
    # 使用 Newton 分差插值
    xd = omega_n[:5]
    yd = Sigma_iw[:5].imag
    dif = newton_divided_differences(xd, yd)
    xv = np.linspace(omega_n[0], omega_n[4], 10)
    yv = evaluate_divided_difference(xd, dif, xv)
    print(f"  自能在 iω_0 处的值: {Sigma_iw[0]:.4f}")
    print(f"  Newton 插值样本 (前3点): {yv[:3]}")
    print()


def run_brillouin_integration():
    """模块 4: 布里渊区积分与四面体法。"""
    print("[模块 4] 布里渊区积分 (Tetrahedron + Lebedev)")
    print("-" * 50)
    lat = TriangularLattice(8, 8, a=1.0)
    kpts = lat.reciprocal_lattice_points()
    # 简单紧束缚能带
    t = 1.0
    energies = np.zeros(len(kpts))
    for i, k in enumerate(kpts):
        kx, ky = k
        energies[i] = -2.0 * t * (np.cos(kx) + np.cos(ky) + np.cos(kx + ky))
    omega_grid = np.linspace(-6.0, 3.0, 100)
    dos = compute_dos_tetrahedron(kpts, energies, omega_grid, eta=0.1)
    # Lebedev 球面测试
    x, y, z, w = lebedev_sphere_grid(50)
    w_sum = np.sum(w)
    print(f"  k 点数量: {len(kpts)}")
    # 排除边界查找峰值
    peak_idx = np.argmax(dos[5:-5]) + 5
    print(f"  DOS 峰值位置: {omega_grid[peak_idx]:.3f}")
    print(f"  Lebedev 权重和: {w_sum:.6f} (理论值 {4*np.pi:.6f})")
    print(f"  BZ 面积: {brillouin_zone_area(lat.bz_vertices):.6f}")
    print()


def run_dynamics():
    """模块 5: Doublon 动力学与 Sawtooth 驱动。"""
    print("[模块 5] 非平衡动力学 (Doublon + Sawtooth 驱动)")
    print("-" * 50)
    t, y = doublon_dynamics_hubbard(U=4.0, t_hop=1.0, beta=2.0, t_max=5.0)
    print(f"  Doublon 密度演化: initial={y[0,0]:.4f}, final={y[-1,0]:.4f}")
    print(f"  Holon 密度演化:   initial={y[0,1]:.4f}, final={y[-1,1]:.4f}")
    # Sawtooth 波测试
    t_test = np.linspace(0, 4 * np.pi, 20)
    saw = [sawtooth_wave(ti, 1.0) for ti in t_test]
    print(f"  Sawtooth 波样本: [{saw[0]:.3f}, {saw[5]:.3f}, {saw[10]:.3f}]")
    print()


def run_disorder():
    """模块 6: 无序采样与热自旋构型。"""
    print("[模块 6] 无序与热涨落采样")
    print("-" * 50)
    nsites = 16
    eps, U = disordered_hubbard_parameters(nsites, W=1.5, U_base=4.0, U_var=0.3)
    print(f"  Anderson 无序: mean={np.mean(eps):.4f}, std={np.std(eps):.4f}")
    print(f"  U 分布: mean={np.mean(U):.4f}, std={np.std(U):.4f}")
    spins = thermal_spin_configuration(nsites, beta=1.0)
    print(f"  热自旋构型: 平均 z 分量={np.mean(spins[:,2]):.4f}")
    boundary_pts = square_surface_sample(8)
    print(f"  方边界采样点: shape={boundary_pts.shape}")
    print()


def run_convergence():
    """模块 7: 自洽迭代收敛分析。"""
    print("[模块 7] 自洽迭代与收敛监控")
    print("-" * 50)
    # 一个简单的自洽问题: x = 0.5 * x + 1
    x0 = np.array([0.0])
    def update(x):
        return np.array([0.5 * x[0] + 1.0])
    x, it, res = self_consistent_iteration(update, x0, tol=1e-10, max_iter=100, mixing="simple", alpha=0.5)
    print(f"  线性自洽: 收敛于 x={x[0]:.6f}, 迭代次数={it}")
    complexity = iteration_complexity_index(res)
    print(f"  迭代复杂度指数: {complexity:.4f}")
    # Collatz 监控
    collatz_t = collatz_stopping_time(27)
    print(f"  Collatz(27) 停止时间: {collatz_t}")
    print()


def run_spectral():
    """模块 8: 谱函数与解析延拓。"""
    print("[模块 8] 谱函数 (Padé + MaxEnt)")
    print("-" * 50)
    beta = 2.0
    omega_n = np.array([(2 * n + 1) * np.pi / beta for n in range(20)])
    # 单极点格林函数
    g_iw = 1.0 / (1j * omega_n + 0.5)
    omega_real = np.linspace(-3.0, 3.0, 100)
    # Padé
    A_pade = pade_spectral_function(omega_n, g_iw, omega_real, eta=0.05)
    sum_rule_pade = float(np.trapezoid(A_pade, omega_real))
    print(f"  Padé 谱函数求和规则: {sum_rule_pade:.4f}")
    # MaxEnt
    A_maxent = maxent_spectral_function(omega_n, g_iw, omega_real, alpha=0.5)
    sum_rule_maxent = float(np.trapezoid(A_maxent, omega_real))
    print(f"  MaxEnt 谱函数求和规则: {sum_rule_maxent:.4f}")
    # 谱矩
    moments = spectral_moments(A_pade, omega_real, max_moment=2)
    print(f"  谱矩 M_0={moments['M_0']:.4f}, M_1={moments['M_1']:.4f}, M_2={moments['M_2']:.4f}")
    print()


def run_dmft_loop():
    """模块 9: 简化的 DMFT 自洽环。"""
    print("[模块 9] 动力学平均场理论 (DMFT) 自洽环")
    print("-" * 50)
    U = 4.0
    t = 1.0
    beta = 2.0
    mu = U / 2.0
    omega_n = np.array([(2 * n + 1) * np.pi / beta for n in range(10)])
    nw = len(omega_n)
    # 初始猜测: 无相互作用
    Sigma = np.zeros(nw, dtype=np.complex128)
    G0_iw = 1.0 / (1j * omega_n + mu)
    def dmft_update(Sigma_old):
        # Dyson 方程
        G = 1.0 / (1j * omega_n + mu - Sigma_old)
        # 平均场近似: Σ ≈ U * <n↑><n↓> - U^2/4 * G
        n_up = 0.5
        n_dn = 0.5
        Sigma_new = U * n_up * n_dn - (U ** 2 / 4.0) * G
        return Sigma_new
    # 简单自洽
    Sigma_new = Sigma.copy()
    residuals = []
    for it in range(30):
        Sigma_old = Sigma_new.copy()
        Sigma_new = dmft_update(Sigma_old)
        res = float(np.linalg.norm(Sigma_new - Sigma_old))
        residuals.append(res)
        if res < 1e-6:
            break
    print(f"  DMFT 自洽: 迭代次数={len(residuals)}, 最终残差={residuals[-1]:.2e}")
    print(f"  收敛后 Σ(iω_0)={Sigma_new[0]:.4f}")
    print()


def main():
    print_banner()
    t0 = time.time()
    np.random.seed(42)

    try:
        run_exact_diagonalization()
        run_dqmc_module()
        run_matsubara_selfenergy()
        run_brillouin_integration()
        run_dynamics()
        run_disorder()
        run_convergence()
        run_spectral()
        run_dmft_loop()
    except Exception as e:
        print(f"[ERROR] 计算过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    t1 = time.time()
    print("=" * 70)
    print(f"  全部计算完成，总耗时: {t1 - t0:.2f} 秒")
    print("=" * 70)


if __name__ == "__main__":
    main()

# ================================================================
# 测试用例（29个，assert模式，涉及随机值均使用固定种子）
# ================================================================

# ---- TC01: TriangularLattice 基本属性验证 ----
lat = TriangularLattice(4, 4, a=1.0)
assert lat.nsites == 16, '[TC01] TriangularLattice 基本属性验证 FAILED'
assert len(lat.neighbors[0]) == 6, '[TC01] TriangularLattice 基本属性验证 FAILED'
assert lat.a == 1.0, '[TC01] TriangularLattice 基本属性验证 FAILED'

# ---- TC02: hex_grid_in_brillouin_zone 生成有效点 ----
lat2 = TriangularLattice(6, 6, a=1.0)
pts = hex_grid_in_brillouin_zone(3, lat2.bz_vertices)
assert pts.shape[1] == 2, '[TC02] hex_grid_in_brillouin_zone 生成有效点 FAILED'
assert len(pts) >= 1, '[TC02] hex_grid_in_brillouin_zone 生成有效点 FAILED'

# ---- TC03: build_hubbard_hamiltonian 返回厄米矩阵 ----
H = build_hubbard_hamiltonian(2, [[1], [0]], 1.0, 4.0, 0.0)
assert H.shape == (16, 16), '[TC03] build_hubbard_hamiltonian 返回厄米矩阵 FAILED'
assert np.allclose(H, H.T.conj()), '[TC03] build_hubbard_hamiltonian 返回厄米矩阵 FAILED'

# ---- TC04: exact_diagonalization 本征值为实数且有序 ----
evals, evecs = exact_diagonalization(H)
assert np.allclose(evals.imag, 0, atol=1e-10), '[TC04] exact_diagonalization 本征值为实数且有序 FAILED'
assert np.all(np.diff(evals) >= -1e-12), '[TC04] exact_diagonalization 本征值为实数且有序 FAILED'

# ---- TC05: compute_ground_state_properties 输出有限值 ----
props = compute_ground_state_properties(2, [[1], [0]], 1.0, 4.0, 0.0)
assert 'E0' in props and 'double_occupancy' in props, '[TC05] compute_ground_state_properties 输出有限值 FAILED'
assert np.isfinite(props['E0']) and np.isfinite(props['double_occupancy']), '[TC05] compute_ground_state_properties 输出有限值 FAILED'

# ---- TC06: thermal_average beta=0 等于算符对角元平均 ----
D = np.diag([0.0, 1.0, 1.0, 2.0])
evals2 = np.array([0.0, 1.0, 1.0, 2.0])
evecs2 = np.eye(4)
val = thermal_average(evals2, evecs2, D, beta=0.0)
expected = np.mean(np.diag(D))
assert np.isclose(val, expected), '[TC06] thermal_average beta=0 等于算符对角元平均 FAILED'

# ---- TC07: lebedev_sphere_grid 权重和近似 4pi ----
x, y, z, w = lebedev_sphere_grid(50)
assert len(w) > 0, '[TC07] lebedev_sphere_grid 权重和近似 4pi FAILED'
assert np.isclose(np.sum(w), 4 * np.pi, rtol=0.05), '[TC07] lebedev_sphere_grid 权重和近似 4pi FAILED'

# ---- TC08: brillouin_zone_area 为正 ----
lat3 = TriangularLattice(4, 4, a=1.0)
area = brillouin_zone_area(lat3.bz_vertices)
assert area > 0, '[TC08] brillouin_zone_area 为正 FAILED'

# ---- TC09: newton_divided_differences 对线性函数精确 ----
xd = np.array([0.0, 1.0, 2.0])
yd = 2.0 * xd + 1.0
dif = newton_divided_differences(xd, yd)
assert np.isclose(dif[0], 1.0), '[TC09] newton_divided_differences 对线性函数精确 FAILED'
assert np.isclose(dif[1], 2.0), '[TC09] newton_divided_differences 对线性函数精确 FAILED'
assert np.isclose(dif[2], 0.0), '[TC09] newton_divided_differences 对线性函数精确 FAILED'

# ---- TC10: evaluate_divided_difference 节点精确命中 ----
xv = np.array([0.0, 1.0, 2.0])
yv = evaluate_divided_difference(xd, dif, xv)
assert np.allclose(yv, yd), '[TC10] evaluate_divided_difference 节点精确命中 FAILED'

# ---- TC11: dyson_equation 零自能返回原格林函数 ----
G0_test = np.array([[[1.0, 0.0], [0.0, 1.0]]], dtype=np.complex128)
Sigma_zero = np.zeros_like(G0_test)
G_test = dyson_equation(G0_test, Sigma_zero)
assert np.allclose(G_test, G0_test), '[TC11] dyson_equation 零自能返回原格林函数 FAILED'

# ---- TC12: build_matsubara_green 形状正确 ----
K_test = np.array([[0.0, -1.0], [-1.0, 0.0]])
G0_mats = build_matsubara_green(2, K_test, mu=0.0, beta=2.0, U=0.0, n_max=2, sigma=0)
assert G0_mats.shape == (5, 2, 2), '[TC12] build_matsubara_green 形状正确 FAILED'

# ---- TC13: pade_spectral_function 求和规则在合理范围 ----
omega_n = np.array([1.0, 3.0, 5.0, 7.0, 9.0]) * np.pi
g_iw = 1.0 / (1j * omega_n + 0.5)
omega_real = np.linspace(-5, 5, 100)
A_pade = pade_spectral_function(omega_n, g_iw, omega_real, eta=0.1)
sum_rule = np.trapezoid(A_pade, omega_real)
assert 0.5 < sum_rule < 2.0, '[TC13] pade_spectral_function 求和规则在合理范围 FAILED'

# ---- TC14: spectral_moments M0 非负 ----
moments = spectral_moments(A_pade, omega_real, max_moment=2)
assert moments['M_0'] >= 0, '[TC14] spectral_moments M0 非负 FAILED'

# ---- TC15: collatz_stopping_time 基础值 ----
assert collatz_stopping_time(1) == 0, '[TC15] collatz_stopping_time 基础值 FAILED'
assert collatz_stopping_time(27) == 111, '[TC15] collatz_stopping_time 基础值 FAILED'

# ---- TC16: self_consistent_iteration 线性问题收敛 ----
def linear_update(x):
    return np.array([0.5 * x[0] + 1.0])
x_sc, it_sc, res_sc = self_consistent_iteration(linear_update, np.array([0.0]), tol=1e-10, max_iter=100, mixing="simple", alpha=0.5)
assert it_sc < 100, '[TC16] self_consistent_iteration 线性问题收敛 FAILED'
assert np.isclose(x_sc[0], 2.0, atol=1e-6), '[TC16] self_consistent_iteration 线性问题收敛 FAILED'

# ---- TC17: iteration_complexity_index 单调残差 ----
res_mono = [1.0, 0.5, 0.25, 0.125]
idx = iteration_complexity_index(res_mono)
assert idx >= 0, '[TC17] iteration_complexity_index 单调残差 FAILED'

# ---- TC18: disordered_hubbard_parameters U 为正 ----
np.random.seed(42)
eps, U_vals = disordered_hubbard_parameters(10, W=2.0, U_base=4.0, U_var=0.5)
assert np.all(U_vals > 0), '[TC18] disordered_hubbard_parameters U 为正 FAILED'
assert len(eps) == 10, '[TC18] disordered_hubbard_parameters U 为正 FAILED'

# ---- TC19: thermal_spin_configuration 自旋单位模长 ----
np.random.seed(42)
spins = thermal_spin_configuration(10, beta=1.0)
norms = np.sqrt(np.sum(spins ** 2, axis=1))
assert np.allclose(norms, 1.0), '[TC19] thermal_spin_configuration 自旋单位模长 FAILED'

# ---- TC20: square_surface_sample 形状正确 ----
np.random.seed(42)
boundary_pts = square_surface_sample(20)
assert boundary_pts.shape == (20, 2), '[TC20] square_surface_sample 形状正确 FAILED'

# ---- TC21: DQMCConfig 参数计算正确 ----
cfg = DQMCConfig(nsites=4, beta=2.0, U=4.0, t=1.0, dtau=0.1)
assert cfg.L >= 1, '[TC21] DQMCConfig 参数计算正确 FAILED'
assert cfg.nsites == 4, '[TC21] DQMCConfig 参数计算正确 FAILED'

# ---- TC22: build_kinetic_matrix 对称且对角元为零 ----
K_mat = build_kinetic_matrix(4, [[1, 3], [0, 2], [1, 3], [0, 2]], 1.0)
assert np.allclose(K_mat, K_mat.T), '[TC22] build_kinetic_matrix 对称且对角元为零 FAILED'
assert np.allclose(np.diag(K_mat), 0.0), '[TC22] build_kinetic_matrix 对称且对角元为零 FAILED'

# ---- TC23: sawtooth_wave 值域约束 ----
t_vals = np.linspace(0, 10, 100)
saw_vals = np.array([sawtooth_wave(ti, 1.0) for ti in t_vals])
assert np.all(saw_vals >= -0.5) and np.all(saw_vals < 0.5), '[TC23] sawtooth_wave 值域约束 FAILED'

# ---- TC24: doublon_dynamics_hubbard 输出非负 ----
t_dd, y_dd = doublon_dynamics_hubbard(U=4.0, t_hop=1.0, beta=2.0, t_max=5.0)
assert np.all(y_dd >= 0), '[TC24] doublon_dynamics_hubbard 输出非负 FAILED'
assert t_dd[0] == 0.0, '[TC24] doublon_dynamics_hubbard 输出非负 FAILED'

# ---- TC25: compute_dos_tetrahedron 输出有限且长度正确 ----
lat4 = TriangularLattice(4, 4, a=1.0)
kpts = lat4.reciprocal_lattice_points()
t = 1.0
energies = np.zeros(len(kpts))
for i, k in enumerate(kpts):
    kx, ky = k
    energies[i] = -2.0 * t * (np.cos(kx) + np.cos(ky) + np.cos(kx + ky))
omega_grid = np.linspace(-6.0, 3.0, 50)
dos = compute_dos_tetrahedron(kpts, energies, omega_grid, eta=0.1)
assert len(dos) == len(omega_grid), '[TC25] compute_dos_tetrahedron 输出有限且长度正确 FAILED'
assert np.all(np.isfinite(dos)), '[TC25] compute_dos_tetrahedron 输出有限且长度正确 FAILED'

# ---- TC26: trinity_triangle_tiling_brillouin_zone 返回正确形状 ----
triangles, weights = trinity_triangle_tiling_brillouin_zone(kpts, lat4.bz_vertices)
assert triangles.shape[1] == 3, '[TC26] trinity_triangle_tiling_brillouin_zone 返回正确形状 FAILED'
assert len(weights) == len(triangles), '[TC26] trinity_triangle_tiling_brillouin_zone 返回正确形状 FAILED'
assert np.isclose(np.sum(weights), 1.0), '[TC26] trinity_triangle_tiling_brillouin_zone 返回正确形状 FAILED'

# ---- TC27: sawtooth_drive_matrix 对角结构 ----
np.random.seed(42)
V_list = sawtooth_drive_matrix(4, amplitude=1.0, omega=1.0, times=np.array([0.0, 1.0]))
assert len(V_list) == 2, '[TC27] sawtooth_drive_matrix 对角结构 FAILED'
assert np.allclose(V_list[0], np.diag(np.diag(V_list[0]))), '[TC27] sawtooth_drive_matrix 对角结构 FAILED'

# ---- TC28: build_hubbard_hamiltonian U=0 能谱对称中心在零 ----
H0 = build_hubbard_hamiltonian(2, [[1], [0]], 1.0, 0.0, 0.0)
evals0, _ = exact_diagonalization(H0)
assert np.isclose(np.sum(evals0), 0.0, atol=1e-10), '[TC28] build_hubbard_hamiltonian U=0 能谱对称中心在零 FAILED'

# ---- TC29: run_dqmc 输出结构正确 ----
np.random.seed(42)
cfg = DQMCConfig(nsites=4, beta=1.0, U=2.0, t=1.0, dtau=0.2)
res_dqmc = run_dqmc(cfg, [[1, 3], [0, 2], [1, 3], [0, 2]], n_warmup=10, n_measure=20)
assert 'double_occupancy' in res_dqmc, '[TC29] run_dqmc 输出结构正确 FAILED'
assert 'kinetic_energy' in res_dqmc, '[TC29] run_dqmc 输出结构正确 FAILED'
assert np.isfinite(res_dqmc['double_occupancy']), '[TC29] run_dqmc 输出结构正确 FAILED'

print('\n全部 29 个测试通过!\n')
