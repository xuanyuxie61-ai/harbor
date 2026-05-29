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
        # [HOLE 2] TODO: 修复三角晶格紧束缚能带色散公式
        # 提示: 三角晶格最近邻跃迁的紧束缚色散关系。
        energies[i] = 0.0  # 占位
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
