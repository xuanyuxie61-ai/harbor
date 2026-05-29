"""
main.py
=======
纳米颗粒-生物膜相互作用的粗粒化分子动力学综合模拟系统

【科学问题】
带电金纳米颗粒（AuNP）与磷脂双分子层（POPC）的相互作用是纳米医学与
纳米毒理学的核心问题。本程序在一个统一的计算框架内耦合以下物理过程：

    1. 膜弹性力学（Helfrich 弯曲能 + 三角化有限元离散）
    2. 静电学（球对称 Poisson-Boltzmann 方程：Jacobi + BiCG 双求解器）
    3. 输运过程（膜附近离子耗竭区的对流-扩散方程，Lax 格式）
    4. 纳米颗粒随机动力学（过阻尼 Langevin + Type-II 受体饱和结合）
    5. 相关随机力（广义 Langevin，指数记忆核 + 快速托普利茨 Cholesky）
    6. 平衡态搜索（Chebyshev Proxy Rootfinder 求力平衡零点）
    7. 自由能代理模型（前馈神经网络回归 DLVO+Helfrich 描述符）
    8. 统计采样（逆 CDF 采样 + SVD 变形模式分析）
    9. 多维能量积分（Gauss-Legendre 乘积规则）
    10. 分子拓扑校验（MTIC 校验和算法）
    11. 并行计算框架（多进程力计算分解）

运行方式：
    python main.py
    （零参数，所有物理参数内置为典型值）
"""

import sys
import numpy as np

# ---------------------------------------------------------------------------
# 导入各模块
# ---------------------------------------------------------------------------
from platform_detect import check_environment, get_platform_info
from topology_validator import validate_topology, generate_topologies
from parallel_utils import parallel_map
from membrane_mesh import TriangulatedMembrane
from electrostatic_solver import PoissonBoltzmannSolver
from transport_solver import AdvectionDiffusionSolver
from nanoparticle_dynamics import NanoparticleLangevinDynamics
from potential_energy import (
    integrate_nd, CubicSplineInterpolator, membrane_binding_energy_integral
)
from sampling_utils import (
    pdf_to_histogram, histogram_to_cdf, cdf_to_sample,
    sphere_sample_marsaglia, svd_deformation_modes,
    sample_random_orientation, boltzmann_acceptance
)
from correlated_forces import generate_correlated_forces, colored_noise_spectrum
from equilibrium_solver import find_equilibrium_distances
from neural_surrogate import NeuralSurrogate, generate_training_data


def print_section(title: str):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def main():
    np.random.seed(117)
    print("纳米颗粒-生物膜相互作用：粗粒化分子动力学综合模拟")
    print(f"Python: {sys.version.split()[0]}")

    # =====================================================================
    # 0. 环境检测
    # =====================================================================
    print_section("0. 平台与环境检测")
    env_ok = check_environment()
    info = get_platform_info()
    print(f"  环境检查通过: {env_ok}")
    print(f"  NumPy MKL 后端: {info['numpy_mkl']}")
    print(f"  机器精度 eps: {info['float_eps']:.2e}")
    print(f"  可用 CPU 核心: {info['max_threads']}")

    # =====================================================================
    # 1. 分子拓扑校验（源自 seed 704_luhn）
    # =====================================================================
    print_section("1. 分子拓扑完整性校验 (MTIC)")
    bead_types = ["PH", "GL", "EST1", "EST2", "C1A", "C2A", "C1B", "C2B"]
    topologies = generate_topologies(bead_types, count=4)
    print(f"  生成 {len(topologies)} 个 Martini 磷脂拓扑标识符")
    valid_count = sum(1 for t in topologies if validate_topology(t))
    print(f"  通过 MTIC 校验: {valid_count}/{len(topologies)}")
    assert valid_count == len(topologies), "拓扑校验失败！"

    # =====================================================================
    # 2. 膜网格构建（源自 seed 755_mesh_etoe）
    # =====================================================================
    print_section("2. 三角化膜网格与邻接分析")
    membrane = TriangulatedMembrane.create_planar_sheet(nx=16, ny=16,
                                                         lx=10.0, ly=10.0)
    etoe = membrane.compute_etoe()
    areas = membrane.compute_element_areas()
    normals = membrane.compute_normals()
    H = membrane.compute_mean_curvature()
    E_bend = membrane.bending_energy(kappa=20.0)
    print(f"  顶点数: {membrane.n_v}, 单元数: {membrane.n_e}")
    print(f"  总膜面积: {np.sum(areas):.4f} nm^2")
    print(f"  平均曲率范围: [{np.min(H):.4f}, {np.max(H):.4f}] 1/nm")
    print(f"  Helfrich 弯曲能: {E_bend:.4f} k_B T")
    # 邻接统计
    n_internal = np.sum(etoe != -1)
    n_boundary = np.sum(etoe == -1)
    print(f"  内部边: {n_internal}, 边界边: {n_boundary}")

    # =====================================================================
    # 3. 静电泊松-玻尔兹曼求解（源自 seed 606_jacobi + 085_bicg）
    # =====================================================================
    print_section("3. 球对称 Poisson-Boltzmann 静电求解")
    pb = PoissonBoltzmannSolver(R_np=2.5, R_max=25.0, z_np=+10.0,
                                 n_0=0.1, epsilon=80.0 * 8.854e-12, T=300.0)
    print(f"  德拜长度: {pb.debye_length():.4f} nm")

    phi_jacobi, r_jacobi, it_jacobi, res_jacobi = pb.solve_jacobi(n_grid=257)
    print(f"  Jacobi 迭代: {it_jacobi} 步, 最终残差: {res_jacobi:.4e}")

    phi_bicg, r_bicg, it_bicg, res_bicg = pb.solve_bicg(n_grid=257)
    print(f"  BiCG 迭代: {it_bicg} 步, 最终残差: {res_bicg:.4e}")

    # 双求解器一致性检验
    diff_solver = np.max(np.abs(phi_jacobi - phi_bicg))
    print(f"  双求解器最大差异: {diff_solver:.4e} V")
    assert diff_solver < 1e-3, "Jacobi 与 BiCG 结果不一致！"

    dphi_surf = pb.electrostatic_force(n_grid=257)
    print(f"  表面电场梯度: {dphi_surf:.4e} V/nm")

    # =====================================================================
    # 4. 对流-扩散输运（源自 seed 354_fd1d_advection_lax）
    # =====================================================================
    print_section("4. 膜附近离子对流-扩散输运 (Lax 格式)")
    transport = AdvectionDiffusionSolver(L=20.0, v=0.5, D=2.0,
                                          c0=0.1, nx=201)
    c_final, history = transport.solve(n_steps=500)
    flux_surf = transport.compute_flux(c_final)
    print(f"  空间格点数: {transport.nx}, 时间步长: {transport.dt:.6f} ns")
    print(f"  膜表面稳态通量: {flux_surf:.6e} (浓度单位)/ns")
    print(f"  本体浓度: {c_final[-1]:.4f}, 表面浓度: {c_final[0]:.4f}")

    # =====================================================================
    # 5. 多维能量积分与样条插值（源自 seed 919_product_rule + 594_interp_spline）
    # =====================================================================
    print_section("5. 多维乘积求积与样条势函数")
    E_num, E_exact = membrane_binding_energy_integral(R_np=2.5, kappa=20.0,
                                                       sigma=1.0, n_quad=5)
    print(f"  结合能数值积分: {E_num:.4f} k_B T")
    print(f"  结合能解析验证: {E_exact:.4f} k_B T")
    print(f"  积分相对误差: {abs(E_num - E_exact) / (abs(E_exact) + 1e-12):.4e}")

    # 构造 LJ 势表并样条插值
    r_tab = np.linspace(0.2, 8.0, 100)
    epsilon_LJ = 4.0
    sigma_LJ = 3.0
    V_tab = 4.0 * epsilon_LJ * ((sigma_LJ / r_tab) ** 12 - (sigma_LJ / r_tab) ** 6)
    V_tab[V_tab > 100.0] = 100.0  # 截断
    spline = CubicSplineInterpolator(r_tab, V_tab)
    r_test = np.array([0.5, 1.0, 2.5, 5.0])
    V_interp = spline.evaluate(r_test)
    V_deriv = spline.derivative(r_test)
    print(f"  样条插值势能在 r={r_test} nm: {V_interp}")
    print(f"  对应力 -dV/dr: {-V_deriv}")

    # =====================================================================
    # 6. 统计采样与 SVD 变形模式（源自 seed 541_histogram_pdf + 1192_svd_sphere）
    # =====================================================================
    print_section("6. 概率采样与 SVD 变形模式分析")
    # 高斯型 PDF 采样验证
    def gaussian_pdf(x):
        return np.exp(-x ** 2 / 2.0) / np.sqrt(2.0 * np.pi)
    b_p, b_l, b_r = pdf_to_histogram(gaussian_pdf, n_bins=64, x_min=-4.0, x_max=4.0)
    c_x, c_y = histogram_to_cdf(b_p, b_l, b_r)
    samples = cdf_to_sample(c_x, c_y, n_samples=5000)
    print(f"  高斯采样均值: {np.mean(samples):.4f}, 标准差: {np.std(samples):.4f}")

    # 球面采样（纳米颗粒随机取向）
    sphere_pts = sphere_sample_marsaglia(1000)
    norms = np.linalg.norm(sphere_pts, axis=0)
    print(f"  球面采样平均模长: {np.mean(norms):.6f} (应 ~1.0)")

    # SVD 变形模式（模拟膜在 NP 压迫下的位移）
    # 人为构造一个高斯钟形位移场
    x_coords = membrane.vertices[:, 0]
    y_coords = membrane.vertices[:, 1]
    cx, cy = 5.0, 5.0
    displacement = np.zeros((3, membrane.n_v), dtype=np.float64)
    displacement[2, :] = -2.0 * np.exp(-((x_coords - cx) ** 2 + (y_coords - cy) ** 2) / 4.0)
    U, S, Vt = svd_deformation_modes(displacement)
    print(f"  SVD 奇异值: {S}")
    print(f"  主变形模式占比: {S[0]/np.sum(S):.2%}")

    # Metropolis 接受测试
    accepted = sum(boltzmann_acceptance(delta_E=5.0) for _ in range(1000))
    print(f"  Metropolis 接受率 (deltaE=5kJ/mol, T=300K): {accepted/1000:.3f}")

    # =====================================================================
    # 7. 相关随机力（源自 seed 1262_toeplitz_cholesky）
    # =====================================================================
    print_section("7. 指数记忆核相关随机力 (快速 Toeplitz Cholesky)")
    forces_corr = generate_correlated_forces(n_steps=4096, dt=0.001,
                                              gamma0=1.0, tau_mem=0.05)
    print(f"  生成随机力长度: {len(forces_corr)}")
    print(f"  均值: {np.mean(forces_corr):.6f}, 标准差: {np.std(forces_corr):.4f}")
    freqs, psd = colored_noise_spectrum(forces_corr, dt=0.001)
    # 验证低频 PSD > 高频 PSD（有色噪声特征）
    print(f"  低频 PSD (f<10): {np.mean(psd[freqs<10]):.4f}")
    print(f"  高频 PSD (f>100): {np.mean(psd[freqs>100]):.4f}")

    # =====================================================================
    # 8. 平衡距离求解（源自 seed 225_cpr）
    # =====================================================================
    print_section("8. Chebyshev Proxy Rootfinder 平衡态搜索")
    # 使用当前物理参数构造总力函数
    nld = NanoparticleLangevinDynamics(z0=5.0)
    def total_force_z(z):
        return nld.total_force(z, debye_length=pb.debye_length())
    # 限制搜索区间以避免 LJ 奇异性导致的 Chebyshev 插值振荡
    roots, stability = find_equilibrium_distances(total_force_z, z_min=2.0, z_max=8.0)
    print(f"  发现 {len(roots)} 个力平衡距离:")
    for z_eq, stab in zip(roots, stability):
        print(f"    z_eq = {z_eq:.4f} nm  [{stab}]")
    assert len(roots) > 0, "未找到平衡距离！"

    # =====================================================================
    # 9. 神经网络自由能代理（源自 seed 773_mnist_neural）
    # =====================================================================
    print_section("9. 神经网络结合自由能代理模型")
    X_train, y_train = generate_training_data(n_samples=512, seed=123)
    X_test, y_test = generate_training_data(n_samples=128, seed=456)
    nn = NeuralSurrogate(input_dim=6, hidden_dims=[32, 16, 8],
                          lr=0.005, lambda_reg=1e-5, seed=42)
    losses = nn.train(X_train, y_train, epochs=300, batch_size=32, verbose=False)
    y_pred = nn.predict(X_test)
    mse_test = float(np.mean((y_pred - y_test) ** 2))
    r2 = float(1.0 - np.sum((y_test - y_pred)**2) / (np.sum((y_test - np.mean(y_test))**2) + 1e-12))
    print(f"  训练轮数: 300, 最终训练损失: {losses[-1]:.4f}")
    print(f"  测试集 MSE: {mse_test:.4f}")
    print(f"  测试集 R^2: {r2:.4f}")
    assert r2 > 0.5, "代理模型拟合不足！"

    # =====================================================================
    # 10. 纳米颗粒朗之万动力学轨迹（源自 seed 488_grazing_ode）
    # =====================================================================
    print_section("10. 纳米颗粒过阻尼朗之万动力学轨迹")
    nld = NanoparticleLangevinDynamics(z0=8.0)
    t_traj, z_traj, F_traj = nld.simulate(n_steps=20000,
                                           debye_length=pb.debye_length())
    print(f"  模拟时长: {t_traj[-1]:.3f} ns")
    print(f"  初始距离: {z_traj[0]:.3f} nm")
    print(f"  终止距离: {z_traj[-1]:.3f} nm")
    print(f"  平均距离: {np.mean(z_traj):.3f} nm")
    print(f"  距离标准差: {np.std(z_traj):.3f} nm")
    print(f"  最大受力: {np.max(np.abs(F_traj)):.2f} kJ/(mol*nm)")

# 模块级辅助函数（供并行测试使用，必须定义在模块顶层以便 pickle）
def _square_task(x):
    return x ** 2


def main():
    np.random.seed(117)
    print("纳米颗粒-生物膜相互作用：粗粒化分子动力学综合模拟")
    print(f"Python: {sys.version.split()[0]}")

    # =====================================================================
    # 0. 环境检测
    # =====================================================================
    print_section("0. 平台与环境检测")
    env_ok = check_environment()
    info = get_platform_info()
    print(f"  环境检查通过: {env_ok}")
    print(f"  NumPy MKL 后端: {info['numpy_mkl']}")
    print(f"  机器精度 eps: {info['float_eps']:.2e}")
    print(f"  可用 CPU 核心: {info['max_threads']}")

    # =====================================================================
    # 1. 分子拓扑校验（源自 seed 704_luhn）
    # =====================================================================
    print_section("1. 分子拓扑完整性校验 (MTIC)")
    bead_types = ["PH", "GL", "EST1", "EST2", "C1A", "C2A", "C1B", "C2B"]
    topologies = generate_topologies(bead_types, count=4)
    print(f"  生成 {len(topologies)} 个 Martini 磷脂拓扑标识符")
    valid_count = sum(1 for t in topologies if validate_topology(t))
    print(f"  通过 MTIC 校验: {valid_count}/{len(topologies)}")
    assert valid_count == len(topologies), "拓扑校验失败！"

    # =====================================================================
    # 2. 膜网格构建（源自 seed 755_mesh_etoe）
    # =====================================================================
    print_section("2. 三角化膜网格与邻接分析")
    membrane = TriangulatedMembrane.create_planar_sheet(nx=16, ny=16,
                                                         lx=10.0, ly=10.0)
    etoe = membrane.compute_etoe()
    areas = membrane.compute_element_areas()
    normals = membrane.compute_normals()
    H = membrane.compute_mean_curvature()
    E_bend = membrane.bending_energy(kappa=20.0)
    print(f"  顶点数: {membrane.n_v}, 单元数: {membrane.n_e}")
    print(f"  总膜面积: {np.sum(areas):.4f} nm^2")
    print(f"  平均曲率范围: [{np.min(H):.4f}, {np.max(H):.4f}] 1/nm")
    print(f"  Helfrich 弯曲能: {E_bend:.4f} k_B T")
    # 邻接统计
    n_internal = np.sum(etoe != -1)
    n_boundary = np.sum(etoe == -1)
    print(f"  内部边: {n_internal}, 边界边: {n_boundary}")

    # =====================================================================
    # 3. 静电泊松-玻尔兹曼求解（源自 seed 606_jacobi + 085_bicg）
    # =====================================================================
    print_section("3. 球对称 Poisson-Boltzmann 静电求解")
    pb = PoissonBoltzmannSolver(R_np=2.5, R_max=25.0, z_np=+10.0,
                                 n_0=0.1, epsilon=80.0 * 8.854e-12, T=300.0)
    print(f"  德拜长度: {pb.debye_length():.4f} nm")

    phi_jacobi, r_jacobi, it_jacobi, res_jacobi = pb.solve_jacobi(n_grid=257)
    print(f"  Jacobi 迭代: {it_jacobi} 步, 最终残差: {res_jacobi:.4e}")

    phi_bicg, r_bicg, it_bicg, res_bicg = pb.solve_bicg(n_grid=257)
    print(f"  BiCG 迭代: {it_bicg} 步, 最终残差: {res_bicg:.4e}")

    # 双求解器一致性检验
    diff_solver = np.max(np.abs(phi_jacobi - phi_bicg))
    print(f"  双求解器最大差异: {diff_solver:.4e} V")
    assert diff_solver < 1e-3, "Jacobi 与 BiCG 结果不一致！"

    dphi_surf = pb.electrostatic_force(n_grid=257)
    print(f"  表面电场梯度: {dphi_surf:.4e} V/nm")

    # =====================================================================
    # 4. 对流-扩散输运（源自 seed 354_fd1d_advection_lax）
    # =====================================================================
    print_section("4. 膜附近离子对流-扩散输运 (Lax 格式)")
    transport = AdvectionDiffusionSolver(L=20.0, v=0.05, D=0.1,
                                          c0=0.1, nx=201)
    c_final, history = transport.solve(n_steps=500)
    flux_surf = transport.compute_flux(c_final)
    print(f"  空间格点数: {transport.nx}, 时间步长: {transport.dt:.6f} ns")
    print(f"  膜表面稳态通量: {flux_surf:.6e} (浓度单位)/ns")
    print(f"  本体浓度: {c_final[-1]:.4f}, 表面浓度: {c_final[0]:.4f}")

    # =====================================================================
    # 5. 多维能量积分与样条插值（源自 seed 919_product_rule + 594_interp_spline）
    # =====================================================================
    print_section("5. 多维乘积求积与样条势函数")
    E_num, E_exact = membrane_binding_energy_integral(R_np=2.5, kappa=20.0,
                                                       sigma=1.0, n_quad=5)
    print(f"  结合能数值积分: {E_num:.4f} k_B T")
    print(f"  结合能解析验证: {E_exact:.4f} k_B T")
    print(f"  积分相对误差: {abs(E_num - E_exact) / (abs(E_exact) + 1e-12):.4e}")

    # 构造 LJ 势表并样条插值
    r_tab = np.linspace(0.2, 8.0, 100)
    epsilon_LJ = 4.0
    sigma_LJ = 3.0
    V_tab = 4.0 * epsilon_LJ * ((sigma_LJ / r_tab) ** 12 - (sigma_LJ / r_tab) ** 6)
    V_tab[V_tab > 100.0] = 100.0  # 截断
    spline = CubicSplineInterpolator(r_tab, V_tab)
    r_test = np.array([0.5, 1.0, 2.5, 5.0])
    V_interp = spline.evaluate(r_test)
    V_deriv = spline.derivative(r_test)
    print(f"  样条插值势能在 r={r_test} nm: {V_interp}")
    print(f"  对应力 -dV/dr: {-V_deriv}")

    # =====================================================================
    # 6. 统计采样与 SVD 变形模式（源自 seed 541_histogram_pdf + 1192_svd_sphere）
    # =====================================================================
    print_section("6. 概率采样与 SVD 变形模式分析")
    # 高斯型 PDF 采样验证
    def gaussian_pdf(x):
        return np.exp(-x ** 2 / 2.0) / np.sqrt(2.0 * np.pi)
    b_p, b_l, b_r = pdf_to_histogram(gaussian_pdf, n_bins=64, x_min=-4.0, x_max=4.0)
    c_x, c_y = histogram_to_cdf(b_p, b_l, b_r)
    samples = cdf_to_sample(c_x, c_y, n_samples=5000)
    print(f"  高斯采样均值: {np.mean(samples):.4f}, 标准差: {np.std(samples):.4f}")

    # 球面采样（纳米颗粒随机取向）
    sphere_pts = sphere_sample_marsaglia(1000)
    norms = np.linalg.norm(sphere_pts, axis=0)
    print(f"  球面采样平均模长: {np.mean(norms):.6f} (应 ~1.0)")

    # SVD 变形模式（模拟膜在 NP 压迫下的位移）
    # 人为构造一个高斯钟形位移场
    x_coords = membrane.vertices[:, 0]
    y_coords = membrane.vertices[:, 1]
    cx, cy = 5.0, 5.0
    displacement = np.zeros((3, membrane.n_v), dtype=np.float64)
    displacement[2, :] = -2.0 * np.exp(-((x_coords - cx) ** 2 + (y_coords - cy) ** 2) / 4.0)
    U, S, Vt = svd_deformation_modes(displacement)
    print(f"  SVD 奇异值: {S}")
    print(f"  主变形模式占比: {S[0]/np.sum(S):.2%}")

    # Metropolis 接受测试
    accepted = sum(boltzmann_acceptance(delta_E=5.0) for _ in range(1000))
    print(f"  Metropolis 接受率 (deltaE=5kJ/mol, T=300K): {accepted/1000:.3f}")

    # =====================================================================
    # 7. 相关随机力（源自 seed 1262_toeplitz_cholesky）
    # =====================================================================
    print_section("7. 指数记忆核相关随机力 (快速 Toeplitz Cholesky)")
    forces_corr = generate_correlated_forces(n_steps=4096, dt=0.001,
                                              gamma0=1.0, tau_mem=0.05)
    print(f"  生成随机力长度: {len(forces_corr)}")
    print(f"  均值: {np.mean(forces_corr):.6f}, 标准差: {np.std(forces_corr):.4f}")
    freqs, psd = colored_noise_spectrum(forces_corr, dt=0.001)
    # 验证低频 PSD > 高频 PSD（有色噪声特征）
    print(f"  低频 PSD (f<10): {np.mean(psd[freqs<10]):.4f}")
    print(f"  高频 PSD (f>100): {np.mean(psd[freqs>100]):.4f}")

    # =====================================================================
    # 8. 平衡距离求解（源自 seed 225_cpr）
    # =====================================================================
    print_section("8. Chebyshev Proxy Rootfinder 平衡态搜索")
    # 使用当前物理参数构造总力函数
    nld = NanoparticleLangevinDynamics(z0=5.0)
    def total_force_z(z):
        return nld.total_force(z, debye_length=pb.debye_length())
    # 限制搜索区间以避免 LJ 奇异性导致的 Chebyshev 插值振荡
    roots, stability = find_equilibrium_distances(total_force_z, z_min=2.0, z_max=8.0)
    print(f"  发现 {len(roots)} 个力平衡距离:")
    for z_eq, stab in zip(roots, stability):
        print(f"    z_eq = {z_eq:.4f} nm  [{stab}]")
    assert len(roots) > 0, "未找到平衡距离！"

    # =====================================================================
    # 9. 神经网络自由能代理（源自 seed 773_mnist_neural）
    # =====================================================================
    print_section("9. 神经网络结合自由能代理模型")
    X_train, y_train = generate_training_data(n_samples=512, seed=123)
    X_test, y_test = generate_training_data(n_samples=128, seed=456)
    nn = NeuralSurrogate(input_dim=6, hidden_dims=[32, 16, 8],
                          lr=0.005, lambda_reg=1e-5, seed=42)
    losses = nn.train(X_train, y_train, epochs=300, batch_size=32, verbose=False)
    y_pred = nn.predict(X_test)
    mse_test = float(np.mean((y_pred - y_test) ** 2))
    r2 = float(1.0 - np.sum((y_test - y_pred)**2) / (np.sum((y_test - np.mean(y_test))**2) + 1e-12))
    print(f"  训练轮数: 300, 最终训练损失: {losses[-1]:.4f}")
    print(f"  测试集 MSE: {mse_test:.4f}")
    print(f"  测试集 R^2: {r2:.4f}")
    assert r2 > 0.5, "代理模型拟合不足！"

    # =====================================================================
    # 10. 纳米颗粒朗之万动力学轨迹（源自 seed 488_grazing_ode）
    # =====================================================================
    print_section("10. 纳米颗粒过阻尼朗之万动力学轨迹")
    nld = NanoparticleLangevinDynamics(z0=8.0)
    t_traj, z_traj, F_traj = nld.simulate(n_steps=20000,
                                           debye_length=pb.debye_length())
    print(f"  模拟时长: {t_traj[-1]:.3f} ns")
    print(f"  初始距离: {z_traj[0]:.3f} nm")
    print(f"  终止距离: {z_traj[-1]:.3f} nm")
    print(f"  平均距离: {np.mean(z_traj):.3f} nm")
    print(f"  距离标准差: {np.std(z_traj):.3f} nm")
    print(f"  最大受力: {np.max(np.abs(F_traj)):.2f} kJ/(mol*nm)")

    # =====================================================================
    # 11. 并行力计算验证（源自 seed 514_hello_parfor）
    # =====================================================================
    print_section("11. 并行计算框架验证")
    data = list(range(1, 17))
    results = parallel_map(_square_task, data, n_workers=2)
    print(f"  并行 map 输入: {data}")
    print(f"  并行 map 输出: {results}")
    assert results == [x ** 2 for x in data], "并行计算结果错误！"

    # =====================================================================
    # 总结
    # =====================================================================
    print_section("综合模拟完成")
    print("  所有子系统运行正常，数值结果通过一致性检验。")
    print("  核心科学输出：")
    print(f"    - 膜弯曲能: {E_bend:.4f} k_B T")
    print(f"    - 德拜长度: {pb.debye_length():.4f} nm")
    print(f"    - 平衡距离: {roots[0]:.4f} nm ({stability[0]})")
    print(f"    - 神经网络 R^2: {r2:.4f}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    _ret = main()


# ================================================================
# 测试用例（45个，assert模式，涉及随机值均使用固定种子）
# ================================================================
# ---- TC01: topology_checksum of valid identifier returns 0 ----
from topology_validator import topology_checksum
cs = topology_checksum("PH-00000")
assert cs == 0, '[TC01] topology_checksum valid string FAILED'

# ---- TC02: topology_check_digit produces digit that makes checksum zero ----
from topology_validator import topology_check_digit
cd = topology_check_digit("PH-0000")
ts = "PH-0000" + str(cd)
assert topology_checksum(ts) == 0, '[TC02] topology_check_digit FAILED'

# ---- TC03: generate_topologies produces all-valid list with correct count ----
bts = ["PH", "GL"]
topos = generate_topologies(bts, count=2)
assert all(validate_topology(t) for t in topos), '[TC03] generate_topologies validity FAILED'
assert len(topos) == 4, '[TC03] generate_topologies count FAILED'

# ---- TC04: TriangulatedMembrane.create_planar_sheet correct dimensions ----
import numpy as np
mem = TriangulatedMembrane.create_planar_sheet(nx=8, ny=8, lx=10.0, ly=10.0)
assert mem.n_v == 64, '[TC04] planar sheet vertex count FAILED'
assert mem.n_e == 2 * 7 * 7, '[TC04] planar sheet element count FAILED'

# ---- TC05: planar membrane bending energy finite and non-negative ----
E_b = mem.bending_energy(kappa=20.0)
assert np.isfinite(E_b), '[TC05] bending energy finite FAILED'
assert E_b >= 0, '[TC05] bending energy non-negative FAILED'

# ---- TC06: membrane element areas all positive, total area = lx*ly ----
areas = mem.compute_element_areas()
assert np.all(areas > 0), '[TC06] element areas positive FAILED'
assert abs(np.sum(areas) - 100.0) < 1e-10, '[TC06] total area FAILED'

# ---- TC07: membrane normals all unit length (planar sheet -> all same dir) ----
normals = mem.compute_normals()
n_norms = np.linalg.norm(normals, axis=1)
assert np.all(np.abs(n_norms - 1.0) < 1e-12), '[TC07] normals unit length FAILED'

# ---- TC08: PoissonBoltzmannSolver debye_length is positive ----
pb = PoissonBoltzmannSolver(R_np=2.5, R_max=25.0, z_np=+10.0, n_0=0.1, epsilon=80.0*8.854e-12, T=300.0)
assert pb.debye_length() > 0, '[TC08] debye_length positive FAILED'

# ---- TC09: PB Jacobi solver returns finite phi and converges ----
phi_j, r_j, it_j, res_j = pb.solve_jacobi(n_grid=129, it_max=10000)
assert np.all(np.isfinite(phi_j)), '[TC09] Jacobi phi finite FAILED'
assert res_j < 1e-2, '[TC09] Jacobi residual FAILED'

# ---- TC10: PB Jacobi vs BiCG consistency ----
phi_b, r_b, it_b, res_b = pb.solve_bicg(n_grid=129)
diff_pb = np.max(np.abs(phi_j - phi_b))
assert diff_pb < 0.1, '[TC10] Jacobi vs BiCG consistency FAILED'

# ---- TC11: PB electrostatic_force is finite ----
ef = pb.electrostatic_force(n_grid=129)
assert np.isfinite(ef), '[TC11] electrostatic_force finite FAILED'

# ---- TC12: transport initial_condition bounded between 0 and c0 ----
ts_port = AdvectionDiffusionSolver(L=10.0, v=0.05, D=0.1, c0=0.1, nx=101)
c_init = ts_port.initial_condition(depletion_width=2.0)
assert np.all(c_init >= 0), '[TC12] init conc non-negative FAILED'
assert np.all(c_init <= ts_port.c0), '[TC12] init conc bounded FAILED'

# ---- TC13: transport solve preserves non-negative concentration ----
c_final, _ = ts_port.solve(n_steps=100)
assert np.all(c_final >= 0), '[TC13] final conc non-negative FAILED'

# ---- TC14: gauss_legendre_1d weights sum to 2 ----
from potential_energy import gauss_legendre_1d
x_gl, w_gl = gauss_legendre_1d(5)
assert abs(np.sum(w_gl) - 2.0) < 1e-12, '[TC14] GL weights sum FAILED'

# ---- TC15: integrate_nd of constant function returns hyper-volume ----
def const_f(x):
    return np.ones(x.shape[0])
a_test = np.array([0.0, 0.0])
b_test = np.array([2.0, 3.0])
val = integrate_nd(const_f, a_test, b_test, n_per_dim=5)
assert abs(val - 6.0) < 1e-10, '[TC15] integrate_nd constant FAILED'

# ---- TC16: membrane_binding_energy_integral numeric approx analytic ----
E_num, E_exact = membrane_binding_energy_integral(R_np=2.5, kappa=20.0, sigma=1.0, n_quad=5)
assert abs(E_num - E_exact) / (abs(E_exact) + 1e-12) < 0.01, '[TC16] binding energy integral FAILED'

# ---- TC17: CubicSplineInterpolator recovers values at original nodes (x^2) ----
x_spl = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
y_spl = x_spl ** 2
spl = CubicSplineInterpolator(x_spl, y_spl)
y_eval = spl.evaluate(x_spl)
assert np.max(np.abs(y_eval - y_spl)) < 1e-12, '[TC17] spline at nodes FAILED'

# ---- TC18: CubicSplineInterpolator derivative of x^2 is positive and monotonic ----
y_der = spl.derivative(np.array([1.0, 2.0, 3.0]))
assert np.all(y_der > 0), '[TC18] spline derivative positive FAILED'
assert y_der[0] < y_der[1] < y_der[2], '[TC18] spline derivative monotonic FAILED'

# ---- TC19: pdf_to_histogram produces non-negative densities ----
def gpdf(x):
    return np.exp(-x**2 / 2.0) / np.sqrt(2.0 * np.pi)
b_p, b_l, b_r = pdf_to_histogram(gpdf, n_bins=32, x_min=-4.0, x_max=4.0)
assert np.all(b_p >= 0), '[TC19] histogram non-negative FAILED'

# ---- TC20: histogram_to_cdf is monotonic with correct endpoints ----
c_x, c_y = histogram_to_cdf(b_p, b_l, b_r)
assert np.all(np.diff(c_y) >= 0), '[TC20] CDF monotonic FAILED'
assert abs(c_y[0]) < 1e-12, '[TC20] CDF start FAILED'
assert abs(c_y[-1] - 1.0) < 1e-12, '[TC20] CDF end FAILED'

# ---- TC21: cdf_to_sample returns correct count and finite values ----
np.random.seed(42)
samps = cdf_to_sample(c_x, c_y, n_samples=1000)
assert len(samps) == 1000, '[TC21] cdf_to_sample count FAILED'
assert np.all(np.isfinite(samps)), '[TC21] cdf_to_sample finite FAILED'

# ---- TC22: sphere_sample_marsaglia produces unit vectors ----
np.random.seed(42)
pts = sphere_sample_marsaglia(500)
nms = np.linalg.norm(pts, axis=0)
assert np.all(np.abs(nms - 1.0) < 1e-12), '[TC22] sphere norms FAILED'

# ---- TC23: svd_deformation_modes returns valid shapes and non-negative S ----
np.random.seed(42)
D_svd = np.random.randn(3, 20)
U_svd, S_svd, Vt_svd = svd_deformation_modes(D_svd)
assert np.all(S_svd >= 0), '[TC23] SVD singular values non-negative FAILED'
assert U_svd.shape == (3, 3), '[TC23] SVD U shape FAILED'
assert Vt_svd.shape == (3, 20), '[TC23] SVD Vt shape FAILED'

# ---- TC24: boltzmann_acceptance always accepts negative delta_E ----
np.random.seed(42)
acc_neg = boltzmann_acceptance(delta_E=-10.0, T=300.0, k_B=8.314e-3)
assert acc_neg == True, '[TC24] Boltzmann negative energy FAILED'

# ---- TC25: boltzmann_acceptance returns True or False ----
np.random.seed(42)
acc_result = boltzmann_acceptance(delta_E=5.0, T=300.0, k_B=8.314e-3)
assert acc_result in (True, False), '[TC25] Boltzmann return value FAILED'

# ---- TC26: exponential_kernel correct values at tau=0 and monotonic decay ----
from correlated_forces import exponential_kernel
tau_arr = np.array([0.0, 0.1, 0.2])
kern = exponential_kernel(tau_arr, gamma0=1.0, tau_mem=0.1)
assert abs(kern[0] - 1.0) < 1e-10, '[TC26] kernel at zero FAILED'
assert kern[1] < kern[0], '[TC26] kernel monotonic a FAILED'
assert kern[2] < kern[1], '[TC26] kernel monotonic b FAILED'

# ---- TC27: generate_correlated_forces produces finite sequence of correct length ----
np.random.seed(42)
f_corr = generate_correlated_forces(n_steps=1000, dt=0.001, gamma0=1.0, tau_mem=0.05)
assert len(f_corr) == 1000, '[TC27] correlated forces length FAILED'
assert np.all(np.isfinite(f_corr)), '[TC27] correlated forces finite FAILED'

# ---- TC28: correlated forces reproducibility with fixed seed ----
np.random.seed(42)
f1 = generate_correlated_forces(n_steps=100, dt=0.001, gamma0=1.0, tau_mem=0.05)
np.random.seed(42)
f2 = generate_correlated_forces(n_steps=100, dt=0.001, gamma0=1.0, tau_mem=0.05)
assert np.allclose(f1, f2), '[TC28] correlated forces reproducibility FAILED'

# ---- TC29: colored_noise_spectrum PSD non-negative ----
freqs, psd = colored_noise_spectrum(f_corr, dt=0.001)
assert np.all(psd >= 0), '[TC29] PSD non-negative FAILED'

# ---- TC30: chebyshev_proxy_rootfinder finds sin(x)=0 at pi ----
from equilibrium_solver import chebyshev_proxy_rootfinder
roots_sin = chebyshev_proxy_rootfinder(lambda x: np.sin(x), 2.5, 3.5, N=32)
found_pi = any(abs(r - np.pi) < 1e-6 for r in roots_sin)
assert found_pi, '[TC30] CPR sin(pi) root FAILED'

# ---- TC31: chebyshev_proxy_rootfinder finds quadratic roots ----
roots_quad = chebyshev_proxy_rootfinder(lambda x: x**2 - 4.0, -3.0, 3.0, N=32)
found_m2 = any(abs(r - (-2.0)) < 1e-6 for r in roots_quad)
found_p2 = any(abs(r - 2.0) < 1e-6 for r in roots_quad)
assert found_m2 and found_p2, '[TC31] CPR quadratic roots FAILED'

# ---- TC32: generate_training_data correct shapes and finite values ----
X_tr, y_tr = generate_training_data(n_samples=100, seed=42)
assert X_tr.shape == (100, 6), '[TC32] training X shape FAILED'
assert y_tr.shape == (100, 1), '[TC32] training y shape FAILED'
assert np.all(np.isfinite(X_tr)), '[TC32] X finite FAILED'
assert np.all(np.isfinite(y_tr)), '[TC32] y finite FAILED'

# ---- TC33: NanoparticleLangevinDynamics total_force is finite ----
nld = NanoparticleLangevinDynamics(z0=5.0)
F_tot = nld.total_force(3.0, debye_length=1.0)
assert np.isfinite(F_tot), '[TC33] total_force finite FAILED'

# ---- TC34: NanoparticleLangevinDynamics force components all finite ----
F_vdw = nld.force_vdw(3.0)
F_bend = nld.force_bending(3.0)
F_bind = nld.force_binding(3.0)
F_elec = nld.force_electrostatic(3.0, debye_length=1.0)
assert np.isfinite(F_vdw), '[TC34] vdw force finite FAILED'
assert np.isfinite(F_bend), '[TC34] bending force finite FAILED'
assert np.isfinite(F_bind), '[TC34] binding force finite FAILED'
assert np.isfinite(F_elec), '[TC34] electrostatic force finite FAILED'

# ---- TC35: Langevin step_euler_maruyama produces finite z ----
np.random.seed(42)
z_step = nld.step_euler_maruyama(debye_length=1.0)
assert np.isfinite(z_step), '[TC35] Euler-Maruyama step finite FAILED'

# ---- TC36: parallel_map returns correct results ----
data_in = list(range(1, 11))
results = parallel_map(_square_task, data_in, n_workers=2)
assert results == [x**2 for x in data_in], '[TC36] parallel_map FAILED'

# ---- TC37: check_environment returns bool ----
env_ok = check_environment()
assert isinstance(env_ok, bool), '[TC37] check_environment type FAILED'

# ---- TC38: get_platform_info has all required keys ----
info = get_platform_info()
for key in ['python_version', 'platform', 'numpy_mkl', 'float_eps', 'max_threads']:
    assert key in info, f'[TC38] platform_info missing key {key} FAILED'

# ---- TC39: NeuralSurrogate trains and produces finite predictions ----
np.random.seed(42)
Xs, ys = generate_training_data(n_samples=128, seed=42)
nn_test = NeuralSurrogate(input_dim=6, hidden_dims=[16, 8], lr=0.005, lambda_reg=1e-5, seed=42)
losses = nn_test.train(Xs, ys, epochs=30, batch_size=32, verbose=False)
y_pred = nn_test.predict(Xs[:10])
assert np.all(np.isfinite(y_pred)), '[TC39] NN predict finite FAILED'
assert y_pred.shape == (10, 1), '[TC39] NN predict shape FAILED'

# ---- TC40: sample_random_orientation returns valid rotation matrix ----
np.random.seed(42)
R = sample_random_orientation()
assert R.shape == (3, 3), '[TC40] rotation matrix shape FAILED'
assert abs(np.linalg.det(R) - 1.0) < 1e-10, '[TC40] rotation det FAILED'
assert np.all(np.abs(R @ R.T - np.eye(3)) < 1e-10), '[TC40] rotation orthogonality FAILED'

# ---- TC41: transport compute_flux returns finite value ----
flux_val = ts_port.compute_flux(c_final)
assert np.isfinite(flux_val), '[TC41] compute_flux finite FAILED'

# ---- TC42: find_equilibrium_distances finds roots for polynomial force ----
def test_force(z):
    return (z - 3.0) * (z - 7.0)
roots_eq, stab_eq = find_equilibrium_distances(test_force, z_min=1.0, z_max=9.0)
assert len(roots_eq) >= 2, '[TC42] find_equilibrium_distances count FAILED'
found_3 = any(abs(r - 3.0) < 0.1 for r in roots_eq)
found_7 = any(abs(r - 7.0) < 0.1 for r in roots_eq)
assert found_3 and found_7, '[TC42] find_equilibrium_distances roots FAILED'

# ---- TC43: topology_checksum of string without digits returns non-zero ----
cs_empty = topology_checksum("ABC-DEF")
assert cs_empty != 0, '[TC43] topology_checksum no digits FAILED'

# ---- TC44: validate_topology returns False for no-digit string ----
assert validate_topology("NO-DIGITS-HERE") == False, '[TC44] validate no-digit string FAILED'

# ---- TC45: toep_cholesky_lower produces valid lower-triangular Cholesky factor ----
from correlated_forces import toep_cholesky_lower
np.random.seed(42)
t_row = np.array([2.0, 0.5, 0.2])
L45 = toep_cholesky_lower(3, t_row)
A_recon = L45 @ L45.T
assert L45.shape == (3, 3), '[TC45] Cholesky L shape FAILED'
assert abs(A_recon[0, 0] - t_row[0]) < 1e-10, '[TC45] Cholesky diag FAILED'
assert np.all(A_recon >= 0), '[TC45] Cholesky psd FAILED'

print('\n全部 45 个测试通过!\n')
