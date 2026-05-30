
import sys
import numpy as np




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




    print_section("0. 平台与环境检测")
    env_ok = check_environment()
    info = get_platform_info()
    print(f"  环境检查通过: {env_ok}")
    print(f"  NumPy MKL 后端: {info['numpy_mkl']}")
    print(f"  机器精度 eps: {info['float_eps']:.2e}")
    print(f"  可用 CPU 核心: {info['max_threads']}")




    print_section("1. 分子拓扑完整性校验 (MTIC)")
    bead_types = ["PH", "GL", "EST1", "EST2", "C1A", "C2A", "C1B", "C2B"]
    topologies = generate_topologies(bead_types, count=4)
    print(f"  生成 {len(topologies)} 个 Martini 磷脂拓扑标识符")
    valid_count = sum(1 for t in topologies if validate_topology(t))
    print(f"  通过 MTIC 校验: {valid_count}/{len(topologies)}")
    assert valid_count == len(topologies), "拓扑校验失败！"




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

    n_internal = np.sum(etoe != -1)
    n_boundary = np.sum(etoe == -1)
    print(f"  内部边: {n_internal}, 边界边: {n_boundary}")




    print_section("3. 球对称 Poisson-Boltzmann 静电求解")
    pb = PoissonBoltzmannSolver(R_np=2.5, R_max=25.0, z_np=+10.0,
                                 n_0=0.1, epsilon=80.0 * 8.854e-12, T=300.0)
    print(f"  德拜长度: {pb.debye_length():.4f} nm")

    phi_jacobi, r_jacobi, it_jacobi, res_jacobi = pb.solve_jacobi(n_grid=257)
    print(f"  Jacobi 迭代: {it_jacobi} 步, 最终残差: {res_jacobi:.4e}")

    phi_bicg, r_bicg, it_bicg, res_bicg = pb.solve_bicg(n_grid=257)
    print(f"  BiCG 迭代: {it_bicg} 步, 最终残差: {res_bicg:.4e}")


    diff_solver = np.max(np.abs(phi_jacobi - phi_bicg))
    print(f"  双求解器最大差异: {diff_solver:.4e} V")
    assert diff_solver < 1e-3, "Jacobi 与 BiCG 结果不一致！"

    dphi_surf = pb.electrostatic_force(n_grid=257)
    print(f"  表面电场梯度: {dphi_surf:.4e} V/nm")




    print_section("4. 膜附近离子对流-扩散输运 (Lax 格式)")
    transport = AdvectionDiffusionSolver(L=20.0, v=0.5, D=2.0,
                                          c0=0.1, nx=201)
    c_final, history = transport.solve(n_steps=500)
    flux_surf = transport.compute_flux(c_final)
    print(f"  空间格点数: {transport.nx}, 时间步长: {transport.dt:.6f} ns")
    print(f"  膜表面稳态通量: {flux_surf:.6e} (浓度单位)/ns")
    print(f"  本体浓度: {c_final[-1]:.4f}, 表面浓度: {c_final[0]:.4f}")




    print_section("5. 多维乘积求积与样条势函数")
    E_num, E_exact = membrane_binding_energy_integral(R_np=2.5, kappa=20.0,
                                                       sigma=1.0, n_quad=5)
    print(f"  结合能数值积分: {E_num:.4f} k_B T")
    print(f"  结合能解析验证: {E_exact:.4f} k_B T")
    print(f"  积分相对误差: {abs(E_num - E_exact) / (abs(E_exact) + 1e-12):.4e}")


    r_tab = np.linspace(0.2, 8.0, 100)
    epsilon_LJ = 4.0
    sigma_LJ = 3.0
    V_tab = 4.0 * epsilon_LJ * ((sigma_LJ / r_tab) ** 12 - (sigma_LJ / r_tab) ** 6)
    V_tab[V_tab > 100.0] = 100.0
    spline = CubicSplineInterpolator(r_tab, V_tab)
    r_test = np.array([0.5, 1.0, 2.5, 5.0])
    V_interp = spline.evaluate(r_test)
    V_deriv = spline.derivative(r_test)
    print(f"  样条插值势能在 r={r_test} nm: {V_interp}")
    print(f"  对应力 -dV/dr: {-V_deriv}")




    print_section("6. 概率采样与 SVD 变形模式分析")

    def gaussian_pdf(x):
        return np.exp(-x ** 2 / 2.0) / np.sqrt(2.0 * np.pi)
    b_p, b_l, b_r = pdf_to_histogram(gaussian_pdf, n_bins=64, x_min=-4.0, x_max=4.0)
    c_x, c_y = histogram_to_cdf(b_p, b_l, b_r)
    samples = cdf_to_sample(c_x, c_y, n_samples=5000)
    print(f"  高斯采样均值: {np.mean(samples):.4f}, 标准差: {np.std(samples):.4f}")


    sphere_pts = sphere_sample_marsaglia(1000)
    norms = np.linalg.norm(sphere_pts, axis=0)
    print(f"  球面采样平均模长: {np.mean(norms):.6f} (应 ~1.0)")



    x_coords = membrane.vertices[:, 0]
    y_coords = membrane.vertices[:, 1]
    cx, cy = 5.0, 5.0
    displacement = np.zeros((3, membrane.n_v), dtype=np.float64)
    displacement[2, :] = -2.0 * np.exp(-((x_coords - cx) ** 2 + (y_coords - cy) ** 2) / 4.0)
    U, S, Vt = svd_deformation_modes(displacement)
    print(f"  SVD 奇异值: {S}")
    print(f"  主变形模式占比: {S[0]/np.sum(S):.2%}")


    accepted = sum(boltzmann_acceptance(delta_E=5.0) for _ in range(1000))
    print(f"  Metropolis 接受率 (deltaE=5kJ/mol, T=300K): {accepted/1000:.3f}")




    print_section("7. 指数记忆核相关随机力 (快速 Toeplitz Cholesky)")
    forces_corr = generate_correlated_forces(n_steps=4096, dt=0.001,
                                              gamma0=1.0, tau_mem=0.05)
    print(f"  生成随机力长度: {len(forces_corr)}")
    print(f"  均值: {np.mean(forces_corr):.6f}, 标准差: {np.std(forces_corr):.4f}")
    freqs, psd = colored_noise_spectrum(forces_corr, dt=0.001)

    print(f"  低频 PSD (f<10): {np.mean(psd[freqs<10]):.4f}")
    print(f"  高频 PSD (f>100): {np.mean(psd[freqs>100]):.4f}")




    print_section("8. Chebyshev Proxy Rootfinder 平衡态搜索")

    nld = NanoparticleLangevinDynamics(z0=5.0)
    def total_force_z(z):
        return nld.total_force(z, debye_length=pb.debye_length())

    roots, stability = find_equilibrium_distances(total_force_z, z_min=2.0, z_max=8.0)
    print(f"  发现 {len(roots)} 个力平衡距离:")
    for z_eq, stab in zip(roots, stability):
        print(f"    z_eq = {z_eq:.4f} nm  [{stab}]")
    assert len(roots) > 0, "未找到平衡距离！"




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


def _square_task(x):
    return x ** 2


def main():
    np.random.seed(117)
    print("纳米颗粒-生物膜相互作用：粗粒化分子动力学综合模拟")
    print(f"Python: {sys.version.split()[0]}")




    print_section("0. 平台与环境检测")
    env_ok = check_environment()
    info = get_platform_info()
    print(f"  环境检查通过: {env_ok}")
    print(f"  NumPy MKL 后端: {info['numpy_mkl']}")
    print(f"  机器精度 eps: {info['float_eps']:.2e}")
    print(f"  可用 CPU 核心: {info['max_threads']}")




    print_section("1. 分子拓扑完整性校验 (MTIC)")
    bead_types = ["PH", "GL", "EST1", "EST2", "C1A", "C2A", "C1B", "C2B"]
    topologies = generate_topologies(bead_types, count=4)
    print(f"  生成 {len(topologies)} 个 Martini 磷脂拓扑标识符")
    valid_count = sum(1 for t in topologies if validate_topology(t))
    print(f"  通过 MTIC 校验: {valid_count}/{len(topologies)}")
    assert valid_count == len(topologies), "拓扑校验失败！"




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

    n_internal = np.sum(etoe != -1)
    n_boundary = np.sum(etoe == -1)
    print(f"  内部边: {n_internal}, 边界边: {n_boundary}")




    print_section("3. 球对称 Poisson-Boltzmann 静电求解")
    pb = PoissonBoltzmannSolver(R_np=2.5, R_max=25.0, z_np=+10.0,
                                 n_0=0.1, epsilon=80.0 * 8.854e-12, T=300.0)
    print(f"  德拜长度: {pb.debye_length():.4f} nm")

    phi_jacobi, r_jacobi, it_jacobi, res_jacobi = pb.solve_jacobi(n_grid=257)
    print(f"  Jacobi 迭代: {it_jacobi} 步, 最终残差: {res_jacobi:.4e}")

    phi_bicg, r_bicg, it_bicg, res_bicg = pb.solve_bicg(n_grid=257)
    print(f"  BiCG 迭代: {it_bicg} 步, 最终残差: {res_bicg:.4e}")


    diff_solver = np.max(np.abs(phi_jacobi - phi_bicg))
    print(f"  双求解器最大差异: {diff_solver:.4e} V")
    assert diff_solver < 1e-3, "Jacobi 与 BiCG 结果不一致！"

    dphi_surf = pb.electrostatic_force(n_grid=257)
    print(f"  表面电场梯度: {dphi_surf:.4e} V/nm")




    print_section("4. 膜附近离子对流-扩散输运 (Lax 格式)")
    transport = AdvectionDiffusionSolver(L=20.0, v=0.05, D=0.1,
                                          c0=0.1, nx=201)
    c_final, history = transport.solve(n_steps=500)
    flux_surf = transport.compute_flux(c_final)
    print(f"  空间格点数: {transport.nx}, 时间步长: {transport.dt:.6f} ns")
    print(f"  膜表面稳态通量: {flux_surf:.6e} (浓度单位)/ns")
    print(f"  本体浓度: {c_final[-1]:.4f}, 表面浓度: {c_final[0]:.4f}")




    print_section("5. 多维乘积求积与样条势函数")
    E_num, E_exact = membrane_binding_energy_integral(R_np=2.5, kappa=20.0,
                                                       sigma=1.0, n_quad=5)
    print(f"  结合能数值积分: {E_num:.4f} k_B T")
    print(f"  结合能解析验证: {E_exact:.4f} k_B T")
    print(f"  积分相对误差: {abs(E_num - E_exact) / (abs(E_exact) + 1e-12):.4e}")


    r_tab = np.linspace(0.2, 8.0, 100)
    epsilon_LJ = 4.0
    sigma_LJ = 3.0
    V_tab = 4.0 * epsilon_LJ * ((sigma_LJ / r_tab) ** 12 - (sigma_LJ / r_tab) ** 6)
    V_tab[V_tab > 100.0] = 100.0
    spline = CubicSplineInterpolator(r_tab, V_tab)
    r_test = np.array([0.5, 1.0, 2.5, 5.0])
    V_interp = spline.evaluate(r_test)
    V_deriv = spline.derivative(r_test)
    print(f"  样条插值势能在 r={r_test} nm: {V_interp}")
    print(f"  对应力 -dV/dr: {-V_deriv}")




    print_section("6. 概率采样与 SVD 变形模式分析")

    def gaussian_pdf(x):
        return np.exp(-x ** 2 / 2.0) / np.sqrt(2.0 * np.pi)
    b_p, b_l, b_r = pdf_to_histogram(gaussian_pdf, n_bins=64, x_min=-4.0, x_max=4.0)
    c_x, c_y = histogram_to_cdf(b_p, b_l, b_r)
    samples = cdf_to_sample(c_x, c_y, n_samples=5000)
    print(f"  高斯采样均值: {np.mean(samples):.4f}, 标准差: {np.std(samples):.4f}")


    sphere_pts = sphere_sample_marsaglia(1000)
    norms = np.linalg.norm(sphere_pts, axis=0)
    print(f"  球面采样平均模长: {np.mean(norms):.6f} (应 ~1.0)")



    x_coords = membrane.vertices[:, 0]
    y_coords = membrane.vertices[:, 1]
    cx, cy = 5.0, 5.0
    displacement = np.zeros((3, membrane.n_v), dtype=np.float64)
    displacement[2, :] = -2.0 * np.exp(-((x_coords - cx) ** 2 + (y_coords - cy) ** 2) / 4.0)
    U, S, Vt = svd_deformation_modes(displacement)
    print(f"  SVD 奇异值: {S}")
    print(f"  主变形模式占比: {S[0]/np.sum(S):.2%}")


    accepted = sum(boltzmann_acceptance(delta_E=5.0) for _ in range(1000))
    print(f"  Metropolis 接受率 (deltaE=5kJ/mol, T=300K): {accepted/1000:.3f}")




    print_section("7. 指数记忆核相关随机力 (快速 Toeplitz Cholesky)")
    forces_corr = generate_correlated_forces(n_steps=4096, dt=0.001,
                                              gamma0=1.0, tau_mem=0.05)
    print(f"  生成随机力长度: {len(forces_corr)}")
    print(f"  均值: {np.mean(forces_corr):.6f}, 标准差: {np.std(forces_corr):.4f}")
    freqs, psd = colored_noise_spectrum(forces_corr, dt=0.001)

    print(f"  低频 PSD (f<10): {np.mean(psd[freqs<10]):.4f}")
    print(f"  高频 PSD (f>100): {np.mean(psd[freqs>100]):.4f}")




    print_section("8. Chebyshev Proxy Rootfinder 平衡态搜索")

    nld = NanoparticleLangevinDynamics(z0=5.0)
    def total_force_z(z):
        return nld.total_force(z, debye_length=pb.debye_length())

    roots, stability = find_equilibrium_distances(total_force_z, z_min=2.0, z_max=8.0)
    print(f"  发现 {len(roots)} 个力平衡距离:")
    for z_eq, stab in zip(roots, stability):
        print(f"    z_eq = {z_eq:.4f} nm  [{stab}]")
    assert len(roots) > 0, "未找到平衡距离！"




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




    print_section("11. 并行计算框架验证")
    data = list(range(1, 17))
    results = parallel_map(_square_task, data, n_workers=2)
    print(f"  并行 map 输入: {data}")
    print(f"  并行 map 输出: {results}")
    assert results == [x ** 2 for x in data], "并行计算结果错误！"




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
    sys.exit(main())
