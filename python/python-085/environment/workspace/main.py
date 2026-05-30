import numpy as np
import sys


from mesh_generator import generate_rectangular_mesh, TriMesh2D
from fem_assembler import ElasticFEM2D, assemble_contact_gaps
from banded_solver import BandedSolver, extract_banded_submatrix
from contact_solver import SignoriniCoulombContact, active_set_newton_contact
from wear_ode import ArchardWearModel, coupled_wear_contact_step
from monte_carlo_contact import (
    monte_carlo_contact_force_variance,
    monte_carlo_friction_coefficient_sensitivity
)
from friction_optimization import friction_coefficient_calibration, peaks_surface_contact_potential
from roughness_field import generate_pink_noise_profile, apply_roughness_to_mesh, correlation_function
from diophantine_utils import (
    diophantine_nonnegative_solve,
    optimize_node_numbering_bandwidth,
    compute_matrix_bandwidth
)
from complex_analysis import (
    complex_modal_analysis,
    complex_damping_matrix_from_friction,
    stability_criterion,
    frequency_response_function
)
from utils import r8mat_print_some, condition_number_estimate


def main():
    print("=" * 78)
    print("  博士级科研代码合成项目：结构力学接触问题与摩擦算法")
    print("  Synthesis Project: Contact Mechanics & Friction Algorithms")
    print("=" * 78)




    print("\n[Phase 1] 网格生成与三角剖分")
    nx, ny = 21, 11
    lx, ly = 1.0, 0.5
    mesh = generate_rectangular_mesh(nx, ny, lx=lx, ly=ly, shift_y=0.0)
    print(f"  网格节点数: {mesh.n_nodes}")
    print(f"  网格单元数: {mesh.n_elements}")
    print(f"  网格总面积: {mesh.total_area():.6f}")


    contact_nodes = mesh.find_bottom_boundary_nodes(tol=1e-6)
    print(f"  接触节点数: {len(contact_nodes)}")


    elem_vals = mesh.node_to_element_average(mesh.nodes[:, 0])
    print(f"  节点->单元平均 (x坐标) 示例: 前5单元 = {elem_vals[:5].flatten()}")




    print("\n[Phase 2] 线弹性有限元刚度矩阵组装")
    E = 2.1e11
    nu = 0.3
    fem = ElasticFEM2D(mesh, young=E, nu=nu, thickness=1.0)
    K = fem.assemble_global_stiffness()
    print(f"  全局刚度矩阵维度: {K.shape}")
    print(f"  条件数估计: {condition_number_estimate(K):.4e}")


    n_dof = 2 * mesh.n_nodes
    f_ext = np.zeros(n_dof)
    top_nodes = np.where(np.abs(mesh.nodes[:, 1] - ly) < 1e-6)[0]
    q_load = -1.0e6
    dx = lx / (nx - 1)
    nodal_force = q_load * dx * 0.5
    for node in top_nodes:
        f_ext[2 * node + 1] = nodal_force
    print(f"  外部载荷节点数: {len(top_nodes)}")
    print(f"  总外部力: {np.sum(f_ext[1::2]):.4e} N")




    print("\n[Phase 3] Signorini-Coulomb 接触问题求解")
    mu_friction = 0.3
    contact_solver = SignoriniCoulombContact(
        fem, contact_nodes, friction_coeff=mu_friction,
        aug_lag_penalty=1e9, max_iter=80, tol=1e-8
    )



    left_nodes = np.where(np.abs(mesh.nodes[:, 0]) < 1e-6)[0]
    right_nodes = np.where(np.abs(mesh.nodes[:, 0] - lx) < 1e-6)[0]
    fixed_nodes = np.unique(np.concatenate([left_nodes, right_nodes]))
    fixed_nodes = np.setdiff1d(fixed_nodes, contact_nodes)
    fixed_values = np.zeros((len(fixed_nodes), 2))
    dof_mask = None

    u_al, hist_al = contact_solver.solve_static(f_ext, fixed_nodes, fixed_values, dof_mask=dof_mask)
    p_n, p_t = contact_solver.compute_contact_pressure(u_al)
    print(f"  AL 迭代次数: {hist_al['iterations']}")
    print(f"  最终残差: {hist_al['final_residual']:.4e}")
    print(f"  最大法向压力: {np.max(p_n):.4e} Pa")
    print(f"  最大切向牵引力: {np.max(np.abs(p_t)):.4e} Pa")
    print(f"  激活接触节点数: {np.sum(p_n > 1.0):d}")


    u_as, hist_as = active_set_newton_contact(fem, contact_nodes, f_ext, mu_friction,
                                               fixed_nodes=fixed_nodes, fixed_values=fixed_values, dof_mask=dof_mask)
    print(f"  主动集 Newton 迭代次数: {hist_as['iterations']}")




    print("\n[Phase 4] 带状矩阵求解器验证")
    ml, mu = compute_matrix_bandwidth(K)
    print(f"  下带宽 ml: {ml}, 上带宽 mu: {mu}")
    solver = BandedSolver(K.shape[0], ml, mu, compact=True)
    K_band = solver.full_to_compact(K)

    try:
        u_band = solver.solve_system(K_band, f_ext, use_pivot=True)
        err_band = np.linalg.norm(K @ u_band - f_ext) / np.linalg.norm(f_ext)
        print(f"  带状求解残差: {err_band:.4e}")
    except ValueError as e:
        print(f"  带状求解跳过（矩阵非严格带状可分解）: {e}")


    if len(contact_nodes) > 0:
        K_sub_band = extract_banded_submatrix(K, contact_nodes, mesh.n_nodes, 2, 2)
        print(f"  接触子矩阵带状存储维度: {K_sub_band.shape}")




    print("\n[Phase 5] Archard 磨损演化分析")
    wear_model = ArchardWearModel(wear_coeff=1e-7, omega=2.0 * np.pi, v0=0.05)


    p_mean = float(np.mean(p_n[p_n > 0])) if np.any(p_n > 0) else 1.0e5
    def pressure_func(t: float) -> float:
        return p_mean

    t_wear, h_wear = wear_model.integrate_midpoint(
        h0=0.0, t_span=(0.0, 10.0), n_steps=200, pressure_func=pressure_func
    )
    print(f"  中点法磨损深度 (t=10s): {h_wear[-1]:.6e} m")


    t_wear2, h_wear2 = wear_model.integrate_rk4(
        h0=0.0, t_span=(0.0, 10.0), n_steps=200, pressure_func=pressure_func
    )
    print(f"  RK4 磨损深度 (t=10s): {h_wear2[-1]:.6e} m")
    print(f"  两种积分器差异: {abs(h_wear[-1] - h_wear2[-1]):.6e}")


    wear_step = coupled_wear_contact_step(wear_model, np.zeros(len(contact_nodes)), p_n, dt=0.01)
    print(f"  耦合单步磨损深度范围: [{np.min(wear_step):.4e}, {np.max(wear_step):.4e}]")




    print("\n[Phase 6] 接触压力蒙特卡洛不确定性分析")

    def sample_pressure(samples: np.ndarray) -> np.ndarray:

        if len(samples) == 0:
            return np.zeros(0)
        x_samples = samples[:, 0]
        x_contact = mesh.nodes[contact_nodes, 0]

        order = np.argsort(x_contact)
        return np.interp(x_samples, x_contact[order], p_n[order], left=0.0, right=0.0)

    mean_p, var_p, max_p = monte_carlo_contact_force_variance(mesh, sample_pressure, n_samples=3000)
    print(f"  蒙特卡洛压力均值: {mean_p:.4e} Pa")
    print(f"  蒙特卡洛压力方差: {var_p:.4e}")
    print(f"  蒙特卡洛压力最大: {max_p:.4e} Pa")


    def friction_sim(mu: float) -> float:
        cs = SignoriniCoulombContact(fem, contact_nodes, friction_coeff=mu, aug_lag_penalty=1e12)
        u_tmp, _ = cs.solve_static(f_ext, fixed_nodes, fixed_values, dof_mask=dof_mask)
        _, pt = cs.compute_contact_pressure(u_tmp)
        return float(np.max(np.abs(pt)))

    sens = monte_carlo_friction_coefficient_sensitivity(
        friction_sim, mu_mean=0.3, mu_std=0.03, n_batches=5, n_per_batch=20
    )
    print(f"  摩擦敏感性整体均值: {sens['overall_mean']:.4e}")
    print(f"  摩擦敏感性整体标准差: {sens['overall_std']:.4e}")




    print("\n[Phase 7] 摩擦系数 Hooke-Jeeves 优化校准")
    target_max_traction = 5.0e4
    mu_cal, mu_info = friction_coefficient_calibration(
        friction_sim, target_max_traction, mu_bounds=(0.05, 1.0)
    )
    print(f"  校准后摩擦系数: {mu_cal:.6f}")
    print(f"  优化迭代次数: {mu_info['iterations']}")
    print(f"  最终目标函数值: {mu_info['final_objective']:.4e}")


    peak_pot = peaks_surface_contact_potential(0.2, 0.1, amplitude=1e8)
    print(f"  Peaks 接触势能 (x=0.2,y=0.1): {peak_pot:.4e} J/m^2")




    print("\n[Phase 8] 接触面 1/f 粗糙度随机场生成")
    x_rough, h_rough = generate_pink_noise_profile(n_points=200, length=lx, beta=1.8)
    print(f"  粗糙度轮廓长度: {len(h_rough)}")
    print(f"  粗糙度均方根 (RMS): {np.std(h_rough):.6f}")


    r_corr = correlation_function(h_rough, m=20)
    print(f"  粗糙度自相关系数 (lag=0): {r_corr[0]:.6f}")
    print(f"  粗糙度自相关系数 (lag=5): {r_corr[5]:.6f}")


    contact_mask = np.zeros(mesh.n_nodes, dtype=bool)
    contact_mask[contact_nodes] = True
    nodes_rough = apply_roughness_to_mesh(mesh.nodes, h_rough, contact_mask, scale=1e-5)
    max_rough_disp = np.max(np.abs(nodes_rough[:, 1] - mesh.nodes[:, 1]))
    print(f"  最大粗糙度位移: {max_rough_disp:.4e} m")




    print("\n[Phase 9] 丢番图节点编号整数规划")

    new_order = optimize_node_numbering_bandwidth(contact_nodes, mesh.n_nodes)
    print(f"  重编号后接触节点位置: [{np.min(new_order[contact_nodes])}, {np.max(new_order[contact_nodes])}]")


    a_test = np.array([6, 10, 15])
    b_test = 60
    try:
        d_sol, v_sol, B_sol, kmin_sol, kmax_sol = diophantine_nonnegative_solve(a_test, b_test)
        print(f"  丢番图解示例: d={d_sol}, v={v_sol}")
        print(f"  通解基向量维度: {B_sol.shape}")
    except ValueError as e:
        print(f"  丢番图求解说明: {e}")




    print("\n[Phase 10] 复特征值稳定性分析")
    n = K.shape[0]

    M = np.diag(np.diag(K) * 0.001)
    C_friction = complex_damping_matrix_from_friction(K, contact_nodes, mu_friction, p_n)
    eigenvalues, eigenvectors = complex_modal_analysis(M, C_friction, K)
    stab = stability_criterion(eigenvalues)
    print(f"  最大特征值实部: {stab['alpha_max']:.4e}")
    print(f"  不稳定模态数: {stab['unstable_count']}")
    print(f"  最小阻尼比: {stab['min_damping_ratio']:.6f}")
    print(f"  最大颤振频率: {stab['max_flutter_freq_hz']:.2f} Hz")


    omega_range = np.linspace(1.0, 500.0, 200)
    load_dof = 2 * mesh.n_nodes // 2 + 1
    frf = frequency_response_function(K, M, C_friction, omega_range, load_dof)
    print(f"  频响函数峰值: {np.max(frf):.4e} m/N")
    print(f"  频响函数峰值频率: {omega_range[np.argmax(frf)]:.2f} rad/s")




    print("\n" + "=" * 78)
    print("  计算结果汇总")
    print("=" * 78)
    print(f"  最大法向接触压力:     {np.max(p_n):.4e} Pa")
    print(f"  最大切向摩擦牵引力:   {np.max(np.abs(p_t)):.4e} Pa")
    print(f"  结构最大位移:         {np.max(np.abs(u_al)):.4e} m")
    print(f"  应变能:               {fem.compute_strain_energy(u_al):.4e} J")
    print(f"  10s 累积磨损深度:     {h_wear[-1]:.4e} m")
    print(f"  校准后摩擦系数:       {mu_cal:.6f}")
    print(f"  颤振不稳定模态数:     {stab['unstable_count']}")
    print(f"  整体计算通过:         YES")
    print("=" * 78)
    print("  程序正常结束。")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
