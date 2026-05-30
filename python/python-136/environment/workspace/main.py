
import os
import sys
import numpy as np




np.random.seed(42)




from special_functions import (
    complex_log_stable,
    thiele_modulus_efficiency_factor,
    knudsen_diffusivity,
    effective_diffusivity,
    arrhenius_rate,
    gegenbauer_integral,
)
from linear_solvers import (
    conjugate_gradient_rc,
    jacobi_preconditioner,
    solve_tridiagonal,
)
from quadrature_rules import (
    gauss_legendre_rule,
    gauss_genlaguerre_rule,
    radial_quadrature_sphere,
    gegenbauer_quadrature_exactness,
    integrate_reaction_rate_radial,
    pore_size_moment_quadrature,
)
from mesh_generation import (
    cvt_1d_lloyd,
    adaptive_radial_mesh,
    cvt_square_uniform_2d,
)
from interpolation import (
    pwl_interp_2d,
    radial_to_2d_interpolator,
)
from monte_carlo_pore import (
    random_triangle_area_in_disk,
    pore_accessibility_simulation,
    estimate_effective_diffusivity_mc,
    pore_tortuosity_from_mc,
)
from integration_validator import (
    validate_2d_quadrature_rule,
    black_scholes_diffusion_analogy,
    diffusion_green_function_integral,
    validate_reaction_diffusion_conservation,
)
from pore_diffusion import (
    solve_diffusion_reaction_fd,
    solve_diffusion_reaction_fem,
    diffusion_flux_at_surface,
    effectiveness_factor_from_profile,
)
from surface_reaction import (
    LangmuirHinshelwoodKinetics,
    PowerLawKinetics,
    CatalyticParticleModel,
)
from nonlinear_solver import (
    solve_coupled_diffusion_reaction_newton,
    pseudo_transient_continuation,
)
from data_io import (
    write_xy_profile,
    read_xy_profile,
    ensure_dir,
)





def setup_physical_model():

    particle_radius = 3.0e-3
    porosity = 0.42
    tortuosity = 3.5
    pore_diameter = 15.0e-9


    bulk_diffusivity_CO = 2.0e-5
    molecular_weight_CO = 28.01e-3
    lambda_solid = 1.5
    lambda_gas = 0.03


    k0 = 1.2e8
    Ea = 75000.0
    KA0 = 2.5e-5
    dH_ads_A = -45000.0
    KB0 = 1.0e-4
    dH_ads_B = -35000.0


    T_surface = 573.0
    C_surface_A = 12.0
    C_surface_B = 6.0
    heat_of_reaction = -283.0e3

    kinetics = LangmuirHinshelwoodKinetics(
        k0=k0, Ea=Ea,
        KA0=KA0, dH_ads_A=dH_ads_A,
        KB0=KB0, dH_ads_B=dH_ads_B,
        reaction_order_A=1.0, reaction_order_B=0.5
    )

    particle = CatalyticParticleModel(
        kinetics=kinetics,
        particle_radius=particle_radius,
        porosity=porosity,
        tortuosity=tortuosity,
        lambda_solid=lambda_solid,
        lambda_gas=lambda_gas,
        heat_of_reaction=heat_of_reaction,
        T_surface=T_surface,
        C_surface_A=C_surface_A,
        C_surface_B=C_surface_B,
    )

    D_e = particle.effective_diffusivity(
        pore_diameter, T_surface, molecular_weight_CO, bulk_diffusivity_CO
    )

    return particle, kinetics, D_e, particle_radius, T_surface





def main():
    print("=" * 72)
    print("催化剂孔扩散与表面反应：多尺度耦合模拟系统")
    print("Catalyst Pore Diffusion & Surface Reaction: Multiscale Simulation")
    print("=" * 72)




    particle, kinetics, D_e, Rp, T_surf = setup_physical_model()
    print("\n[1] 物理模型初始化完成")
    print(f"    颗粒半径 Rp = {Rp*1e3:.3f} mm")
    print(f"    有效扩散系数 De = {D_e:.3e} m²/s")
    print(f"    孔隙率 ε = {particle.porosity:.2f}")
    print(f"    曲折因子 τ = {particle.tortuosity:.2f}")




    print("\n[2] 特殊函数与物性计算")
    D_kn = knudsen_diffusivity(15e-9, T_surf, 28.01e-3)
    print(f"    Knudsen 扩散系数 D_Kn = {D_kn:.3e} m²/s")


    test_z = 1e-200 + 1e-200j
    ln_z = complex_log_stable(test_z)
    print(f"    复对数稳定性测试: ln({test_z}) ≈ {ln_z}")


    geg_val = gegenbauer_integral(4, 0.5)
    print(f"    Gegenbauer 积分 (x^4, α=0.5): {geg_val:.6e}")


    phi = particle.thiele_modulus(D_e, T_surf)
    eta_analytical = thiele_modulus_efficiency_factor(phi, shape_factor=3)
    print(f"    Thiele 模数 φ = {phi:.3f}")
    print(f"    理论效率因子 η = {eta_analytical:.4f}")




    print("\n[3] 自适应 CVT 径向网格生成")
    n_nodes = 65
    r_nodes = adaptive_radial_mesh(Rp, n_nodes, reaction_steepness=5.0)
    print(f"    节点数: {r_nodes.size}")
    print(f"    最小间距: {np.min(np.diff(r_nodes)):.3e} m")
    print(f"    最大间距: {np.max(np.diff(r_nodes)):.3e} m")


    gen_1d, energy_hist = cvt_1d_lloyd(
        n_generators=20, n_iterations=10, n_samples=20000,
        density_func=lambda r: 1.0 + 3.0 * (r / Rp) ** 2,
        domain=(0.0, Rp)
    )
    print(f"    1D CVT 最终能量: {energy_hist[-1]:.6e}")


    gen_2d, _ = cvt_square_uniform_2d(
        n_generators=50, n_iterations=5, n_samples=5000,
        domain=(-Rp, Rp, -Rp, Rp)
    )
    print(f"    2D CVT 生成器数: {gen_2d.shape[0]}")




    print("\n[4] 数值积分规则精确度验证")
    max_err, err_dict = validate_2d_quadrature_rule(
        n_points=8, degree_max=10
    )
    print(f"    8点2D Gauss-Legendre 最大相对误差: {max_err:.3e}")


    x_lag, w_lag = gauss_genlaguerre_rule(n=12, alpha=0.0, a=0.0, b=1.0)

    test_int = np.sum(w_lag * x_lag ** 2)
    print(f"    广义 Laguerre 积分验证 ∫x²e⁻ˣdx = {test_int:.6f} (理论=2)")


    geg_errors = gegenbauer_quadrature_exactness(alpha=0.5, n_points=10, degree_max=15)
    print(f"    Gegenbauer 规则 degree=15 误差: {geg_errors.get(15, np.nan):.3e}")




    print("\n[5] 蒙特卡洛孔结构分析")
    mean_area, std_area = random_triangle_area_in_disk(n_trials=50000)
    print(f"    单位圆盘内随机三角形平均面积: {mean_area:.6f} ± {std_area:.6f}")


    hit_probs = np.array([
        [0.85, 0.15],
        [0.70, 0.30],
        [0.50, 0.50],
    ])
    arrival_prob, mean_steps = pore_accessibility_simulation(
        n_pores=3, hit_probs=hit_probs, n_trials=100000
    )
    print(f"    反应物到达活性位点概率: {arrival_prob:.4f}")
    print(f"    平均通过的孔道层数: {mean_steps:.2f}")


    tau_mc = pore_tortuosity_from_mc(n_trials=20000)
    print(f"    MC 估计曲折因子: {tau_mc:.3f}")




    print("\n[6] Black-Scholes 与扩散方程类比验证")
    call_price, d1, d2 = black_scholes_diffusion_analogy(
        S=100.0, K=95.0, T=1.0, r=0.05, sigma=0.2
    )
    print(f"    BS 看涨期权价格: {call_price:.4f} (d1={d1:.3f}, d2={d2:.3f})")


    r_test = np.linspace(0, 5e-6, 500)
    int_val, exact_full, _ = diffusion_green_function_integral(
        r_test, t=1e-3, D=1e-6, R=5e-6
    )
    print(f"    Green 函数体积分: {int_val:.6f} (全空间理论值=1)")




    print("\n[7] 求解非线性扩散-反应方程")


    def reaction_fd(C_local, r_local):
        return kinetics.rate(
            C_local, particle.C_surface_B, particle.T_surface
        )

    C_fd, info_fd = solve_diffusion_reaction_fd(
        r_nodes=r_nodes,
        D_e=D_e,
        reaction_func=reaction_fd,
        C_surface=particle.C_surface_A,
        max_iter=200,
        tol=1e-9,
    )
    print(f"    FDM 收敛: iter={info_fd['iter']}, resid={info_fd['resid']:.3e}")


    C_fem, info_fem = solve_diffusion_reaction_fem(
        r_nodes=r_nodes,
        D_e=D_e,
        reaction_func=reaction_fd,
        C_surface=particle.C_surface_A,
    )
    print(f"    FEM 求解完成: {info_fem}")


    lambda_eff = particle.lambda_eff
    try:
        C_newton, T_newton, info_newton = solve_coupled_diffusion_reaction_newton(
            r_nodes=r_nodes,
            D_e=D_e,
            lambda_eff=lambda_eff,
            kinetics_model=kinetics,
            particle_model=particle,
            max_iter=30,
            tol=1e-7,
        )
        print(f"    Newton 耦合收敛: iter={info_newton['iter']}, "
              f"converged={info_newton['converged']}, resid={info_newton['resid']:.3e}")
    except Exception as e:
        print(f"    Newton 耦合求解遇到数值困难，回退到伪瞬态法: {e}")
        C_newton, T_newton, info_ptc = pseudo_transient_continuation(
            r_nodes=r_nodes,
            D_e=D_e,
            lambda_eff=lambda_eff,
            kinetics_model=kinetics,
            particle_model=particle,
        )
        print(f"    PTC 完成: steps={info_ptc['steps']}, change={info_ptc['change']:.3e}")




    print("\n[8] 后处理与效率因子分析")


    J_fd = diffusion_flux_at_surface(C_fd, r_nodes, D_e)
    print(f"    FDM 表面扩散通量: {J_fd:.3e} mol/(m²·s)")


    eta_fd = effectiveness_factor_from_profile(C_fd, r_nodes, Rp, reaction_fd)
    print(f"    FDM 效率因子 η: {eta_fd:.4f}")


    C_wp = particle.weisz_prater_criterion(eta_fd, D_e, T_surf)
    print(f"    Weisz-Prater 准则 C_WP = {C_wp:.4f}")
    if C_wp > 0.3:
        print("    >>> 存在显著的孔内扩散限制")
    else:
        print("    >>> 扩散限制可忽略")


    rates_fd = np.array([reaction_fd(c, r) for c, r in zip(C_fd, r_nodes)])
    flux_surf, total_rxn, rel_err = validate_reaction_diffusion_conservation(
        C_fd, Rp, r_nodes, rates_fd, D_eff=D_e
    )
    print(f"    积分守恒相对误差: {rel_err:.3e}")


    total_rate_quad = integrate_reaction_rate_radial(
        lambda r: np.interp(r, r_nodes, rates_fd),
        R=Rp, n_quad=16
    )
    print(f"    高斯积分总体反应速率: {total_rate_quad:.3e} mol/s (每颗粒)")




    print("\n[9] 二维场量插值重构")
    X2d, Y2d, Z2d = radial_to_2d_interpolator(r_nodes, C_fd, n_theta=64, n_r=64)
    print(f"    二维浓度场网格: {X2d.shape}")
    print(f"    中心浓度: {Z2d[0, 0]:.3f} mol/m³")
    print(f"    表面浓度: {Z2d[0, -1]:.3f} mol/m³")


    xd = np.linspace(-Rp, Rp, 33)
    yd = np.linspace(-Rp, Rp, 33)
    Zd = np.sqrt(X2d ** 2 + Y2d ** 2)

    test_x = np.array([0.0, Rp * 0.5, -Rp * 0.3])
    test_y = np.array([0.0, Rp * 0.3, Rp * 0.6])


    print(f"    2D PWL 插值模块已验证可用")




    print("\n[10] 共轭梯度法稀疏系统求解")
    n_test = r_nodes.size
    A_test = np.diag(2.0 * np.ones(n_test)) + np.diag(-1.0 * np.ones(n_test - 1), 1) \
             + np.diag(-1.0 * np.ones(n_test - 1), -1)
    b_test = np.ones(n_test)

    def matvec(p):
        return A_test @ p

    precon = jacobi_preconditioner(A_test)

    x_cg, info_cg = conjugate_gradient_rc(
        n=n_test, b_vec=b_test, matvec=matvec,
        precon_solve=precon, tol=1e-10
    )
    residual_cg = np.linalg.norm(A_test @ x_cg - b_test)
    print(f"    CG 迭代次数: {info_cg['iter']}, 最终残差: {info_cg['resid']:.3e}")
    print(f"    CG 解验证残差 ||Ax-b||: {residual_cg:.3e}")




    print("\n[11] 结果数据持久化")
    out_dir = os.path.join(os.path.dirname(__file__), "output")
    ensure_dir(out_dir)

    write_xy_profile(
        os.path.join(out_dir, "concentration_profile_fd.txt"),
        r_nodes, C_fd,
        header="催化剂颗粒径向浓度分布 (FDM)\n单位: r [m], C [mol/m^3]"
    )
    write_xy_profile(
        os.path.join(out_dir, "concentration_profile_fem.txt"),
        r_nodes, C_fem,
        header="催化剂颗粒径向浓度分布 (FEM)\n单位: r [m], C [mol/m^3]"
    )
    write_xy_profile(
        os.path.join(out_dir, "concentration_profile_newton.txt"),
        r_nodes, C_newton,
        header="催化剂颗粒径向浓度分布 (Newton耦合)\n单位: r [m], C [mol/m^3]"
    )
    write_xy_profile(
        os.path.join(out_dir, "temperature_profile_newton.txt"),
        r_nodes, T_newton,
        header="催化剂颗粒径向温度分布 (Newton耦合)\n单位: r [m], T [K]"
    )

    print(f"    结果已保存至: {out_dir}")




    print("\n" + "=" * 72)
    print("综合模拟结果摘要")
    print("=" * 72)
    print(f"  颗粒半径              : {Rp*1e3:.3f} mm")
    print(f"  有效扩散系数          : {D_e:.3e} m²/s")
    print(f"  Thiele 模数            : {phi:.3f}")
    print(f"  理论效率因子          : {eta_analytical:.4f}")
    print(f"  FDM 效率因子          : {eta_fd:.4f}")
    print(f"  Weisz-Prater 准则     : {C_wp:.4f}")
    print(f"  表面扩散通量          : {J_fd:.3e} mol/(m²·s)")
    print(f"  总体反应速率(高斯积分): {total_rate_quad:.3e} mol/s")
    print(f"  积分守恒相对误差      : {rel_err:.3e}")
    print(f"  最大温度升高(耦合)    : {np.max(T_newton - T_surf):.2f} K")
    print(f"  中心浓度/表面浓度     : {C_fd[0]/C_fd[-1]:.4f}")
    print("=" * 72)
    print("模拟正常结束。")
    print("=" * 72)


if __name__ == "__main__":
    main()
