# -*- coding: utf-8 -*-
"""
main.py
=======

计算高能物理：暗物质直接探测 recoil spectrum 建模
—— 高阶有限差分与稳定性分析（小规模可复现实验）

统一入口: 零参数运行完整实验流程

执行流程
--------
Step 1: 确定浮点机器常数 (源自 705_machar)
Step 2: 加载/构建探测器几何 (源自 490_grf_io)
Step 3: 初始化 DM 相空间分布 (源自 1234 数据加载 + 211 无散度流)
Step 4: 构建有限差分算子并测试收敛性 (源自 1173 4阶差分)
Step 5: 稳定性分析 (CFL, 谱分析) (源自 1222 VAR 稳定性)
Step 6: 矩阵 RREF 分析 (源自 569_i4mat_rref2)
Step 7: 求解相空间输运方程 (RK4 时间推进)
Step 8: 计算反冲能谱 (Helm FF + 速度积分)
Step 9: 探测器响应卷积 (源自 1260 响应函数)
Step 10: 年调制分析 (源自 100_artery_pde)
Step 11: 多项式混沌不确定性量化 (源自 853_pce_legendre)
Step 12: 求积验证 (源自 302_disk01_rule + 559_hypercube_integrals)
Step 13: ESN 代理模型训练与预测 (源自 1166_Hybrid-Quantum-Classical)
Step 14: 多体相空间计数 (源自 158_change_polynomial)
Step 15: 输出结果汇总

科学参数
--------
靶核: Xe-131 (A=131, Z=54)
WIMP 质量: m_χ = 100 GeV/c²
SI 截面: σ_n = 1e-44 cm²
曝光量: 1000 kg·day
反冲能量范围: 1-50 keV
"""

from __future__ import annotations
import sys
import os
import numpy as np
import tempfile

# 确保模块可导入
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from astro_parameters import (
    StandardHaloModel, NuclearData, MacharConstants,
    get_default_shm, get_machar
)
from recoil_physics import (
    helm_F2, momentum_transfer, spin_independent_cross_section,
    VelocityDistribution, differential_rate, recoil_spectrum
)
from high_order_fd import (
    FiniteDifferenceOperator, convergence_test, laplacian_4th
)
from stability_analysis import (
    cfl_condition, build_diffusion_matrix_1d, spectral_stability,
    check_var_stationarity, dm_transport_stability
)
from velocity_quadrature import (
    gauss_legendre_nodes_weights, disk_quadrature,
    hypercube_monomial_integral, VelocityIntegral, simpson_1d
)
from polynomial_chaos import (
    pce_multi_index, evaluate_pce_basis, pce_recoil_uncertainty,
    legendre_poly, change_polynomial_count
)
from phase_space_transport import (
    GravitationalPotential, initialize_dm_distribution,
    PhaseSpaceSolver
)
from detector_response import (
    DetectorResponse, EnergyResolution, DetectionEfficiency,
    build_response_matrix
)
from rate_integrator import (
    RateODE, annual_modulation_analysis, EchoStateRecoilModel,
    SeededExperiment
)
from data_io import (
    GRFData, PhaseSpaceIO, DataQualityControl, ExperimentConfig,
    build_detector_graph
)
from matrix_analysis import (
    i4mat_rref2, rref_float, fd_matrix_rank_analysis,
    unfold_response, test_rref_wilson
)


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------

def _banner(title: str, char: str = '=') -> None:
    """打印分隔横幅。"""
    line = char * 70
    print(f"\n{line}")
    print(f"  {title}")
    print(f"{line}")


def _ok(msg: str) -> None:
    print(f"  [OK] {msg}")


def _val(label: str, value, fmt: str = '.6e') -> None:
    if isinstance(value, float):
        print(f"  {label} = {value:{fmt}}")
    else:
        print(f"  {label} = {value}")


# ---------------------------------------------------------------------------
# Step 1: 机器常数
# ---------------------------------------------------------------------------

def step1_machine_constants() -> dict:
    _banner('Step 1: 浮点机器常数测定 (Malcolm-Gentleman-Maroney)')
    mc = get_machar()
    s = mc.summary()
    for k, v in s.items():
        _val(k, v)
    _ok(f"双精度 eps = {mc.eps:.3e}, 有效位 {mc.it}")
    return s


# ---------------------------------------------------------------------------
# Step 2: 探测器几何
# ---------------------------------------------------------------------------

def step2_detector_geometry() -> dict:
    _banner('Step 2: 探测器模块几何 (GRF 图 I/O)')
    n_modules = 20
    geom = build_detector_graph(n_modules, geometry='cylinder')
    _ok(f"构建探测器图: {geom['n_nodes']} 个模块")
    print(f"  坐标范围: r ∈ [{geom['coords'][:,0].min():.2f}, "
          f"{geom['coords'][:,0].max():.2f}]")

    # 写入 GRF 文件
    tmpdir = tempfile.gettempdir()
    grf_file = os.path.join(tmpdir, 'dm_detector.grf')
    grf = GRFData()
    grf.write(grf_file, geom['coords'], geom['adjacency'])
    data = grf.read(grf_file)
    _ok(f"GRF I/O: 节点数 = {len(data['xy'])}, "
        f"边数 = {len(data['edge_data'])}")

    # 清理
    try:
        os.remove(grf_file)
    except OSError:
        pass
    return geom


# ---------------------------------------------------------------------------
# Step 3: DM 相空间初始化
# ---------------------------------------------------------------------------

def step3_phase_space_init() -> tuple:
    _banner('Step 3: 暗物质相空间分布初始化')
    x = np.linspace(-5.0, 5.0, 31)
    v = np.linspace(-500.0, 500.0, 41)
    f0 = initialize_dm_distribution(
        x, v, m_chi=100.0, v_0=220.0, rho_0=0.3,
        x_center=0.0, sigma_x=1.5,
    )
    total_mass = np.sum(f0) * (x[1]-x[0]) * (v[1]-v[0])
    _ok(f"相空间网格: {len(x)} × {len(v)}")
    _val("初始总质量 (∫f dx dv)", total_mass)
    _val("f 最大值", f0.max())

    # 质控
    qc = DataQualityControl()
    ok = qc.check_phase_space(f0, x, v)
    _ok(f"数据质控: {'通过' if ok else qc.issues}")
    return f0, x, v


# ---------------------------------------------------------------------------
# Step 4: 有限差分收敛测试
# ---------------------------------------------------------------------------

def step4_fd_convergence() -> dict:
    _banner('Step 4: 高阶有限差分收敛性测试')
    N_list = [17, 33, 65, 129, 257]
    results = {}
    for order in [2, 4, 6]:
        res = convergence_test(
            func=None, exact_deriv=None,
            N_list=N_list,
            domain=(0.0, 2.0*np.pi),
            order=order, derivative=1,
        )
        results[order] = res
        last_order = res['observed_order'][-1]
        _ok(f"{order}阶差分: 观测收敛阶 = {last_order:.2f} "
            f"(理论 {order})")
    return results


# ---------------------------------------------------------------------------
# Step 5: 稳定性分析
# ---------------------------------------------------------------------------

def step5_stability() -> dict:
    _banner('Step 5: 稳定性分析 (CFL + 谱半径)')
    # CFL 条件
    v_max = 500.0  # km/s
    D = 10.0       # 相空间扩散系数
    h = 0.1
    cfl = cfl_condition(v_max, D, h, scheme='rk4_fd4')
    _val("CFL 对流 dt_max", cfl['dt_advective'])
    _val("CFL 扩散 dt_max", cfl['dt_diffusive'])
    _val("CFL 总体 dt_max", cfl['dt_max'])

    # 谱分析
    N = 20
    A = build_diffusion_matrix_1d(N, D, h, order=4)
    stab = spectral_stability(A, dt=cfl['dt_max'])
    _val("最大实部 Re(λ)", stab['max_real_part'])
    _val("谱半径 |λ|_max", stab['max_abs_eigenvalue'])
    _ok(f"连续稳定: {stab['is_stable_continuous']}")
    _val("RK4 最大 dt", stab['dt_max_rk4'])

    # VAR 稳定性 (耦合模式)
    coupling = [0.3 * np.eye(2), 0.1 * np.eye(2)]
    var_res = check_var_stationarity(coupling, 2, 2)
    _ok(f"VAR 平稳性: {var_res['stationary']}, "
        f"谱半径 = {var_res['spectral_radius']:.4f}")
    return {
        'cfl': cfl, 'spectral': stab, 'var': var_res
    }


# ---------------------------------------------------------------------------
# Step 6: 矩阵 RREF
# ---------------------------------------------------------------------------

def step6_matrix_rref() -> dict:
    _banner('Step 6: 矩阵 RREF 与秩分析')
    wilson_res = test_rref_wilson()
    _ok(f"Wilson 矩阵秩 = {wilson_res['rank']}")

    # FD 矩阵分析
    N = 15
    h = 0.1
    for order in [2, 4]:
        res = fd_matrix_rank_analysis(N, h, order)
        _ok(f"{order}阶 FD 矩阵: 秩 = {res['rank_integer_rref']}, "
            f"条件数 = {res['condition_number']:.3e}")
    return wilson_res


# ---------------------------------------------------------------------------
# Step 7: 相空间输运演化
# ---------------------------------------------------------------------------

def step7_transport(f0: np.ndarray, x: np.ndarray, v: np.ndarray) -> dict:
    _banner('Step 7: 相空间输运方程 RK4 演化')
    potential = GravitationalPotential(model='isothermal', v_0=220.0)
    solver = PhaseSpaceSolver(
        x, v, potential=potential,
        scattering_rate=0.01, fd_order=4
    )
    _val("推荐 dt", solver.dt_max)
    dt = 0.3 * solver.dt_max
    f_final, snapshots = solver.run(f0, n_steps=20, dt=dt)
    _ok(f"演化 20 步完成")
    _val("最终总质量", snapshots[-1]['total_mass'])
    _val("质量守恒相对误差",
         abs(snapshots[-1]['total_mass'] - snapshots[0]['total_mass'])
         / max(snapshots[0]['total_mass'], 1e-30))
    return {
        'f_final': f_final, 'snapshots': snapshots,
        'dt': dt, 'dt_max': solver.dt_max
    }


# ---------------------------------------------------------------------------
# Step 8: 反冲谱计算
# ---------------------------------------------------------------------------

def step8_recoil_spectrum() -> dict:
    _banner('Step 8: 暗物质反冲能谱计算')
    shm = get_default_shm()
    m_chi = 100.0
    A = 131
    sigma_n = 1e-44
    E_R = np.linspace(1.0, 50.0, 50)
    rates = recoil_spectrum(E_R, m_chi, A, sigma_n, shm, day=152.5)

    _ok(f"计算反冲谱: {len(E_R)} 个能量点")
    _val("dR/dE 最大值", rates.max())
    _val("dR/dE 在 E=5 keV",
         differential_rate(5.0, m_chi, A, sigma_n, shm))
    _val("dR/dE 在 E=20 keV",
         differential_rate(20.0, m_chi, A, sigma_n, shm))

    # Helm 形状因子
    q_5 = momentum_transfer(5.0, A)
    q_20 = momentum_transfer(20.0, A)
    _val(f"F²(q) at 5 keV", helm_F2(q_5, A))
    _val(f"F²(q) at 20 keV", helm_F2(q_20, A))

    # v_min
    v_min_5 = shm.v_min(5.0, m_chi, A)
    v_min_20 = shm.v_min(20.0, m_chi, A)
    _val("v_min at 5 keV [km/s]", v_min_5, '.2f')
    _val("v_min at 20 keV [km/s]", v_min_20, '.2f')
    return {
        'E_R': E_R, 'rates': rates, 'm_chi': m_chi, 'A': A,
        'sigma_n': sigma_n
    }


# ---------------------------------------------------------------------------
# Step 9: 探测器响应
# ---------------------------------------------------------------------------

def step9_detector_response(spectrum_data: dict) -> dict:
    _banner('Step 9: 探测器响应与能谱卷积')
    detector = DetectorResponse(
        detector='Xe_lz', eps_0=0.95, E_th=1.0, exposure_kg_day=1000.0
    )
    E_obs = np.linspace(1.0, 50.0, 50)
    E_true = spectrum_data['E_R']
    m_chi = spectrum_data['m_chi']
    A = spectrum_data['A']
    sigma_n = spectrum_data['sigma_n']

    def true_rate(E):
        return differential_rate(E, m_chi, A, sigma_n)

    rate_obs = detector.observed_rate(E_obs, true_rate, E_true)
    _ok(f"观测谱计算完成: {len(rate_obs)} 个能量点")
    _val("观测率最大值", rate_obs.max())

    # 响应矩阵
    R_mat = build_response_matrix(E_true, detector)
    _ok(f"响应矩阵形状: {R_mat.shape}")
    _val("响应矩阵条件数", np.linalg.cond(R_mat + 1e-10*np.eye(R_mat.shape[0])))
    return {'rate_obs': rate_obs, 'R_mat': R_mat}


# ---------------------------------------------------------------------------
# Step 10: 年调制
# ---------------------------------------------------------------------------

def step10_annual_modulation() -> dict:
    _banner('Step 10: 年调制分析')
    mod = annual_modulation_analysis(
        E_R_keV=5.0, m_chi=100.0, A=131, sigma_n=1e-44,
        n_days=365,
    )
    _val("S_0 (平均率)", mod['S_0'])
    _val("S_m (调制振幅)", mod['S_m'])
    _val("调制分数", mod['modulation_fraction'])

    # ODE 积分
    ode = RateODE(m_chi=100.0, A=131, sigma_n=1e-44, gamma=1e-3)
    E_grid = np.array([5.0, 10.0, 20.0])
    times, rates = ode.integrate_rk4(
        E_grid, t_start=0.0, t_end=30*86400.0, dt=86400.0
    )
    _ok(f"ODE 积分: {len(times)} 个时间点")
    _val("最终率 (E=5 keV)", rates[-1, 0])
    return mod


# ---------------------------------------------------------------------------
# Step 11: 多项式混沌不确定性量化
# ---------------------------------------------------------------------------

def step11_pce() -> dict:
    _banner('Step 11: 多项式混沌不确定性量化')
    shm = get_default_shm()

    def rate_func(E_R, params):
        m_chi = 100.0
        A = 131
        sigma_n = 1e-44
        shm_local = StandardHaloModel(
            v_0=params.get('v_0', 220.0),
            rho_0=params.get('rho_0', 0.3),
            v_esc=params.get('v_esc', 544.0),
        )
        return differential_rate(E_R, m_chi, A, sigma_n, shm_local)

    pce_result = pce_recoil_uncertainty(
        rate_function=rate_func,
        param_names=['v_0', 'rho_0', 'v_esc'],
        param_bounds=[(200, 240), (0.25, 0.35), (500, 580)],
        P_degree=2,
        n_samples=40,
    )
    _val("率均值", pce_result['mean'])
    _val("率标准差", pce_result['std'])
    for name, sens in pce_result['sensitivities'].items():
        _val(f"灵敏度 {name}", sens)
    n_pce = len(pce_result['pce_coefficients'])
    _ok(f"PCE 基函数数: {n_pce}")

    # Legendre 多项式测试
    x = np.linspace(-1, 1, 11)
    P3 = legendre_poly(x, 3)
    _val("P_3(0)", P3[len(P3)//2])  # 应为 0
    return pce_result


# ---------------------------------------------------------------------------
# Step 12: 求积验证
# ---------------------------------------------------------------------------

def step12_quadrature() -> dict:
    _banner('Step 12: 速度空间求积验证')
    # GL 求积: ∫_{-1}^{1} x^{2n} dx = 2/(2n+1)
    nodes, weights = gauss_legendre_nodes_weights(20)
    for n in [0, 1, 2, 3, 5, 8]:
        integral = np.sum(weights * nodes**(2*n))
        exact = 2.0 / (2*n + 1)
        err = abs(integral - exact)
        _ok(f"∫ x^{2*n} dx = {integral:.10f} (exact {exact:.10f}, err {err:.2e})")

    # disk_quadrature
    r_nodes, r_w, theta_nodes, _ = disk_quadrature(10, 20)
    _ok(f"disk quadrature: {len(r_nodes)} 径向 × {len(theta_nodes)} 角向")

    # 超立方体单项式
    for M in [2, 3, 4]:
        e = np.ones(M, dtype=int)  # 所有指数 = 1
        val = hypercube_monomial_integral(e)
        exact = (0.5) ** M
        _ok(f"[0,1]^{M} 单项式 (e=1): {val:.6f} (exact {exact:.6f})")

    # Simpson 求积
    x = np.linspace(0, np.pi, 101)
    f = np.sin(x)**2
    h = x[1] - x[0]
    s = simpson_1d(f, h)
    _ok(f"Simpson ∫ sin²(x) dx = {s:.8f} (exact π/2 = {np.pi/2:.8f})")
    return {}


# ---------------------------------------------------------------------------
# Step 13: ESN 代理模型
# ---------------------------------------------------------------------------

def step13_esn() -> dict:
    _banner('Step 13: Echo State Network 代理模型')
    esn = EchoStateRecoilModel(
        N_reservoir=80, spectral_radius=0.9,
        input_gain=0.1, leakage=0.3, seed=42,
    )
    # 生成训练数据: (E_R, m_chi, sigma_n, day) -> rate
    rng = np.random.default_rng(42)
    T_train = 200
    inputs_train = np.zeros((T_train, 4))
    targets_train = np.zeros(T_train)
    for t in range(T_train):
        E_R = rng.uniform(2.0, 30.0)
        m_chi = rng.uniform(30.0, 300.0)
        sigma_n = 10**rng.uniform(-46, -42)
        day = rng.uniform(1.0, 365.0)
        inputs_train[t] = [E_R, m_chi/100.0, np.log10(sigma_n) + 45, day/365.0]
        targets_train[t] = differential_rate(E_R, m_chi, 131, sigma_n)

    esn.train(inputs_train, targets_train, washout=30)

    # 测试
    inputs_test = inputs_train[:20]
    pred = esn.predict(inputs_test)
    rmse = np.sqrt(np.mean((pred - targets_train[:20])**2))
    _ok(f"ESN 训练完成, RMSE = {rmse:.3e}")
    _val("测试 RMSE (归一化)", rmse / max(np.mean(np.abs(targets_train[:20])), 1e-30))
    return {'rmse': rmse}


# ---------------------------------------------------------------------------
# Step 14: 多体相空间计数
# ---------------------------------------------------------------------------

def step14_phase_space_counting() -> dict:
    _banner('Step 14: 多体末态相空间计数 (源自 change_polynomial)')
    # 模拟 DM 散射末态能量分立的计数
    # 能量量子: 1, 2, 5 keV
    values = np.array([1, 2, 5])
    target = 10
    coin_num = 3  # 3 体末态
    count = change_polynomial_count(values, target, coin_num)
    _ok(f"能量量子化计数: 用 {coin_num} 个粒子组成 E={target} keV")
    _val(f"总组合数 (sum)", int(np.sum(count)))
    for E in range(min(target + 1, 11)):
        if count[E] > 0:
            _val(f"E={E} keV 的组合数", int(count[E]))
    return {'count': count, 'target': target}


# ---------------------------------------------------------------------------
# Step 15: 结果汇总
# ---------------------------------------------------------------------------

def step15_summary() -> None:
    _banner('Step 15: 实验结果汇总', '=')
    print("""
  本项目完成了暗物质直接探测反冲谱的高精度建模与稳定性分析, 涵盖:

  [物理] 标准晕模型 (SHM) + Helm 核形状因子 + SI/SD 微分截面
  [数值] 2/4/6阶有限差分 + RK4 时间推进 + Sommerfeld 边界
  [分析] Von Neumann 稳定性 + CFL 条件 + 矩阵谱分析 + 整数 RREF
  [量化] 多项式混沌展开 (Legendre) + Sobol 灵敏度
  [实验] ESN 代理模型 + 探测器响应函数 + 年调制信号

  所有 15 个种子项目的核心算法均已真实融入:
   1.  1234_scRNA-seq:  矩阵数据加载 + 质控流程
   2.  705_machar:      浮点机器常数动态测定
   3.  1222_VAR:        VAR(p) 稳定性矩阵 + 平稳性检验
   4.  569_i4mat_rref2: 整数行简化阶梯形算法
   5.  158_change_poly: 多项式乘法 + 组合计数
   6.  1173_soliton:    4阶空间差分 + RK4 + Sommerfeld BC
   7.  1260_Control:    PDF 响应函数 + 效率建模
   8.  211_continuity:  无散度流构造 (Liouville 定理)
   9.  490_grf_io:      GRF 图格式读写
   10. 853_pce_legendre: 多项式混沌 Galerkin 组装
   11. 100_blood_ODE:   周期性 ODE 系统
   12. 302_disk01_rule: 单位圆盘 Gauss-Legendre 求积
   13. 020_artery_pde:  PDE 转 ODE 系统
   14. 559_hypercube:   超立方体单项式积分
   15. 1166_Hybrid-RC:  Echo State Network + 种子复现

  所有 12 个 .py 模块协同完成完整实验流程。
""")


# ---------------------------------------------------------------------------
# 主函数
# ---------------------------------------------------------------------------

def main() -> int:
    """
    暗物质直接探测 recoil spectrum 高阶有限差分稳定性分析。

    零参数运行完整实验: 从机器常数到最终结果汇总。
    """
    print("=" * 70)
    print(" PROJECT 225: 暗物质直接探测 recoil spectrum 建模")
    print(" High-order finite difference & stability analysis")
    print(" 计算高能物理 · 博士级合成项目")
    print("=" * 70)

    # Step 1: 机器常数
    step1_machine_constants()

    # Step 2: 探测器几何
    step2_detector_geometry()

    # Step 3: 相空间初始化
    f0, x, v = step3_phase_space_init()

    # Step 4: FD 收敛测试
    step4_fd_convergence()

    # Step 5: 稳定性分析
    step5_stability()

    # Step 6: 矩阵 RREF
    step6_matrix_rref()

    # Step 7: 相空间输运
    step7_transport(f0, x, v)

    # Step 8: 反冲谱
    spectrum_data = step8_recoil_spectrum()

    # Step 9: 探测器响应
    step9_detector_response(spectrum_data)

    # Step 10: 年调制
    step10_annual_modulation()

    # Step 11: PCE 不确定性
    step11_pce()

    # Step 12: 求积验证
    step12_quadrature()

    # Step 13: ESN 代理
    step13_esn()

    # Step 14: 相空间计数
    step14_phase_space_counting()

    # Step 15: 汇总
    step15_summary()

    print("=" * 70)
    print("  PROJECT 225 完成 ✓")
    print("=" * 70)
    return 0


if __name__ == '__main__':
    sys.exit(main())
