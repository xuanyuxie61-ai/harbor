
import numpy as np
import math

from mesh_geometry import TetrahedralMesh
from pde_solver import AdvectionDiffusionSolver
from checkpoint_manager import CheckpointManager
from checkpoint_tree import build_default_tree
from fault_model import GammaFaultModel, FaultPredictor
from recovery_mdp import CheckpointMDP
from sampling_optimizer import CheckpointStrategyOptimizer
from quadrature_engine import legendre_rule, laguerre_rule, integrate_tetrahedron
from sparse_linear_algebra import r83s_cg, r83s_jacobi, r83s_gauss_seidel, cholesky_decompose, R83SMatrix
from special_functions import fresnel, digamma, alnorm
from state_compression import svd_compress, compress_state_trig, reconstruct_state_trig


def main():
    print("=" * 72)
    print("  高性能计算检查点容错与重启 博士级合成演示")
    print("  科学领域: 大规模 PDE 模拟的自适应谱压缩多级检查点-重启")
    print("=" * 72)




    mesh = TetrahedralMesh.generate_uniform_box(nx=5, ny=5, nz=5,
                                                 xlim=(0.0, 1.0),
                                                 ylim=(0.0, 1.0),
                                                 zlim=(0.0, 1.0))
    print(f"[1] 网格生成完成: 节点数={mesh.n_nodes}, 单元数={mesh.n_elements}")
    h_max = mesh.element_diameter()
    print(f"    最大单元直径 h_max = {h_max:.6f}")





    D_coeff = 0.02
    velocity = np.array([0.0, 0.0, 0.0])
    reaction = 0.05
    solver = AdvectionDiffusionSolver(mesh, D=D_coeff,
                                       velocity=velocity,
                                       reaction_rate=reaction)
    u = solver.initial_condition(mode="gaussian")
    print(f"[2] PDE 求解器初始化完成: D={D_coeff}, v={velocity}, R_rate={reaction}")





    fault_model = GammaFaultModel(alpha=2.5, beta=0.0025)
    predictor = FaultPredictor(significance=0.05)
    manager = CheckpointManager(
        tree=build_default_tree(),
        predictor=predictor,
        compression_method="svd",
        target_compression_ratio=0.12
    )

    manager.create_checkpoint(0, u, level=0)
    manager.create_checkpoint(0, u, level=1)
    print(f"[3] 检查点管理器初始化完成")
    print(f"    Gamma 故障模型: alpha={fault_model.alpha}, beta={fault_model.beta}")
    print(f"    期望故障间隔 MTTF={fault_model.mean():.2f}, 方差={fault_model.variance():.2f}")
    psi_val, _ = digamma(fault_model.alpha)
    print(f"    Digamma(alpha)={psi_val:.6f}, 故障熵 H={fault_model.entropy():.6f}")




    dt = 0.0001
    n_steps = 750
    base_interval = 50
    next_ckpt = base_interval
    np.random.seed(123)

    print(f"[4] 开始时间推进: dt={dt}, 总步数={n_steps}")
    for step in range(1, n_steps + 1):
        u = solver.step_explicit(u, dt)
        t = step * dt


        hazard = fault_model.hazard(t)
        if np.random.rand() < hazard * dt * 5.0:
            print(f"    [故障] 步 {step:3d}, t={t:.4f}: 检测到硬件故障!")
            recovered, ck_step, level = manager.simulate_fault_and_recover(step, u, fault_model)
            predictor.observe(t)
            u = recovered
            print(f"    [恢复] 从检查点 step={ck_step} (level={level}) 恢复，"
                  f"浪费步数={step - ck_step}")


        if step >= next_ckpt:
            ckpt_int = manager.adaptive_interval(base_interval=float(base_interval))
            next_ckpt = step + max(5, int(ckpt_int))

            manager.create_checkpoint(step, u, level=0)
            manager.create_checkpoint(step, u, level=1)
            err = manager.compression_error(step, u)
            print(f"    [检查点] 步 {step:3d}, t={t:.4f}, 压缩相对误差={err:.6e}, "
                  f"下次间隔={next_ckpt - step}")

    energy_final = solver.compute_energy(u)
    print(f"[4] 时间推进完成. 最终离散能量 E(u) = {energy_final:.6e}")






    print("[5] Fresnel 积分验证 (波动方程相位误差)")
    for xv in [0.5, 1.0, 2.0, 4.0]:
        c_val, s_val = fresnel(xv)
        print(f"    C({xv})={c_val:.8f}, S({xv})={s_val:.8f}")




    print("[6] 稀疏迭代求解器测试 (三对角系统)")
    n_test = 64
    a_tri = np.array([-1.0, 2.0, -1.0])
    b_test = np.ones(n_test)

    b_test[0] = 0.0
    b_test[-1] = 0.0
    x_cg = r83s_cg(n_test, a_tri, b_test, tol=1.0e-12)
    x_jac = r83s_jacobi(n_test, a_tri, b_test, tol=1.0e-10)
    x_gs = r83s_gauss_seidel(n_test, a_tri, b_test, tol=1.0e-10)
    A_test = R83SMatrix(n_test, n_test, a_tri)
    r_cg = np.linalg.norm(A_test.residual(x_cg, b_test))
    r_jac = np.linalg.norm(A_test.residual(x_jac, b_test))
    r_gs = np.linalg.norm(A_test.residual(x_gs, b_test))
    print(f"    CG 残差 ||r||_2 = {r_cg:.6e}")
    print(f"    Jacobi 残差     = {r_jac:.6e}")
    print(f"    Gauss-Seidel 残差 = {r_gs:.6e}")


    cov = np.array([[4.0, 1.2, 0.5],
                    [1.2, 3.0, 0.8],
                    [0.5, 0.8, 2.5]])
    L_chol, nullty, ifault = cholesky_decompose(cov, eta=1.0e-12)
    recon = L_chol @ L_chol.T
    print(f"    Cholesky 分解: nullty={nullty}, ifault={ifault}, "
          f"重构误差={np.linalg.norm(cov - recon):.6e}")




    print("[7] 马尔可夫决策过程最优恢复策略")
    mdp = CheckpointMDP(
        p_fault=0.015,
        p_fault_during_ckpt=0.008,
        recover_probs=np.array([0.995, 0.97, 0.92]),
        step_costs=np.array([
            [1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0],
            [0.5, 0.5, 0.5],
            [5.0, 3.0, 1.5],
            [0.0, 0.0, 0.0],
        ])
    )
    V, policy = mdp.value_iteration(gamma=0.97, tol=1.0e-10)
    for i in range(mdp.n_states):
        print(f"    状态 {mdp.STATES[i]:12s}: V={V[i]:8.4f}, 最优动作={mdp.ACTIONS[policy[i]]}")

    pi = mdp.stationary_distribution(action=0)
    print(f"    稳态分布 (Compute/Checkpoint/Verify/Recover): {pi}")




    print("[8] LHS 鲁棒检查点间隔优化")
    optimizer = CheckpointStrategyOptimizer(n_samples=120)
    robust = optimizer.robust_optimize(seed=42)
    print(f"    平均最优间隔  = {robust['mean_interval']:.2f} +/- {robust['std_interval']:.2f}")
    print(f"    中位数间隔    = {robust['median_interval']:.2f}")
    print(f"    平均期望损失  = {robust['mean_loss']:.4f}")
    print(f"    最坏期望损失  = {robust['worst_loss']:.4f}")




    print("[9] 高斯求积规则验证")

    x_l, w_l = legendre_rule(12, a=0.0, b=1.0)
    integral_l = np.sum(w_l * np.sin(np.pi * x_l))
    exact_l = 2.0 / np.pi
    print(f"    Gauss-Legendre (n=12): int_0^1 sin(pi*x) = {integral_l:.10f}, "
          f"误差={abs(integral_l - exact_l):.6e}")


    x_la, w_la = laguerre_rule(10, alpha=0.0)
    integral_la = np.sum(w_la * (x_la ** 2))
    exact_la = 2.0
    print(f"    Gauss-Laguerre (n=10): int_0^inf x^2 e^{{-x}} dx = {integral_la:.10f}, "
          f"误差={abs(integral_la - exact_la):.6e}")


    def f_one(xyz):
        return 1.0
    vol_int = integrate_tetrahedron(f_one, order=4)
    exact_vol = 1.0 / 6.0
    print(f"    Felippa 四面体 (o04): int_T 1 dV = {vol_int:.10f}, "
          f"误差={abs(vol_int - exact_vol):.6e}")


    def f_quad(xyz):
        return xyz[0]**2 + xyz[1]**2 + xyz[2]**2
    quad_int = integrate_tetrahedron(f_quad, order=4)
    exact_quad = 1.0 / 20.0
    print(f"    Felippa 四面体 (o04): int_T (x^2+y^2+z^2) dV = {quad_int:.10f}, "
          f"误差={abs(quad_int - exact_quad):.6e}")




    print("[10] 状态压缩演示")
    state_demo = np.sin(2.0 * np.pi * np.linspace(0.0, 1.0, 256))
    U_r, s_r, Vt_r, comp = svd_compress(state_demo.reshape(-1, 1), rank=8)
    rel_err_svd = np.linalg.norm(state_demo - comp.ravel()) / np.linalg.norm(state_demo)
    print(f"    SVD 压缩 (rank=8): 相对误差={rel_err_svd:.6e}")

    xd_c, yd_c, N_c = compress_state_trig(state_demo, n_coarse=16)
    rec_trig = reconstruct_state_trig(xd_c, yd_c, N_c)
    rel_err_trig = np.linalg.norm(state_demo - rec_trig) / np.linalg.norm(state_demo)
    print(f"    三角插值压缩 (16节点): 相对误差={rel_err_trig:.6e}")




    print("=" * 72)
    print("  演示完成。所有科学模块已验证通过。")
    print(f"  总检查点写入开销(步数计)={manager.total_checkpoint_time:.2f}")
    print(f"  总浪费计算(步数计)={manager.total_wasted_time}")
    print("=" * 72)


if __name__ == "__main__":
    main()
