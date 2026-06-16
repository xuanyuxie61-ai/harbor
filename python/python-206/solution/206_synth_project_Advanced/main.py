"""
main.py  --  贝叶斯模型校准统一入口
===============================================================
科学问题:
    Gray-Scott 反应扩散系统的贝叶斯参数校准.
    给定稀疏带噪观测 y_obs, 推断反应速率 (f, k)
    和扩散系数 (Du, Dv) 的后验分布.
流程:
    1. 初始化数值基底 (机器常数)
    2. 生成合成观测数据 (sim + real)
    3. 构建先验几何 (magic-4 协方差, canalization 掩码, ...)
    4. 训练 GP 代理 (保真核 + ResNet 编码器)
    5. 运行 Jacobian-free Newton-Krylov MAP 估计
    6. 运行约束 MAP (对数屏障)
    7. 运行自适应 MCMC (流形 mixup 提议)
    8. 计算贝叶斯证据 (正磁盘求积)
    9. 序贯实验设计 (优惠券收集)
   10. Sim-to-Real 域自适应
   11. 输出诊断报告
"""
from __future__ import annotations
import math
import sys
from typing import Dict, List

# --- 子模块 ---
from numerical_base import NUMERICS
from forward_model import GrayScottForward
from synthetic_data import SyntheticDataGenerator
from prior_geometry import PriorGeometry
from surrogate_model import FidelityGPSurrogate
from map_solver import MAPSolver
from constrained_calibration import ConstrainedMAPSolver
from bayesian_calibration import BayesianCalibration
from evidence_quadrature import compute_evidence, bayes_factor, interpret_bayes_factor
from experimental_design import (
    SequentialDesign, coupon_collector_expected,
    coupon_collector_monte_carlo
)
from domain_adapter import SimToRealCalibration


# ======================================================================
# 阶段 1: 数值基底
# ======================================================================
def stage_numerical_base() -> None:
    print("=" * 60)
    print("阶段 1: 数值基底初始化")
    print("=" * 60)
    print(NUMERICS.summary())
    print()


# ======================================================================
# 阶段 2: 合成数据
# ======================================================================
def stage_synthetic_data() -> Dict[str, object]:
    print("=" * 60)
    print("阶段 2: 合成观测数据生成")
    print("=" * 60)
    theta_true = {"Du": 0.16, "Dv": 0.08, "f": 0.04, "k": 0.06}
    print(f"真实参数: {theta_true}")
    gen = SyntheticDataGenerator(
        theta_true=theta_true, nx=11, ny=11,
        obs_noise_std=0.02, model_discrepancy=0.005, seed=42
    )
    y_sim, theta_vec = gen.generate_sim_data()
    y_real, _ = gen.generate_real_data()
    print(f"sim 观测 (n={len(y_sim)}): "
          + ", ".join(f"{yi:.4f}" for yi in y_sim[:4]) + " ...")
    print(f"real 观测 (n={len(y_real)}): "
          + ", ".join(f"{yi:.4f}" for yi in y_real[:4]) + " ...")
    print()
    return {"theta_true": theta_true, "theta_vec": theta_vec,
            "y_sim": y_sim, "y_real": y_real, "gen": gen}


# ======================================================================
# 阶段 3: 先验几何
# ======================================================================
def stage_prior_geometry(dim: int = 4) -> PriorGeometry:
    print("=" * 60)
    print("阶段 3: 先验几何构建")
    print("=" * 60)
    prior = PriorGeometry(dim=dim, sigma2=0.25, alpha=0.15,
                          budget=4, n_channels=4, seed=0)
    print(f"参数维度: {dim}")
    print(f"激活参数: {prior.n_active} / {dim}")
    print(f"激活掩码: {prior.active_mask}")
    print(f"先验 log det(Sigma): {prior.log_det:.4f}")
    print(f"离散校准设计数: {len(prior.designs)}")
    print(f"对称轨道代表数: {len(prior.orbit_reps)}")
    print()
    return prior


# ======================================================================
# 阶段 4: 代理训练
# ======================================================================
def stage_surrogate_training(data: Dict) -> FidelityGPSurrogate:
    print("=" * 60)
    print("阶段 4: GP 代理训练 (保真核 + ResNet 编码)")
    print("=" * 60)
    gen: SyntheticDataGenerator = data["gen"]
    thetas_train, ys_train = gen.generate_training_set(n_samples=30)
    print(f"训练样本数: {len(thetas_train)}")
    surrogate = FidelityGPSurrogate(latent_dim=6, n_obs=gen.n_obs,
                                     sigma_n=0.05, mix_weight=0.7)
    surrogate.fit(thetas_train, ys_train)
    loss = surrogate.train_loss()
    print(f"代理训练 MSE: {loss:.6f}")
    print()
    return surrogate


# ======================================================================
# 阶段 5: MAP 估计 (JFNK)
# ======================================================================
def stage_map_solver(data: Dict, prior: PriorGeometry,
                     surrogate: FidelityGPSurrogate) -> List[float]:
    print("=" * 60)
    print("阶段 5: JFNK MAP 估计")
    print("=" * 60)
    dim = prior.dim
    n_obs = surrogate.n_obs
    obs_data = data["y_sim"]
    obs_noise_std = 0.02
    Sigma_obs_inv_diag = 1.0 / (obs_noise_std ** 2)

    def J_func(theta):
        y_pred = surrogate.predict(theta)
        misfit = 0.5 * Sigma_obs_inv_diag * sum(
            (obs_data[i] - y_pred[i]) ** 2 for i in range(n_obs))
        reg = -prior.log_pdf(theta)
        return misfit + reg

    def grad_J(theta):
        eps = max(NUMERICS.eps ** 0.5, 1e-6)
        g = [0.0] * dim
        for d in range(dim):
            theta_p = list(theta)
            theta_p[d] += eps
            g[d] = (J_func(theta_p) - J_func(theta)) / eps
        return g

    theta0 = prior.sample()
    solver = MAPSolver(dim=dim, maxit=30, gmres_restart=15)
    theta_map, hist = solver.solve(grad_J, J_func, theta0)
    print(f"MAP 估计: [{', '.join(f'{t:.4f}' for t in theta_map)}]")
    if hist:
        print(f"  最终 J={hist[-1]['J']:.4f}, "
              f"grad_norm={hist[-1]['grad_norm']:.4e}")
    print()
    return theta_map


# ======================================================================
# 阶段 6: 约束 MAP
# ======================================================================
def stage_constrained_map(data: Dict, prior: PriorGeometry,
                          surrogate: FidelityGPSurrogate) -> List[float]:
    print("=" * 60)
    print("阶段 6: 约束 MAP (对数屏障)")
    print("=" * 60)
    dim = prior.dim
    n_obs = surrogate.n_obs
    obs_data = data["y_sim"]
    Sigma_obs_inv_diag = 1.0 / (0.02 ** 2)

    def J_func(theta):
        y_pred = surrogate.predict(theta)
        misfit = 0.5 * Sigma_obs_inv_diag * sum(
            (obs_data[i] - y_pred[i]) ** 2 for i in range(n_obs))
        reg = -prior.log_pdf(theta)
        return misfit + reg

    def grad_J(theta):
        eps = max(NUMERICS.eps ** 0.5, 1e-6)
        g = [0.0] * dim
        for d in range(dim):
            theta_p = list(theta)
            theta_p[d] += eps
            g[d] = (J_func(theta_p) - J_func(theta)) / eps
        return g

    theta0 = prior.sample()
    solver = ConstrainedMAPSolver(dim=dim, max_outer=5, max_inner=20)
    theta_cmap, hist = solver.solve(J_func, grad_J, theta0)
    print(f"约束 MAP: [{', '.join(f'{t:.4f}' for t in theta_cmap)}]")
    if hist:
        print(f"  最终 J={hist[-1]['J']:.4f}, min_h={hist[-1]['min_h']:.4f}")
    print()
    return theta_cmap


# ======================================================================
# 阶段 7: MCMC 采样
# ======================================================================
def stage_mcmc(data: Dict, prior: PriorGeometry,
               surrogate: FidelityGPSurrogate) -> Dict[str, object]:
    print("=" * 60)
    print("阶段 7: 自适应 MCMC (流形 mixup 提议)")
    print("=" * 60)
    cal = BayesianCalibration(
        dim=prior.dim, n_obs=surrogate.n_obs,
        prior=prior, surrogate=surrogate,
        obs_data=data["y_sim"], obs_noise_std=0.02, seed=5
    )
    diag = cal.run_mcmc(n_samples=300, n_warmup=100, n_chains=2)
    print(f"R-hat 均值: {diag['r_hat_mean']:.4f}")
    print(f"R-hat 最大: {diag['r_hat_max']:.4f}")
    print(f"ESS 均值: {diag['ess_mean']:.1f}")
    print(f"接受率: {diag['acceptance_mean']:.3f}")
    post_mean = diag["post_mean"]
    post_std = diag["post_std"]
    labels = ["log(Du)", "log(Dv)", "f", "k"]
    for i, lab in enumerate(labels):
        print(f"  {lab}: {post_mean[i]:.4f} +/- {post_std[i]:.4f}")
    print()
    return diag


# ======================================================================
# 阶段 8: 贝叶斯证据
# ======================================================================
def stage_evidence(prior: PriorGeometry,
                   surrogate: FidelityGPSurrogate,
                   obs_data: List[float]) -> None:
    print("=" * 60)
    print("阶段 8: 贝叶斯证据 (正磁盘求积)")
    print("=" * 60)
    n_obs = len(obs_data)
    Sigma_obs_inv_diag = 1.0 / (0.02 ** 2)

    def log_post_1(theta):
        if len(theta) < 2:
            theta = theta + [0.0] * (2 - len(theta))
        theta_full = list(theta) + [0.04, 0.05]
        y_pred = surrogate.predict(theta_full[:4])
        misfit = 0.5 * Sigma_obs_inv_diag * sum(
            (obs_data[i] - y_pred[i]) ** 2 for i in range(n_obs))
        lp = prior.log_pdf(theta_full[:4])
        return -(misfit - lp)

    def log_post_2(theta):
        if len(theta) < 2:
            theta = theta + [0.0] * (2 - len(theta))
        theta_full = list(theta) + [0.08, 0.10]
        y_pred = surrogate.predict(theta_full[:4])
        misfit = 0.5 * Sigma_obs_inv_diag * sum(
            (obs_data[i] - y_pred[i]) ** 2 for i in range(n_obs))
        lp = prior.log_pdf(theta_full[:4])
        return -(misfit - lp)

    try:
        Z1 = compute_evidence(log_post_1, dim=2, n_quad=8)
        Z2 = compute_evidence(log_post_2, dim=2, n_quad=8)
        log_Z1 = NUMERICS.clamp_log(Z1)
        log_Z2 = NUMERICS.clamp_log(Z2)
        log_bf = bayes_factor(log_Z1, log_Z2)
        print(f"模型 1 证据: log Z1 = {log_Z1:.4f}")
        print(f"模型 2 证据: log Z2 = {log_Z2:.4f}")
        print(f"log Bayes 因子: {log_bf:.4f}")
        print(f"解释: {interpret_bayes_factor(log_bf)}")
    except Exception as e:
        print(f"证据计算跳过 (维度限制): {e}")
    print()


# ======================================================================
# 阶段 9: 序贯实验设计
# ======================================================================
def stage_experimental_design() -> None:
    print("=" * 60)
    print("阶段 9: 序贯实验设计 (优惠券收集)")
    print("=" * 60)
    k = 4
    N = 20
    print(f"通道数 k={k}, 总预算 N={N}")
    print(f"理论期望 E[T] = {coupon_collector_expected(k):.2f}")
    mc = coupon_collector_monte_carlo(k, n_trials=200, seed=0)
    print(f"蒙特卡洛: mean={mc['mean']:.2f}, var={mc['var']:.2f}")
    design = SequentialDesign(n_channels=k, total_budget=N,
                              threshold=3, seed=0)
    counts = design.run_design()
    summary = design.summary()
    print(f"自适应分配: {counts}")
    print(f"覆盖度: {summary['coverage']:.2f}")
    print()


# ======================================================================
# 阶段 10: Sim-to-Real 域自适应
# ======================================================================
def stage_domain_adaptation(data: Dict,
                            surrogate: FidelityGPSurrogate) -> None:
    print("=" * 60)
    print("阶段 10: Sim-to-Real 域自适应")
    print("=" * 60)
    gen: SyntheticDataGenerator = data["gen"]
    thetas, _ = gen.generate_training_set(n_samples=15)
    sim_z = [surrogate.encoder.forward(t) for t in thetas]
    real_z = [[z[d] + 0.1 for d in range(len(z))] for z in sim_z]
    s2r = SimToRealCalibration(latent_dim=6, seed=3)
    summary = s2r.run_alignment(sim_z, real_z, n_iters=20)
    print(f"Sim 样本数: {summary['n_sim']}")
    print(f"Real 样本数: {summary['n_real']}")
    print(f"MMD 对齐前: {summary['mmd_before']:.6f}")
    print(f"MMD 对齐后: {summary['mmd_after']:.6f}")
    print()


# ======================================================================
# 主流程
# ======================================================================
def main() -> None:
    print()
    print("*" * 60)
    print("*  PROJECT 206: 贝叶斯模型校准                              *")
    print("*  Gray-Scott 反应扩散系统的不确定性量化                     *")
    print("*" * 60)
    print()

    stage_numerical_base()
    data = stage_synthetic_data()
    prior = stage_prior_geometry(dim=4)
    surrogate = stage_surrogate_training(data)
    theta_map = stage_map_solver(data, prior, surrogate)
    theta_cmap = stage_constrained_map(data, prior, surrogate)
    diag = stage_mcmc(data, prior, surrogate)
    stage_evidence(prior, surrogate, data["y_sim"])
    stage_experimental_design()
    stage_domain_adaptation(data, surrogate)

    print("=" * 60)
    print("校准流程完成")
    print("=" * 60)
    print("项目结构:")
    print("  - numerical_base.py      : 机器常数 (705_machar)")
    print("  - forward_model.py       : Gray-Scott FEM (486+402)")
    print("  - prior_geometry.py      : 先验几何 (709+054+1159+1414)")
    print("  - surrogate_model.py     : GP 代理 (1074+1102)")
    print("  - proposal_engine.py     : 流形提议 (1104)")
    print("  - evidence_quadrature.py : 证据积分 (304)")
    print("  - map_solver.py          : JFNK MAP (617)")
    print("  - constrained_calibration.py : 约束校准 (1260)")
    print("  - experimental_design.py : 序贯设计 (449)")
    print("  - domain_adapter.py      : Sim-to-Real (1079)")
    print("  - bayesian_calibration.py: MCMC 引擎")
    print("  - synthetic_data.py      : 合成数据")
    print("  - main.py                : 统一入口")
    print()
    print("全部 15 个种子项目已融入.")


if __name__ == "__main__":
    main()
