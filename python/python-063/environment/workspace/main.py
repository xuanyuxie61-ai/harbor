"""
================================================================================
古气候代理数据同化与地球系统状态重建 (Paleoclimate Proxy Data Assimilation)
================================================================================

本项目围绕"气候科学：古气候代理数据同化"展开，是一个博士级自然科学计算问题。
核心任务：利用多种古气候代理记录（冰芯δ18O、树轮宽度、珊瑚Sr/Ca等），
结合集合卡尔曼滤波（EnKF）与随机气候动力学模型，重建过去1000年的全球温度场。

数学模型：
1. 气候状态演化：dX/dt = f(X,θ) + σ(X)·dW  （随机微分方程，Milstein离散）
2. 代理响应模型：Y_i = h_i(X) + ε_i,  ε_i ~ LogNormal_truncated(μ,σ,a,b)
3. 数据同化：X^a = X^f + K·(Y - H·X^f),  K = P^f H^T (H P^f H^T + R)^{-1}
4. 空间积分：∫_Ω T(λ,φ,t) dΩ ≈ Σ_j w_j T(λ_j, φ_j, t)  （Witherden金字塔积分）

运行方式：零参数直接运行
    python main.py
================================================================================
"""

import numpy as np
import time

from earth_mesh import SphericalTriMesh, compute_global_mean_temperature
from climate_sde import StochasticClimateModel, ClimateForcing
from sobol_sampler import SobolSequence
from proxy_response import ProxyResponseModel, ProxyCalibration
from ensemble_kalman_filter import EnsembleKalmanFilter
from proxy_network import ProxyNetwork, ClimateEventRanker
from observation_strategy import OptimalObservationStrategy, MultiProxyCompetition
from reconstruction_engine import ReconstructionEngine
from utils import ensure_positive_definite, regularize_covariance, print_summary


def main():
    print("=" * 80)
    print("古气候代理数据同化与地球系统状态重建")
    print("Paleoclimate Proxy Data Assimilation & Earth System State Reconstruction")
    print("=" * 80)
    np.random.seed(42)

    # ========================================================================
    # 1. 构建球面三角网格（地球表面离散化）
    # ========================================================================
    print("\n[Step 1] 构建球面三角网格 ...")
    mesh = SphericalTriMesh(refinement_level=3)
    n_nodes = mesh.n_nodes
    n_elements = mesh.n_elements
    print(f"  网格节点数: {n_nodes}")
    print(f"  网格单元数: {n_elements}")

    # ========================================================================
    # 2. 初始化气候SDE模型
    # ========================================================================
    print("\n[Step 2] 初始化随机气候动力学模型 ...")
    climate = StochasticClimateModel(
        n_grid=n_nodes,
        diffusion_coeff=1.2e-6,      # 热扩散系数 (°C^2 / yr)
        damping_coeff=0.02,           # 牛顿冷却系数 (1/yr)
        forcing_amplitude=0.15,       # 外部强迫幅度
        noise_intensity=0.08,         # 维纳过程强度
        dt_years=1.0,                 # 时间步长 (年)
    )
    print(f"  状态维度: {n_nodes}")
    print(f"  时间步长: {climate.dt_years} 年")

    # ========================================================================
    # 3. 生成真实气候轨迹（"自然"运行）
    # ========================================================================
    print("\n[Step 3] 生成真实气候轨迹 (1000年) ...")
    n_years = 1000
    true_states = np.zeros((n_years + 1, n_nodes))
    true_states[0, :] = climate.initial_state()
    for t in range(n_years):
        true_states[t + 1, :] = climate.milstein_step(true_states[t, :])
    print(f"  模拟完成: {n_years} 年")

    # ========================================================================
    # 4. 初始化Sobol序列用于集合生成
    # ========================================================================
    print("\n[Step 4] 使用Sobol准随机序列初始化集合 ...")
    ensemble_size = 64
    sobol = SobolSequence(dim=n_nodes)
    perturbation = sobol.generate(n=ensemble_size)
    ensemble_states = np.zeros((n_years + 1, ensemble_size, n_nodes))
    for e in range(ensemble_size):
        ensemble_states[0, e, :] = climate.initial_state() + 0.5 * (perturbation[e] - 0.5)
    print(f"  集合大小: {ensemble_size}")

    # ========================================================================
    # 5. 定义代理响应模型与校准
    # ========================================================================
    print("\n[Step 5] 定义代理响应模型 ...")
    proxy_cal = ProxyCalibration()
    proxy_types = ["ice_core_d18O", "tree_ring_width", "coral_SrCa", "sediment_MgCa"]
    proxy_models = {}
    for ptype in proxy_types:
        proxy_models[ptype] = ProxyResponseModel(
            proxy_type=ptype,
            sensitivity=proxy_cal.get_sensitivity(ptype),
            bias=proxy_cal.get_bias(ptype),
            noise_sigma=proxy_cal.get_noise(ptype),
            truncation_lower=proxy_cal.get_truncation_lower(ptype),
            truncation_upper=proxy_cal.get_truncation_upper(ptype),
        )
    print(f"  代理类型: {proxy_types}")

    # ========================================================================
    # 6. 构建代理观测网络
    # ========================================================================
    print("\n[Step 6] 构建代理观测网络 ...")
    network = ProxyNetwork(mesh=mesh, n_proxies_per_type=6)
    network.build_spatial_network()
    n_proxies = network.n_proxies
    print(f"  总代理站点数: {n_proxies}")

    # ========================================================================
    # 7. 生成代理观测（含截断对数正态噪声）
    # ========================================================================
    print("\n[Step 7] 生成代理观测数据 ...")
    observations = np.zeros((n_years + 1, n_proxies))
    for t in range(n_years + 1):
        observations[t, :] = network.generate_observations(
            true_states[t, :], proxy_models
        )
    print(f"  观测矩阵维度: {observations.shape}")

    # ========================================================================
    # 8. 最优观测策略（贪心选择信息增益最大站点）
    # ========================================================================
    print("\n[Step 8] 执行最优观测策略 ...")
    obs_strategy = OptimalObservationStrategy(
        network=network, proxy_models=proxy_models
    )
    selected_indices = obs_strategy.greedy_select(
        n_select=min(20, n_proxies), covariance_threshold=0.7
    )
    print(f"  选定观测站点数: {len(selected_indices)}")

    # ========================================================================
    # 9. 多代理竞争模型（评估不同代理类型的重建可信度）
    # ========================================================================
    print("\n[Step 9] 多代理竞争与可信度评估 ...")
    competition = MultiProxyCompetition(proxy_types=proxy_types)
    credibility = competition.evaluate_credibility(
        true_states=true_states,
        observations=observations,
        network=network,
        proxy_models=proxy_models,
    )
    for ptype, cred in credibility.items():
        print(f"  {ptype:20s} 可信度: {cred:.4f}")

    # ========================================================================
    # 10. 集合卡尔曼滤波同化
    # ========================================================================
    print("\n[Step 10] 执行集合卡尔曼滤波同化 ...")
    enkf = EnsembleKalmanFilter(
        ensemble_size=ensemble_size,
        state_dim=n_nodes,
        obs_dim=len(selected_indices),
        inflation_factor=1.05,
        localization_radius=2500.0,  # km
    )

    assimilated_states = np.zeros_like(true_states)
    assimilated_states[0, :] = np.mean(ensemble_states[0, :, :], axis=0)

    for t in range(1, n_years + 1):
        # 预报步：集合向前积分
        for e in range(ensemble_size):
            ensemble_states[t, e, :] = climate.milstein_step(
                ensemble_states[t - 1, e, :]
            )

        # 每10年进行一次同化
        if t % 10 == 0:
            y_obs = observations[t, selected_indices]
            H = network.build_observation_operator(
                selected_indices, n_nodes, proxy_models
            )
            ensemble_states[t, :, :] = enkf.analysis_step(
                ensemble_states[t, :, :],
                y_obs,
                H,
                mesh=mesh,
                time_index=t,
            )

        assimilated_states[t, :] = np.mean(ensemble_states[t, :, :], axis=0)

    print(f"  同化完成，总步数: {n_years}")

    # ========================================================================
    # 11. 重建引擎与误差分析
    # ========================================================================
    print("\n[Step 11] 重建质量评估 ...")
    engine = ReconstructionEngine(mesh=mesh, true_states=true_states)
    metrics = engine.evaluate_reconstruction(assimilated_states)

    print(f"  全球平均温度 RMSE: {metrics['global_rmse']:.6f} °C")
    print(f"  空间平均相关系数: {metrics['spatial_corr']:.6f}")
    print(f"  趋势误差: {metrics['trend_error']:.6f} °C/世纪")
    print(f"  集合离散度: {metrics['ensemble_spread']:.6f}")

    # ========================================================================
    # 12. 气候事件排序验证（使用Ulam距离）
    # ========================================================================
    print("\n[Step 12] 气候事件排序验证 ...")
    ranker = ClimateEventRanker()
    event_accuracy = ranker.evaluate_event_ordering(
        true_series=np.mean(true_states, axis=1),
        recon_series=np.mean(assimilated_states, axis=1),
        n_events=20,
    )
    print(f"  极端事件排序准确率: {event_accuracy:.4f}")

    # ========================================================================
    # 13. 全球平均温度时间序列与空间积分
    # ========================================================================
    print("\n[Step 13] 计算全球平均温度时间序列 ...")
    true_global = np.zeros(n_years + 1)
    recon_global = np.zeros(n_years + 1)
    for t in range(n_years + 1):
        true_global[t] = compute_global_mean_temperature(mesh, true_states[t, :])
        recon_global[t] = compute_global_mean_temperature(mesh, assimilated_states[t, :])

    warming_true = (true_global[-1] - true_global[0]) / (n_years / 100.0)
    warming_recon = (recon_global[-1] - recon_global[0]) / (n_years / 100.0)
    print(f"  真实百年变暖趋势: {warming_true:.4f} °C/世纪")
    print(f"  重建百年变暖趋势: {warming_recon:.4f} °C/世纪")
    print(f"  趋势偏差: {abs(warming_true - warming_recon):.4f} °C/世纪")

    # ========================================================================
    # 14. 最终报告
    # ========================================================================
    print("\n" + "=" * 80)
    print("古气候重建完成!")
    print("=" * 80)
    print_summary(metrics, warming_true, warming_recon, event_accuracy)
    print("=" * 80)


if __name__ == "__main__":
    start = time.time()
    main()
    elapsed = time.time() - start
    print(f"\n总运行时间: {elapsed:.2f} 秒")
