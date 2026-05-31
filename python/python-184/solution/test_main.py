"""
================================================================================
博士级科研代码合成项目：时间序列预测与异常检测的多物理场耦合计算框架
================================================================================

本项目基于 15 个种子项目的核心算法，融合构造了一个面向
"数据科学：时间序列预测与异常检测" 的前沿博士级计算项目。

核心科学问题：
    针对具有非线性动力学、多尺度耦合与未知噪声结构的复杂时间序列，
    构建一个集预测、平滑、重构、验证于一体的多物理场计算框架。
    框架涵盖：
    - 自回归建模与特征根稳定性分析（Toeplitz + Newton-Maehly）
    - 图结构异常检测（邻接矩阵谱分析 + PageRank）
    - RBF 非均匀插值重建（核方法）
    - PDE 时空平滑（热方程 FEM + 反应-扩散）
    - 生化非线性 ODE 动力学建模
    - 广义 Gauss-Hermite 贝叶斯积分
    - 时延嵌入几何质量分析
    - 数值鲁棒性评估与 Monte Carlo 概率界
    - 制造解方法（MMS）验证
    - GF(2) 离散模式检测
    - 三角形对称求积特征提取

运行方式：
    python main.py    （零参数运行）
================================================================================
"""

import numpy as np
import sys
import time

# 确保模块导入路径
sys.path.insert(0, __file__.rsplit('/', 1)[0] if '/' in __file__ else '.')

from toeplitz_solver import ToeplitzSolver
from ar_predictor import ARPredictor
from graph_anomaly_detector import GraphAnomalyDetector
from rbf_reconstructor import RBFReconstructor
from pde_spatiotemporal_model import PDE1DHeatExplicit, ReactionDiffusion1D
from nonlinear_ode_dynamics import BiochemicalODE, ExtendedBrusselator
from quadrature_bayesian import GenHermiteQuadrature
from embedding_geometry import EmbeddingGeometry
from numerical_robustness import NumericalRobustness
from manufactured_verification import ManufacturedVerification
from discrete_pattern_kernel import DiscretePatternKernel
from triangular_feature_integrator import TriangularFeatureIntegrator


def generate_synthetic_series(n: int = 512, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """
    生成合成时间序列：
    - 基础信号：Brusselator 振荡器 + 线性趋势 + 季节性
    - 注入结构性异常：突变、漂移、振荡频率变化
    - 加性高斯噪声
    """
    np.random.seed(seed)
    t = np.linspace(0.0, 20.0, n)

    # 基础动力学：Brusselator 振荡
    bruss = ExtendedBrusselator(a=1.0, b=2.8, D1=0.0, D2=0.0)
    y0 = np.array([1.0, 1.0])
    _, traj = bruss.integrate(y0, (0.0, 20.0), n - 1)
    base = traj[:, 0]  # 取 x1 分量

    # 线性趋势 + 季节性
    trend = 0.05 * t
    seasonal = 0.3 * np.sin(2.0 * np.pi * t / 3.0)
    signal = base + trend + seasonal

    # 注入异常
    anomaly_mask = np.zeros(n, dtype=bool)
    # 异常 1：突发尖峰
    anomaly_mask[120:125] = True
    signal[120:125] += 3.5
    # 异常 2：水平漂移
    anomaly_mask[250:270] = True
    signal[250:270] += 1.8
    # 异常 3：频率突变（替换为高频信号）
    anomaly_mask[380:390] = True
    signal[380:390] = 2.0 * np.sin(8.0 * np.pi * t[380:390]) + np.mean(signal)

    # 加性噪声
    noise = np.random.normal(0.0, 0.15, n)
    series = signal + noise

    return t, series, anomaly_mask


def main():
    print("=" * 80)
    print("博士级时间序列预测与异常检测多物理场耦合计算框架")
    print("=" * 80)
    start_time = time.time()

    # ============================================================
    # 1. 数据生成
    # ============================================================
    print("\n[阶段 1] 生成合成复杂时间序列...")
    t, series, true_anomalies = generate_synthetic_series(n=512, seed=42)
    print(f"  序列长度: {len(series)}")
    print(f"  注入异常点数: {np.sum(true_anomalies)}")

    # ============================================================
    # 2. Toeplitz + Levinson-Durbin AR 建模与预测
    # ============================================================
    print("\n[阶段 2] Toeplitz 系统求解与 AR(p) 建模...")
    ar_order = 8
    ar_model = ARPredictor(order=ar_order)
    ar_model.fit(series)
    print(f"  AR({ar_order}) 系数: {np.round(ar_model.ar_coefs, 4)}")
    print(f"  噪声标准差估计: {ar_model.sigma_e:.4f}")

    # 特征根分析（Newton-Maehly）
    stability = ar_model.stability_analysis()
    print(f"  系统稳定性: {'稳定' if stability['stable'] else '不稳定'}")
    print(f"  检测到 {len(stability['modes'])} 个动力学模态:")
    for idx, mode in enumerate(stability['modes'][:4]):
        print(f"    模态 {idx+1}: |z|={mode['modulus']:.4f}, "
              f"τ={mode['time_constant']:.2f}, f={mode['frequency']:.4f}, ζ={mode['damping_ratio']:.4f}")

    # 多步预测
    forecast_steps = 20
    pred, lower, upper = ar_model.forecast_interval(series, steps=forecast_steps, confidence=0.95)
    print(f"  {forecast_steps} 步预测完成，预测区间覆盖率待验证")

    # ============================================================
    # 3. 图结构异常检测
    # ============================================================
    print("\n[阶段 3] 图邻接矩阵谱异常检测...")
    # 构造滑动窗口特征用于图分析
    window = 10
    X_graph = np.array([series[i:i+window] for i in range(len(series) - window + 1)])
    graph_det = GraphAnomalyDetector(k_neighbors=5)
    graph_scores = graph_det.detect(X_graph, method="pagerank")
    # 将窗口得分映射回原始序列
    graph_scores_full = np.zeros(len(series))
    for i, score in enumerate(graph_scores):
        graph_scores_full[i:i+window] += score
    graph_scores_full /= window
    graph_scores_full = (graph_scores_full - graph_scores_full.min()) / (graph_scores_full.max() - graph_scores_full.min() + 1e-12)

    # 连通分量分析
    n_comp, labels = graph_det.connected_components()
    print(f"  相似度图连通分量数: {n_comp}")
    print(f"  最大分量占比: {np.max(np.bincount(labels)) / len(labels) * 100:.1f}%")
    print(f"  图异常得分范围: [{graph_scores_full.min():.4f}, {graph_scores_full.max():.4f}]")

    # ============================================================
    # 4. RBF 重建与异常评分
    # ============================================================
    print("\n[阶段 4] RBF 径向基函数重建与留一法异常评分...")
    rbf = RBFReconstructor(kernel="gaussian", shape_param=0.5, regularization=1e-8)
    rbf_scores = rbf.anomaly_score(t, series)
    # 重建完整序列（用于缺失值插补演示）
    missing_idx = np.array([100, 200, 300, 400])
    observed_t = np.delete(t, missing_idx)
    observed_v = np.delete(series, missing_idx)
    reconstructed = rbf.reconstruct_series(observed_t, observed_v, t)
    recon_error = np.abs(series - reconstructed)
    print(f"  RBF 重建误差 (L2): {np.sqrt(np.mean(recon_error**2)):.4f}")
    print(f"  RBF 异常得分范围: [{rbf_scores.min():.4f}, {rbf_scores.max():.4f}]")

    # ============================================================
    # 5. PDE 时空平滑
    # ============================================================
    print("\n[阶段 5] PDE 时空平滑（热方程 FEM + 反应-扩散）...")
    heat_solver = PDE1DHeatExplicit(kappa=0.5)
    u_smoothed = heat_solver.solve(t, series.copy(), dt=0.01, n_steps=50)
    print(f"  热方程平滑前后 L2 差: {np.sqrt(np.mean((u_smoothed - series)**2)):.4f}")

    rd_solver = ReactionDiffusion1D(D=0.3, rho=0.1, K=series.max(), mu=0.05, c_s=0.5)
    u_rd = rd_solver.solve(t, series.copy(), dt=0.005, n_steps=40, scheme="heun")
    print(f"  反应-扩散演化后范围: [{u_rd.min():.4f}, {u_rd.max():.4f}]")

    # ============================================================
    # 6. 非线性生化 ODE 动力学
    # ============================================================
    print("\n[阶段 6] 生化反应网络 ODE 动力学建模...")
    bio = BiochemicalODE(kf=1.0, kr=0.1, kcat=0.5)
    y0_bio = np.array([1.0, 2.0, 0.0, 0.0])
    t_bio, y_bio = bio.integrate_rk4(y0_bio, (0.0, 20.0), 200)
    h_final = bio.conserved_quantities(y_bio[-1])
    h_initial = bio.conserved_quantities(y0_bio)
    print(f"  初始守恒量: E_tot={h_initial[0]:.4f}, S_tot={h_initial[1]:.4f}")
    print(f"  最终守恒量: E_tot={h_final[0]:.4f}, S_tot={h_final[1]:.4f}")
    print(f"  守恒偏差: {np.abs(h_final - h_initial)}")

    # Brusselator Lyapunov 指数
    bruss = ExtendedBrusselator(a=1.0, b=2.8)
    y0_bruss = np.array([1.0, 1.0])
    lyap = bruss.lyapunov_exponent_numerical(y0_bruss, (0.0, 50.0), 5000)
    print(f"  Brusselator 最大 Lyapunov 指数: {lyap:.4f} ({'混沌' if lyap > 0.01 else '规则'})")

    # ============================================================
    # 7. 广义 Gauss-Hermite 贝叶斯积分
    # ============================================================
    print("\n[阶段 7] 广义 Gauss-Hermite 贝叶斯预测积分...")
    gh = GenHermiteQuadrature(alpha=0.0, a=0.0, b=1.0, n=16)
    nodes, weights = gh.compute_rule()
    print(f"  求积节点数: {len(nodes)}")

    # 预测分布矩计算：假设参数后验为 N(μ, σ²)
    post_mean = np.mean(series)
    post_std = np.std(series)
    pred_func = lambda theta: theta + 0.1 * np.sin(theta)  # 非线性预测函数
    m1, m2 = gh.predictive_moments(post_mean, post_std, pred_func)
    pred_var = m2 - m1**2
    print(f"  预测均值 E[f(θ)]: {m1:.4f}")
    print(f"  预测方差 Var[f(θ)]: {pred_var:.4f}")

    # 验证标准正态矩
    gh_std = GenHermiteQuadrature(alpha=0.0, a=0.0, b=1.0, n=20)
    gh_std.compute_rule()
    moment2 = gh_std.integrate(lambda x: x**2) / np.sqrt(np.pi)
    print(f"  二阶矩验证 (应为 0.5): {moment2:.6f}")

    # ============================================================
    # 8. 时延嵌入几何质量分析
    # ============================================================
    print("\n[阶段 8] 时延嵌入几何质量分析与 FNN 维度估计...")
    geom = EmbeddingGeometry(embedding_dim=3, delay=2)
    X_embed = geom.delay_embed(series)
    print(f"  嵌入矩阵形状: {X_embed.shape}")

    embed_scores = geom.local_triangulation_quality(X_embed, k_neighbors=6)
    print(f"  几何异常得分范围: [{embed_scores.min():.4f}, {embed_scores.max():.4f}]")

    est_dim = geom.embedding_dimension_estimate(series, max_dim=8, threshold=0.1)
    print(f"  FNN 估计最小嵌入维度: {est_dim}")

    # ============================================================
    # 9. 数值鲁棒性分析
    # ============================================================
    print("\n[阶段 9] 数值鲁棒性与 Monte Carlo 概率界...")
    nr = NumericalRobustness()

    # 机器精度分析
    x_test = 1.0
    x_next = nr.next_float(x_test)
    x_prev = nr.prev_float(x_test)
    print(f"  nextafter(1.0) = {x_next:.16e}, gap = {x_next - 1.0:.2e}")
    print(f"  prevfloat(1.0) = {x_prev:.16e}, gap = {1.0 - x_prev:.2e}")

    # Regula Falsi 求异常阈值（对齐长度）
    n_scores = len(series)
    embed_scores_aligned = np.zeros(n_scores)
    embed_off = (n_scores - len(embed_scores)) // 2
    embed_scores_aligned[embed_off:embed_off + len(embed_scores)] = embed_scores
    if embed_off > 0:
        embed_scores_aligned[:embed_off] = embed_scores[0]
        embed_scores_aligned[embed_off + len(embed_scores):] = embed_scores[-1]
    combined_scores = 0.3 * graph_scores_full + 0.3 * rbf_scores + 0.4 * embed_scores_aligned
    threshold = nr.threshold_by_quantile_root(combined_scores, target_fpr=0.05)
    print(f"  Regula Falsi 求得异常阈值 (FPR=5%): {threshold:.4f}")

    # 矩阵条件数分析
    sample_cov = np.cov(X_embed.T)
    sens = nr.condition_number_sensitivity(sample_cov)
    print(f"  嵌入协方差条件数: {sens['condition_number']:.2e}")
    print(f"  双精度可解性: {sens['solvable_in_double_precision']}")

    # 高维球 Monte Carlo
    mc_val = nr.ball_monte_carlo_integral(lambda x: np.exp(-np.sum(x**2)), dim=5, n_samples=20000)
    print(f"  5 维球 Monte Carlo 积分估计: {mc_val:.6f}")

    # ============================================================
    # 10. 制造解方法 (MMS) 验证
    # ============================================================
    print("\n[阶段 10] 制造解方法 (MMS) PDE 验证...")
    mms = ManufacturedVerification(kappa=0.5)
    x_test = np.linspace(0.0, 1.0, 41)
    res_mms = mms.verify_heat_fem(x_test, dt=0.001, n_steps=100)
    print(f"  热方程 MMS L2 误差: {res_mms['l2_error']:.6e}")
    print(f"  相对误差: {res_mms['relative_l2']:.6e}")

    conv = mms.convergence_study(n_grids=[11, 21, 41, 81], dt=0.001, n_steps=100)
    print(f"  估计空间收敛阶: {conv['estimated_spatial_order']:.2f}")

    rd_mms = mms.verify_reaction_diffusion(x_test, dt=0.001, n_steps=100, D=0.1)
    print(f"  反应-扩散 MMS 验证通过: {rd_mms['verified']}")

    # ============================================================
    # 11. GF(2) 离散模式检测
    # ============================================================
    print("\n[阶段 11] GF(2) 离散卷积异常模式检测...")
    dpk = DiscretePatternKernel(window_size=5)
    binary = dpk.binarize(series, threshold=threshold)
    burst = dpk.detect_burst_pattern(binary, burst_length=3)
    alternating = dpk.detect_alternating_pattern(binary)
    P_markov = dpk.transition_matrix_analysis(binary)
    H_rate = dpk.entropy_rate(binary)

    print(f"  二值化后异常比例: {np.mean(binary)*100:.1f}%")
    print(f"  检测到的 burst 模式数: {np.sum(burst)}")
    print(f"  检测到的交替模式数: {np.sum(alternating)}")
    print(f"  Markov 转移矩阵:\n{np.round(P_markov, 4)}")
    print(f"  熵率: {H_rate:.4f} bits/symbol")

    # ============================================================
    # 12. 三角形对称求积特征提取
    # ============================================================
    print("\n[阶段 12] 三角形对称求积 2D 特征提取...")
    tfi = TriangularFeatureIntegrator(order=5)

    # 验证求积精度
    verif = tfi.verify_monomial(2, 1)
    print(f"  单项式 x^2 y 积分验证: 精确={verif['exact']:.6f}, 数值={verif['numerical']:.6f}, 误差={verif['error']:.2e}")

    # 三元组特征提取
    triplets = np.array([
        [series[i], series[i+1], series[i+2]]
        for i in range(0, len(series)-4, 3)
    ])
    tri_features = tfi.extract_triangular_features(triplets)
    print(f"  三角形积分特征范围: [{tri_features.min():.4f}, {tri_features.max():.4f}]")

    # ============================================================
    # 13. 综合异常评分与性能评估
    # ============================================================
    print("\n[阶段 13] 综合异常检测性能评估...")
    # 多方法融合得分
    n = len(series)
    # 对齐长度（embed_scores 较短，其余截断或填充）
    min_len = min(len(graph_scores_full), len(rbf_scores), len(embed_scores))
    # 使用原始长度的索引，对 embed_scores 进行边界填充
    embed_scores_full = np.zeros(n)
    embed_offset = (n - len(embed_scores)) // 2
    embed_scores_full[embed_offset:embed_offset + len(embed_scores)] = embed_scores
    # 对边缘做简单延拓
    if embed_offset > 0:
        embed_scores_full[:embed_offset] = embed_scores[0]
        embed_scores_full[embed_offset + len(embed_scores):] = embed_scores[-1]

    g_s = graph_scores_full
    r_s = rbf_scores
    e_s = embed_scores_full

    # Z-score 标准化后加权融合
    def zscore(x):
        return (x - np.mean(x)) / (np.std(x) + 1e-12)

    fused = 0.25 * zscore(g_s) + 0.35 * zscore(r_s) + 0.40 * zscore(e_s)
    fused = (fused - fused.min()) / (fused.max() - fused.min() + 1e-12)

    # 基于阈值判定异常
    detected = fused > threshold
    true_pos = np.sum(detected & true_anomalies)
    false_pos = np.sum(detected & (~true_anomalies))
    false_neg = np.sum((~detected) & true_anomalies)

    precision = true_pos / (true_pos + false_pos + 1e-12)
    recall = true_pos / (true_pos + false_neg + 1e-12)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)

    print(f"  阈值: {threshold:.4f}")
    print(f"  检出异常数: {np.sum(detected)}, 实际异常数: {np.sum(true_anomalies)}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")

    # AUC-ROC 近似（通过遍历阈值）
    thresholds = np.linspace(0.0, 1.0, 101)
    tprs = []
    fprs = []
    for th in thresholds:
        det = fused > th
        tp = np.sum(det & true_anomalies)
        fp = np.sum(det & (~true_anomalies))
        fn = np.sum((~det) & true_anomalies)
        tn = np.sum((~det) & (~true_anomalies))
        tprs.append(tp / (tp + fn + 1e-12))
        fprs.append(fp / (fp + tn + 1e-12))
    # 确保 fprs 单调递增以正确计算 AUC
    fprs_arr = np.array(fprs)
    tprs_arr = np.array(tprs)
    sort_idx = np.argsort(fprs_arr)
    auc = np.trapezoid(tprs_arr[sort_idx], fprs_arr[sort_idx])
    print(f"  AUC-ROC:   {auc:.4f}")

    # ============================================================
    # 完成
    # ============================================================
    elapsed = time.time() - start_time
    print("\n" + "=" * 80)
    print(f"全部计算完成，耗时: {elapsed:.2f} 秒")
    print("=" * 80)

    # 返回关键结果字典（便于外部调用）
    return {
        "ar_coefs": ar_model.ar_coefs,
        "stability": stability['stable'],
        "forecast": pred,
        "graph_components": n_comp,
        "rbf_recon_error": float(np.sqrt(np.mean(recon_error**2))),
        "pde_smooth_l2": float(np.sqrt(np.mean((u_smoothed - series)**2))),
        "bio_conservation_error": float(np.max(np.abs(h_final - h_initial))),
        "lyapunov_exponent": lyap,
        "predictive_variance": pred_var,
        "embedding_dimension": est_dim,
        "condition_number": sens['condition_number'],
        "mms_l2_error": res_mms['l2_error'],
        "entropy_rate": H_rate,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "auc_roc": auc,
        "elapsed_time": elapsed
    }


if __name__ == "__main__":
    results = main()

# ================================================================
# 测试用例（40个，assert模式，涉及随机值均使用固定种子）
# ================================================================
# ---- TC01: generate_synthetic_series 返回三元组 ----
t1, s1, a1 = generate_synthetic_series(n=64, seed=42)
assert isinstance(t1, np.ndarray) and isinstance(s1, np.ndarray) and isinstance(a1, np.ndarray), '[TC01] generate_synthetic_series 返回类型 FAILED'
assert len(t1) == 64 and len(s1) == 64 and len(a1) == 64, '[TC01] 序列长度校验 FAILED'

# ---- TC02: generate_synthetic_series 可复现性 ----
import numpy as np
np.random.seed(123)
t2a, s2a, a2a = generate_synthetic_series(n=64, seed=123)
np.random.seed(123)
t2b, s2b, a2b = generate_synthetic_series(n=64, seed=123)
assert np.allclose(t2a, t2b) and np.allclose(s2a, s2b) and np.allclose(a2a, a2b), '[TC02] generate_synthetic_series 可复现性 FAILED'

# ---- TC03: generate_synthetic_series 输出值均为有限值 ----
t3, s3, a3 = generate_synthetic_series(n=64, seed=7)
assert np.all(np.isfinite(t3)) and np.all(np.isfinite(s3)), '[TC03] generate_synthetic_series 有限值校验 FAILED'

# ---- TC04: ARPredictor fit 返回自身且系数有限 ----
ar4 = ARPredictor(order=4)
_, s4, _ = generate_synthetic_series(n=128, seed=99)
result4 = ar4.fit(s4)
assert result4 is ar4, '[TC04] ARPredictor fit 返回值 FAILED'
assert len(ar4.ar_coefs) == 4, '[TC04] AR 系数个数 FAILED'
assert np.all(np.isfinite(ar4.ar_coefs)), '[TC04] AR 系数有限性 FAILED'

# ---- TC05: ARPredictor predict 输出形状正确 ----
ar5 = ARPredictor(order=3)
_, s5, _ = generate_synthetic_series(n=128, seed=42)
ar5.fit(s5)
pred5 = ar5.predict(s5, steps=10)
assert len(pred5) == 10, '[TC05] ARPredictor predict 输出长度 FAILED'
assert np.all(np.isfinite(pred5)), '[TC05] ARPredictor predict 有限性 FAILED'

# ---- TC06: ARPredictor forecast_interval 上下界关系 ----
ar6 = ARPredictor(order=4)
_, s6, _ = generate_synthetic_series(n=128, seed=55)
ar6.fit(s6)
pred6, lower6, upper6 = ar6.forecast_interval(s6, steps=8, confidence=0.95)
assert len(pred6) == 8 and len(lower6) == 8 and len(upper6) == 8, '[TC06] forecast_interval 输出形状 FAILED'
assert np.all(lower6 <= pred6) and np.all(pred6 <= upper6), '[TC06] forecast_interval 上下界关系 FAILED'

# ---- TC07: ARPredictor stability_analysis 返回字典含预期键 ----
ar7 = ARPredictor(order=4)
_, s7, _ = generate_synthetic_series(n=128, seed=77)
ar7.fit(s7)
stab7 = ar7.stability_analysis()
assert isinstance(stab7, dict), '[TC07] stability_analysis 返回类型 FAILED'
assert 'stable' in stab7 and 'modes' in stab7, '[TC07] stability_analysis 缺键 FAILED'

# ---- TC08: ToeplitzSolver solve_yule_walker 有限系数 ----
import numpy as np
np.random.seed(42)
ts8 = ToeplitzSolver()
autocorr8 = np.array([1.0, 0.5, 0.2, 0.05])
ar8, k8 = ts8.solve_yule_walker(autocorr8)
assert len(ar8) == 3 and len(k8) == 3, '[TC08] solve_yule_walker 输出长度 FAILED'
assert np.all(np.isfinite(ar8)) and np.all(np.isfinite(k8)), '[TC08] solve_yule_walker 有限性 FAILED'

# ---- TC09: ToeplitzSolver schur_cohn_stability ----
ts9 = ToeplitzSolver()
k_stable9 = np.array([0.3, -0.5, 0.2])
assert ts9.schur_cohn_stability(k_stable9) == True, '[TC09] schur_cohn_stability 稳定判定 FAILED'
k_unstable9 = np.array([0.3, 1.5, 0.2])
assert ts9.schur_cohn_stability(k_unstable9) == False, '[TC09] schur_cohn_stability 不稳定判定 FAILED'

# ---- TC10: ToeplitzSolver solve_toeplitz 输出有限且形状正确 ----
ts10 = ToeplitzSolver()
t10 = np.array([2.0, 1.0, 0.5])
T10 = ts10.autocorr_to_toeplitz(t10)
b10 = T10 @ np.array([1.0, 2.0, 3.0])
x10 = ts10.solve_toeplitz(t10, b10)
assert len(x10) == 3, '[TC10] solve_toeplitz 输出长度 FAILED'
assert np.all(np.isfinite(x10)), '[TC10] solve_toeplitz 有限性 FAILED'

# ---- TC11: GraphAnomalyDetector detect 输出在 [0,1] ----
import numpy as np
np.random.seed(42)
gad11 = GraphAnomalyDetector(k_neighbors=3)
X11 = np.random.randn(50, 4)
scores11 = gad11.detect(X11, method='pagerank')
assert len(scores11) == 50, '[TC11] detect 输出长度 FAILED'
assert np.all(scores11 >= 0.0) and np.all(scores11 <= 1.0), '[TC11] detect 范围约束 FAILED'

# ---- TC12: GraphAnomalyDetector connected_components 正整数 ----
import numpy as np
np.random.seed(42)
gad12 = GraphAnomalyDetector(k_neighbors=3)
X12 = np.random.randn(30, 3)
gad12.build_graph(X12)
n_comp12, labels12 = gad12.connected_components()
assert n_comp12 >= 1, '[TC12] connected_components 计数 FAILED'
assert len(labels12) == 30, '[TC12] connected_components 标签长度 FAILED'

# ---- TC13: GraphAnomalyDetector spectral_anomaly_score 输出在 [0,1] ----
import numpy as np
np.random.seed(42)
gad13 = GraphAnomalyDetector(k_neighbors=5)
X13 = np.random.randn(40, 3)
gad13.build_graph(X13)
spec13 = gad13.spectral_anomaly_score()
assert len(spec13) == 40, '[TC13] spectral_anomaly_score 长度 FAILED'
assert np.all(spec13 >= 0.0) and np.all(spec13 <= 1.0), '[TC13] spectral_anomaly_score 范围 FAILED'

# ---- TC14: RBFReconstructor fit+predict 有限输出 ----
import numpy as np
np.random.seed(42)
rbf14 = RBFReconstructor(kernel='gaussian', shape_param=0.5, regularization=1e-8)
centers14 = np.linspace(0, 10, 20).reshape(-1, 1)
values14 = np.sin(centers14[:, 0])
rbf14.fit(centers14, values14)
points14 = np.linspace(0, 10, 50).reshape(-1, 1)
pred14 = rbf14.predict(points14)
assert len(pred14) == 50, '[TC14] RBF predict 输出长度 FAILED'
assert np.all(np.isfinite(pred14)), '[TC14] RBF predict 有限性 FAILED'

# ---- TC15: RBFReconstructor reconstruct_series 输出形状 ----
import numpy as np
np.random.seed(42)
rbf15 = RBFReconstructor(kernel='gaussian', shape_param=0.5, regularization=1e-8)
t_obs15 = np.array([0.0, 1.0, 3.0, 4.0, 6.0, 7.0, 9.0, 10.0])
v_obs15 = np.sin(t_obs15)
t_all15 = np.linspace(0, 10, 51)
recon15 = rbf15.reconstruct_series(t_obs15, v_obs15, t_all15)
assert len(recon15) == 51, '[TC15] reconstruct_series 输出长度 FAILED'
assert np.all(np.isfinite(recon15)), '[TC15] reconstruct_series 有限性 FAILED'

# ---- TC16: RBFReconstructor anomaly_score 输出在 [0,1] ----
import numpy as np
np.random.seed(42)
rbf16 = RBFReconstructor(kernel='gaussian', shape_param=0.5, regularization=1e-8)
t16 = np.linspace(0, 10, 60)
s16 = np.sin(t16) + 0.1 * np.random.randn(60)
scores16 = rbf16.anomaly_score(t16, s16)
assert len(scores16) == 60, '[TC16] anomaly_score 长度 FAILED'
assert np.all(scores16 >= 0.0) and np.all(scores16 <= 1.0), '[TC16] anomaly_score 范围 FAILED'

# ---- TC17: PDE1DHeatExplicit solve 有限输出 ----
heat17 = PDE1DHeatExplicit(kappa=0.5)
x17 = np.linspace(0, 1, 31)
u0_17 = np.sin(np.pi * x17)
u17 = heat17.solve(x17, u0_17, dt=0.001, n_steps=50)
assert len(u17) == 31, '[TC17] PDE1DHeat solve 输出长度 FAILED'
assert np.all(np.isfinite(u17)), '[TC17] PDE1DHeat solve 有限性 FAILED'

# ---- TC18: ReactionDiffusion1D solve 输出形状 ----
rd18 = ReactionDiffusion1D(D=0.3, rho=0.1, K=2.0, mu=0.05, c_s=0.5)
x18 = np.linspace(0, 1, 25)
u0_18 = 0.5 + 0.3 * np.sin(2 * np.pi * x18)
u18 = rd18.solve(x18, u0_18, dt=0.005, n_steps=30, scheme='heun')
assert len(u18) == 25, '[TC18] ReactionDiffusion1D solve 输出长度 FAILED'
assert np.all(np.isfinite(u18)), '[TC18] ReactionDiffusion1D solve 有限性 FAILED'

# ---- TC19: BiochemicalODE 守恒量非负 ----
bio19 = BiochemicalODE(kf=1.0, kr=0.1, kcat=0.5)
y19 = np.array([1.0, 2.0, 0.0, 0.0])
h19 = bio19.conserved_quantities(y19)
assert len(h19) == 2, '[TC19] conserved_quantities 长度 FAILED'
assert np.all(h19 >= 0.0), '[TC19] 守恒量非负 FAILED'

# ---- TC20: BiochemicalODE integrate_rk4 初始/最终守恒近似 ----
bio20 = BiochemicalODE(kf=1.0, kr=0.1, kcat=0.5)
y0_20 = np.array([1.0, 2.0, 0.0, 0.0])
t20, y20 = bio20.integrate_rk4(y0_20, (0.0, 5.0), 100)
h_init20 = bio20.conserved_quantities(y0_20)
h_final20 = bio20.conserved_quantities(y20[-1])
assert y20.shape == (101, 4), '[TC20] integrate_rk4 输出形状 FAILED'
assert np.all(np.abs(h_final20 - h_init20) < 1e-2), '[TC20] 守恒量偏差过大 FAILED'

# ---- TC21: ExtendedBrusselator integrate 输出形状 ----
bruss21 = ExtendedBrusselator(a=1.0, b=2.8)
y0_21 = np.array([1.0, 1.0])
t21, y21 = bruss21.integrate(y0_21, (0.0, 10.0), 100)
assert y21.shape == (101, 2), '[TC21] ExtendedBrusselator integrate 输出形状 FAILED'
assert np.all(np.isfinite(y21)), '[TC21] ExtendedBrusselator integrate 有限性 FAILED'

# ---- TC22: ExtendedBrusselator lyapunov_exponent 有限标量 ----
bruss22 = ExtendedBrusselator(a=1.0, b=2.8)
y0_22 = np.array([1.0, 1.0])
lyap22 = bruss22.lyapunov_exponent_numerical(y0_22, (0.0, 10.0), 200)
assert np.isfinite(lyap22), '[TC22] Lyapunov 指数有限性 FAILED'

# ---- TC23: GenHermiteQuadrature compute_rule 权重和 ≈ √π ----
gh23 = GenHermiteQuadrature(alpha=0.0, a=0.0, b=1.0, n=16)
nodes23, weights23 = gh23.compute_rule()
assert len(nodes23) == 16 and len(weights23) == 16, '[TC23] compute_rule 输出长度 FAILED'
assert np.abs(np.sum(weights23) - np.sqrt(np.pi)) < 1e-10, '[TC23] 权重和校验 FAILED'

# ---- TC24: GenHermiteQuadrature integrate 常数函数 ----
gh24 = GenHermiteQuadrature(alpha=0.0, a=0.0, b=1.0, n=20)
gh24.compute_rule()
I_const24 = gh24.integrate(lambda x: np.ones_like(x))
assert np.abs(I_const24 - np.sqrt(np.pi)) < 1e-10, '[TC24] 常数函数积分 FAILED'

# ---- TC25: GenHermiteQuadrature integrate x^2 ----
gh25 = GenHermiteQuadrature(alpha=0.0, a=0.0, b=1.0, n=30)
gh25.compute_rule()
I_x2_25 = gh25.integrate(lambda x: x**2)
expected25 = 0.5 * np.sqrt(np.pi)
assert np.abs(I_x2_25 - expected25) < 1e-6, '[TC25] x^2 积分 FAILED'

# ---- TC26: EmbeddingGeometry delay_embed 形状正确 ----
geom26 = EmbeddingGeometry(embedding_dim=3, delay=2)
s26 = np.sin(np.linspace(0, 4*np.pi, 100))
X26 = geom26.delay_embed(s26)
expected26 = 100 - (3 - 1) * 2
assert X26.shape == (expected26, 3), '[TC26] delay_embed 形状 FAILED'

# ---- TC27: EmbeddingGeometry embedding_dimension_estimate 正整数 ----
geom27 = EmbeddingGeometry(embedding_dim=3, delay=2)
s27 = np.sin(np.linspace(0, 4*np.pi, 200)) + 0.05 * np.random.randn(200)
import numpy as np
np.random.seed(42)
dim27 = geom27.embedding_dimension_estimate(s27, max_dim=6, threshold=0.2)
assert isinstance(dim27, (int, np.integer)), '[TC27] embedding_dimension_estimate 类型 FAILED'
assert dim27 > 0, '[TC27] embedding_dimension_estimate 非正 FAILED'

# ---- TC28: NumericalRobustness next_float > x (for x > 0) ----
nr28 = NumericalRobustness()
x28 = 1.0
nx28 = nr28.next_float(x28)
assert nx28 > x28, '[TC28] next_float 单调性 FAILED'

# ---- TC29: NumericalRobustness prev_float < x (for x > 0) ----
nr29 = NumericalRobustness()
x29 = 1.0
px29 = nr29.prev_float(x29)
assert px29 < x29, '[TC29] prev_float 单调性 FAILED'

# ---- TC30: NumericalRobustness threshold_by_quantile_root 在 [0,1] ----
nr30 = NumericalRobustness()
scores30 = np.random.rand(200)
import numpy as np
np.random.seed(42)
th30 = nr30.threshold_by_quantile_root(scores30, target_fpr=0.1)
assert 0.0 <= th30 <= 1.0, '[TC30] threshold_by_quantile_root 范围 FAILED'

# ---- TC31: ManufacturedVerification verify_heat_fem 返回预期键 ----
mms31 = ManufacturedVerification(kappa=0.5)
x31 = np.linspace(0.0, 1.0, 21)
res31 = mms31.verify_heat_fem(x31, dt=0.001, n_steps=50)
assert 'l2_error' in res31 and 'relative_l2' in res31, '[TC31] verify_heat_fem 缺键 FAILED'
assert res31['l2_error'] > 0, '[TC31] L2 误差非正 FAILED'

# ---- TC32: ManufacturedVerification convergence_study 估计收敛阶 ----
mms32 = ManufacturedVerification(kappa=0.5)
conv32 = mms32.convergence_study(n_grids=[11, 21, 41], dt=0.0005, n_steps=50)
assert 'estimated_spatial_order' in conv32, '[TC32] convergence_study 缺键 FAILED'
assert np.isfinite(conv32['estimated_spatial_order']), '[TC32] 估计收敛阶有限性 FAILED'

# ---- TC33: DiscretePatternKernel binarize 输出 0/1 ----
dpk33 = DiscretePatternKernel(window_size=5)
s33 = np.sin(np.linspace(0, 4*np.pi, 80)) + 0.2 * np.random.randn(80)
import numpy as np
np.random.seed(42)
binary33 = dpk33.binarize(s33)
assert np.all((binary33 == 0) | (binary33 == 1)), '[TC33] binarize 输出非 0/1 FAILED'

# ---- TC34: DiscretePatternKernel entropy_rate 有限值 ----
dpk34 = DiscretePatternKernel(window_size=5)
s34 = np.sin(np.linspace(0, 4*np.pi, 100)) + 0.2 * np.random.randn(100)
import numpy as np
np.random.seed(42)
binary34 = dpk34.binarize(s34)
ent34 = dpk34.entropy_rate(binary34)
assert np.isfinite(ent34), '[TC34] entropy_rate 有限性 FAILED'
assert 0.0 <= ent34 <= 2.0, '[TC34] entropy_rate 范围 FAILED'

# ---- TC35: TriangularFeatureIntegrator verify_monomial 高精度 ----
tfi35 = TriangularFeatureIntegrator(order=5)
verif35 = tfi35.verify_monomial(2, 1)
assert verif35['error'] < 1e-10, '[TC35] verify_monomial 精度 FAILED'

# ---- TC36: TriangularFeatureIntegrator extract_triangular_features 输出 [0,1] ----
tfi36 = TriangularFeatureIntegrator(order=5)
triplets36 = np.array([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0], [0.5, 1.5, 2.5]])
feat36 = tfi36.extract_triangular_features(triplets36)
assert len(feat36) == 3, '[TC36] extract_triangular_features 长度 FAILED'
assert np.all(feat36 >= 0.0) and np.all(feat36 <= 1.0), '[TC36] extract_triangular_features 范围 FAILED'

# ---- TC37: main 集成测试：结果字典含预期键 ----
assert isinstance(results, dict), '[TC37] main 返回类型 FAILED'
expected_keys = ['ar_coefs', 'stability', 'forecast', 'rbf_recon_error', 'pde_smooth_l2',
                 'bio_conservation_error', 'lyapunov_exponent', 'predictive_variance',
                 'embedding_dimension', 'condition_number', 'mms_l2_error', 'entropy_rate',
                 'precision', 'recall', 'f1_score', 'auc_roc', 'elapsed_time']
for k in expected_keys:
    assert k in results, f'[TC37] main 结果缺键 {k} FAILED'

# ---- TC38: main 集成测试：性能指标在 [0,1] ----
assert 0.0 <= results['precision'] <= 1.0, '[TC38] precision 范围 FAILED'
assert 0.0 <= results['recall'] <= 1.0, '[TC38] recall 范围 FAILED'
assert 0.0 <= results['f1_score'] <= 1.0, '[TC38] f1_score 范围 FAILED'
assert 0.0 <= results['auc_roc'] <= 1.0, '[TC38] auc_roc 范围 FAILED'

# ---- TC39: main 集成测试：forecast 是有限值 ----
assert np.all(np.isfinite(results['forecast'])), '[TC39] forecast 有限性 FAILED'

# ---- TC40: main 集成测试：elapsed_time > 0 ----
assert results['elapsed_time'] > 0, '[TC40] elapsed_time 非正 FAILED'

print('\n全部 40 个测试通过!\n')
