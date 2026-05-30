
import numpy as np
import sys
import time


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
    np.random.seed(seed)
    t = np.linspace(0.0, 20.0, n)


    bruss = ExtendedBrusselator(a=1.0, b=2.8, D1=0.0, D2=0.0)
    y0 = np.array([1.0, 1.0])
    _, traj = bruss.integrate(y0, (0.0, 20.0), n - 1)
    base = traj[:, 0]


    trend = 0.05 * t
    seasonal = 0.3 * np.sin(2.0 * np.pi * t / 3.0)
    signal = base + trend + seasonal


    anomaly_mask = np.zeros(n, dtype=bool)

    anomaly_mask[120:125] = True
    signal[120:125] += 3.5

    anomaly_mask[250:270] = True
    signal[250:270] += 1.8

    anomaly_mask[380:390] = True
    signal[380:390] = 2.0 * np.sin(8.0 * np.pi * t[380:390]) + np.mean(signal)


    noise = np.random.normal(0.0, 0.15, n)
    series = signal + noise

    return t, series, anomaly_mask


def main():
    print("=" * 80)
    print("博士级时间序列预测与异常检测多物理场耦合计算框架")
    print("=" * 80)
    start_time = time.time()




    print("\n[阶段 1] 生成合成复杂时间序列...")
    t, series, true_anomalies = generate_synthetic_series(n=512, seed=42)
    print(f"  序列长度: {len(series)}")
    print(f"  注入异常点数: {np.sum(true_anomalies)}")
















    raise NotImplementedError("Hole_3: 请完成 AR 建模与预测调用代码")




    print("\n[阶段 3] 图邻接矩阵谱异常检测...")

    window = 10
    X_graph = np.array([series[i:i+window] for i in range(len(series) - window + 1)])
    graph_det = GraphAnomalyDetector(k_neighbors=5)
    graph_scores = graph_det.detect(X_graph, method="pagerank")

    graph_scores_full = np.zeros(len(series))
    for i, score in enumerate(graph_scores):
        graph_scores_full[i:i+window] += score
    graph_scores_full /= window
    graph_scores_full = (graph_scores_full - graph_scores_full.min()) / (graph_scores_full.max() - graph_scores_full.min() + 1e-12)


    n_comp, labels = graph_det.connected_components()
    print(f"  相似度图连通分量数: {n_comp}")
    print(f"  最大分量占比: {np.max(np.bincount(labels)) / len(labels) * 100:.1f}%")
    print(f"  图异常得分范围: [{graph_scores_full.min():.4f}, {graph_scores_full.max():.4f}]")




    print("\n[阶段 4] RBF 径向基函数重建与留一法异常评分...")
    rbf = RBFReconstructor(kernel="gaussian", shape_param=0.5, regularization=1e-8)
    rbf_scores = rbf.anomaly_score(t, series)

    missing_idx = np.array([100, 200, 300, 400])
    observed_t = np.delete(t, missing_idx)
    observed_v = np.delete(series, missing_idx)
    reconstructed = rbf.reconstruct_series(observed_t, observed_v, t)
    recon_error = np.abs(series - reconstructed)
    print(f"  RBF 重建误差 (L2): {np.sqrt(np.mean(recon_error**2)):.4f}")
    print(f"  RBF 异常得分范围: [{rbf_scores.min():.4f}, {rbf_scores.max():.4f}]")




    print("\n[阶段 5] PDE 时空平滑（热方程 FEM + 反应-扩散）...")
    heat_solver = PDE1DHeatExplicit(kappa=0.5)
    u_smoothed = heat_solver.solve(t, series.copy(), dt=0.01, n_steps=50)
    print(f"  热方程平滑前后 L2 差: {np.sqrt(np.mean((u_smoothed - series)**2)):.4f}")

    rd_solver = ReactionDiffusion1D(D=0.3, rho=0.1, K=series.max(), mu=0.05, c_s=0.5)
    u_rd = rd_solver.solve(t, series.copy(), dt=0.005, n_steps=40, scheme="heun")
    print(f"  反应-扩散演化后范围: [{u_rd.min():.4f}, {u_rd.max():.4f}]")




    print("\n[阶段 6] 生化反应网络 ODE 动力学建模...")
    bio = BiochemicalODE(kf=1.0, kr=0.1, kcat=0.5)
    y0_bio = np.array([1.0, 2.0, 0.0, 0.0])
    t_bio, y_bio = bio.integrate_rk4(y0_bio, (0.0, 20.0), 200)
    h_final = bio.conserved_quantities(y_bio[-1])
    h_initial = bio.conserved_quantities(y0_bio)
    print(f"  初始守恒量: E_tot={h_initial[0]:.4f}, S_tot={h_initial[1]:.4f}")
    print(f"  最终守恒量: E_tot={h_final[0]:.4f}, S_tot={h_final[1]:.4f}")
    print(f"  守恒偏差: {np.abs(h_final - h_initial)}")


    bruss = ExtendedBrusselator(a=1.0, b=2.8)
    y0_bruss = np.array([1.0, 1.0])
    lyap = bruss.lyapunov_exponent_numerical(y0_bruss, (0.0, 50.0), 5000)
    print(f"  Brusselator 最大 Lyapunov 指数: {lyap:.4f} ({'混沌' if lyap > 0.01 else '规则'})")




    print("\n[阶段 7] 广义 Gauss-Hermite 贝叶斯预测积分...")
    gh = GenHermiteQuadrature(alpha=0.0, a=0.0, b=1.0, n=16)
    nodes, weights = gh.compute_rule()
    print(f"  求积节点数: {len(nodes)}")


    post_mean = np.mean(series)
    post_std = np.std(series)
    pred_func = lambda theta: theta + 0.1 * np.sin(theta)
    m1, m2 = gh.predictive_moments(post_mean, post_std, pred_func)
    pred_var = m2 - m1**2
    print(f"  预测均值 E[f(θ)]: {m1:.4f}")
    print(f"  预测方差 Var[f(θ)]: {pred_var:.4f}")


    gh_std = GenHermiteQuadrature(alpha=0.0, a=0.0, b=1.0, n=20)
    gh_std.compute_rule()
    moment2 = gh_std.integrate(lambda x: x**2) / np.sqrt(np.pi)
    print(f"  二阶矩验证 (应为 0.5): {moment2:.6f}")




    print("\n[阶段 8] 时延嵌入几何质量分析与 FNN 维度估计...")
    geom = EmbeddingGeometry(embedding_dim=3, delay=2)
    X_embed = geom.delay_embed(series)
    print(f"  嵌入矩阵形状: {X_embed.shape}")

    embed_scores = geom.local_triangulation_quality(X_embed, k_neighbors=6)
    print(f"  几何异常得分范围: [{embed_scores.min():.4f}, {embed_scores.max():.4f}]")

    est_dim = geom.embedding_dimension_estimate(series, max_dim=8, threshold=0.1)
    print(f"  FNN 估计最小嵌入维度: {est_dim}")




    print("\n[阶段 9] 数值鲁棒性与 Monte Carlo 概率界...")
    nr = NumericalRobustness()


    x_test = 1.0
    x_next = nr.next_float(x_test)
    x_prev = nr.prev_float(x_test)
    print(f"  nextafter(1.0) = {x_next:.16e}, gap = {x_next - 1.0:.2e}")
    print(f"  prevfloat(1.0) = {x_prev:.16e}, gap = {1.0 - x_prev:.2e}")


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


    sample_cov = np.cov(X_embed.T)
    sens = nr.condition_number_sensitivity(sample_cov)
    print(f"  嵌入协方差条件数: {sens['condition_number']:.2e}")
    print(f"  双精度可解性: {sens['solvable_in_double_precision']}")


    mc_val = nr.ball_monte_carlo_integral(lambda x: np.exp(-np.sum(x**2)), dim=5, n_samples=20000)
    print(f"  5 维球 Monte Carlo 积分估计: {mc_val:.6f}")




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




    print("\n[阶段 12] 三角形对称求积 2D 特征提取...")
    tfi = TriangularFeatureIntegrator(order=5)


    verif = tfi.verify_monomial(2, 1)
    print(f"  单项式 x^2 y 积分验证: 精确={verif['exact']:.6f}, 数值={verif['numerical']:.6f}, 误差={verif['error']:.2e}")


    triplets = np.array([
        [series[i], series[i+1], series[i+2]]
        for i in range(0, len(series)-4, 3)
    ])
    tri_features = tfi.extract_triangular_features(triplets)
    print(f"  三角形积分特征范围: [{tri_features.min():.4f}, {tri_features.max():.4f}]")




    print("\n[阶段 13] 综合异常检测性能评估...")

    n = len(series)

    min_len = min(len(graph_scores_full), len(rbf_scores), len(embed_scores))

    embed_scores_full = np.zeros(n)
    embed_offset = (n - len(embed_scores)) // 2
    embed_scores_full[embed_offset:embed_offset + len(embed_scores)] = embed_scores

    if embed_offset > 0:
        embed_scores_full[:embed_offset] = embed_scores[0]
        embed_scores_full[embed_offset + len(embed_scores):] = embed_scores[-1]

    g_s = graph_scores_full
    r_s = rbf_scores
    e_s = embed_scores_full


    def zscore(x):
        return (x - np.mean(x)) / (np.std(x) + 1e-12)

    fused = 0.25 * zscore(g_s) + 0.35 * zscore(r_s) + 0.40 * zscore(e_s)
    fused = (fused - fused.min()) / (fused.max() - fused.min() + 1e-12)


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

    fprs_arr = np.array(fprs)
    tprs_arr = np.array(tprs)
    sort_idx = np.argsort(fprs_arr)
    auc = np.trapezoid(tprs_arr[sort_idx], fprs_arr[sort_idx])
    print(f"  AUC-ROC:   {auc:.4f}")




    elapsed = time.time() - start_time
    print("\n" + "=" * 80)
    print(f"全部计算完成，耗时: {elapsed:.2f} 秒")
    print("=" * 80)


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
