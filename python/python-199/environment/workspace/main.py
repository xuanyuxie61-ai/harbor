"""
main.py — 高性能计算：内存受限外排序算法
==========================================
统一入口，零参数运行。

科学问题：超大规模异构科学数据流的内存受限自适应外排序与
分布式重分布算法。

模拟场景：高能物理实验（LHC-like）产生的粒子事件数据流，
数据量（~10^5条记录）远超内存容量（~500条记录），需在外部
存储约束下完成全序排列，供后续物理分析使用。

完整流程：
    1. 生成异构科学数据集（蒙特卡洛 + ODE + 混沌动力学）
    2. Toeplitz变换预处理（内存受限矩阵运算）
    3. 自适应采样与分区估计（直方图 + RBF + FEM + 多边形采样）
    4. 外排序核心引擎（替换选择 + k路归并）
    5. 分区优化与约束满足（覆盖约束 + 素数哈希）
    6. 数值稳定性与误差分析（Euler误差 + Logistic分岔 + SIR稳定性）
    7. 数据流动力学预测（动脉PDE + Lax-Wendroff格式）
    8. 验证与性能评估
"""

import math
import random
import time
from typing import List, Tuple, Dict

from utils import (
    generate_primes, hash_family_seed, robust_division,
    entropy_of_distribution, kldivergence
)
from data_generator import (
    generate_heterogeneous_dataset, exact_circle_integral,
    SIRDataFlow, logistic_chaotic_sequence
)
from toeplitz_solver import R8LTTSolver, toeplitz_transform_keys
from adaptive_sampler import (
    HistogramPDFSampler, RBFInterpolator, FEM1DProjector,
    PolygonPartitionSampler, adaptive_partition_estimation
)
from external_sort_engine import (
    ReplacementSelection, KWayMerge,
    PiecewiseConstantPartition, ExternalSortPipeline,
    verify_sorted
)
from partition_optimizer import (
    CoverConstraintSolver, HashPartitioner,
    AdaptivePartitionOptimizer, optimal_page_alignment
)
from stability_analyzer import (
    EulerErrorAnalysis, LogisticBifurcationAnalyzer,
    SIRStabilityAnalyzer, SortStabilityMonitor
)
from stream_dynamics import (
    ArteryFlowModel, LaxWendroffBuffer, predict_optimal_buffer_size
)


def print_section(title: str):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def main():
    random.seed(199)

    # ============================================================
    # 参数配置
    # ============================================================
    TOTAL_RECORDS = 5000       # 总数据记录数
    MEMORY_CAPACITY = 300      # 内存容量（记录数）
    K_WAY = 4                  # 归并路数
    NUM_PARTITIONS = 8         # 分区数
    KEY_INDEX = 0              # 排序键索引
    PAGE_SIZE = 64             # 页大小（记录数）

    print("高性能计算：内存受限外排序算法")
    print(f"配置: N={TOTAL_RECORDS}, M={MEMORY_CAPACITY}, k={K_WAY}, P={NUM_PARTITIONS}")
    print(f"内存受限比: N/M = {TOTAL_RECORDS / MEMORY_CAPACITY:.2f}")

    # ============================================================
    # Phase 1: 异构科学数据生成
    # ============================================================
    print_section("Phase 1: 异构科学数据生成")

    t0 = time.time()
    dataset = generate_heterogeneous_dataset(
        total_records=TOTAL_RECORDS,
        memory_limit=MEMORY_CAPACITY,
        seed=199
    )
    t1 = time.time()
    print(f"  生成 {len(dataset)} 条记录，耗时 {t1 - t0:.4f} s")

    # 验证Gamma函数积分公式
    e1, e2 = 2, 2
    mc_val = exact_circle_integral(e1, e2)
    print(f"  圆周积分 I({e1},{e2}) = {mc_val:.6f} (解析解)")

    # TODO: Hole_3a — SIR 模型初始化与基本再生数计算
    # 提示：需确保参数与 stability_analyzer.SIRStabilityAnalyzer 一致
    sir = SIRDataFlow(alpha=0.3, beta=0.1, gamma=0.05, N=1000.0)
    # TODO: 调用 basic_reproduction_number() 并输出结果
    pass

    # Logistic混沌验证
    logistic_seq = logistic_chaotic_sequence(1000, r=3.9, x0=0.314159)
    mean_log = sum(logistic_seq) / len(logistic_seq)
    print(f"  Logistic序列均值 = {mean_log:.4f} (理论期望 ≈ 0.5)")

    # ============================================================
    # Phase 2: Toeplitz变换预处理
    # ============================================================
    print_section("Phase 2: Toeplitz变换预处理")

    raw_keys = [rec[KEY_INDEX] for rec in dataset]
    transformed_keys = toeplitz_transform_keys(raw_keys, decay=0.95)
    print(f"  原始键值范围: [{min(raw_keys):.4f}, {max(raw_keys):.4f}]")
    print(f"  Toeplitz变换后范围: [{min(transformed_keys):.4f}, {max(transformed_keys):.4f}]")

    # 构造前缀和Toeplitz矩阵并求解
    n_test = min(50, len(raw_keys))
    test_vec = raw_keys[:n_test]
    solver = R8LTTSolver([0.95 ** i for i in range(n_test)])
    matvec_result = solver.matvec(test_vec)
    print(f"  Toeplitz矩阵-向量乘法示例 (前3项): {matvec_result[:3]}")

    # ============================================================
    # Phase 3: 自适应采样与分区估计
    # ============================================================
    print_section("Phase 3: 自适应采样与分区估计")

    sample_keys = random.sample(raw_keys, min(1000, len(raw_keys)))

    # 直方图分位数
    hist = HistogramPDFSampler(sample_keys, num_bins=128)
    q25 = hist.quantile(0.25)
    q50 = hist.quantile(0.50)
    q75 = hist.quantile(0.75)
    print(f"  直方图分位数: Q25={q25:.4f}, Q50={q50:.4f}, Q75={q75:.4f}")

    # RBF插值平滑
    x_rbf = [hist.x_min + i * (hist.x_max - hist.x_min) / 30.0 for i in range(31)]
    y_rbf = [hist.quantile(i / 30.0) for i in range(31)]
    rbf = RBFInterpolator(x_rbf, y_rbf, r0=(hist.x_max - hist.x_min) / 8.0, kernel_type=1)
    rbf_q50 = rbf.evaluate(q50)
    print(f"  RBF插值验证 @ Q50: {rbf_q50:.4f}")

    # FEM投影
    fem_nodes = [hist.x_min + i * (hist.x_max - hist.x_min) / 20.0 for i in range(21)]
    fem = FEM1DProjector(fem_nodes)

    def cdf_approx(x: float) -> float:
        return (x - hist.x_min) / (hist.x_max - hist.x_min) if hist.x_max > hist.x_min else 0.5

    fem_proj = fem.project(cdf_approx)
    print(f"  FEM投影节点值范围: [{min(fem_proj):.4f}, {max(fem_proj):.4f}]")

    # 多边形采样（二维键空间模拟）
    poly = PolygonPartitionSampler([
        (0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)
    ])
    poly_samples = poly.sample(10, seed=42)
    print(f"  多边形均匀采样 (前3点): {poly_samples[:3]}")

    # 综合自适应边界估计
    boundaries = adaptive_partition_estimation(sample_keys, NUM_PARTITIONS, use_rbf_refinement=True)
    print(f"  自适应分区边界: {[f'{b:.4f}' for b in boundaries]}")

    # ============================================================
    # Phase 4: 外排序核心引擎
    # ============================================================
    print_section("Phase 4: 外排序核心引擎")

    pipeline = ExternalSortPipeline(
        memory_capacity=MEMORY_CAPACITY,
        k_way=K_WAY,
        key_index=KEY_INDEX
    )

    t2 = time.time()
    sorted_data = pipeline.sort(dataset)
    t3 = time.time()

    is_sorted = verify_sorted(sorted_data, KEY_INDEX)
    print(f"  排序验证: {'通过' if is_sorted else '失败'}")
    print(f"  排序耗时: {t3 - t2:.4f} s")
    print(f"  模拟I/O次数: {pipeline.io_count}")

    # 理论I/O下界
    theoretical_io = pipeline.theoretical_io_cost(TOTAL_RECORDS, PAGE_SIZE)
    print(f"  理论I/O下界 (Aggarwal-Vitter): {theoretical_io:.2f}")

    # ============================================================
    # Phase 5: 分区优化与约束满足
    # ============================================================
    print_section("Phase 5: 分区优化与约束满足")

    # 分段常数分区
    pwc = PiecewiseConstantPartition(boundaries)
    partitions = pwc.partition_records(sorted_data, KEY_INDEX)
    partition_sizes = [len(p) for p in partitions]
    print(f"  各分区大小: {partition_sizes}")
    print(f"  分区熵: {entropy_of_distribution([s / sum(partition_sizes) for s in partition_sizes]):.4f}")

    # 哈希分区
    hasher = HashPartitioner(NUM_PARTITIONS, prime_seed_idx=7)
    hash_parts = hasher.partition(sorted_data, KEY_INDEX)
    hash_sizes = [len(p) for p in hash_parts]
    print(f"  哈希分区大小: {hash_sizes}")

    # 覆盖约束求解
    solver_cc = CoverConstraintSolver(NUM_PARTITIONS, MEMORY_CAPACITY)
    assignment = solver_cc.greedy_assign(partition_sizes)
    balance = solver_cc.balance_score(partition_sizes, assignment)
    print(f"  贪心分配均衡评分: {balance:.4f}")

    # 页对齐优化
    opt_block = optimal_page_alignment(TOTAL_RECORDS, MEMORY_CAPACITY, PAGE_SIZE)
    print(f"  最优页对齐块大小: {opt_block} 记录")

    # ============================================================
    # Phase 6: 数值稳定性与误差分析
    # ============================================================
    print_section("Phase 6: 数值稳定性与误差分析")

    # Euler截断误差分析
    def f_artery(t: float, y: float) -> float:
        return -0.5 * y + math.sin(t)
    def df_dt(t: float, y: float) -> float:
        return math.cos(t)
    def df_dy(t: float, y: float) -> float:
        return -0.5

    euler_analyzer = EulerErrorAnalysis(f_artery, df_dt, df_dy, y0=1.0, t0=0.0, t_end=10.0)
    lte = euler_analyzer.local_truncation_error(h=0.01, t=1.0, y=0.8)
    geb = euler_analyzer.global_error_bound(h=0.01, L=0.5, M2=2.0)
    rec_h = euler_analyzer.recommend_step(target_error=1e-4, L=0.5, M2=2.0)
    print(f"  Euler LTE (h=0.01): {lte:.6e}")
    print(f"  Euler 全局误差上界: {geb:.6e}")
    print(f"  推荐步长 (target=1e-4): {rec_h:.6f}")

    # Logistic分岔分析
    logistic_analyzer = LogisticBifurcationAnalyzer(r=3.9, x0=0.5)
    lyap = logistic_analyzer.lyapunov_exponent(n_iter=2000)
    chaotic = logistic_analyzer.is_chaotic()
    print(f"  Logistic Lyapunov指数 (r=3.9): {lyap:.4f}")
    print(f"  混沌判定: {'是' if chaotic else '否'}")

    # TODO: Hole_3b — SIR 稳定性分析与平衡点计算
    # 提示：参数需与上方 SIRDataFlow 保持一致
    sir_stab = SIRStabilityAnalyzer(alpha=0.3, beta=0.1, gamma=0.05, N=1000.0)
    # TODO: 调用 reproduction_number() 和 endemic_equilibrium() 并输出结果
    pass

    # 排序稳定性监控
    monitor = SortStabilityMonitor(NUM_PARTITIONS, MEMORY_CAPACITY)
    runs = ReplacementSelection(MEMORY_CAPACITY).generate_runs(dataset, KEY_INDEX)
    run_lengths = [len(r) for r in runs]
    diag = monitor.diagnose(partition_sizes, run_lengths, predicted_peak=MEMORY_CAPACITY * 1.2)
    print(f"  稳定性诊断: {diag}")

    # ============================================================
    # Phase 7: 数据流动力学预测
    # ============================================================
    print_section("Phase 7: 数据流动力学预测")

    # 动脉PDE模型
    artery = ArteryFlowModel(
        alpha=2.0, beta=0.8, gamma=1.0,
        a=1.0, b=0.5, omega=1.5, x=1.0, dp_dx=1.0
    )
    t_vals, u_vals, v_vals = artery.simulate_euler(
        u0=0.1, v0=0.0, t_end=20.0, n_steps=200
    )
    amp = artery.analytical_amplitude()
    print(f"  动脉模型稳态振幅: {amp:.4f}")
    print(f"  缓冲区占用终值: {u_vals[-1]:.4f}")

    # 最优缓冲区预测
    opt_buffer = predict_optimal_buffer_size(
        alpha=2.0, beta=0.8, gamma=1.0, omega=1.5, safety_factor=1.5
    )
    print(f"  预测最优缓冲区大小: {opt_buffer:.2f}")

    # Lax-Wendroff数据流模拟
    nx = 20
    lw = LaxWendroffBuffer(nx=nx, c=0.5, dx=0.1, dt=0.05)
    rho0 = [0.5 + 0.3 * math.sin(2.0 * math.pi * i / nx) for i in range(nx)]
    history = lw.simulate(rho0, n_steps=50)
    final_total = sum(history[-1])
    initial_total = sum(history[0])
    print(f"  Lax-Wendroff守恒验证: 初始总量={initial_total:.4f}, 终了总量={final_total:.4f}")
    print(f"  Courant数: {lw.compute_courant_number():.4f}")

    # ============================================================
    # Phase 8: 综合评估与总结
    # ============================================================
    print_section("Phase 8: 综合评估与总结")

    total_time = t3 - t0
    throughput = TOTAL_RECORDS / total_time if total_time > 0 else 0.0

    print(f"  总记录数: {TOTAL_RECORDS}")
    print(f"  内存容量: {MEMORY_CAPACITY} 记录")
    print(f"  排序键索引: {KEY_INDEX}")
    print(f"  排序正确性: {'通过' if is_sorted else '失败'}")
    print(f"  归并段数: {len(runs)}")
    print(f"  理论平均段长: ~{2 * MEMORY_CAPACITY} (替换选择)")
    print(f"  实际平均段长: {sum(run_lengths) / len(run_lengths):.1f}")
    print(f"  模拟I/O次数: {pipeline.io_count}")
    print(f"  理论I/O下界: {theoretical_io:.2f}")
    print(f"  分区倾斜度: {monitor.partition_skew(partition_sizes):.4f}")
    print(f"  总耗时: {total_time:.4f} s")
    print(f"  吞吐量: {throughput:.1f} records/s")

    # 输出前5条和后5条记录验证
    print(f"\n  排序后前5条记录 (复合键):")
    for rec in sorted_data[:5]:
        print(f"    key={rec[KEY_INDEX]:.6f}, ts={rec[1]:.4f}, E={rec[5]:.4f}")
    print(f"  排序后后5条记录 (复合键):")
    for rec in sorted_data[-5:]:
        print(f"    key={rec[KEY_INDEX]:.6f}, ts={rec[1]:.4f}, E={rec[5]:.4f}")

    print("\n" + "=" * 60)
    print("  外排序任务完成。")
    print("=" * 60)

    return {
        "sorted": is_sorted,
        "records": TOTAL_RECORDS,
        "memory": MEMORY_CAPACITY,
        "time": total_time,
        "io_count": pipeline.io_count,
        "partitions": partition_sizes,
        "runs": len(runs)
    }


if __name__ == "__main__":
    result = main()
