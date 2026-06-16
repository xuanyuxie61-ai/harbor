"""
mc_replicas.py — Monte Carlo 副本与 k-means 聚类 (映射自 kmeans_fast)
=============================================================
本模块实现 PDF 不确定度的 Monte Carlo 方法:
  (1) MC 副本生成与拟合;
  (2) PDF 集合的 k-means 聚类 (映射自 kmeans_fast);
  (3) 聚类中心的统计意义分析;
  (4) MC 不确定度与 Hessian 不确定度的比较.

映射自 kmeans_fast (Elkan 加速 k-means):
  原始: 使用三角不等式加速最近中心查找
  本模块: 在 PDF 参数空间中, 对 MC 副本进行聚类

核心公式 (MC 副本):
    d_i^{(r)} = d_i + σ_i × N_r(0,1)
    对每个副本 r, 拟合得到 p^{(r)}, 然后:
    f_MC(x) = (1/N_rep) Σ_r f(x; p^{(r)})
    Δf_MC(x) = √[(1/N_rep) Σ_r (f(x;p^{(r)}) − f_MC(x))²]

核心公式 (k-means 目标):
    J = Σ_{i=1}^N min_k ||x_i − c_k||²
    Elkan 加速: 利用三角不等式跳过不必要的距离计算
        若 d(x_i, c_{assigned}) ≤ 0.5 × min_{j≠assigned} d(c_j, c_k),
        则 x_i 仍属于 c_{assigned}.

核心公式 (Elkan 三角不等式加速):
    d(x, c_j) ≥ d(x, c_k) − d(c_k, c_j)
    → 若 d(x, c_k) < 0.5 d(c_k, c_j), x 不可能属于 c_j
"""
from __future__ import annotations
import math
import random
from typing import Dict, List, Tuple, Optional

from phys_consts import EPS_NUMERICAL
from pdf_param import default_params, params_to_vector, vector_to_params, evaluate_all_flavors, normalize_parameters
from pdf_fit import fit_pdf_params
from experimental_data import DataSet, mc_replica_generation


# ============================================================
# 1. MC 副本拟合
# ============================================================
def fit_mc_replicas(
    base_data: DataSet,
    n_replicas: int = 5,
    init_params: Optional[Dict[str, float]] = None,
    fit_max_iter: int = 20,
    base_seed: int = 1000,
    x_grid: Optional[List[float]] = None,
    verbose: bool = False
) -> List[Dict[str, float]]:
    """
    生成并拟合 MC 副本:
      1) 对 r = 1, ..., N_rep:
         a) 生成副本 d^{(r)} = d + σ × N(0,1)
         b) 拟合得到 p^{(r)}

    参数:
        base_data:    原始数据集
        n_replicas:   副本数
        init_params:  拟合初始参数
        fit_max_iter: 每个副本的最大拟合迭代
        base_seed:    基础随机种子
        x_grid:       求和规则网格
        verbose:      打印进度
    返回:
        拟合后的参数列表 [p^{(1)}, p^{(2)}, ...]
    """
    if init_params is None:
        init_params = default_params()

    fitted_params = []
    for r in range(n_replicas):
        seed = base_seed + r * 137
        replica_data = mc_replica_generation(base_data, seed=seed)
        if verbose:
            print(f"  MC 副本 {r + 1}/{n_replicas}: 拟合中...")
        p_fit, chi2_hist, n_iter = fit_pdf_params(
            replica_data, init_params=dict(init_params),
            max_iter=fit_max_iter, x_grid=x_grid, verbose=False)
        fitted_params.append(p_fit)
        if verbose:
            chi2_final = chi2_hist[-1] if chi2_hist else float('nan')
            print(f"    → χ² = {chi2_final:.4f}, 迭代 {n_iter} 次")

    return fitted_params


# ============================================================
# 2. MC PDF 不确定度
# ============================================================
def mc_pdf_uncertainty(
    fitted_params_list: List[Dict[str, float]],
    x_grid: List[float],
    flavor: str = 'g'
) -> Tuple[List[float], List[float], List[float]]:
    """
    计算 MC PDF 不确定度:
        f_mean(x) = (1/N) Σ_r f(x; p^{(r)})
        Δf_MC(x) = √[(1/N) Σ_r (f(x;p^{(r)}) − f_mean(x))²]

    返回:
        (f_mean, f_upper, f_lower)
    """
    n_rep = len(fitted_params_list)
    n_x = len(x_grid)
    flavor_idx = {'uv': 0, 'dv': 1, 'g': 2, 's': 3}.get(flavor, 2)

    # 收集所有 PDF 值
    f_matrix = []
    for params in fitted_params_list:
        f_row = []
        for x in x_grid:
            vals = evaluate_all_flavors(x, params)
            f_row.append(vals[flavor_idx])
        f_matrix.append(f_row)

    # 计算均值与标准差
    f_mean = [0.0] * n_x
    f_std_sq = [0.0] * n_x

    for i in range(n_x):
        vals = [f_matrix[r][i] for r in range(n_rep)]
        mean = sum(vals) / n_rep
        f_mean[i] = mean
        var = sum((v - mean) ** 2 for v in vals) / max(n_rep - 1, 1)
        f_std_sq[i] = var

    f_upper = [f_mean[i] + math.sqrt(f_std_sq[i]) for i in range(n_x)]
    f_lower = [f_mean[i] - math.sqrt(f_std_sq[i]) for i in range(n_x)]
    return f_mean, f_upper, f_lower


# ============================================================
# 3. k-means 聚类 (映射自 kmeans_fast)
# ============================================================
def euclidean_distance(a: List[float], b: List[float]) -> float:
    """欧氏距离"""
    return math.sqrt(sum((ai - bi) ** 2 for ai, bi in zip(a, b)))


def kmeans_clustering(
    data_vectors: List[List[float]], k: int,
    max_iter: int = 50, seed: int = 42,
    use_elkan: bool = True
) -> Tuple[List[List[float]], List[int], float]:
    """
    k-means 聚类 (映射自 kmeans_fast, Elkan 加速):

    算法:
      1) 初始化: 随机选 k 个中心 (或 kmeans++ 初始化)
      2) E步: 将每个点分配到最近中心
      3) M步: 重新计算中心 = 聚类均值
      4) 重复直到收敛或达到 max_iter

    Elkan 加速 (method=2):
      - 维护每个点到其分配中心的距离上界;
      - 维护中心间距离 nndist[j] = 0.5 min_k d(c_j, c_k);
      - 若 d(x_i, c_assigned) ≤ nndist[assigned], 跳过重新分配.

    参数:
        data_vectors: N 个 M 维数据点
        k:            聚类数
        max_iter:     最大迭代
        seed:         随机种子
        use_elkan:    是否使用 Elkan 加速
    返回:
        (centers, assignments, inertia)
    """
    rng = random.Random(seed)
    n = len(data_vectors)
    if n == 0 or k <= 0:
        return [], [], 0.0
    m = len(data_vectors[0])
    k = min(k, n)

    # 初始化: kmeans++ 风格
    centers = [list(data_vectors[rng.randint(0, n - 1)])]
    for _ in range(1, k):
        # 选择距已有中心最远的点
        dists = [min(euclidean_distance(x, c) for c in centers) for x in data_vectors]
        total = sum(d * d for d in dists)
        if total < EPS_NUMERICAL:
            centers.append(list(data_vectors[rng.randint(0, n - 1)]))
            continue
        r = rng.random() * total
        cum = 0.0
        for i, d in enumerate(dists):
            cum += d * d
            if cum >= r:
                centers.append(list(data_vectors[i]))
                break
        else:
            centers.append(list(data_vectors[-1]))

    assignments = [0] * n
    for iteration in range(max_iter):
        # E步
        changed = 0

        if use_elkan and k > 1:
            # Elkan 加速: 计算中心间距离
            cent_dist = [[0.0] * k for _ in range(k)]
            for j1 in range(k):
                for j2 in range(j1 + 1, k):
                    d = euclidean_distance(centers[j1], centers[j2])
                    cent_dist[j1][j2] = d
                    cent_dist[j2][j1] = d

            nndist = [min(cent_dist[j][kk] for kk in range(k) if kk != j) / 2.0
                      for j in range(k)]

        for i in range(n):
            if use_elkan and k > 1:
                # Elkan 三角不等式检查
                assigned = assignments[i]
                d_assigned = euclidean_distance(data_vectors[i], centers[assigned])
                if d_assigned <= nndist[assigned]:
                    continue  # 跳过: 不可能改变

            # 完整距离计算
            min_dist = float('inf')
            min_center = 0
            for j in range(k):
                d = euclidean_distance(data_vectors[i], centers[j])
                if d < min_dist:
                    min_dist = d
                    min_center = j
            if assignments[i] != min_center:
                assignments[i] = min_center
                changed += 1

        # M步
        new_centers = [[0.0] * m for _ in range(k)]
        counts = [0] * k
        for i in range(n):
            j = assignments[i]
            counts[j] += 1
            for d in range(m):
                new_centers[j][d] += data_vectors[i][d]
        for j in range(k):
            if counts[j] > 0:
                for d in range(m):
                    new_centers[j][d] /= counts[j]
            else:
                new_centers[j] = list(centers[j])

        centers = new_centers
        if changed == 0:
            break

    # 计算惯性 (inertia = Σ ||x_i − c_{assigned}||²)
    inertia = 0.0
    for i in range(n):
        d = euclidean_distance(data_vectors[i], centers[assignments[i]])
        inertia += d * d

    return centers, assignments, inertia


# ============================================================
# 4. PDF 副本聚类分析
# ============================================================
def cluster_pdf_replicas(
    fitted_params_list: List[Dict[str, float]],
    k: int = 2
) -> Tuple[List[List[float]], List[int], float, Dict]:
    """
    对 MC 拟合的 PDF 参数进行 k-means 聚类:

    1) 将每个参数集转换为向量;
    2) 标准化 (减均值, 除标准差);
    3) 使用 k-means 聚类;
    4) 分析聚类结构.

    返回:
        (centers, assignments, inertia, stats)
    """
    n_rep = len(fitted_params_list)
    if n_rep == 0:
        return [], [], 0.0, {}

    # 转换为向量
    vectors = [params_to_vector(p) for p in fitted_params_list]
    m = len(vectors[0])

    # 标准化
    means = [0.0] * m
    stds = [0.0] * m
    for d in range(m):
        vals = [v[d] for v in vectors]
        means[d] = sum(vals) / n_rep
        stds[d] = math.sqrt(sum((v - means[d]) ** 2 for v in vals) / max(n_rep - 1, 1))
        if stds[d] < EPS_NUMERICAL:
            stds[d] = 1.0

    vectors_norm = [[(vectors[i][d] - means[d]) / stds[d] for d in range(m)]
                    for i in range(n_rep)]

    # k-means
    centers, assignments, inertia = kmeans_clustering(vectors_norm, k=k)

    stats = {
        'n_replicas': n_rep,
        'n_clusters': k,
        'inertia': inertia,
        'cluster_sizes': [assignments.count(j) for j in range(k)],
    }
    return centers, assignments, inertia, stats
