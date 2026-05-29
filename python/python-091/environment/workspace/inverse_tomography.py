"""
超声断层反演与整数精确线性代数模块

基于种子项目 569_i4mat_rref2 和 668_levenshtein_distance 的核心算法，
为超声层析成像提供介质参数反演、精确秩分析和序列比对功能。

物理模型:
超声层析反演的目标是从测量数据重建介质的声速分布 c(x,y)。

射线理论近似（高频极限）:
    T = ∫_L 1/c(x,y) ds
其中 T 为沿射线 L 的传播时间，ds 为弧长微元。

线性化反演:
令 m(x,y) = 1/c(x,y) - 1/c₀ 为慢度扰动，则:
    ΔT_i = Σ_j A_{ij} · m_j
其中 A_{ij} 为第 i 条射线穿过第 j 个像素的 path length。

离散化后得到线性系统:
    A·m = Δt
其中 A (M×N) 为投影矩阵，M 为射线数，N 为像素数。

整数精确线性代数:
当投影矩阵 A 为整数矩阵（像素path length以整数格点表示）时，
使用整数行最简形（IRREF）可避免浮点舍入误差，精确判定矩阵秩
和欠定性。

Levenshtein距离:
用于比较不同扫描位置获取的A-scan回波序列的相似性，
识别组织类型的变化。
"""

import numpy as np
from typing import Tuple, List, Dict


def i4mat_rref2(A: np.ndarray, max_iter: int = 10000) -> Tuple[np.ndarray, int]:
    """计算整数矩阵的行最简形（Integer Reduced Row Echelon Form, IRREF）。

    算法步骤:
    1. 选主元：在当前列中寻找非零元素
    2. 行交换：将主元行移到当前行
    3. 符号标准化：使主元为正
    4. GCD约分：提取整行公因子
    5. 消元：用整数运算消去同列其他元素

    与浮点RREF不同，IRREF仅使用整数运算（加、减、GCD），
    完全避免舍入误差，适合精确线性代数分析。

    参数:
        A: 整数矩阵
        max_iter: 最大迭代次数（数值鲁棒性保护）

    返回:
        A_rref: 行最简形矩阵
        rank: 矩阵秩
    """
    A_work = A.copy().astype(np.int64)
    m, n = A_work.shape
    rank = 0

    for col in range(n):
        if rank >= m:
            break

        # 寻找主元
        pivot_row = -1
        for row in range(rank, m):
            if A_work[row, col] != 0:
                pivot_row = row
                break

        if pivot_row == -1:
            continue

        # 行交换
        if pivot_row != rank:
            A_work[[pivot_row, rank]] = A_work[[rank, pivot_row]]

        # 符号标准化
        if A_work[rank, col] < 0:
            A_work[rank] = -A_work[rank]

        # GCD约分（提取整行公因子）
        row_vals = A_work[rank]
        nonzero_vals = row_vals[row_vals != 0]
        if len(nonzero_vals) > 0:
            g = np.gcd.reduce(np.abs(nonzero_vals))
            if g > 1:
                A_work[rank] = A_work[rank] // g

        # 消去同列其他元素
        for row in range(m):
            if row != rank and A_work[row, col] != 0:
                # 计算消元系数
                factor = A_work[row, col]
                pivot = A_work[rank, col]

                # 使用lcm避免分数
                lcm_val = np.lcm(abs(factor), abs(pivot))
                if lcm_val == 0:
                    continue

                row_mult = lcm_val // abs(factor)
                pivot_mult = lcm_val // abs(pivot)

                if factor * pivot < 0:
                    pivot_mult = -pivot_mult

                A_work[row] = row_mult * A_work[row] - pivot_mult * A_work[rank]

                # 再次GCD约分
                nonzero_vals = A_work[row][A_work[row] != 0]
                if len(nonzero_vals) > 0:
                    g = np.gcd.reduce(np.abs(nonzero_vals))
                    if g > 1:
                        A_work[row] = A_work[row] // g

        rank += 1

    return A_work, rank


def levenshtein_distance(s1: List, s2: List) -> int:
    """计算两个序列的Levenshtein编辑距离。

    动态规划递推:
        D[i,j] = min(
            D[i-1,j] + 1,      (删除)
            D[i,j-1] + 1,      (插入)
            D[i-1,j-1] + cost  (替换，cost=0 if s1[i-1]==s2[j-1] else 1)
        )

    时间复杂度: O(|s1|·|s2|)
    空间复杂度: O(|s2|)（滚动数组优化）

    参数:
        s1, s2: 输入序列（可为字符串、列表或数组）

    返回:
        distance: 编辑距离
    """
    m, n = len(s1), len(s2)

    if m == 0:
        return n
    if n == 0:
        return m

    # 使用滚动数组优化空间
    prev = np.arange(n + 1, dtype=int)
    curr = np.zeros(n + 1, dtype=int)

    for i in range(1, m + 1):
        curr[0] = i
        for j in range(1, n + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            curr[j] = min(
                prev[j] + 1,      # 删除
                curr[j - 1] + 1,  # 插入
                prev[j - 1] + cost # 替换
            )
        prev, curr = curr, prev

    return int(prev[n])


def build_projection_matrix(n_rays: int, n_pixels_x: int, n_pixels_y: int,
                            domain_size: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
    """构建超声层析反演的投影矩阵。

    使用射线理论，将每个像素内的路径长度作为矩阵元素。
    为使用整数精确代数，将路径长度量化为整数格点。

    参数:
        n_rays: 每个方向的射线数
        n_pixels_x, n_pixels_y: 像素网格尺寸
        domain_size: 模拟域边长 (m)

    返回:
        A: (M, N) 投影矩阵，M = 2*n_rays（水平和垂直方向）
        ray_angles: (M,) 射线角度
    """
    pixel_size = domain_size / max(n_pixels_x, n_pixels_y)
    n_pixels = n_pixels_x * n_pixels_y
    M = 2 * n_rays

    A = np.zeros((M, n_pixels), dtype=int)
    ray_angles = np.zeros(M)

    # 水平射线（平行于x轴）
    for r in range(n_rays):
        y_ray = domain_size * (r + 0.5) / n_rays

        for px in range(n_pixels_x):
            for py in range(n_pixels_y):
                pixel_y_min = py * domain_size / n_pixels_y
                pixel_y_max = (py + 1) * domain_size / n_pixels_y

                if pixel_y_min <= y_ray < pixel_y_max:
                    pixel_idx = py * n_pixels_x + px
                    # 路径长度量化为整数
                    A[r, pixel_idx] = int(domain_size / (n_pixels_x * pixel_size))

        ray_angles[r] = 0.0

    # 垂直射线（平行于y轴）
    for r in range(n_rays):
        x_ray = domain_size * (r + 0.5) / n_rays

        for px in range(n_pixels_x):
            for py in range(n_pixels_y):
                pixel_x_min = px * domain_size / n_pixels_x
                pixel_x_max = (px + 1) * domain_size / n_pixels_x

                if pixel_x_min <= x_ray < pixel_x_max:
                    pixel_idx = py * n_pixels_x + px
                    A[n_rays + r, pixel_idx] = int(domain_size / (n_pixels_y * pixel_size))

        ray_angles[n_rays + r] = np.pi / 2.0

    return A, ray_angles


def solve_tomography_svd(A: np.ndarray, travel_times: np.ndarray,
                         regularization: float = 1e-4) -> np.ndarray:
    """使用SVD正则化求解超声层析反演问题。

    最小二乘问题:
        min ||A·m - Δt||² + λ²·||m||²

    SVD解:
        m = V·diag(σ_i / (σ_i² + λ²))·Uᵀ·Δt

    参数:
        A: (M, N) 投影矩阵
        travel_times: (M,) 传播时间测量值
        regularization: Tikhonov正则化参数 λ

    返回:
        m: (N,) 慢度扰动重建
    """
    # SVD分解
    U, s, Vt = np.linalg.svd(A.astype(float), full_matrices=False)

    # 滤波因子（Tikhonov正则化）
    filter_factors = s / (s**2 + regularization**2)

    # 伪逆求解
    m = Vt.T @ (filter_factors * (U.T @ travel_times))

    return m


def analyze_system_identifiability(A: np.ndarray) -> Dict:
    """分析反演系统的可识别性（Identifiability）。

    使用整数RREF精确分析矩阵的秩和零空间结构，
    判断哪些像素参数可以被唯一确定。

    参数:
        A: 投影矩阵

    返回:
        分析结果字典
    """
    # 整数RREF分析
    A_rref, rank = i4mat_rref2(A)

    m, n = A.shape
    nullity = n - rank

    # 浮点SVD辅助分析
    A_float = A.astype(float)
    U, s, Vt = np.linalg.svd(A_float, full_matrices=False)

    # 条件数
    nonzero_singular = s[s > 1e-10]
    if len(nonzero_singular) > 0:
        condition_number = nonzero_singular[0] / nonzero_singular[-1]
    else:
        condition_number = np.inf

    return {
        'matrix_shape': (m, n),
        'rank': rank,
        'nullity': nullity,
        'is_overdetermined': m > n,
        'is_underdetermined': m < n,
        'condition_number': float(condition_number),
        'n_nonzero_singular': len(nonzero_singular),
        'identifiability_ratio': float(rank / n) if n > 0 else 0.0
    }


def classify_tissue_sequences(scan_sequences: List[List[float]],
                              reference_sequences: List[List[float]],
                              labels: List[str]) -> Dict:
    """使用Levenshtein距离对超声A-scan序列进行组织分类。

    将连续的A-scan信号量化为符号序列，然后计算与参考序列的编辑距离，
    最小距离对应的参考类别即为分类结果。

    参数:
        scan_sequences: 待分类的A-scan符号序列列表
        reference_sequences: 参考符号序列列表
        labels: 参考序列对应的组织标签

    返回:
        分类结果字典
    """
    if len(reference_sequences) != len(labels):
        raise ValueError("参考序列数与标签数必须相同")

    results = []

    for scan in scan_sequences:
        distances = []
        for ref in reference_sequences:
            # 将浮点序列量化为整数符号（用于编辑距离计算）
            scan_quantized = [int(round(x * 10)) for x in scan]
            ref_quantized = [int(round(x * 10)) for x in ref]
            dist = levenshtein_distance(scan_quantized, ref_quantized)
            distances.append(dist)

        min_idx = int(np.argmin(distances))
        results.append({
            'predicted_label': labels[min_idx],
            'min_distance': distances[min_idx],
            'all_distances': distances
        })

    return {
        'n_scans': len(scan_sequences),
        'n_classes': len(labels),
        'classifications': results
    }
