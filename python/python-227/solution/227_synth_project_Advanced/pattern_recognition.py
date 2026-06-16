"""
pattern_recognition.py — 径迹模式识别与种子生成
=================================================

融合种子项目:
    [669_levenshtein_matrix]   : Levenshtein 编辑距离矩阵
    [158_change_polynomial]    : 多项式组合计数 → Hough 变换投票
    [1122_PepHiRe]            : Ladderpath 层次分解 → 径迹拓扑分析

物理问题:
    给定一组探测器击中点，识别出属于同一粒子径迹的子集

    主要挑战:
    1. 组合爆炸: N 个击中中可能的径迹候选数为 C(N, k)
    2. 假阳性: 鬼击中、噪声击中、δ射线
    3. 效率: 需要找到所有真实径迹

    方法:
    1. 种子生成: 从 3 个击中构建初始径迹种子
    2. 种子扩展: 通过编辑距离匹配扩展种子
    3. 层次分解: 将复杂径迹拓扑分解为基本单元
"""

import math
import itertools
from typing import List, Tuple, Dict, Optional


# ============================================================
# [669_levenshtein_matrix] 编辑距离 → 击中模式匹配
# ============================================================
def levenshtein_distance_matrix(seq_a, seq_b):
    """
    基于 [669_levenshtein_matrix] 的 Levenshtein 距离矩阵

    原算法:
        D(i+1, j+1) = min(
            D(i, j+1) + 1,          # 删除
            D(i+1, j) + 1,          # 插入
            D(i, j) + cost          # 替换
        )
        其中 cost = 0 if s[i] == t[j] else 1

    映射到径迹识别:
    将每层的击中模式编码为序列:
        seq = [hit_id_layer0, hit_id_layer1, ...]

    两条候选径迹的"编辑距离"衡量它们的拓扑相似度:
    - 距离 0 = 完全相同的层命中模式
    - 距离 1 = 差一层 (可能丢失一个击中或合并)
    - 距离 > 2 = 不太可能是同一条径迹

    Parameters
    ----------
    seq_a : list
        序列 A (例如: 层击中 ID 列表)
    seq_b : list
        序列 B

    Returns
    -------
    list of list : 完整的距离矩阵 D
    """
    m = len(seq_a)
    n = len(seq_b)

    # 初始化 (m+1) × (n+1) 矩阵
    D = [[0] * (n + 1) for _ in range(m + 1)]

    # 边界条件 (与 [669] 一致)
    for i in range(m + 1):
        D[i][0] = i  # 删除代价
    for j in range(n + 1):
        D[0][j] = j  # 插入代价

    # DP 递推 (与 [669] 完全一致)
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            # 替换代价
            if seq_a[i - 1] == seq_b[j - 1]:
                sub_cost = 0
            else:
                sub_cost = 1

            D[i][j] = min(
                D[i - 1][j] + 1,        # 删除
                D[i][j - 1] + 1,        # 插入
                D[i - 1][j - 1] + sub_cost  # 替换/匹配
            )

    return D


def track_pattern_similarity(track_a_hits, track_b_hits, n_layers):
    """
    计算两条径迹候选的层命中模式相似度

    将每层的命中情况编码为二进制字符串:
        '1' = 该层有击中
        '0' = 该层无击中

    然后计算 Levenshtein 距离

    Parameters
    ----------
    track_a_hits : set of int
        径迹 A 命中的层编号集合
    track_b_hits : set of int
        径迹 B 命中的层编号集合
    n_layers : int
        总层数

    Returns
    -------
    dict : {
        'distance': int,
        'similarity': float,  # 1 - distance/max_len
        'match_layers': list,
    }
    """
    # 编码为序列
    seq_a = ['1' if i in track_a_hits else '0' for i in range(n_layers)]
    seq_b = ['1' if i in track_b_hits else '0' for i in range(n_layers)]

    D = levenshtein_distance_matrix(seq_a, seq_b)
    distance = D[len(seq_a)][len(seq_b)]

    max_len = max(len(seq_a), len(seq_b))
    similarity = 1.0 - distance / max_len if max_len > 0 else 1.0

    # 共同命中层
    match_layers = sorted(track_a_hits & track_b_hits)

    return {
        'distance': distance,
        'similarity': similarity,
        'match_layers': match_layers,
        'n_common': len(match_layers),
    }


# ============================================================
# [158_change_polynomial] 多项式组合 → 种子计数
# ============================================================
def count_seed_combinations(layer_hit_counts, n_seed_layers=3):
    """
    基于 [158_change_polynomial] 的多项式乘法计数种子组合

    原算法:
        用多项式 p(x) = x^{v1} + x^{v2} + ... 编码硬币面值
        p(x)^n 的系数给出组合数

    映射到种子生成:
    给定每层的击中数 [h_0, h_1, ..., h_{N-1}]
    选择 3 层 (n_seed_layers) 的所有可能种子数:

    用生成函数:
        G(x) = Π_{i=0}^{N-1} (1 + h_i · x)

    G(x) 中 x^k 的系数 = 选择 k 层各取一个击中的组合数

    特别地，选择 3 层的种子数:
        N_seeds = Σ_{i<j<k} h_i · h_j · h_k

    Parameters
    ----------
    layer_hit_counts : list of int
        每层的击中数
    n_seed_layers : int
        种子所需的层数

    Returns
    -------
    dict : {
        'total_seeds': int,
        'seeds_per_layer_combo': dict,
        'polynomial_coeffs': list,
    }
    """
    n_layers = len(layer_hit_counts)

    # 构建生成多项式 (与 [158] 的多项式乘法一致)
    # p(x) = Π (1 + h_i · x)
    poly = [1.0]  # 初始 = 1

    for h_i in layer_hit_counts:
        # 乘以 (1 + h_i · x)
        factor = [1.0, float(h_i)]
        poly = _polynomial_multiply(poly, factor)

    # x^n_seed_layers 的系数 = 选择 n_seed_layers 层的组合数
    if n_seed_layers < len(poly):
        total_seeds = int(round(poly[n_seed_layers]))
    else:
        total_seeds = 0

    # 枚举具体的层组合
    seeds_per_combo = {}
    for combo in itertools.combinations(range(n_layers), n_seed_layers):
        n_combo = 1
        for layer_idx in combo:
            n_combo *= layer_hit_counts[layer_idx]
        seeds_per_combo[combo] = n_combo

    return {
        'total_seeds': total_seeds,
        'seeds_per_layer_combo': seeds_per_combo,
        'polynomial_coeffs': poly,
    }


def _polynomial_multiply(p, q):
    """
    基于 [158] 的多项式乘法

    (p·q)_k = Σ_{i+j=k} p_i · q_j
    """
    n = len(p)
    m = len(q)
    result = [0.0] * (n + m - 1)
    for i in range(n):
        for j in range(m):
            result[i + j] += p[i] * q[j]
    # 去除尾部零
    while len(result) > 1 and abs(result[-1]) < 1e-15:
        result.pop()
    return result


# ============================================================
# [1122_PepHiRe] Ladderpath 层次分解 → 径迹拓扑分析
# ============================================================
class TrackLadderpath:
    """
    基于 [1122_PepHiRe] 的 Ladderpath 序列分解

    原算法将一组字符串分解为重复子串 (ladderons) 的层次结构:
        STRMAT 类维护:
        - Head: 剩余未分解的字符串片段
        - Group: ladderon 成员关系
        - ladderonBook: ladderon → [groupID, occurrences]

    映射到径迹识别:
    将径迹的层命中序列分解为重复模式:
    - 基本单元 (basic unit): 单层击中
    - 重复模式 (ladderon): 等间距的多层击中

    例如:
        径迹 A 命中层 [0, 2, 4, 6] → ladderon "0_2" 重复 4 次
        径迹 B 命中层 [0, 1, 2, 3, 4] → ladderon "0_1" 重复 5 次

    这有助于:
    1. 识别径迹类型 (直线 vs 螺旋)
    2. 发现共享子结构的径迹族
    3. 高效存储和检索径迹模板
    """

    def __init__(self):
        self.ladderon_book = {}
        self.decomposition = []

    def decompose(self, hit_sequence):
        """
        将击中序列分解为 ladderon 层次结构

        Parameters
        ----------
        hit_sequence : list of int
            命中的层编号列表 (已排序)

        Returns
        -------
        list of dict : 分解结果
        """
        if not hit_sequence:
            return []

        # 转换为字符串表示
        seq_str = '_'.join(str(h) for h in sorted(hit_sequence))
        chars = list(seq_str)

        # 贪心查找最长重复子串 (简化版 [1122] 的 ladderpath)
        ladderons = []
        remaining = list(hit_sequence)

        # 查找等间距模式
        while len(remaining) >= 2:
            best_ladderon = None
            best_count = 0
            best_span = []

            # 尝试不同的间距
            for gap in range(1, len(remaining)):
                for start_idx in range(len(remaining)):
                    # 从 start_idx 开始，以 gap 为间距提取子序列
                    pattern = [remaining[start_idx]]
                    span = [start_idx]
                    prev = remaining[start_idx]

                    for idx in range(start_idx + 1, len(remaining)):
                        if remaining[idx] - prev == gap:
                            pattern.append(remaining[idx])
                            span.append(idx)
                            prev = remaining[idx]

                    if len(pattern) >= 2 and len(pattern) > best_count:
                        best_ladderon = tuple(pattern)
                        best_count = len(pattern)
                        best_span = span

            if best_ladderon and best_count >= 2:
                ladderons.append({
                    'pattern': best_ladderon,
                    'count': best_count,
                    'gap': best_ladderon[1] - best_ladderon[0] if len(best_ladderon) > 1 else 0,
                    'layers': list(best_ladderon),
                })
                # 从剩余中移除已匹配的
                for idx in sorted(best_span, reverse=True):
                    if idx < len(remaining):
                        remaining.pop(idx)
            else:
                break

        # 剩余的单层击中作为 basic units
        for h in remaining:
            ladderons.append({
                'pattern': (h,),
                'count': 1,
                'gap': 0,
                'layers': [h],
            })

        # 更新 ladderon book
        for lad in ladderons:
            key = str(lad['pattern'])
            if key not in self.ladderon_book:
                self.ladderon_book[key] = {
                    'pattern': lad['pattern'],
                    'occurrences': 0,
                    'total_count': 0,
                }
            self.ladderon_book[key]['occurrences'] += 1
            self.ladderon_book[key]['total_count'] += lad['count']

        self.decomposition = ladderons
        return ladderons

    def complexity_index(self):
        """
        计算复杂度指数 (类似 [1122] 的 ladderpath index)

        index3 = (n_ladderons, n_basic, max_depth)
        复杂度越低 = 径迹越规则
        """
        if not self.decomposition:
            return (0, 0, 0)

        n_ladderons = len([d for d in self.decomposition if d['count'] >= 2])
        n_basic = len([d for d in self.decomposition if d['count'] == 1])
        max_depth = max(d['count'] for d in self.decomposition) if self.decomposition else 0

        return (n_ladderons, n_basic, max_depth)


# ============================================================
# 种子生成器
# ============================================================
def generate_track_seeds(hits_per_layer, detector_layers, max_seeds=1000):
    """
    从各层击中生成径迹种子

    算法:
    1. 选择 3 个连续的层 (或近似连续)
    2. 从每层取一个击中
    3. 计算种子参数 (曲率、方向)
    4. 过滤不物理的种子

    Parameters
    ----------
    hits_per_layer : dict
        {layer_id: [hit_dict, ...]}
    detector_layers : list
        DetectorLayer 列表
    max_seeds : int
        最大种子数

    Returns
    -------
    list of dict : 种子列表
    """
    seeds = []

    # 获取有击中的层
    active_layers = sorted([lid for lid, hits in hits_per_layer.items() if hits])

    if len(active_layers) < 3:
        return seeds

    # 计数种子组合 (使用 [158] 的方法)
    hit_counts = [len(hits_per_layer[lid]) for lid in active_layers]
    combo_info = count_seed_combinations(hit_counts, n_seed_layers=3)

    # 限制种子数
    total_possible = combo_info['total_seeds']

    # 生成种子: 遍历连续三层组合
    for layer_triplet_idx in range(len(active_layers) - 2):
        l0 = active_layers[layer_triplet_idx]
        l1 = active_layers[layer_triplet_idx + 1]
        l2 = active_layers[layer_triplet_idx + 2]

        hits0 = hits_per_layer.get(l0, [])
        hits1 = hits_per_layer.get(l1, [])
        hits2 = hits_per_layer.get(l2, [])

        for h0 in hits0:
            for h1 in hits1:
                for h2 in hits2:
                    if len(seeds) >= max_seeds:
                        return seeds

                    # 构建种子
                    seed = _build_seed(h0, h1, h2, l0, l1, l2, detector_layers)
                    if seed is not None:
                        seeds.append(seed)

    return seeds


def _build_seed(hit0, hit1, hit2, layer0, layer1, layer2, detector_layers):
    """
    从三个击中构建径迹种子

    使用圆拟合估计曲率 (横向) 和直线拟合估计方向 (纵向)
    """
    # 获取击中坐标 (转换为 (x, y) 平面)
    r0, z0 = hit0.get('r_meas', 0), hit0.get('z_meas', 0)
    r1, z1 = hit1.get('r_meas', 0), hit1.get('z_meas', 0)
    r2, z2 = hit2.get('r_meas', 0), hit2.get('z_meas', 0)

    # 假设 φ = 0 简化 (径迹在 r-z 平面投影)
    x0, y0 = r0, 0.0
    x1, y1 = r1, 0.0
    x2, y2 = r2, 0.0

    # 圆拟合: 通过三点求圆
    curvature = _circle_curvature_2d(x0, y0, x1, y1, x2, y2)

    if curvature is None:
        return None

    # 检查曲率物理合理性
    # κ = q/p_T, |κ| < κ_max → p_T > p_T_min
    kappa_max = 0.1  # 1/mm (对应 p_T_min ~ 3 GeV at 2T)
    if abs(curvature) > kappa_max:
        return None  # 动量过低

    # 纵向方向 tan(λ) = dz/ds
    ds01 = math.sqrt((x1 - x0)**2 + (y1 - y0)**2)
    ds12 = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

    if ds01 + ds12 < 1e-10:
        return None

    dz_total = z2 - z0
    ds_total = ds01 + ds12
    tan_lambda = dz_total / ds_total if ds_total > 1e-10 else 0.0

    # 限制 dip 角
    if abs(tan_lambda) > 10.0:
        return None

    return {
        'curvature': curvature,
        'tan_lambda': tan_lambda,
        'phi_0': math.atan2(y0, x0),
        'd0': 0.0,  # 需要参考点
        'z0': z0,
        'layers': [layer0, layer1, layer2],
        'hits': [hit0, hit1, hit2],
    }


def _circle_curvature_2d(x1, y1, x2, y2, x3, y3):
    """
    通过三点计算圆的曲率

    κ = 1/R = 4·Area / (a·b·c)

    其中 Area 为三角形面积，a,b,c 为边长
    """
    # 边长
    a = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    b = math.sqrt((x3 - x2)**2 + (y3 - y2)**2)
    c = math.sqrt((x1 - x3)**2 + (y1 - y3)**2)

    # 三角形面积 (叉积)
    area = abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)) / 2.0

    # 曲率
    denom = a * b * c
    if denom < 1e-15:
        return None

    curvature = 4.0 * area / denom

    # 带符号: 由叉积判断弯曲方向
    cross = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
    if cross < 0:
        curvature = -curvature

    return curvature
