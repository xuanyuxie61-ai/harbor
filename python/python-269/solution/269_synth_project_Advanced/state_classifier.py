"""
state_classifier.py — 量子态分类与递归分析
============================================================

本模块实现量子态的分类算法:

1. 基于参与比的分类:
    - PR/N > 0.1: 扩展态 (extended)
    - PR/N < 0.01: 局域态 (localized)
    - 中间: 临界态 (critical)

2. Collatz 型递归序列 (源自 collatz_recursive):
    用于生成态的分类序列:
        a_{n+1} = a_n / 2  (if a_n even)
        a_{n+1} = 3a_n + 1 (if a_n odd)

    在量子霍尔语境下, 此递归用于分析能谱的层次结构:
        将能级指标映射到分类序列, 揭示能级的自相似分形结构.

3. Bernstein 多项式光滑 (源自 bernstein_polynomial):
    使用 Bernstein 多项式对离散的态分类结果进行光滑:
        B_n(f)(x) = Σ_{k=0}^n f(k/n) · C(n,k) · x^k · (1-x)^{n-k}

    光滑后的分类函数用于:
    - 识别相变点 (分类函数的拐点)
    - 估计临界指数

4. 边界词分析 (源自 boundary_word_equilateral):
    对于六角格点 flake, 分析边界态的拓扑性质:
    - 边界词描述了边界的路径
    - 合法的边界词对应物理上可实现的边界
    - 边界词的等价类对应不同的边界终止

参考文献:
    [1] Abrahams, E. et al. PRL 42, 673 (1979) [Scaling theory]
    [2] Kritchevsky et al. "Quantum Hall Edge States" (2020)
"""

import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from wavelet_downsample import participation_ratio, inverse_participation_ratio


def classify_states(eigenvalues: np.ndarray,
                     eigenvectors: np.ndarray,
                     N_total: int,
                     threshold_extended: float = 0.1,
                     threshold_localized: float = 0.01) -> Dict[str, Any]:
    """基于参与比对量子态进行分类

    分类准则:
        PR/N > threshold_extended   → 扩展态
        PR/N < threshold_localized  → 局域态
        其他                         → 临界态

    Args:
        eigenvalues: 本征值
        eigenvectors: 本征态矩阵
        N_total: 总格点数
        threshold_extended: 扩展态阈值
        threshold_localized: 局域态阈值
    Returns:
        分类结果
    """
    n_states = len(eigenvalues)
    classifications = []
    pr_values = []
    ipr_values = []

    for n in range(n_states):
        psi = eigenvectors[:, n]
        pr = participation_ratio(psi)
        ipr = inverse_participation_ratio(psi)
        pr_norm = pr / N_total

        pr_values.append(pr)
        ipr_values.append(ipr)

        if pr_norm > threshold_extended:
            label = "extended"
        elif pr_norm < threshold_localized:
            label = "localized"
        else:
            label = "critical"

        classifications.append({
            'level': n,
            'energy': eigenvalues[n],
            'PR': pr,
            'PR_norm': pr_norm,
            'IPR': ipr,
            'type': label,
        })

    # 统计
    n_extended = sum(1 for c in classifications if c['type'] == 'extended')
    n_localized = sum(1 for c in classifications if c['type'] == 'localized')
    n_critical = sum(1 for c in classifications if c['type'] == 'critical')

    return {
        'classifications': classifications,
        'n_extended': n_extended,
        'n_localized': n_localized,
        'n_critical': n_critical,
        'n_total': n_states,
        'pr_values': np.array(pr_values),
        'ipr_values': np.array(ipr_values),
    }


def collatz_classification(energy_level: int, max_steps: int = 100) -> Dict[str, Any]:
    """Collatz 递归分类序列 (源自 collatz_recursive)

    将能级指标 n 通过 Collatz 递归映射到分类序列:
        a_0 = n
        a_{k+1} = a_k/2  (if a_k even)
        a_{k+1} = 3a_k + 1 (if a_k odd)

    序列的特征 (到达 1 的步数、最大值) 被用作态的分类指纹:
    - 快速收敛 → 规则态 (体扩展态)
    - 长序列 → 不规则态 (局域态或临界态)

    这虽然是非物理的数学映射, 但提供了一种独特的态指纹方法.

    Args:
        energy_level: 能级指标 (正整数)
        max_steps: 最大递归步数
    Returns:
        分类序列信息
    """
    if energy_level < 1:
        return {'sequence': [0], 'length': 1, 'max_value': 0,
                'converged': False}

    sequence = [energy_level]
    current = energy_level
    steps = 0

    while current != 1 and steps < max_steps:
        if current % 2 == 0:
            current = current // 2
        else:
            current = 3 * current + 1
        sequence.append(current)
        steps += 1

    return {
        'sequence': sequence,
        'length': len(sequence),
        'max_value': max(sequence),
        'converged': current == 1,
        'parity_pattern': [s % 2 for s in sequence],
    }


def bernstein_smooth_classification(pr_normalized: np.ndarray,
                                     degree: int = 10) -> np.ndarray:
    """使用 Bernstein 多项式光滑分类函数

    B_n(f)(x) = Σ_{k=0}^n f(k/n) · C(n,k) · x^k · (1-x)^{n-k}

    将离散的 PR 归一化值映射到连续的分类函数,
    用于识别相变点 (分类函数的拐点).

    Bernstein 多项式的性质:
    - 保形性: 光滑后不引入新的极值
    - 收敛性: B_n(f) → f (一致收敛, 当 n → ∞)
    - 数值稳定性: 避免 Runge 现象

    Args:
        pr_normalized: PR/N 值数组 (在 [0, 1] 范围内)
        degree: Bernstein 多项式阶数
    Returns:
        光滑后的分类函数值
    """
    from scipy.special import comb

    x = np.linspace(0, 1, len(pr_normalized))
    n = degree
    result = np.zeros_like(x)

    for k in range(n + 1):
        # Bernstein 基函数
        t_k = k / n if n > 0 else 0
        # 找到最接近的 PR 值
        idx = np.argmin(np.abs(pr_normalized - t_k))
        f_k = pr_normalized[idx] if idx < len(pr_normalized) else 0.0

        # B_{n,k}(x) = C(n,k) x^k (1-x)^{n-k}
        basis = comb(n, k) * (x ** k) * ((1 - x) ** (n - k))
        result += f_k * basis

    return result


def boundary_word_analysis(boundary_sites: np.ndarray,
                            center: np.ndarray) -> Dict[str, Any]:
    """边界词分析 (源自 boundary_word_equilateral)

    对于六角格点 flake 的边界, 编码边界路径为方向序列:
        方向编码 (六角格点 6 方向):
            0: (0, 1)
            1: (1, 0)
            2: (1, -1)
            3: (0, -1)
            4: (-1, 0)
            5: (-1, 1)

    边界词的性质:
    - 合法性: 右步数 = 左步数, 上步数 = 下步数
    - 等价类: 旋转等价的边界词属于同一类
    - 代表元: 字典序最小的边界词

    边界词的拓扑信息:
    - 边界词的总转角 = 2π (简单闭合曲线)
    - 边界词的自交叉数 = 边界拓扑的复杂度

    Args:
        boundary_sites: 边界格点坐标 (N × 2)
        center: 中心坐标
    Returns:
        边界词分析结果
    """
    if len(boundary_sites) < 3:
        return {'word': '', 'legal': False, 'length': 0}

    # 计算从中心到每个边界点的角度
    angles = np.arctan2(boundary_sites[:, 1] - center[1],
                        boundary_sites[:, 0] - center[0])

    # 排序角度
    sorted_idx = np.argsort(angles)
    sorted_sites = boundary_sites[sorted_idx]

    # 计算方向序列
    directions = []
    for i in range(len(sorted_sites)):
        next_i = (i + 1) % len(sorted_sites)
        diff = sorted_sites[next_i] - sorted_sites[i]

        # 量化到 6 个方向
        angle = np.arctan2(diff[1], diff[0])
        direction = int(round(angle / (np.pi / 3))) % 6
        directions.append(str(direction))

    word = ''.join(directions)

    # 检查合法性
    right = word.count('1') + word.count('2')
    left = word.count('4') + word.count('5')
    up = word.count('0') + word.count('5')
    down = word.count('2') + word.count('3')
    legal = (right == left) and (up == down)

    # 找代表元 (字典序最小)
    n = len(word)
    rotations = [word[i:] + word[:i] for i in range(n)]
    representative = min(rotations) if rotations else ''

    return {
        'word': word,
        'length': len(word),
        'legal': legal,
        'representative': representative,
        'n_right': right,
        'n_left': left,
        'n_up': up,
        'n_down': down,
    }


def spectral_rigidity(eigenvalues: np.ndarray,
                       L_range: np.ndarray) -> np.ndarray:
    """谱刚性 Δ₃(L) 分析

    Δ₃(L) 衡量能谱在尺度 L 上偏离均匀分布的程度:
        Δ₃(L) = (1/L) min_{A,B} ∫_0^L [N(E) - AE - B]² dE

    对于 Poisson 谱 (局域态): Δ₃(L) ~ L/15
    对于 GOE 谱 (扩展态): Δ₃(L) ~ (1/π²)ln(L)
    对于 GUE 谱: Δ₃(L) ~ (1/2π²)ln(L)

    在量子霍尔转变点, Δ₃(L) 介于两者之间.

    Args:
        eigenvalues: 本征值 (已排序)
        L_range: L 值范围
    Returns:
        Δ₃(L) 数组
    """
    evals_sorted = np.sort(eigenvalues)
    N = len(evals_sorted)
    delta3 = np.zeros(len(L_range))

    for iL, L in enumerate(L_range):
        if L > N or L < 2:
            delta3[iL] = np.nan
            continue

        # 滑动窗口
        n_windows = N - int(L) + 1
        delta3_sum = 0.0

        for start in range(0, n_windows, max(1, n_windows // 10)):
            end = start + int(L)
            if end > N:
                break
            window = evals_sorted[start:end]

            # 线性拟合 N(E) = AE + B
            E = np.arange(len(window))
            if len(E) >= 2:
                A, B = np.polyfit(E, window, 1)
                residual = window - (A * E + B)
                delta3_sum += np.mean(residual ** 2)

        if n_windows > 0:
            delta3[iL] = delta3_sum / max(1, n_windows // max(1, n_windows // 10))

    return delta3
