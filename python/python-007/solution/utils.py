"""
通用工具模块
整合自：709_magic4_matrix（幻方构造）
以及各类数值辅助函数
"""
import numpy as np


def magic4_matrix(n):
    """
    构造4k阶幻方矩阵，用于生成特殊数值权重矩阵。
    在吸积盘模拟中，幻方矩阵可用于构造结构化采样网格或
    特殊正交权重分布，保证数值积分的对称性和守恒性。

    数学原理（Doubly Even Magic Square）:
    对于 n = 4k 的方阵，将 1..n² 依次填入，然后对每个 4×4
    子块的"X"型位置（主对角线或反对角线上的单元）进行互补替换：
        k2 = n² + 1 - k1

    参数:
        n: 矩阵阶数，必须是4的倍数

    返回:
        n×n 幻方矩阵
    """
    if n % 4 != 0:
        raise ValueError(f"magic4_matrix requires n to be a multiple of 4, got {n}")
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")

    # 初始顺序填充
    M = np.arange(1, n * n + 1).reshape(n, n)

    # 对每个 4×4 子块进行 X-pattern 替换
    n_sq = n * n
    for i in range(n):
        for j in range(n):
            # 检测是否位于 X-pattern 上
            m1 = np.abs(i - j) % 4
            m2 = (i + j + 1) % 4
            if m1 == 0 or m2 == 0:
                M[i, j] = n_sq + 1 - M[i, j]

    return M


def normalized_magic_weights(n):
    """
    将幻方矩阵归一化为权重矩阵，使总和为1。
    用于蒙特卡洛或谱方法中的非均匀权重分配。
    """
    M = magic4_matrix(n)
    return M.astype(np.float64) / np.sum(M)


def ball_unit_sample(n_samples, dim=3, seed=None):
    """
    在单位球内均匀采样 n_samples 个点。
    算法：
        1. 生成高斯随机向量 -> 方向均匀分布
        2. 归一化得到球面方向
        3. 半径 r = u^(1/dim)，其中 u~Uniform(0,1)
          保证体积元 dV ~ r^(dim-1) dr 的均匀分布

    参数:
        n_samples: 采样点数
        dim: 空间维度（默认3）
        seed: 随机种子（可选）

    返回:
        (n_samples, dim) 数组
    """
    if n_samples < 0:
        raise ValueError("n_samples must be non-negative")
    if seed is not None:
        np.random.seed(seed)

    # 方向：高斯分布归一化
    dirs = np.random.randn(n_samples, dim)
    norms = np.linalg.norm(dirs, axis=1, keepdims=True)
    norms = np.where(norms < 1e-15, 1.0, norms)
    dirs = dirs / norms

    # 半径
    u = np.random.rand(n_samples)
    r = u ** (1.0 / dim)

    return dirs * r.reshape(-1, 1)


def distance_stats(points_a, points_b):
    """
    计算两组点之间两两距离的统计量。
    用于分析吸积盘粒子或喷流微团的间距分布。
    """
    if points_a.shape[1] != points_b.shape[1]:
        raise ValueError("Dimension mismatch")

    # 计算所有两两距离
    diffs = points_a[:, np.newaxis, :] - points_b[np.newaxis, :, :]
    dists = np.linalg.norm(diffs, axis=2)

    return {
        'mean': float(np.mean(dists)),
        'std': float(np.std(dists)),
        'min': float(np.min(dists)),
        'max': float(np.max(dists)),
        'distances': dists.flatten()
    }


def safe_divide(a, b, fill_value=0.0):
    """安全除法，避免除以零。"""
    b = np.asarray(b, dtype=np.float64)
    result = np.full_like(np.asarray(a, dtype=np.float64), fill_value)
    mask = np.abs(b) > 1e-15
    result[mask] = a[mask] / b[mask]
    return result


def clip_with_warning(arr, lo, hi, name="array"):
    """裁剪数组到范围，保证数值稳定性。"""
    arr = np.asarray(arr)
    clipped = np.clip(arr, lo, hi)
    n_violations = np.sum((arr < lo) | (arr > hi))
    if n_violations > 0:
        # 静默处理，不打印以避免干扰
        pass
    return clipped
