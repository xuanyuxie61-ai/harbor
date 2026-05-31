"""
variomino_clot.py
基于 1389_variomino 的变体多格拼板矩阵操作思想，
构建纤维蛋白 clot 微观结构的网格表示与拓扑分析工具。

科学背景：
    纤维蛋白 clot 由交联的纤维蛋白单体组成三维网络。
    在二维截面上， clot 结构可表示为带有权重（纤维密度）的
    多格拼板（variomino）。

    矩阵操作包括：
    - 凝聚（condense）：去除空白区域，提取 clot 核心
    - 嵌入（embed）：将 clot 结构嵌入血管截面
    - 变换（transform）：旋转/翻转以分析各向异性

数学模型：
    1. clot 密度矩阵 P_{ij} ∈ [0,1]：
       P_{ij} = 0  表示血浆（无纤维）
       P_{ij} > 0  表示纤维蛋白密度

    2. 凝聚操作：
       移除全零行/列，得到最小外接矩形。

    3. 孔隙连通性（基于 embed 思想）：
       使用 DFS/BFS 标记连通的孔隙区域，
       评估 clot 的渗透性。
"""

import numpy as np


class VariominoClot:
    """
    纤维蛋白 clot 的多格矩阵表示与操作。
    """

    def __init__(self, density_matrix):
        """
        参数:
            density_matrix : ndarray, 二维纤维蛋白密度矩阵 [0,1]
        """
        self.P = np.asarray(density_matrix, dtype=float)
        if self.P.ndim != 2:
            raise ValueError("density_matrix 必须为二维数组")

    def condense(self, threshold=1e-12):
        """
        基于 1389_variomino/variomino_condense 的凝聚操作：
        移除边缘全零行/列，保留 clot 核心。

        参数:
            threshold : float, 判定为零的阈值

        返回:
            Q : ndarray, 凝聚后的矩阵
        """
        Q = self.P.copy()
        # 移除上方全零行
        while Q.shape[0] > 0 and np.all(Q[0, :] <= threshold):
            Q = Q[1:, :]
        # 移除下方全零行
        while Q.shape[0] > 0 and np.all(Q[-1, :] <= threshold):
            Q = Q[:-1, :]
        # 移除左侧全零列
        while Q.shape[1] > 0 and np.all(Q[:, 0] <= threshold):
            Q = Q[:, 1:]
        # 移除右侧全零列
        while Q.shape[1] > 0 and np.all(Q[:, -1] <= threshold):
            Q = Q[:, :-1]
        if Q.size == 0:
            Q = np.zeros((1, 1))
        return Q

    def embed_in_domain(self, domain_shape, center=None):
        """
        将 clot 结构嵌入更大的血管截面域中。

        参数:
            domain_shape : tuple, (H, W) 目标域大小
            center       : tuple or None, 嵌入中心位置

        返回:
            embedded : ndarray, 嵌入后的矩阵
        """
        dh, dw = domain_shape
        Q = self.condense()
        qh, qw = Q.shape
        if qh > dh or qw > dw:
            raise ValueError("clot 尺寸超过目标域")

        if center is None:
            cy, cx = dh // 2, dw // 2
        else:
            cy, cx = center

        top = max(0, cy - qh // 2)
        left = max(0, cx - qw // 2)
        bottom = min(dh, top + qh)
        right = min(dw, left + qw)

        embedded = np.zeros(domain_shape, dtype=float)
        # 调整以避免越界
        q_top = max(0, -top)
        q_left = max(0, -left)
        q_bottom = q_top + (bottom - top)
        q_right = q_left + (right - left)
        embedded[top:bottom, left:right] = Q[q_top:q_bottom, q_left:q_right]
        return embedded

    def rotate90(self, k=1):
        """
        旋转 clot 结构 k×90 度。
        """
        return np.rot90(self.P, k=k)

    def flip_horizontal(self):
        """
        水平翻转。
        """
        return np.fliplr(self.P)

    def flip_vertical(self):
        """
        垂直翻转。
        """
        return np.flipud(self.P)

    def count_pore_clusters(self, threshold=0.05):
        """
        使用 BFS 统计孔隙（低密度区域）的连通分量数。

        参数:
            threshold : float, 判定为孔隙的密度上限

        返回:
            n_clusters : int, 连通孔隙区域数
            labels     : ndarray, 标记矩阵
        """
        binary = (self.P <= threshold).astype(int)
        h, w = binary.shape
        labels = np.zeros((h, w), dtype=int)
        label_id = 0

        for i in range(h):
            for j in range(w):
                if binary[i, j] == 1 and labels[i, j] == 0:
                    label_id += 1
                    # BFS
                    queue = [(i, j)]
                    labels[i, j] = label_id
                    head = 0
                    while head < len(queue):
                        y, x = queue[head]
                        head += 1
                        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                            ny, nx = y + dy, x + dx
                            if 0 <= ny < h and 0 <= nx < w:
                                if binary[ny, nx] == 1 and labels[ny, nx] == 0:
                                    labels[ny, nx] = label_id
                                    queue.append((ny, nx))
        return label_id, labels

    def compute_anisotropy(self):
        """
        计算 clot 结构的各向异性指数：
            A = |λ1 - λ2| / (λ1 + λ2)
        其中 λ1, λ2 为惯性张量的特征值。
        """
        P = self.condense()
        if P.size == 0:
            return 0.0
        h, w = P.shape
        ys, xs = np.mgrid[:h, :w]
        total_mass = P.sum()
        if total_mass < 1e-30:
            return 0.0
        cy = np.sum(ys * P) / total_mass
        cx = np.sum(xs * P) / total_mass

        Ixx = np.sum((ys - cy) ** 2 * P) / total_mass
        Iyy = np.sum((xs - cx) ** 2 * P) / total_mass
        Ixy = np.sum((ys - cy) * (xs - cx) * P) / total_mass

        inertia = np.array([[Ixx, Ixy], [Ixy, Iyy]])
        eigs = np.linalg.eigvalsh(inertia)
        eigs = np.sort(eigs)[::-1]
        if np.sum(eigs) < 1e-30:
            return 0.0
        anisotropy = (eigs[0] - eigs[1]) / (eigs[0] + eigs[1])
        return float(anisotropy)

    @staticmethod
    def generate_random_clot(height=30, width=30, n_fibers=8, fiber_length=8, seed=42):
        """
        生成随机纤维蛋白 clot 结构。
        """
        rng = np.random.default_rng(seed)
        P = np.zeros((height, width), dtype=float)
        for _ in range(n_fibers):
            y0 = rng.integers(0, height)
            x0 = rng.integers(0, width)
            angle = rng.uniform(0, 2 * np.pi)
            for step in range(fiber_length):
                y = int(round(y0 + step * np.sin(angle)))
                x = int(round(x0 + step * np.cos(angle)))
                if 0 <= y < height and 0 <= x < width:
                    P[y, x] = min(1.0, P[y, x] + 0.3 + rng.uniform(0, 0.4))
        # 添加扩散
        from scipy.ndimage import gaussian_filter
        P = gaussian_filter(P, sigma=1.0)
        P = np.clip(P, 0, 1)
        return P


def demo_variomino():
    """
    演示 clot 结构的矩阵操作。
    """
    P = VariominoClot.generate_random_clot(height=40, width=40, n_fibers=12, seed=42)
    clot = VariominoClot(P)

    print("=" * 60)
    print("纤维蛋白 clot 多格结构分析")
    print("=" * 60)

    Q = clot.condense()
    print(f"原始矩阵大小: {clot.P.shape}")
    print(f"凝聚后大小: {Q.shape}")

    n_pores, labels = clot.count_pore_clusters(threshold=0.1)
    print(f"孔隙连通区域数: {n_pores}")

    ani = clot.compute_anisotropy()
    print(f"结构各向异性指数: {ani:.4f}")

    embedded = clot.embed_in_domain((60, 60), center=(30, 30))
    print(f"嵌入域大小: {embedded.shape}")

    return clot


if __name__ == "__main__":
    demo_variomino()
