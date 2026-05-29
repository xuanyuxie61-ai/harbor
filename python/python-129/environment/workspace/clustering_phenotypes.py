"""
clustering_phenotypes.py
基于 039_asa113 的交换优化聚类思想，
构建血凝表型数据的非层次聚类与优化工具。

科学背景：
    患者的凝血功能存在显著个体差异，形成不同的"凝血表型"（phenotypes）。
    通过凝血指标（PT, aPTT, D-dimer, Fibrinogen, Platelet count 等）
    对患者进行聚类，可识别出血栓高风险/低风险亚群。

    本模块实现一种基于交换（swap）优化的迭代聚类算法：
    在类别间交换样本以最小化类内离散度。

数学模型：
    1. 类内离散度（WCSS）：
       J = Σ_{k=1}^K Σ_{i∈C_k} ||x_i - μ_k||²

    2. 交换准则：
       对对象 i ∈ C_l 和 j ∈ C_m，计算交换后的 ΔJ。
       若 ΔJ < 0，则执行交换。

    3. 迭代直至收敛（与 039_asa113 的 swap 逻辑一致）。
"""

import numpy as np


class SwapClustering:
    """
    基于交换优化的非层次聚类算法。
    改编自 039_asa113/swap.m 的核心逻辑。
    """

    def __init__(self, n_clusters=3, max_iter=200, tol=1e-8):
        """
        参数:
            n_clusters : int, 聚类数 K
            max_iter   : int, 最大迭代次数
            tol        : float, 收敛容差
        """
        if n_clusters < 2:
            raise ValueError("n_clusters 必须 >= 2")
        self.K = n_clusters
        self.max_iter = max_iter
        self.tol = tol

    def _compute_wcss(self, X, labels, centroids):
        """
        计算类内离散度总和。
        """
        wcss = 0.0
        for k in range(self.K):
            members = X[labels == k]
            if len(members) > 0:
                diffs = members - centroids[k]
                wcss += np.sum(diffs ** 2)
        return wcss

    def _update_centroids(self, X, labels):
        """
        更新类中心。
        """
        centroids = np.zeros((self.K, X.shape[1]))
        for k in range(self.K):
            members = X[labels == k]
            if len(members) > 0:
                centroids[k] = np.mean(members, axis=0)
            else:
                # 空类处理：随机选择一个点
                centroids[k] = X[np.random.randint(0, len(X))]
        return centroids

    def fit(self, X, init_labels=None):
        """
        执行交换优化聚类。

        参数:
            X           : ndarray, shape (n_samples, n_features)
            init_labels : ndarray or None, 初始标签

        返回:
            labels     : ndarray, 聚类标签
            centroids  : ndarray, 类中心
            wcss       : float, 最终类内离散度
            n_swaps    : int, 执行的交换次数
        """
        X = np.asarray(X, dtype=float)
        n_samples, n_features = X.shape
        if n_samples < self.K:
            raise ValueError("样本数必须 >= 聚类数")

        if init_labels is None:
            labels = np.random.default_rng(42).integers(0, self.K, size=n_samples)
        else:
            labels = np.asarray(init_labels, dtype=int).copy()

        centroids = self._update_centroids(X, labels)
        wcss = self._compute_wcss(X, labels, centroids)
        n_swaps = 0

        for it in range(self.max_iter):
            improved = False
            # 遍历所有可能的交换对
            for i in range(n_samples):
                li = labels[i]
                for j in range(i + 1, n_samples):
                    lj = labels[j]
                    if li == lj:
                        continue
                    # 计算交换 i 和 j 的类别后的 WCSS 变化
                    # 快速计算：仅更新涉及的两个类
                    old_wcss = wcss
                    # 临时交换
                    labels[i], labels[j] = lj, li
                    new_centroids = self._update_centroids(X, labels)
                    new_wcss = self._compute_wcss(X, labels, new_centroids)

                    if new_wcss < old_wcss - self.tol:
                        # 接受交换
                        centroids = new_centroids
                        wcss = new_wcss
                        n_swaps += 1
                        improved = True
                    else:
                        # 拒绝交换，恢复
                        labels[i], labels[j] = li, lj

            if not improved:
                break

        return labels, centroids, wcss, n_swaps

    def predict(self, X, centroids):
        """
        对新样本进行分类。
        """
        X = np.asarray(X, dtype=float)
        dists = np.zeros((X.shape[0], self.K))
        for k in range(self.K):
            dists[:, k] = np.sum((X - centroids[k]) ** 2, axis=1)
        return np.argmin(dists, axis=1)


def generate_coagulation_phenotypes(n_samples=200, seed=42):
    """
    生成模拟的血凝表型数据。
    特征：PT(s), aPTT(s), D-dimer(mg/L), Fibrinogen(g/L), Platelet(10^9/L)
    """
    rng = np.random.default_rng(seed)
    # 3个预设表型中心
    centers = np.array([
        [12.0, 30.0, 0.3, 3.0, 250.0],   # 正常
        [10.5, 25.0, 1.5, 4.5, 350.0],   # 高凝
        [15.0, 45.0, 0.8, 1.5, 120.0],   # 低凝/出血倾向
    ])
    scales = np.array([
        [0.8, 3.0, 0.1, 0.3, 30.0],
        [0.6, 2.5, 0.3, 0.4, 40.0],
        [1.2, 5.0, 0.2, 0.3, 25.0],
    ])

    X = []
    true_labels = []
    for k in range(3):
        n_k = n_samples // 3
        samples = rng.normal(loc=centers[k], scale=scales[k], size=(n_k, 5))
        X.append(samples)
        true_labels.extend([k] * n_k)

    X = np.vstack(X)
    true_labels = np.array(true_labels)
    # 随机打乱
    perm = rng.permutation(len(X))
    return X[perm], true_labels[perm]


def demo_clustering():
    """
    演示血凝表型的交换优化聚类。
    """
    X, true_labels = generate_coagulation_phenotypes(n_samples=150, seed=42)

    # 标准化
    X_norm = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-12)

    clusterer = SwapClustering(n_clusters=3, max_iter=100)
    labels, centroids, wcss, n_swaps = clusterer.fit(X_norm)

    print("=" * 60)
    print("血凝表型交换优化聚类分析")
    print("=" * 60)
    print(f"最终 WCSS: {wcss:.4f}")
    print(f"执行交换次数: {n_swaps}")

    # 计算与真实标签的匹配度（不考虑标签排列）
    from itertools import permutations
    best_acc = 0.0
    for perm in permutations(range(3)):
        mapped = np.array([perm[l] for l in labels])
        acc = np.mean(mapped == true_labels)
        best_acc = max(best_acc, acc)
    print(f"聚类准确率（最优排列）: {best_acc:.2%}")

    # 各类统计
    for k in range(3):
        members = X[labels == k]
        if len(members) > 0:
            print(f"\n类别 {k} (n={len(members)}):")
            print(f"  PT mean = {members[:, 0].mean():.2f} s")
            print(f"  aPTT mean = {members[:, 1].mean():.2f} s")
            print(f"  D-dimer mean = {members[:, 2].mean():.2f} mg/L")

    return clusterer, labels, centroids


if __name__ == "__main__":
    demo_clustering()
