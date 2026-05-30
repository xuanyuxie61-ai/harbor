
import numpy as np


class SwapClustering:

    def __init__(self, n_clusters=3, max_iter=200, tol=1e-8):
        if n_clusters < 2:
            raise ValueError("n_clusters 必须 >= 2")
        self.K = n_clusters
        self.max_iter = max_iter
        self.tol = tol

    def _compute_wcss(self, X, labels, centroids):
        wcss = 0.0
        for k in range(self.K):
            members = X[labels == k]
            if len(members) > 0:
                diffs = members - centroids[k]
                wcss += np.sum(diffs ** 2)
        return wcss

    def _update_centroids(self, X, labels):
        centroids = np.zeros((self.K, X.shape[1]))
        for k in range(self.K):
            members = X[labels == k]
            if len(members) > 0:
                centroids[k] = np.mean(members, axis=0)
            else:

                centroids[k] = X[np.random.randint(0, len(X))]
        return centroids

    def fit(self, X, init_labels=None):
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

            for i in range(n_samples):
                li = labels[i]
                for j in range(i + 1, n_samples):
                    lj = labels[j]
                    if li == lj:
                        continue


                    old_wcss = wcss

                    labels[i], labels[j] = lj, li
                    new_centroids = self._update_centroids(X, labels)
                    new_wcss = self._compute_wcss(X, labels, new_centroids)

                    if new_wcss < old_wcss - self.tol:

                        centroids = new_centroids
                        wcss = new_wcss
                        n_swaps += 1
                        improved = True
                    else:

                        labels[i], labels[j] = li, lj

            if not improved:
                break

        return labels, centroids, wcss, n_swaps

    def predict(self, X, centroids):
        X = np.asarray(X, dtype=float)
        dists = np.zeros((X.shape[0], self.K))
        for k in range(self.K):
            dists[:, k] = np.sum((X - centroids[k]) ** 2, axis=1)
        return np.argmin(dists, axis=1)


def generate_coagulation_phenotypes(n_samples=200, seed=42):
    rng = np.random.default_rng(seed)

    centers = np.array([
        [12.0, 30.0, 0.3, 3.0, 250.0],
        [10.5, 25.0, 1.5, 4.5, 350.0],
        [15.0, 45.0, 0.8, 1.5, 120.0],
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

    perm = rng.permutation(len(X))
    return X[perm], true_labels[perm]


def demo_clustering():
    X, true_labels = generate_coagulation_phenotypes(n_samples=150, seed=42)


    X_norm = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-12)

    clusterer = SwapClustering(n_clusters=3, max_iter=100)
    labels, centroids, wcss, n_swaps = clusterer.fit(X_norm)

    print("=" * 60)
    print("血凝表型交换优化聚类分析")
    print("=" * 60)
    print(f"最终 WCSS: {wcss:.4f}")
    print(f"执行交换次数: {n_swaps}")


    from itertools import permutations
    best_acc = 0.0
    for perm in permutations(range(3)):
        mapped = np.array([perm[l] for l in labels])
        acc = np.mean(mapped == true_labels)
        best_acc = max(best_acc, acc)
    print(f"聚类准确率（最优排列）: {best_acc:.2%}")


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
