
import numpy as np
from scipy.sparse import csr_matrix


class CoagulationNetworkGraph:

    def __init__(self):
        self.node_names = [
            "TF", "VII", "VIIa", "TF_VIIa",
            "IX", "IXa", "VIII", "VIIIa",
            "X", "Xa", "V", "Va",
            "II", "IIa", "Fibrinogen", "Fibrin",
            "XI", "XIa", "XII", "XIIa",
            "PC", "APC", "PS", "TM",
            "ATIII", "TFPI", "Plasmin", "tPA"
        ]
        self.n_nodes = len(self.node_names)
        self.name_to_idx = {name: i for i, name in enumerate(self.node_names)}
        self.adj = np.zeros((self.n_nodes, self.n_nodes), dtype=int)
        self._build_edges()

    def _build_edges(self):
        edges = [
            ("TF", "TF_VIIa"),
            ("VIIa", "TF_VIIa"),
            ("TF_VIIa", "IXa"),
            ("TF_VIIa", "Xa"),
            ("IXa", "Xa"),
            ("VIIIa", "IXa"),
            ("Xa", "IIa"),
            ("Va", "IIa"),
            ("IIa", "Fibrin"),
            ("IIa", "APC"),
            ("APC", "Va"),
            ("XIa", "IXa"),
            ("XIIa", "XIa"),
            ("tPA", "Plasmin"),
            ("Plasmin", "Fibrin"),
            ("ATIII", "Xa"),
            ("ATIII", "IIa"),
            ("ATIII", "IXa"),
            ("TFPI", "Xa"),
            ("TFPI", "TF_VIIa"),
        ]
        for src, dst in edges:
            if src in self.name_to_idx and dst in self.name_to_idx:
                i = self.name_to_idx[src]
                j = self.name_to_idx[dst]
                self.adj[i, j] = 1

    def build_sparse_matrix(self):
        rows, cols = np.where(self.adj > 0)
        data = np.ones(len(rows), dtype=int)
        return csr_matrix((data, (rows, cols)), shape=(self.n_nodes, self.n_nodes))

    def pagerank(self, alpha=0.85, max_iter=200, tol=1e-12):
        n = self.n_nodes
        A = self.adj.astype(float)

        out_deg = A.sum(axis=1)
        S = np.zeros_like(A)
        for i in range(n):
            if out_deg[i] > 0:
                S[i, :] = A[i, :] / out_deg[i]
            else:
                S[i, :] = 1.0 / n

        pi = np.ones(n) / n
        for _ in range(max_iter):
            pi_new = alpha * pi @ S + (1.0 - alpha) / n
            if np.linalg.norm(pi_new - pi, ord=1) < tol:
                break
            pi = pi_new
        return pi

    def jaccard_distance_matrix(self):
        n = self.n_nodes
        dist = np.zeros((n, n))
        for i in range(n):
            Ni = set(np.where(self.adj[i, :] > 0)[0])
            Ni.update(np.where(self.adj[:, i] > 0)[0])
            for j in range(i + 1, n):
                Nj = set(np.where(self.adj[j, :] > 0)[0])
                Nj.update(np.where(self.adj[:, j] > 0)[0])
                inter = len(Ni & Nj)
                union = len(Ni | Nj)
                if union == 0:
                    d = 0.0
                else:
                    d = 1.0 - inter / union
                dist[i, j] = d
                dist[j, i] = d

        return dist

    def hierarchical_clustering(self, dist_matrix):
        n = dist_matrix.shape[0]
        clusters = {i: [i] for i in range(n)}
        linkage = []
        active = list(range(n))
        next_id = n

        while len(active) > 1:
            min_d = np.inf
            pair = (-1, -1)
            for idx_i in range(len(active)):
                i = active[idx_i]
                for idx_j in range(idx_i + 1, len(active)):
                    j = active[idx_j]
                    ci = clusters[i]
                    cj = clusters[j]
                    d_min = np.min(dist_matrix[np.ix_(ci, cj)])
                    if d_min < min_d:
                        min_d = d_min
                        pair = (i, j)

            i, j = pair
            linkage.append((i, j, min_d, len(clusters[i]) + len(clusters[j])))
            clusters[next_id] = clusters[i] + clusters[j]
            active.remove(i)
            active.remove(j)
            active.append(next_id)
            next_id += 1

        return linkage, clusters


def analyze_network():
    net = CoagulationNetworkGraph()
    pr = net.pagerank()

    print("=" * 60)
    print("血凝级联反应网络 PageRank 分析")
    print("=" * 60)
    ranked = sorted(zip(net.node_names, pr), key=lambda x: x[1], reverse=True)
    for name, score in ranked[:10]:
        print(f"  {name:20s}: {score:.6f}")

    dist = net.jaccard_distance_matrix()
    linkage, clusters = net.hierarchical_clustering(dist)
    print(f"\n层次聚类完成，共 {len(linkage)} 次合并")
    print(f"最终类包含节点数: {len(clusters[linkage[-1][0]]) + len(clusters[linkage[-1][1]])}")
    return net, pr, dist, linkage


if __name__ == "__main__":
    analyze_network()
