
import numpy as np

try:
    from scipy.spatial.distance import cdist, squareform
    from scipy.cluster.hierarchy import linkage, fcluster
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False


class LipidDistanceMatrix:

    def __init__(self, nx, ny, spatial_weight=2.0, sigma=1.5):
        self.nx = nx
        self.ny = ny
        self.n = nx * ny
        self.spatial_weight = spatial_weight
        self.sigma = sigma

    def compute_distance_matrix(self, feature_vectors, positions=None):
        feat = np.asarray(feature_vectors)
        n = feat.shape[0]
        if positions is None:
            pos = np.array([[i, j] for i in range(self.nx) for j in range(self.ny)])
        else:
            pos = np.asarray(positions)

        if _HAS_SCIPY:
            d_feat = cdist(feat, feat, metric='euclidean')
            dx = np.abs(pos[:, 0][:, None] - pos[:, 0][None, :])
            dy = np.abs(pos[:, 1][:, None] - pos[:, 1][None, :])
            dx = np.minimum(dx, self.nx - dx)
            dy = np.minimum(dy, self.ny - dy)
            d_spatial = np.sqrt(dx ** 2 + dy ** 2)
            d_total = d_feat + self.spatial_weight * np.exp(-d_spatial ** 2 / (2.0 * self.sigma ** 2))
            return d_total


        dist = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                d_feat = np.linalg.norm(feat[i] - feat[j])
                dx = abs(pos[i, 0] - pos[j, 0])
                dy = abs(pos[i, 1] - pos[j, 1])
                dx = min(dx, self.nx - dx)
                dy = min(dy, self.ny - dy)
                d_spatial = np.sqrt(dx ** 2 + dy ** 2)
                d_total = d_feat + self.spatial_weight * np.exp(-d_spatial ** 2 / (2.0 * self.sigma ** 2))
                dist[i, j] = d_total
                dist[j, i] = d_total
        return dist


class HierarchicalClustering:

    def __init__(self, distance_matrix):
        self.dist = np.asarray(distance_matrix)
        self.n = self.dist.shape[0]
        if self.dist.shape[0] != self.dist.shape[1]:
            raise ValueError("距离矩阵必须是方阵。")

    def cluster(self):
        if _HAS_SCIPY:

            condensed = squareform(self.dist, checks=False)
            Z = linkage(condensed, method='single')
            return Z


        linkage_list = []
        current_label = self.n
        active = set(range(self.n))
        label_to_points = {i: [i] for i in range(self.n)}

        for _ in range(self.n - 1):
            if len(active) < 2:
                break
            min_dist = np.inf
            pair = (-1, -1)
            active_list = sorted(active)
            for a_idx, a in enumerate(active_list):
                for b in active_list[a_idx + 1:]:
                    d_ab = np.min([self.dist[i, j]
                                   for i in label_to_points[a]
                                   for j in label_to_points[b]])
                    if d_ab < min_dist:
                        min_dist = d_ab
                        pair = (a, b)

            if pair[0] == -1:
                break

            a, b = pair
            linkage_list.append([a, b, min_dist, len(label_to_points[a]) + len(label_to_points[b])])

            new_points = label_to_points[a] + label_to_points[b]
            label_to_points[current_label] = new_points
            active.remove(a)
            active.remove(b)
            active.add(current_label)
            current_label += 1

        return np.array(linkage_list)

    def cut_tree(self, linkage_matrix, n_clusters):
        n = self.n
        if n_clusters < 1 or n_clusters > n:
            raise ValueError("n_clusters 范围无效。")

        if _HAS_SCIPY:
            from scipy.cluster.hierarchy import fcluster
            labels = fcluster(linkage_matrix, n_clusters, criterion='maxclust')
            return labels - 1


        parent = np.arange(2 * n)

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x, y):
            rx, ry = find(x), find(y)
            if rx != ry:
                parent[rx] = ry

        n_merges = len(linkage_matrix)
        n_current = n
        for i in range(n_merges):
            if n_current <= n_clusters:
                break
            a, b = int(linkage_matrix[i, 0]), int(linkage_matrix[i, 1])
            ra, rb = find(a), find(b)
            if ra != rb:
                union(a, b)
                n_current -= 1

        cluster_map = {}
        cluster_ids = np.zeros(n, dtype=int)
        next_id = 0
        for i in range(n):
            root = find(i)
            if root not in cluster_map:
                cluster_map[root] = next_id
                next_id += 1
            cluster_ids[i] = cluster_map[root]

        return cluster_ids

    def domain_size_distribution(self, linkage_matrix, n_clusters):
        labels = self.cut_tree(linkage_matrix, n_clusters)
        unique, counts = np.unique(labels, return_counts=True)
        return counts

    def interface_energy_estimate(self, linkage_matrix, temperature=300.0, kb=0.008314):
        if len(linkage_matrix) == 0:
            return 0.0
        h_max = np.max(linkage_matrix[:, 2])
        l0 = 1.0
        gamma = kb * temperature * h_max / l0
        return float(gamma)


def chain_letter_style_symmetrization(dist_matrix):
    D = np.asarray(dist_matrix)
    return 0.5 * (D + D.T)


def order_parameter_to_feature_vector(S2_local, area_ratio, density_head):
    S2_local = np.asarray(S2_local).ravel()
    area_ratio = np.asarray(area_ratio).ravel()
    density_head = np.asarray(density_head).ravel()
    n = len(S2_local)
    feats = np.zeros((n, 3))
    feats[:, 0] = S2_local
    feats[:, 1] = area_ratio
    feats[:, 2] = density_head
    return feats
