
import numpy as np
from utils import sparse_adjacency_to_laplacian


class BrainConnectomeGraph:

    def __init__(self, n_regions=50, connection_prob=0.15, weight_dist='lognormal',
                 random_state=None):
        if random_state is not None:
            np.random.seed(random_state)
        self.n_regions = n_regions
        self.connection_prob = connection_prob

        self.adjacency = self._build_adjacency(weight_dist)
        self.laplacian = sparse_adjacency_to_laplacian(self.adjacency)
        self.degree_matrix = np.diag(np.sum(self.adjacency, axis=1))

    def _build_adjacency(self, weight_dist):
        n = self.n_regions
        A = np.zeros((n, n), dtype=float)
        for i in range(n):
            for j in range(i + 1, n):
                if np.random.rand() < self.connection_prob:
                    if weight_dist == 'lognormal':

                        w = np.random.lognormal(mean=-1.0, sigma=1.0)
                    else:
                        w = np.random.uniform(0.1, 1.0)
                    A[i, j] = w
                    A[j, i] = w

        for i in range(n):
            if np.sum(A[i, :]) == 0:
                j = np.random.randint(0, n)
                if j == i:
                    j = (j + 1) % n
                w = 0.5
                A[i, j] = w
                A[j, i] = w
        return A

    def compute_fiedler_value(self):
        eigenvalues = np.linalg.eigvalsh(self.laplacian)

        eigenvalues = np.sort(eigenvalues)
        if len(eigenvalues) > 1:
            return float(eigenvalues[1])
        return 0.0

    def compute_effective_resistance(self, i, j):
        L_pinv = np.linalg.pinv(self.laplacian)
        return float(L_pinv[i, i] + L_pinv[j, j] - 2.0 * L_pinv[i, j])

    def compute_communicability(self, t=1.0):
        from scipy.linalg import expm
        return expm(t * self.adjacency)

    def simulate_diffusion(self, initial_concentration, dt=0.01, n_steps=1000):
        c = np.asarray(initial_concentration, dtype=float).copy()
        history = np.zeros((n_steps + 1, self.n_regions), dtype=float)
        history[0] = c.copy()
        for k in range(n_steps):
            c = c - dt * (self.laplacian @ c)
            history[k + 1] = c.copy()
        return history


class NeuralPercolationAnalyzer:

    def __init__(self, shape=(64, 64)):
        self.shape = shape
        self.M, self.N = shape

    def simulate_site_percolation(self, p, random_state=None):
        if random_state is not None:
            np.random.seed(random_state)
        occupied = np.random.rand(self.M, self.N) < p
        labels = np.zeros((self.M, self.N), dtype=int)
        label_id = 0
        sizes = []

        for i in range(self.M):
            for j in range(self.N):
                if occupied[i, j] and labels[i, j] == 0:
                    label_id += 1
                    stack = [(i, j)]
                    labels[i, j] = label_id
                    count = 0
                    while stack:
                        ci, cj = stack.pop()
                        count += 1

                        for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                            ni, nj = ci + di, cj + dj
                            if 0 <= ni < self.M and 0 <= nj < self.N:
                                if occupied[ni, nj] and labels[ni, nj] == 0:
                                    labels[ni, nj] = label_id
                                    stack.append((ni, nj))
                    sizes.append(count)
        return occupied, labels, sizes

    def find_spanning_cluster(self, labels):
        M, N = labels.shape
        spanning = False
        spanning_labels = set()

        left_labels = set(labels[:, 0])
        right_labels = set(labels[:, -1])
        horiz = left_labels & right_labels
        horiz.discard(0)
        if horiz:
            spanning = True
            spanning_labels.update(horiz)

        top_labels = set(labels[0, :])
        bottom_labels = set(labels[-1, :])
        vert = top_labels & bottom_labels
        vert.discard(0)
        if vert:
            spanning = True
            spanning_labels.update(vert)
        return spanning, spanning_labels

    def estimate_critical_threshold(self, n_samples=20, p_values=None):
        if p_values is None:
            p_values = np.linspace(0.4, 0.7, 16)
        span_probs = []
        for p in p_values:
            spans = 0
            for _ in range(n_samples):
                _, labels, _ = self.simulate_site_percolation(p)
                is_spanning, _ = self.find_spanning_cluster(labels)
                if is_spanning:
                    spans += 1
            span_probs.append(spans / n_samples)
        span_probs = np.array(span_probs)

        above_half = np.where(span_probs >= 0.5)[0]
        below_half = np.where(span_probs < 0.5)[0]
        if len(above_half) == 0 or len(below_half) == 0:
            p_c = p_values[np.argmin(np.abs(span_probs - 0.5))]
        else:
            i_low = below_half[-1]
            i_high = above_half[0]
            p_low, p_high = p_values[i_low], p_values[i_high]
            prob_low, prob_high = span_probs[i_low], span_probs[i_high]
            if abs(prob_high - prob_low) < 1e-10:
                p_c = 0.5 * (p_low + p_high)
            else:
                p_c = p_low + (0.5 - prob_low) * (p_high - p_low) / (prob_high - prob_low)
        return p_c, p_values, span_probs

    def compute_fractal_dimension(self, labels, target_label):
        mask = (labels == target_label)
        coords = np.argwhere(mask)
        if len(coords) < 10:
            return 0.0
        max_box = min(self.M, self.N) // 2
        if max_box < 2:
            max_box = 2
        box_sizes = 2 ** np.arange(1, int(np.log2(max_box)) + 1)
        counts = []
        for box_size in box_sizes:

            bins_i = coords[:, 0] // box_size
            bins_j = coords[:, 1] // box_size
            unique_boxes = set(zip(bins_i, bins_j))
            counts.append(len(unique_boxes))
        counts = np.array(counts, dtype=float)
        box_sizes = np.array(box_sizes, dtype=float)
        valid = counts > 0
        if np.sum(valid) < 2:
            return 0.0

        log_boxes = np.log(box_sizes[valid])
        log_counts = np.log(counts[valid])
        coeffs = np.polyfit(log_boxes, log_counts, 1)
        D = -coeffs[0]
        return float(D)


class ConnectomePercolationBridge:

    def __init__(self, connectome, grid_shape=(32, 32)):
        self.connectome = connectome
        self.grid_shape = grid_shape
        self.percolation = NeuralPercolationAnalyzer(grid_shape)

    def map_connectome_to_grid(self, active_regions, activation_strength):
        M, N = self.grid_shape
        grid = np.zeros((M, N), dtype=float)
        n_regions = self.connectome.n_regions
        for r in range(n_regions):

            ci = np.random.randint(0, M)
            cj = np.random.randint(0, N)
            radius = max(1, int(np.sqrt(M * N / n_regions) * 0.5))
            for i in range(max(0, ci - radius), min(M, ci + radius + 1)):
                for j in range(max(0, cj - radius), min(N, cj + radius + 1)):
                    if (i - ci) ** 2 + (j - cj) ** 2 <= radius ** 2:
                        if r in active_regions:
                            grid[i, j] = activation_strength[r]
        return grid

    def analyze_information_spread(self, seed_region, steps=50):
        n = self.connectome.n_regions
        activation = np.zeros(n, dtype=float)
        activation[seed_region] = 1.0
        history = [activation.copy()]

        A_norm = self.connectome.adjacency.copy()
        row_sums = np.sum(A_norm, axis=1)
        row_sums[row_sums == 0] = 1.0
        A_norm = A_norm / row_sums[:, None]
        for _ in range(steps):
            activation = 0.8 * activation + 0.2 * (A_norm @ activation)
            activation = np.clip(activation, 0.0, 1.0)
            history.append(activation.copy())

        active_regions = [i for i, a in enumerate(activation) if a > 0.3]
        grid = self.map_connectome_to_grid(active_regions, activation)

        occupied = grid > np.mean(grid)
        labels = np.zeros(grid.shape, dtype=int)
        label_id = 0
        sizes = []
        M, N = grid.shape
        for i in range(M):
            for j in range(N):
                if occupied[i, j] and labels[i, j] == 0:
                    label_id += 1
                    stack = [(i, j)]
                    labels[i, j] = label_id
                    count = 0
                    while stack:
                        ci, cj = stack.pop()
                        count += 1
                        for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                            ni, nj = ci + di, cj + dj
                            if 0 <= ni < M and 0 <= nj < N:
                                if occupied[ni, nj] and labels[ni, nj] == 0:
                                    labels[ni, nj] = label_id
                                    stack.append((ni, nj))
                    sizes.append(count)
        is_spanning, _ = self.percolation.find_spanning_cluster(labels)
        max_size = max(sizes) if sizes else 0
        return {
            'activation_history': np.array(history),
            'final_activation': activation,
            'max_cluster_size': max_size,
            'is_spanning': is_spanning,
            'grid_labels': labels
        }
