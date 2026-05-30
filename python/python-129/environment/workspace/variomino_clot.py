
import numpy as np


class VariominoClot:

    def __init__(self, density_matrix):
        self.P = np.asarray(density_matrix, dtype=float)
        if self.P.ndim != 2:
            raise ValueError("density_matrix 必须为二维数组")

    def condense(self, threshold=1e-12):
        Q = self.P.copy()

        while Q.shape[0] > 0 and np.all(Q[0, :] <= threshold):
            Q = Q[1:, :]

        while Q.shape[0] > 0 and np.all(Q[-1, :] <= threshold):
            Q = Q[:-1, :]

        while Q.shape[1] > 0 and np.all(Q[:, 0] <= threshold):
            Q = Q[:, 1:]

        while Q.shape[1] > 0 and np.all(Q[:, -1] <= threshold):
            Q = Q[:, :-1]
        if Q.size == 0:
            Q = np.zeros((1, 1))
        return Q

    def embed_in_domain(self, domain_shape, center=None):
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

        q_top = max(0, -top)
        q_left = max(0, -left)
        q_bottom = q_top + (bottom - top)
        q_right = q_left + (right - left)
        embedded[top:bottom, left:right] = Q[q_top:q_bottom, q_left:q_right]
        return embedded

    def rotate90(self, k=1):
        return np.rot90(self.P, k=k)

    def flip_horizontal(self):
        return np.fliplr(self.P)

    def flip_vertical(self):
        return np.flipud(self.P)

    def count_pore_clusters(self, threshold=0.05):
        binary = (self.P <= threshold).astype(int)
        h, w = binary.shape
        labels = np.zeros((h, w), dtype=int)
        label_id = 0

        for i in range(h):
            for j in range(w):
                if binary[i, j] == 1 and labels[i, j] == 0:
                    label_id += 1

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

        from scipy.ndimage import gaussian_filter
        P = gaussian_filter(P, sigma=1.0)
        P = np.clip(P, 0, 1)
        return P


def demo_variomino():
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
