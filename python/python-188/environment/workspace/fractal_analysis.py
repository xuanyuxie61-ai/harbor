
import numpy as np


class MandelbrotSemanticBoundary:

    def __init__(self, escape_radius: float = 2.0):
        self.escape_radius = float(escape_radius)

    def iterate_point(self, c: complex, max_iter: int = 100) -> int:
        z = 0.0 + 0.0j
        for i in range(max_iter):
            z = z * z + c
            if abs(z) > self.escape_radius:
                return i + 1
        return max_iter

    def compute_region(self, x_min: float = -2.0, x_max: float = 1.0,
                       y_min: float = -1.5, y_max: float = 1.5,
                       nx: int = 101, ny: int = 101,
                       max_iter: int = 50) -> np.ndarray:
        X = np.linspace(x_min, x_max, nx)
        Y = np.linspace(y_min, y_max, ny)

        result = np.zeros((ny, nx), dtype=int)
        for j in range(ny):
            for i in range(nx):
                c = complex(X[i], Y[j])
                result[j, i] = self.iterate_point(c, max_iter)

        return result

    def estimate_fractal_dimension(self, x_min: float = -2.0, x_max: float = 0.5,
                                   y_min: float = -1.0, y_max: float = 1.0,
                                   resolutions: list = None) -> float:
        if resolutions is None:
            resolutions = [32, 64, 128, 256]

        counts = []
        epsilons = []

        for nx in resolutions:
            ny = nx
            result = self.compute_region(x_min, x_max, y_min, y_max,
                                         nx, ny, max_iter=100)


            boundary = np.zeros_like(result, dtype=bool)
            for j in range(1, ny - 1):
                for i in range(1, nx - 1):
                    if result[j, i] < 100:

                        neighbors = [
                            result[j - 1, i], result[j + 1, i],
                            result[j, i - 1], result[j, i + 1]
                        ]
                        if any(n == 100 for n in neighbors):
                            boundary[j, i] = True

            count = np.sum(boundary)
            epsilon = (x_max - x_min) / nx
            counts.append(count)
            epsilons.append(epsilon)


        log_eps = np.log(1.0 / np.array(epsilons))
        log_counts = np.log(np.array(counts))


        A = np.vstack([log_eps, np.ones(len(log_eps))]).T
        D, _ = np.linalg.lstsq(A, log_counts, rcond=None)[0]
        return float(D)


class IFSSemanticTransformer:

    def __init__(self, transformations: list = None, probabilities: list = None,
                 seed: int = 42):
        self.rng = np.random.default_rng(seed)

        if transformations is None:

            self.transformations = [
                (np.array([[0.0, 0.0], [0.0, 0.5]]), np.array([0.5, 0.0])),
                (np.array([[0.1, 0.0], [0.0, 0.1]]), np.array([0.45, 0.15])),
                (np.array([[0.42, -0.42], [0.42, 0.42]]), np.array([0.29, -0.01])),
                (np.array([[0.42, 0.42], [-0.42, 0.42]]), np.array([0.29, 0.41]))
            ]
            self.probabilities = np.array([0.25, 0.25, 0.25, 0.25])
        else:
            self.transformations = transformations
            self.probabilities = np.array(probabilities)


        self.probabilities = self.probabilities / np.sum(self.probabilities)

    def transform_embedding(self, embedding: np.ndarray, num_iterations: int = 100) -> np.ndarray:
        embedding = np.asarray(embedding, dtype=float)


        if len(embedding) > 2:

            x = embedding[:2].copy()
        else:
            x = embedding.copy()
            if len(x) < 2:
                x = np.concatenate([x, np.zeros(2 - len(x))])


        x_norm = np.linalg.norm(x)
        if x_norm > 1e-15:
            x = x / x_norm
        x = np.clip(x, 0.0, 1.0)


        points = [x.copy()]
        for _ in range(num_iterations):
            idx = self.rng.choice(len(self.transformations), p=self.probabilities)
            A, b = self.transformations[idx]
            x = A @ x + b
            points.append(x.copy())

        return np.array(points)

    def compute_attractor_bounding_box(self, num_samples: int = 10000) -> dict:
        points = []
        x = np.array([0.5, 0.5])
        for _ in range(num_samples):
            idx = self.rng.choice(len(self.transformations), p=self.probabilities)
            A, b = self.transformations[idx]
            x = A @ x + b
            points.append(x.copy())

        points = np.array(points)
        return {
            'x_min': float(points[:, 0].min()),
            'x_max': float(points[:, 0].max()),
            'y_min': float(points[:, 1].min()),
            'y_max': float(points[:, 1].max()),
            'center': points.mean(axis=0),
            'std': points.std(axis=0)
        }

    def lyapunov_exponent(self, num_iterations: int = 5000) -> float:
        x = np.array([0.5, 0.5])
        v = np.array([1.0, 0.0])
        lyap_sum = 0.0

        for _ in range(num_iterations):
            idx = self.rng.choice(len(self.transformations), p=self.probabilities)
            A, b = self.transformations[idx]
            x = A @ x + b
            v = A @ v
            v_norm = np.linalg.norm(v)
            if v_norm > 1e-15:
                lyap_sum += np.log(v_norm)
                v = v / v_norm

        return lyap_sum / num_iterations


def demo():
    print("=" * 60)
    print("语义嵌入分形与混沌分析演示")
    print("=" * 60)


    print("\n--- Mandelbrot 语义边界分析 ---")
    mbd = MandelbrotSemanticBoundary(escape_radius=2.0)
    result = mbd.compute_region(x_min=-1.0, x_max=-0.6, y_min=0.0, y_max=0.4,
                                nx=51, ny=51, max_iter=30)
    print(f"逃逸时间矩阵形状: {result.shape}")
    print(f"最大逃逸次数: {result.max()}")
    print(f"属于集合的点数: {np.sum(result >= 30)}")
    print(f"逃逸点比例: {np.sum(result < 30) / result.size * 100:.2f}%")


    print("\n分形维数估计...")
    D = mbd.estimate_fractal_dimension(
        x_min=-1.0, x_max=0.5, y_min=-1.0, y_max=1.0,
        resolutions=[32, 64, 128]
    )
    print(f"估计分形维数: {D:.4f}")
    print(f"(理论值约 2.0)")


    print("\n--- IFS 语义嵌入变换 ---")
    ifs = IFSSemanticTransformer(seed=42)
    rng = np.random.default_rng(42)
    embedding = rng.standard_normal(10)
    points = ifs.transform_embedding(embedding, num_iterations=500)
    print(f"嵌入维度: {len(embedding)}")
    print(f"IFS变换点数: {len(points)}")
    print(f"变换后范围: x=[{points[:, 0].min():.4f}, {points[:, 0].max():.4f}], "
          f"y=[{points[:, 1].min():.4f}, {points[:, 1].max():.4f}]")

    bbox = ifs.compute_attractor_bounding_box(num_samples=5000)
    print(f"\n吸引子包围盒:")
    print(f"  x: [{bbox['x_min']:.4f}, {bbox['x_max']:.4f}]")
    print(f"  y: [{bbox['y_min']:.4f}, {bbox['y_max']:.4f}]")
    print(f"  中心: [{bbox['center'][0]:.4f}, {bbox['center'][1]:.4f}]")

    lyap = ifs.lyapunov_exponent(num_iterations=5000)
    print(f"\nLyapunov指数: {lyap:.6f}")
    if lyap > 0:
        print("  => 系统呈现混沌行为")
    else:
        print("  => 系统稳定收敛")

    print("\n模块运行完成")
    return mbd, ifs


if __name__ == "__main__":
    demo()
