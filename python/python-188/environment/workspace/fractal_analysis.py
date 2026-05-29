"""
分形与混沌分析模块：语义嵌入空间的复杂几何结构分析

原项目映射: 710_mandelbrot, 1290_tree_chaos

科学背景:
    1. Mandelbrot集合:
       对于复数 c，迭代:
           z_{n+1} = z_n^2 + c
       如果 |z_n| 始终保持有界，则 c 属于 Mandelbrot 集。
       
       逃逸时间算法:
           记录迭代多少次后 |z| > 2（逃逸半径）。
    
    2. 迭代函数系统 (IFS):
       使用一组仿射变换:
           x_{n+1} = A_k * x_n + b_k
       其中 k 以一定概率选择。
       
       树形混沌IFS:
           A0 = [[0, 0], [0, 0.5]],  b0 = [0.5, 0]
           A1 = [[0.1, 0], [0, 0.1]], b1 = [0.45, 0.15]
           A2 = [[0.42, -0.42], [0.42, 0.42]], b2 = [0.29, -0.01]
           A3 = [[0.42, 0.42], [-0.42, 0.42]], b3 = [0.29, 0.41]

在NLP语义嵌入中的应用:
    - 分析语义嵌入空间的边界复杂度和分形维数
    - 使用IFS进行语义嵌入的多尺度变换和增强
    - 通过混沌动力学理解语义漂移的不可预测性
"""

import numpy as np


class MandelbrotSemanticBoundary:
    """
    Mandelbrot 语义边界分析器。
    
    分析语义嵌入空间中类似Mandelbrot集的复杂边界结构。
    """

    def __init__(self, escape_radius: float = 2.0):
        self.escape_radius = float(escape_radius)

    def iterate_point(self, c: complex, max_iter: int = 100) -> int:
        """
        计算单点的逃逸迭代次数。
        
        Returns
        -------
        int
            逃逸前的迭代次数，如果始终有界则返回 max_iter。
        """
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
        """
        计算矩形区域内的逃逸时间。
        
        Returns
        -------
        np.ndarray
            ny x nx 的逃逸时间矩阵。
        """
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
        """
        使用盒计数法估计分形维数。
        
        对于尺度 epsilon，计算覆盖边界所需的盒子数 N(epsilon)。
        分形维数: D = -lim_{epsilon->0} log(N) / log(epsilon)
        """
        if resolutions is None:
            resolutions = [32, 64, 128, 256]

        counts = []
        epsilons = []

        for nx in resolutions:
            ny = nx
            result = self.compute_region(x_min, x_max, y_min, y_max,
                                         nx, ny, max_iter=100)
            # 边界点：逃逸时间既不是0也不是max_iter
            # 或者相邻点有不同逃逸时间
            boundary = np.zeros_like(result, dtype=bool)
            for j in range(1, ny - 1):
                for i in range(1, nx - 1):
                    if result[j, i] < 100:
                        # 检查是否与内部点相邻
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

        # 线性拟合 log(N) vs log(1/epsilon)
        log_eps = np.log(1.0 / np.array(epsilons))
        log_counts = np.log(np.array(counts))

        # 最小二乘拟合
        A = np.vstack([log_eps, np.ones(len(log_eps))]).T
        D, _ = np.linalg.lstsq(A, log_counts, rcond=None)[0]
        return float(D)


class IFSSemanticTransformer:
    """
    IFS语义嵌入变换器。
    
    使用迭代函数系统进行语义嵌入的多尺度变换。
    """

    def __init__(self, transformations: list = None, probabilities: list = None,
                 seed: int = 42):
        """
        Parameters
        ----------
        transformations : list
            仿射变换矩阵列表，每个元素为 (A, b) 元组。
        probabilities : list
            每个变换的选择概率。
        """
        self.rng = np.random.default_rng(seed)

        if transformations is None:
            # 默认树形IFS
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

        # 归一化概率
        self.probabilities = self.probabilities / np.sum(self.probabilities)

    def transform_embedding(self, embedding: np.ndarray, num_iterations: int = 100) -> np.ndarray:
        """
        对语义嵌入进行IFS变换。
        
        将高维嵌入投影到2D后进行IFS迭代。
        """
        embedding = np.asarray(embedding, dtype=float)

        # 投影到2D (使用PCA风格的前两个主成分)
        if len(embedding) > 2:
            # 简化为前两个维度
            x = embedding[:2].copy()
        else:
            x = embedding.copy()
            if len(x) < 2:
                x = np.concatenate([x, np.zeros(2 - len(x))])

        # 归一化到单位正方形
        x_norm = np.linalg.norm(x)
        if x_norm > 1e-15:
            x = x / x_norm
        x = np.clip(x, 0.0, 1.0)

        # IFS迭代
        points = [x.copy()]
        for _ in range(num_iterations):
            idx = self.rng.choice(len(self.transformations), p=self.probabilities)
            A, b = self.transformations[idx]
            x = A @ x + b
            points.append(x.copy())

        return np.array(points)

    def compute_attractor_bounding_box(self, num_samples: int = 10000) -> dict:
        """
        计算吸引子的包围盒。
        """
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
        """
        估计最大Lyapunov指数。
        
        lambda = lim_{n->inf} (1/n) * sum_{k=1}^n ln(||J_k * v_k|| / ||v_k||)
        """
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
    """模块功能演示"""
    print("=" * 60)
    print("语义嵌入分形与混沌分析演示")
    print("=" * 60)

    # Mandelbrot
    print("\n--- Mandelbrot 语义边界分析 ---")
    mbd = MandelbrotSemanticBoundary(escape_radius=2.0)
    result = mbd.compute_region(x_min=-1.0, x_max=-0.6, y_min=0.0, y_max=0.4,
                                nx=51, ny=51, max_iter=30)
    print(f"逃逸时间矩阵形状: {result.shape}")
    print(f"最大逃逸次数: {result.max()}")
    print(f"属于集合的点数: {np.sum(result >= 30)}")
    print(f"逃逸点比例: {np.sum(result < 30) / result.size * 100:.2f}%")

    # 分形维数估计（使用较低分辨率加速）
    print("\n分形维数估计...")
    D = mbd.estimate_fractal_dimension(
        x_min=-1.0, x_max=0.5, y_min=-1.0, y_max=1.0,
        resolutions=[32, 64, 128]
    )
    print(f"估计分形维数: {D:.4f}")
    print(f"(理论值约 2.0)")

    # IFS
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
