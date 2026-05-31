"""
process_sampling.py
===================
模拟超构表面纳米加工工艺中的随机涨落与空间不确定性。

本模块融合项目 1006_random_data（各种几何区域上的均匀随机采样）
与 291_discrete_pdf_sample_2d（二维离散概率分布采样）的核心算法，
对纳米柱的高度、宽度、位置和侧壁角等工艺参数进行随机扰动采样，
以评估制造容差对光学性能的影响。

科学背景：
在电子束光刻（EBL）和反应离子刻蚀（RIE）过程中，纳米结构存在
多种随机误差源：
1. 线边缘粗糙度（LER, Line Edge Roughness）
2. 刻蚀深度不均匀性
3. 侧壁角偏离（Sidewall Angle Deviation）
4. 位置套刻误差（Overlay Error）

这些误差可以用二维空间随机场描述：
    δh(x,y) ~ GP(0, C_h(r)),   C_h(r) = σ_h² exp(-r² / 2l_c²)
其中 GP 表示高斯随机场，l_c 为相关长度。

本模块提供：
- 纳米柱截面上的均匀随机采样（用于 Monte-Carlo 误差统计）
- 离散工艺参数分布的二维采样
- 工艺误差随机场的生成与重构
"""

import numpy as np


class ProcessSampler:
    """
    纳米加工工艺随机采样与误差建模。
    """

    def __init__(self, seed=None):
        if seed is not None:
            np.random.seed(seed)

    # ------------------------------------------------------------------
    # 几何区域上的均匀采样（源自 1006_random_data）
    # ------------------------------------------------------------------
    @staticmethod
    def uniform_in_triangle(n, v1, v2, v3):
        """
        在三角形内生成 n 个均匀分布的随机点（Turk 规则 #1）。

        参数化映射：
            P = a V1 + b V2 + c V3
        其中 a = 1 - sqrt(r2), b = (1-r1)*sqrt(r2), c = r1*sqrt(r2)。
        """
        points = np.zeros((n, 2), dtype=np.float64)
        r = np.random.rand(n, 2)
        a = 1.0 - np.sqrt(r[:, 1])
        b = (1.0 - r[:, 0]) * np.sqrt(r[:, 1])
        c = r[:, 0] * np.sqrt(r[:, 1])
        points = (a[:, None] * v1[None, :] +
                  b[:, None] * v2[None, :] +
                  c[:, None] * v3[None, :])
        return points

    @staticmethod
    def uniform_in_annulus(n, r1, r2, center=(0.0, 0.0)):
        """
        在圆环 r1 ≤ ρ ≤ r2 内均匀采样。
        逆 CDF 方法：ρ = sqrt(r1² + U (r2² - r1²))。
        """
        u = np.random.rand(n)
        rho = np.sqrt(r1 ** 2 + u * (r2 ** 2 - r1 ** 2))
        theta = 2.0 * np.pi * np.random.rand(n)
        x = center[0] + rho * np.cos(theta)
        y = center[1] + rho * np.sin(theta)
        return np.column_stack([x, y])

    @staticmethod
    def uniform_in_polygon_convex(n, vertices):
        """
        在凸多边形内均匀采样。
        方法：将多边形三角化后，按面积加权选择三角形，
        再在三角形内均匀采样。
        """
        vertices = np.array(vertices, dtype=np.float64)
        n_v = len(vertices)
        if n_v < 3:
            raise ValueError("多边形至少需要 3 个顶点")

        # 以第一个顶点为基准三角化
        triangles = []
        areas = []
        for i in range(1, n_v - 1):
            tri = [vertices[0], vertices[i], vertices[i + 1]]
            # 面积（叉积的一半）
            area = 0.5 * abs(
                (tri[1][0] - tri[0][0]) * (tri[2][1] - tri[0][1]) -
                (tri[2][0] - tri[0][0]) * (tri[1][1] - tri[0][1])
            )
            triangles.append(tri)
            areas.append(area)

        areas = np.array(areas)
        total_area = np.sum(areas)
        if total_area < 1e-18:
            raise ValueError("多边形面积为零")
        probs = areas / total_area

        # 按面积加权选择三角形
        choices = np.random.choice(len(triangles), size=n, p=probs)
        points = np.zeros((n, 2), dtype=np.float64)
        for i in range(n):
            tri = triangles[choices[i]]
            pts = ProcessSampler.uniform_in_triangle(1, tri[0], tri[1], tri[2])
            points[i] = pts[0]
        return points

    # ------------------------------------------------------------------
    # 离散二维概率分布采样（源自 291_discrete_pdf_sample_2d）
    # ------------------------------------------------------------------
    @staticmethod
    def sample_discrete_cdf_2d(n_samples, pdf_grid, x_range, y_range):
        """
        基于二维离散 PDF 网格进行采样。

        Parameters
        ----------
        n_samples : int
        pdf_grid : ndarray, shape (nx, ny)
            非负概率密度函数值（无需归一化）
        x_range : tuple (xmin, xmax)
        y_range : tuple (ymin, ymax)

        Returns
        -------
        samples : ndarray, shape (n_samples, 2)
        """
        nx, ny = pdf_grid.shape
        pdf_flat = pdf_grid.flatten()
        cdf = np.cumsum(pdf_flat)
        if cdf[-1] < 1e-18:
            cdf = np.arange(1, len(cdf) + 1, dtype=np.float64)
        cdf = cdf / cdf[-1]

        u = np.random.rand(n_samples)
        # 二分查找每个 u 对应的 CDF 区间
        indices = np.searchsorted(cdf, u)
        indices = np.clip(indices, 0, len(cdf) - 1)

        # 将一维索引转回二维网格索引
        ix = indices // ny
        iy = indices % ny

        xmin, xmax = x_range
        ymin, ymax = y_range
        dx = (xmax - xmin) / nx
        dy = (ymax - ymin) / ny

        # 在网格单元内均匀采样
        samples = np.zeros((n_samples, 2), dtype=np.float64)
        samples[:, 0] = xmin + (ix + np.random.rand(n_samples)) * dx
        samples[:, 1] = ymin + (iy + np.random.rand(n_samples)) * dy
        return samples

    # ------------------------------------------------------------------
    # 工艺误差随机场生成
    # ------------------------------------------------------------------
    def generate_height_error_field(self, x_grid, y_grid,
                                     sigma_h=5.0e-9, correlation_length=1.0e-6):
        """
        生成高度误差的二维高斯随机场。

        功率谱密度（高斯型）：
            S_h(k) = σ_h² (l_c² / 2π) exp(-k² l_c² / 2)

        使用谱方法生成：
            δh(x,y) = IFFT[ sqrt(S_h(k)) * N(0,1) ]
        """
        nx = len(x_grid)
        ny = len(y_grid)
        dx = x_grid[1] - x_grid[0] if nx > 1 else 1.0
        dy = y_grid[1] - y_grid[0] if ny > 1 else 1.0

        # 频率网格
        kx = 2.0 * np.pi * np.fft.fftfreq(nx, d=dx)
        ky = 2.0 * np.pi * np.fft.fftfreq(ny, d=dy)
        KX, KY = np.meshgrid(kx, ky, indexing='ij')
        k2 = KX ** 2 + KY ** 2

        # 功率谱
        psd = sigma_h ** 2 * (correlation_length ** 2 / (2.0 * np.pi)) * np.exp(
            -k2 * correlation_length ** 2 / 2.0
        )
        # 零频修正
        psd[0, 0] = 0.0

        # 随机相位
        noise_real = np.random.randn(nx, ny)
        noise_imag = np.random.randn(nx, ny)
        noise = noise_real + 1.0j * noise_imag

        # 频域滤波
        spectrum = np.sqrt(psd) * noise
        field = np.fft.ifft2(spectrum).real
        return field

    def sample_process_variations(self, n_samples, base_params,
                                   sigma_params, correlation_matrix=None):
        """
        对纳米柱参数进行多变量高斯采样。

        Parameters
        ----------
        n_samples : int
        base_params : ndarray, shape (n_params,)
            [h, w, x, y, sidewall_angle]
        sigma_params : ndarray, shape (n_params,)
        correlation_matrix : ndarray, shape (n_params, n_params), optional
            相关性矩阵（默认单位矩阵）

        Returns
        -------
        samples : ndarray, shape (n_samples, n_params)
        """
        n_params = len(base_params)
        cov = np.diag(sigma_params ** 2)
        if correlation_matrix is not None:
            # 将相关性转为协方差
            for i in range(n_params):
                for j in range(n_params):
                    cov[i, j] = correlation_matrix[i, j] * sigma_params[i] * sigma_params[j]

        # Cholesky 分解生成相关随机变量
        L = np.linalg.cholesky(cov + 1e-12 * np.eye(n_params))
        z = np.random.randn(n_samples, n_params)
        deviations = z @ L.T
        samples = base_params[None, :] + deviations
        return samples

    def lanczos_resampling(self, field, x_src, y_src, x_dst, y_dst):
        """
        使用 Lanczos 重采样将误差场插值到任意目标位置。
        用于将连续误差场赋给离散纳米柱位置。
        """
        nx, ny = field.shape
        dx = x_src[1] - x_src[0]
        dy = y_src[1] - y_src[0]
        a = 2  # Lanczos 核阶数

        def lanczos_kernel(v, a_val):
            v = np.abs(v)
            return np.where(v < a_val, np.sinc(v) * np.sinc(v / a_val), 0.0)

        values = np.zeros(len(x_dst), dtype=np.float64)
        for i in range(len(x_dst)):
            xd = x_dst[i]
            yd = y_dst[i]
            # 找到最近的源网格点
            ix0 = int(np.floor((xd - x_src[0]) / dx))
            iy0 = int(np.floor((yd - y_src[0]) / dy))
            val = 0.0
            w_sum = 0.0
            for ix in range(max(0, ix0 - a + 1), min(nx, ix0 + a + 1)):
                for iy in range(max(0, iy0 - a + 1), min(ny, iy0 + a + 1)):
                    wx = lanczos_kernel((xd - x_src[ix]) / dx, a)
                    wy = lanczos_kernel((yd - y_src[iy]) / dy, a)
                    w = wx * wy
                    val += w * field[ix, iy]
                    w_sum += w
            if w_sum > 1e-15:
                values[i] = val / w_sum
            else:
                values[i] = 0.0
        return values


def demo():
    """演示：生成工艺误差场并采样纳米柱参数。"""
    ps = ProcessSampler(seed=42)

    # 1. 在纳米柱截面上均匀采样（用于 Monte-Carlo LER 分析）
    tri_points = ps.uniform_in_triangle(1000,
                                        np.array([0.0, 0.0]),
                                        np.array([0.3e-6, 0.0]),
                                        np.array([0.0, 0.3e-6]))
    print(f"[process_sampling] 三角形内采样: mean=({tri_points[:,0].mean():.3e}, "
          f"{tri_points[:,1].mean():.3e})")

    # 2. 生成高度误差随机场
    x_grid = np.linspace(-5e-6, 5e-6, 128)
    y_grid = np.linspace(-5e-6, 5e-6, 128)
    h_field = ps.generate_height_error_field(x_grid, y_grid,
                                              sigma_h=5e-9,
                                              correlation_length=0.8e-6)
    print(f"[process_sampling] 高度误差场: mean={h_field.mean():.3e} m, "
          f"std={h_field.std():.3e} m")

    # 3. 多变量工艺参数采样
    base = np.array([0.6e-6, 0.3e-6, 0.0, 0.0, 90.0])  # h, w, x, y, angle
    sigma = np.array([0.02e-6, 0.01e-6, 0.005e-6, 0.005e-6, 2.0])
    corr = np.eye(5)
    corr[0, 1] = corr[1, 0] = 0.3  # 高度与宽度正相关
    samples = ps.sample_process_variations(500, base, sigma, corr)
    print(f"[process_sampling] 参数采样均值: h={samples[:,0].mean():.3e}, "
          f"w={samples[:,1].mean():.3e}")
    print(f"[process_sampling] 参数采样标准差: h={samples[:,0].std():.3e}, "
          f"w={samples[:,1].std():.3e}")

    # 4. 将误差场插值到纳米柱位置
    pillar_x = np.random.uniform(-4e-6, 4e-6, size=50)
    pillar_y = np.random.uniform(-4e-6, 4e-6, size=50)
    h_errors = ps.lanczos_resampling(h_field, x_grid, y_grid, pillar_x, pillar_y)
    print(f"[process_sampling] 纳米柱高度误差插值: mean={h_errors.mean():.3e}, "
          f"std={h_errors.std():.3e}")
    return h_field, samples


if __name__ == "__main__":
    demo()
