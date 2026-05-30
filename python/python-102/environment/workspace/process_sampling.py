
import numpy as np


class ProcessSampler:

    def __init__(self, seed=None):
        if seed is not None:
            np.random.seed(seed)




    @staticmethod
    def uniform_in_triangle(n, v1, v2, v3):
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
        u = np.random.rand(n)
        rho = np.sqrt(r1 ** 2 + u * (r2 ** 2 - r1 ** 2))
        theta = 2.0 * np.pi * np.random.rand(n)
        x = center[0] + rho * np.cos(theta)
        y = center[1] + rho * np.sin(theta)
        return np.column_stack([x, y])

    @staticmethod
    def uniform_in_polygon_convex(n, vertices):
        vertices = np.array(vertices, dtype=np.float64)
        n_v = len(vertices)
        if n_v < 3:
            raise ValueError("多边形至少需要 3 个顶点")


        triangles = []
        areas = []
        for i in range(1, n_v - 1):
            tri = [vertices[0], vertices[i], vertices[i + 1]]

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


        choices = np.random.choice(len(triangles), size=n, p=probs)
        points = np.zeros((n, 2), dtype=np.float64)
        for i in range(n):
            tri = triangles[choices[i]]
            pts = ProcessSampler.uniform_in_triangle(1, tri[0], tri[1], tri[2])
            points[i] = pts[0]
        return points




    @staticmethod
    def sample_discrete_cdf_2d(n_samples, pdf_grid, x_range, y_range):
        nx, ny = pdf_grid.shape
        pdf_flat = pdf_grid.flatten()
        cdf = np.cumsum(pdf_flat)
        if cdf[-1] < 1e-18:
            cdf = np.arange(1, len(cdf) + 1, dtype=np.float64)
        cdf = cdf / cdf[-1]

        u = np.random.rand(n_samples)

        indices = np.searchsorted(cdf, u)
        indices = np.clip(indices, 0, len(cdf) - 1)


        ix = indices // ny
        iy = indices % ny

        xmin, xmax = x_range
        ymin, ymax = y_range
        dx = (xmax - xmin) / nx
        dy = (ymax - ymin) / ny


        samples = np.zeros((n_samples, 2), dtype=np.float64)
        samples[:, 0] = xmin + (ix + np.random.rand(n_samples)) * dx
        samples[:, 1] = ymin + (iy + np.random.rand(n_samples)) * dy
        return samples




    def generate_height_error_field(self, x_grid, y_grid,
                                     sigma_h=5.0e-9, correlation_length=1.0e-6):
        nx = len(x_grid)
        ny = len(y_grid)
        dx = x_grid[1] - x_grid[0] if nx > 1 else 1.0
        dy = y_grid[1] - y_grid[0] if ny > 1 else 1.0


        kx = 2.0 * np.pi * np.fft.fftfreq(nx, d=dx)
        ky = 2.0 * np.pi * np.fft.fftfreq(ny, d=dy)
        KX, KY = np.meshgrid(kx, ky, indexing='ij')
        k2 = KX ** 2 + KY ** 2


        psd = sigma_h ** 2 * (correlation_length ** 2 / (2.0 * np.pi)) * np.exp(
            -k2 * correlation_length ** 2 / 2.0
        )

        psd[0, 0] = 0.0


        noise_real = np.random.randn(nx, ny)
        noise_imag = np.random.randn(nx, ny)
        noise = noise_real + 1.0j * noise_imag


        spectrum = np.sqrt(psd) * noise
        field = np.fft.ifft2(spectrum).real
        return field

    def sample_process_variations(self, n_samples, base_params,
                                   sigma_params, correlation_matrix=None):
        n_params = len(base_params)
        cov = np.diag(sigma_params ** 2)
        if correlation_matrix is not None:

            for i in range(n_params):
                for j in range(n_params):
                    cov[i, j] = correlation_matrix[i, j] * sigma_params[i] * sigma_params[j]


        L = np.linalg.cholesky(cov + 1e-12 * np.eye(n_params))
        z = np.random.randn(n_samples, n_params)
        deviations = z @ L.T
        samples = base_params[None, :] + deviations
        return samples

    def lanczos_resampling(self, field, x_src, y_src, x_dst, y_dst):
        nx, ny = field.shape
        dx = x_src[1] - x_src[0]
        dy = y_src[1] - y_src[0]
        a = 2

        def lanczos_kernel(v, a_val):
            v = np.abs(v)
            return np.where(v < a_val, np.sinc(v) * np.sinc(v / a_val), 0.0)

        values = np.zeros(len(x_dst), dtype=np.float64)
        for i in range(len(x_dst)):
            xd = x_dst[i]
            yd = y_dst[i]

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
    ps = ProcessSampler(seed=42)


    tri_points = ps.uniform_in_triangle(1000,
                                        np.array([0.0, 0.0]),
                                        np.array([0.3e-6, 0.0]),
                                        np.array([0.0, 0.3e-6]))
    print(f"[process_sampling] 三角形内采样: mean=({tri_points[:,0].mean():.3e}, "
          f"{tri_points[:,1].mean():.3e})")


    x_grid = np.linspace(-5e-6, 5e-6, 128)
    y_grid = np.linspace(-5e-6, 5e-6, 128)
    h_field = ps.generate_height_error_field(x_grid, y_grid,
                                              sigma_h=5e-9,
                                              correlation_length=0.8e-6)
    print(f"[process_sampling] 高度误差场: mean={h_field.mean():.3e} m, "
          f"std={h_field.std():.3e} m")


    base = np.array([0.6e-6, 0.3e-6, 0.0, 0.0, 90.0])
    sigma = np.array([0.02e-6, 0.01e-6, 0.005e-6, 0.005e-6, 2.0])
    corr = np.eye(5)
    corr[0, 1] = corr[1, 0] = 0.3
    samples = ps.sample_process_variations(500, base, sigma, corr)
    print(f"[process_sampling] 参数采样均值: h={samples[:,0].mean():.3e}, "
          f"w={samples[:,1].mean():.3e}")
    print(f"[process_sampling] 参数采样标准差: h={samples[:,0].std():.3e}, "
          f"w={samples[:,1].std():.3e}")


    pillar_x = np.random.uniform(-4e-6, 4e-6, size=50)
    pillar_y = np.random.uniform(-4e-6, 4e-6, size=50)
    h_errors = ps.lanczos_resampling(h_field, x_grid, y_grid, pillar_x, pillar_y)
    print(f"[process_sampling] 纳米柱高度误差插值: mean={h_errors.mean():.3e}, "
          f"std={h_errors.std():.3e}")
    return h_field, samples


if __name__ == "__main__":
    demo()
