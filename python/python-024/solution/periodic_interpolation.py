r"""
periodic_interpolation.py
========================
三角插值与周期性数据重排模块。
在太阳耀斑环（coronal loop）的环形几何中，
角向坐标 phi 具有 2*pi 周期性，适合使用三角插值处理。

核心数学模型
------------
1. 三角插值（Trigonometric Interpolation）:
    对于周期为 2*pi 的函数 f(phi)，在等距节点 phi_j = 2*pi j/N 上的
    三角插值多项式为:

        P_N(phi) = sum_{j=0}^{N-1} f_j * C_j(phi)

    其中 C_j(phi) 为三角基函数（cardinal functions）:

        当 N 为奇数时:
            C_j(phi) = sin(N/2 * (phi-phi_j)) / [N * sin((phi-phi_j)/2)]

        当 N 为偶数时:
            C_j(phi) = sin(N/2 * (phi-phi_j)) / [N * tan((phi-phi_j)/2)]

    在节点处 C_j(phi_k) = delta_{jk}。

2. 周期性数据重排:
    将一维序列数据按角向-径向顺序重排为二维网格阵列，
    对应于从极坐标 (r, phi) 到笛卡尔网格的映射。

3. 在 MHD 中的应用:
    对于环形装置或日冕环中的扰动，角向模式数为 m:
        xi(r, phi) = xi_m(r) * exp(i m phi)

    通过三角插值可在任意 phi 处重构扰动场。

融入原项目:
- 1356_trig_interp: 三角基函数、三角插值
- 214_contour_sequence4: 序列数据重排、网格化 reshape
"""

import numpy as np
from typing import Tuple, Optional


class TrigonometricInterpolator:
    """
    等距节点的三角插值器。
    """

    def __init__(self, phi_nodes: np.ndarray, values: np.ndarray):
        """
        phi_nodes: 等距节点，假设在 [0, 2*pi) 上均匀分布。
        values: 节点处的函数值。
        """
        self.phi = np.asarray(phi_nodes, dtype=float)
        self.f = np.asarray(values, dtype=float)
        self.N = len(self.phi)
        if self.N < 2:
            raise ValueError("至少需要 2 个节点")
        self.h = 2.0 * np.pi / self.N
        # 验证等距
        diffs = np.diff(np.sort(self.phi))
        if not np.allclose(diffs, self.h, atol=1e-10):
            # 如果不等距，强制重新生成等距节点并插值
            self.phi = np.linspace(0.0, 2.0 * np.pi, self.N, endpoint=False)

    def _cardinal(self, phi_eval: np.ndarray, j: int) -> np.ndarray:
        """
        计算第 j 个三角基函数在 phi_eval 处的值。
        公式:
            N 为奇数: C_j(phi) = sin(N/2 * Delta phi) / (N * sin(Delta phi/2))
            N 为偶数: C_j(phi) = sin(N/2 * Delta phi) / (N * tan(Delta phi/2))
        其中 Delta phi = phi - phi_j。
        """
        phi_eval = np.asarray(phi_eval, dtype=float)
        dphi = phi_eval - self.phi[j]
        # 归一化到 [-pi, pi]
        dphi = np.mod(dphi + np.pi, 2.0 * np.pi) - np.pi

        eps = 1e-14
        if self.N % 2 == 1:
            denom = self.N * np.sin(dphi / 2.0 + eps)
        else:
            denom = self.N * np.tan(dphi / 2.0 + eps)
        num = np.sin(self.N * dphi / 2.0)
        C = num / denom
        # 在节点处修正为 1
        C[np.abs(dphi) < 1e-12] = 1.0
        return C

    def interpolate(self, phi_eval: np.ndarray) -> np.ndarray:
        """
        在 phi_eval 处求插值。
        """
        phi_eval = np.asarray(phi_eval, dtype=float)
        result = np.zeros_like(phi_eval, dtype=float)
        for j in range(self.N):
            result += self.f[j] * self._cardinal(phi_eval, j)
        return result

    def derivative(self, phi_eval: np.ndarray) -> np.ndarray:
        """
        计算插值函数的导数 dP/d phi。
        使用频域方法（FFT）更高效且数值稳定。
        """
        phi_eval = np.asarray(phi_eval, dtype=float)
        # FFT 到频域
        f_hat = np.fft.fft(self.f)
        # 导数: i*k*f_hat
        k = np.fft.fftfreq(self.N, d=self.h) * 2.0 * np.pi * self.N
        df_hat = 1j * k * f_hat
        df = np.fft.ifft(df_hat).real
        # 用 df 的值做三角插值
        interp = TrigonometricInterpolator(self.phi, df)
        return interp.interpolate(phi_eval)


class GridReshaper:
    """
    数据重排与网格化工具，对应 contour_sequence4 的核心思想。
    """

    @staticmethod
    def polar_to_cartesian_reshaper(r: np.ndarray,
                                     phi: np.ndarray,
                                     field_rphi: np.ndarray,
                                     nx: int = 64,
                                     ny: int = 64,
                                     x_range: Tuple[float, float] = (-1.0, 1.0),
                                     y_range: Tuple[float, float] = (-1.0, 1.0)) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        将极坐标 (r, phi) 上的场重排为笛卡尔网格 (x, y)。

        Parameters
        ----------
        r : 1D array, 径向坐标
        phi : 1D array, 角向坐标（均匀分布）
        field_rphi : 2D array, shape (nr, nphi)
        nx, ny : 笛卡尔网格分辨率

        Returns
        -------
        X, Y, Z : 2D arrays
        """
        r = np.asarray(r)
        phi = np.asarray(phi)
        field = np.asarray(field_rphi)
        nr, nphi = field.shape
        if len(r) != nr or len(phi) != nphi:
            raise ValueError("field_rphi 维度与 r, phi 不匹配")

        x = np.linspace(x_range[0], x_range[1], nx)
        y = np.linspace(y_range[0], y_range[1], ny)
        X, Y = np.meshgrid(x, y, indexing='ij')

        # 极坐标转换
        Rq = np.sqrt(X ** 2 + Y ** 2)
        Phiq = np.arctan2(Y, X)
        Phiq = np.mod(Phiq + 2.0 * np.pi, 2.0 * np.pi)

        # 双线性插值（径向-角向）
        Z = np.zeros_like(X)
        dr = r[1] - r[0] if nr > 1 else 1.0
        dphi = phi[1] - phi[0] if nphi > 1 else 1.0

        for i in range(nx):
            for j in range(ny):
                ri = Rq[i, j]
                pi = Phiq[i, j]
                # 径向索引
                ir = int(np.floor((ri - r[0]) / dr))
                ir = np.clip(ir, 0, nr - 2)
                wr = (ri - r[ir]) / dr
                wr = np.clip(wr, 0.0, 1.0)
                # 角向索引（循环）
                ip = int(np.floor((pi - phi[0]) / dphi)) % nphi
                wp = (pi - phi[ip]) / dphi
                wp = np.clip(wp, 0.0, 1.0)
                ip2 = (ip + 1) % nphi

                # 双线性插值
                f00 = field[ir, ip]
                f10 = field[ir + 1, ip] if ir + 1 < nr else field[ir, ip]
                f01 = field[ir, ip2]
                f11 = field[ir + 1, ip2] if ir + 1 < nr else field[ir, ip2]
                Z[i, j] = (1 - wr) * (1 - wp) * f00 + wr * (1 - wp) * f10 + \
                          (1 - wr) * wp * f01 + wr * wp * f11
        return X, Y, Z

    @staticmethod
    def vector_to_grid(values: np.ndarray, nx: int, ny: int,
                       orientation: str = 'row') -> np.ndarray:
        """
        将一维向量按行优先或列优先重排为二维网格。
        对应 contour_sequence4 中的 reshape 操作。
        """
        values = np.asarray(values)
        if len(values) != nx * ny:
            raise ValueError(f"values 长度 {len(values)} 不等于 nx*ny={nx*ny}")
        if orientation == 'row':
            return values.reshape((nx, ny), order='C')
        elif orientation == 'col':
            return values.reshape((ny, nx), order='F').T
        else:
            raise ValueError("orientation 必须是 'row' 或 'col'")


def demo_periodic():
    """
    演示三角插值与数据重排。
    """
    print("\n[PeriodicInterpolation] 演示: 三角插值")
    N = 16
    phi_nodes = np.linspace(0.0, 2.0 * np.pi, N, endpoint=False)
    # 测试函数: f(phi) = cos(2*phi) + 0.5*sin(3*phi)
    f_exact = np.cos(2.0 * phi_nodes) + 0.5 * np.sin(3.0 * phi_nodes)
    interp = TrigonometricInterpolator(phi_nodes, f_exact)
    phi_fine = np.linspace(0.0, 2.0 * np.pi, 200)
    f_interp = interp.interpolate(phi_fine)
    f_exact_fine = np.cos(2.0 * phi_fine) + 0.5 * np.sin(3.0 * phi_fine)
    err = np.max(np.abs(f_interp - f_exact_fine))
    print(f"  N={N} 节点, 插值误差: {err:.3e}")

    # 导数测试
    df_exact_fine = -2.0 * np.sin(2.0 * phi_fine) + 1.5 * np.cos(3.0 * phi_fine)
    df_interp = interp.derivative(phi_fine)
    err_d = np.max(np.abs(df_interp - df_exact_fine))
    print(f"  导数插值误差: {err_d:.3e}")

    print("\n[PeriodicInterpolation] 演示: 极坐标重排")
    nr, nphi = 32, 32
    r = np.linspace(0.1, 1.0, nr)
    phi = np.linspace(0.0, 2.0 * np.pi, nphi, endpoint=False)
    # 构造测试场: f = J_0(2*pi*r) * cos(2*phi)
    from scipy.special import jv
    Rg, Pg = np.meshgrid(r, phi, indexing='ij')
    field = jv(0, 2.0 * np.pi * Rg) * np.cos(2.0 * Pg)
    X, Y, Z = GridReshaper.polar_to_cartesian_reshaper(r, phi, field, nx=64, ny=64)
    print(f"  重排后网格大小: {Z.shape}")
    print(f"  重排场值范围: [{np.min(Z):.3f}, {np.max(Z):.3f}]")


if __name__ == "__main__":
    demo_periodic()
