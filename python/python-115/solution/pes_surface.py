"""
pes_surface.py
势能面（Potential Energy Surface, PES）径向基函数插值模块

核心功能：
- 多维径向基函数（RBF）插值
- 多二次、逆多二次、薄板样条、高斯核函数
- 权重计算与插值评估
- 用于从稀疏从头算数据构建连续势能面

科学背景：
酶催化反应势能面 V(q) 是 3N-6 维超曲面（N 为原子数）。
直接计算每一点的能量代价极高（DFT ~ 小时/点）。
RBF 插值从稀疏采样点 {x_i, V_i} 构建连续近似：

    Ṽ(x) = Σ_{i=1}^{N_d} w_i φ(||x - x_i||)

其中 φ(r) 为径向基函数，权重 w 通过解线性方程组确定：

    A w = V,    A_{ij} = φ(||x_i - x_j||)

常用 RBF 核函数：
1. 多二次（Multiquadric, MQ）：
    φ(r) = √(r² + r₀²)
    适用于平缓变化势能面

2. 逆多二次（Inverse Multiquadric, IMQ）：
    φ(r) = 1/√(r² + r₀²)
    具有正定性，插值稳定性好

3. 薄板样条（Thin-Plate Spline, TPS）：
    φ(r) = r² log(r/r₀)   (r ≠ 0)
    φ(0) = 0
    适用于弯曲变形模式

4. 高斯（Gaussian）：
    φ(r) = exp(-r²/(2r₀²))
    局部性强，适合陡峭势能面

形状参数 r₀ 的选择：
    r₀ ≈ 0.5 * d_max / N_d^{1/m}
    d_max 为最大采样距离，m 为维度
"""

import numpy as np


class RBFKernel:
    """径向基函数核函数族"""

    @staticmethod
    def multiquadric(r, r0):
        """多二次：φ(r) = √(r² + r₀²)"""
        return np.sqrt(r ** 2 + r0 ** 2)

    @staticmethod
    def inverse_multiquadric(r, r0):
        """逆多二次：φ(r) = 1/√(r² + r₀²)"""
        return 1.0 / np.sqrt(r ** 2 + r0 ** 2)

    @staticmethod
    def thin_plate_spline(r, r0):
        """薄板样条：φ(r) = r² log(r/r₀)"""
        v = np.zeros_like(r, dtype=float)
        mask = r > 1e-15
        v[mask] = r[mask] ** 2 * np.log(r[mask] / r0)
        return v

    @staticmethod
    def gaussian(r, r0):
        """高斯：φ(r) = exp(-r²/(2r₀²))"""
        return np.exp(-0.5 * r ** 2 / r0 ** 2)


class PESInterpolator:
    """
    势能面 RBF 插值器
    """

    def __init__(self, m, nd, xd, r0, kernel_name='gaussian'):
        """
        参数：
            m: 空间维度
            nd: 数据点数量
            xd: 数据点坐标 (m, nd)
            r0: 形状参数
            kernel_name: 'multiquadric', 'inverse_multiquadric', 'thin_plate_spline', 'gaussian'
        """
        self.m = m
        self.nd = nd
        self.xd = np.asarray(xd, dtype=float)
        self.r0 = r0

        kernel_map = {
            'multiquadric': RBFKernel.multiquadric,
            'inverse_multiquadric': RBFKernel.inverse_multiquadric,
            'thin_plate_spline': RBFKernel.thin_plate_spline,
            'gaussian': RBFKernel.gaussian
        }
        if kernel_name not in kernel_map:
            raise ValueError(f"未知核函数: {kernel_name}")
        self.phi = kernel_map[kernel_name]
        self.weights = None

    def compute_weights(self, fd):
        """
        计算 RBF 插值权重

        解线性方程组：
            A w = f_d
        其中 A_{ij} = φ(||x_i - x_j||)

        参数：
            fd: 数据点函数值 (nd,)
        """
        fd = np.asarray(fd, dtype=float)
        if fd.shape[0] != self.nd:
            raise ValueError(f"数据点数量不匹配: {fd.shape[0]} != {self.nd}")

        A = np.zeros((self.nd, self.nd), dtype=float)
        for i in range(self.nd):
            d = self.xd - self.xd[:, i:i + 1]
            r = np.sqrt(np.sum(d ** 2, axis=0))
            A[i, :] = self.phi(r, self.r0)

        # 正则化：添加小量到对角线以提高数值稳定性
        reg = 1e-10 * np.eye(self.nd)
        self.weights = np.linalg.solve(A + reg, fd)

    def interpolate(self, xi):
        """
        在插值点处评估势能

        参数：
            xi: 插值点坐标 (m, ni) 或 (m,)
        返回：
            fi: 插值函数值 (ni,) 或标量
        """
        if self.weights is None:
            raise RuntimeError("必须先调用 compute_weights 计算权重")

        xi = np.asarray(xi, dtype=float)
        if xi.ndim == 1:
            xi = xi.reshape(-1, 1)
        ni = xi.shape[1]

        fi = np.zeros(ni, dtype=float)
        for i in range(ni):
            d = self.xd - xi[:, i:i + 1]
            r = np.sqrt(np.sum(d ** 2, axis=0))
            v = self.phi(r, self.r0)
            fi[i] = np.dot(v, self.weights)

        return fi if ni > 1 else fi[0]

    def gradient(self, xi, h=1e-5):
        """
        数值计算势能梯度 ∇V(x)

        公式：
            ∂V/∂x_k ≈ [V(x + h*e_k) - V(x - h*e_k)] / (2h)
        """
        xi = np.asarray(xi, dtype=float)
        if xi.ndim == 1:
            xi = xi.reshape(-1, 1)
        m, ni = xi.shape
        grad = np.zeros((m, ni), dtype=float)

        for k in range(m):
            e_k = np.zeros(m)
            e_k[k] = 1.0
            x_plus = xi + h * e_k.reshape(-1, 1)
            x_minus = xi - h * e_k.reshape(-1, 1)
            grad[k, :] = (self.interpolate(x_plus) - self.interpolate(x_minus)) / (2.0 * h)

        return grad

    def hessian(self, xi, h=1e-4):
        """
        数值计算势能 Hessian H_{kl} = ∂²V/(∂x_k ∂x_l)
        """
        xi = np.asarray(xi, dtype=float).flatten()
        m = len(xi)
        H = np.zeros((m, m), dtype=float)

        for k in range(m):
            for l in range(k, m):
                e_k = np.zeros(m)
                e_l = np.zeros(m)
                e_k[k] = 1.0
                e_l[l] = 1.0

                f_pp = self.interpolate(xi + h * e_k + h * e_l)
                f_pm = self.interpolate(xi + h * e_k - h * e_l)
                f_mp = self.interpolate(xi - h * e_k + h * e_l)
                f_mm = self.interpolate(xi - h * e_k - h * e_l)

                H[k, l] = (f_pp - f_pm - f_mp + f_mm) / (4.0 * h ** 2)
                H[l, k] = H[k, l]

        return H


def estimate_r0(xd):
    """
    自动估计 RBF 形状参数 r₀

    公式：
        r₀ = 0.5 * (d_max / N_d^{1/m})
    其中 d_max 为数据点间最大距离
    """
    xd = np.asarray(xd, dtype=float)
    m, nd = xd.shape

    max_dist = 0.0
    for i in range(nd):
        for j in range(i + 1, nd):
            d = np.linalg.norm(xd[:, i] - xd[:, j])
            if d > max_dist:
                max_dist = d

    r0 = 0.5 * max_dist / (nd ** (1.0 / m))
    return max(r0, 1e-3)
