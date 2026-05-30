
import numpy as np


class RBFKernel:

    @staticmethod
    def multiquadric(r, r0):
        return np.sqrt(r ** 2 + r0 ** 2)

    @staticmethod
    def inverse_multiquadric(r, r0):
        return 1.0 / np.sqrt(r ** 2 + r0 ** 2)

    @staticmethod
    def thin_plate_spline(r, r0):
        v = np.zeros_like(r, dtype=float)
        mask = r > 1e-15
        v[mask] = r[mask] ** 2 * np.log(r[mask] / r0)
        return v

    @staticmethod
    def gaussian(r, r0):
        return np.exp(-0.5 * r ** 2 / r0 ** 2)


class PESInterpolator:

    def __init__(self, m, nd, xd, r0, kernel_name='gaussian'):
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
        fd = np.asarray(fd, dtype=float)
        if fd.shape[0] != self.nd:
            raise ValueError(f"数据点数量不匹配: {fd.shape[0]} != {self.nd}")

        A = np.zeros((self.nd, self.nd), dtype=float)
        for i in range(self.nd):
            d = self.xd - self.xd[:, i:i + 1]
            r = np.sqrt(np.sum(d ** 2, axis=0))
            A[i, :] = self.phi(r, self.r0)


        reg = 1e-10 * np.eye(self.nd)
        self.weights = np.linalg.solve(A + reg, fd)

    def interpolate(self, xi):
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
