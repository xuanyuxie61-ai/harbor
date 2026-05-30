
import numpy as np
from math import sqrt, log


class LatinHypercubeSampler:

    def __init__(self, dim_num, point_num, seed=None):
        if dim_num <= 0 or point_num <= 0:
            raise ValueError("dim_num和point_num必须为正整数")
        self.dim_num = dim_num
        self.point_num = point_num
        self.rng = np.random.default_rng(seed)

    def sample_uniform(self):
        x = self.rng.random((self.dim_num, self.point_num))
        for i in range(self.dim_num):
            perm = self.rng.permutation(self.point_num)
            for j in range(self.point_num):
                x[i, j] = (perm[j] + x[i, j]) / self.point_num
        return x

    def sample_normal(self, mean=None, cov=None):

        u = self.sample_uniform()

        eps = 1e-10
        u = np.clip(u, eps, 1.0 - eps)
        z = np.zeros_like(u)
        for i in range(self.dim_num):
            for j in range(self.point_num):
                z[i, j] = _inverse_cdf_normal(u[i, j])


        if cov is not None:
            cov = np.asarray(cov, dtype=np.float64)
            if cov.shape != (self.dim_num, self.dim_num):
                raise ValueError(f"协方差矩阵维度不匹配")

            try:
                L = np.linalg.cholesky(cov)
            except np.linalg.LinAlgError:

                eigvals, eigvecs = np.linalg.eigh(cov)
                eigvals = np.maximum(eigvals, 1e-12)
                L = eigvecs @ np.diag(np.sqrt(eigvals))
            z = L @ z

        if mean is not None:
            mean = np.asarray(mean, dtype=np.float64).reshape(self.dim_num, 1)
            z = z + mean

        return z

    def sample_for_heston(self, rho, point_num=None):
        if point_num is None:
            point_num = self.point_num
        old_point_num = self.point_num
        self.point_num = point_num
        cov = np.array([[1.0, rho], [rho, 1.0]], dtype=np.float64)
        samples = self.sample_normal(cov=cov)
        self.point_num = old_point_num
        return samples


def _inverse_cdf_normal(p):
    if p <= 0.0 or p >= 1.0:
        raise ValueError("p必须在(0,1)开区间内")


    if p < 0.5:
        q = p
        sign = -1.0
    else:
        q = 1.0 - p
        sign = 1.0

    if q < 1e-300:

        t = sqrt(-2.0 * log(q))
        return sign * (t - (log(t) + 0.5 * log(4.0 * np.pi)) / t)


    a1 = -3.969683028665376e+01
    a2 =  2.209460984245205e+02
    a3 = -2.759285104469687e+02
    a4 =  1.383577518672690e+02
    a5 = -3.066479806614716e+01
    a6 =  2.506628277459239e+00

    b1 = -5.447609879822406e+01
    b2 =  1.615858368580409e+02
    b3 = -1.556989798598866e+02
    b4 =  6.680131188771972e+01
    b5 = -1.328068155288572e+01

    c1 = -7.784894002430293e-03
    c2 = -3.223964580411365e-01
    c3 = -2.400758277161838e+00
    c4 = -2.549732539343734e+00
    c5 =  4.374664141464968e+00
    c6 =  2.938163982698783e+00

    d1 =  7.784695709041462e-03
    d2 =  3.224671290700398e-01
    d3 =  2.445134137142996e+00
    d4 =  3.754408661907416e+00

    p_low = 0.02425
    p_high = 1.0 - p_low

    if p < p_low:

        q_sqrt = sqrt(-2.0 * log(q))
        x = (((((c1 * q_sqrt + c2) * q_sqrt + c3) * q_sqrt + c4) * q_sqrt + c5) * q_sqrt + c6) \
            / ((((d1 * q_sqrt + d2) * q_sqrt + d3) * q_sqrt + d4) * q_sqrt + 1.0)
    elif p <= p_high:

        r = q - 0.5
        r2 = r * r
        x = (((((a1 * r2 + a2) * r2 + a3) * r2 + a4) * r2 + a5) * r2 + a6) * r \
            / (((((b1 * r2 + b2) * r2 + b3) * r2 + b4) * r2 + b5) * r2 + 1.0)
    else:

        q_sqrt = sqrt(-2.0 * log(q))
        x = -(((((c1 * q_sqrt + c2) * q_sqrt + c3) * q_sqrt + c4) * q_sqrt + c5) * q_sqrt + c6) \
            / ((((d1 * q_sqrt + d2) * q_sqrt + d3) * q_sqrt + d4) * q_sqrt + 1.0)



    e = 0.5 * (1.0 + np.sign(x) * np.sqrt(1.0 - np.exp(-2.0/np.pi * x * x))) - p

    return sign * abs(x)
