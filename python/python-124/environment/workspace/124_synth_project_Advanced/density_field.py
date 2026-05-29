"""
density_field.py
骨密度场插值与材料属性映射模块

融合来源：
- 1214_test_interp_nd: N维插值测试问题集（切比雪夫级数求值 csevl, inits）
- 313_dot_l2: L2内积计算

科学背景：
骨密度 rho(x,y) 在骨骼截面上是非均匀分布的。本项目使用切比雪夫级数
展开来参数化骨密度场，并通过L2内积将密度场与有限元基函数耦合。

核心数学公式：
1. 切比雪夫级数展开：
   f(x) = sum_{k=0}^{N-1} c_k T_k(x),  x ∈ [-1, 1]
   其中 T_k(x) 为第一类切比雪夫多项式。

2. Clenshaw 递推求值（csevl）：
   b0 = 2x*b1 - b2 + c_k, 反向递推
   f(x) = 0.5*(b0 - b2)

3. L2 内积（dot_l2）：
   <f, g>_{L2} = ∫_a^b f(x) g(x) dx
   用于计算密度场与形函数的耦合系数。
"""

import numpy as np
from scipy import integrate
from typing import Callable, Tuple, Optional


# ===================================================================
# 切比雪夫级数求值（来自 1214_test_interp_nd 的 csevl）
# ===================================================================
def csevl(x: float, cs: np.ndarray, n: Optional[int] = None) -> float:
    """
    使用 Clenshaw 递推求切比雪夫级数值。

    数学推导：
    切比雪夫多项式满足递推关系：
        T_0(x) = 1
        T_1(x) = x
        T_{k+1}(x) = 2x T_k(x) - T_{k-1}(x)

    级数 f(x) = Σ c_k T_k(x) 可通过反向递推计算：
        b_{N+1} = b_N = 0
        b_k = 2x b_{k+1} - b_{k+2} + c_k,  k = N-1,...,0
        f(x) = 0.5 * (b_0 - b_2)

    Parameters
    ----------
    x : float
        求值点，需在 [-1.1, 1.1] 内
    cs : np.ndarray
        切比雪夫系数
    n : int, optional
        使用的前 n 个系数

    Returns
    -------
    float
        级数值
    """
    if n is None:
        n = len(cs)
    if n < 1:
        raise ValueError("csevl: Number of terms must be >= 1.")
    if n > 1000:
        raise ValueError("csevl: Number of terms must be <= 1000.")
    if x < -1.1 or x > 1.1:
        raise ValueError(f"csevl: X outside valid range [-1.1, 1.1], got {x}")

    b0 = 0.0
    b1 = 0.0
    b2 = 0.0

    for i in range(n - 1, -1, -1):
        b2 = b1
        b1 = b0
        b0 = 2.0 * x * b1 - b2 + cs[i]

    value = 0.5 * (b0 - b2)
    return value


def inits(cs: np.ndarray, eta: float = 1e-12) -> int:
    """
    根据精度 eta 截断切比雪夫系数，返回有效项数。

    算法：从尾项开始扫描，当 |cs[i]| > eta 时停止。

    Parameters
    ----------
    cs : np.ndarray
        切比雪夫系数
    eta : float
        截断容差

    Returns
    -------
    int
        保留的系数个数
    """
    n = len(cs)
    for i in range(n - 1, -1, -1):
        if abs(cs[i]) > eta:
            return i + 1
    return 1


# ===================================================================
# L2 内积计算（来自 313_dot_l2）
# ===================================================================
def dot_l2(f: Callable[[float], float], g: Callable[[float], float],
           a: float, b: float, epsabs: float = 1e-14,
           epsrel: float = 1e-12) -> float:
    """
    计算两个一元函数在 [a, b] 上的 L2 内积：
        <f, g> = ∫_a^b f(x) g(x) dx

    Parameters
    ----------
    f, g : callable
        被积函数
    a, b : float
        积分上下限
    epsabs, epsrel : float
        绝对与相对容差

    Returns
    -------
    float
        L2 内积值
    """
    if a >= b:
        raise ValueError("Interval [a,b] must satisfy a < b.")

    def integrand(x: float) -> float:
        return f(x) * g(x)

    val, err = integrate.quad(integrand, a, b, limit=100,
                              epsabs=epsabs, epsrel=epsrel)
    if err > 1e-8:
        raise RuntimeError(f"L2 inner product integration failed with error {err}")
    return val


# ===================================================================
# 骨密度场类
# ===================================================================
class BoneDensityField:
    """
    骨密度场表示。

    使用二维切比雪夫级数展开参数化骨密度分布：
        rho(xi, eta) = Σ_{i=0}^{Nx-1} Σ_{j=0}^{Ny-1} c_{ij} T_i(xi) T_j(eta)

    其中 (xi, eta) ∈ [-1, 1]^2 为参考坐标，通过等参映射与物理坐标关联。
    """

    def __init__(self, cheb_coeffs: Optional[np.ndarray] = None,
                 nx_cheb: int = 8, ny_cheb: int = 8):
        """
        Parameters
        ----------
        cheb_coeffs : np.ndarray, optional
            二维切比雪夫系数矩阵，形状 (nx_cheb, ny_cheb)
        nx_cheb, ny_cheb : int
            切比雪夫级数截断阶数
        """
        self.nx_cheb = nx_cheb
        self.ny_cheb = ny_cheb

        if cheb_coeffs is not None:
            if cheb_coeffs.shape != (nx_cheb, ny_cheb):
                raise ValueError("cheb_coeffs shape mismatch")
            self.coeffs = cheb_coeffs.copy()
        else:
            # 默认：中心密度高，边缘密度低的骨密度分布
            self.coeffs = self._generate_default_coeffs()

    def _generate_default_coeffs(self) -> np.ndarray:
        """
        生成默认骨密度切比雪夫系数，模拟真实骨骼密度分布。
        """
        nx, ny = self.nx_cheb, self.ny_cheb
        coeffs = np.zeros((nx, ny))

        # 基线密度
        coeffs[0, 0] = 1.0
        # 边缘衰减项
        if nx > 2:
            coeffs[2, 0] = -0.15
        if ny > 2:
            coeffs[0, 2] = -0.15
        # 中心增强
        if nx > 1 and ny > 1:
            coeffs[1, 1] = 0.05
        # 高阶修正
        if nx > 4 and ny > 4:
            coeffs[4, 0] = 0.02
            coeffs[0, 4] = 0.02

        return coeffs

    def evaluate(self, xi: float, eta: float) -> float:
        """
        在参考坐标 (xi, eta) ∈ [-1, 1]^2 上求骨密度值。

        算法：先对 xi 方向用 csevl 求每行的值，再对 eta 方向综合。
        """
        if not (-1.0 <= xi <= 1.0 and -1.0 <= eta <= 1.0):
            # 边界裁剪
            xi = max(-1.0, min(1.0, xi))
            eta = max(-1.0, min(1.0, eta))

        nx = inits(self.coeffs[:, 0], eta=1e-12)
        ny = inits(self.coeffs[0, :], eta=1e-12)

        # 逐行求切比雪夫级数
        row_vals = np.zeros(ny)
        for j in range(ny):
            row_vals[j] = csevl(xi, self.coeffs[:, j], nx)

        # 对结果再求切比雪夫级数
        val = csevl(eta, row_vals, ny)
        return val

    def evaluate_physical(self, x: float, y: float,
                          xlim: Tuple[float, float] = (0.0, 20.0),
                          ylim: Tuple[float, float] = (0.0, 30.0)) -> float:
        """
        在物理坐标 (x, y) 上求骨密度值。

        坐标变换：
            xi  = 2 * (x - x_min) / (x_max - x_min) - 1
            eta = 2 * (y - y_min) / (y_max - y_min) - 1
        """
        x_min, x_max = xlim
        y_min, y_max = ylim

        if x_max <= x_min or y_max <= y_min:
            raise ValueError("Invalid physical domain bounds.")

        xi = 2.0 * (x - x_min) / (x_max - x_min) - 1.0
        eta = 2.0 * (y - y_min) / (y_max - y_min) - 1.0

        return self.evaluate(xi, eta)

    def evaluate_batch(self, xy: np.ndarray,
                       xlim: Tuple[float, float] = (0.0, 20.0),
                       ylim: Tuple[float, float] = (0.0, 30.0)) -> np.ndarray:
        """
        批量求值。

        Parameters
        ----------
        xy : np.ndarray, shape (2, N)
            物理坐标数组

        Returns
        -------
        np.ndarray, shape (N,)
            密度值
        """
        if xy.shape[0] != 2:
            raise ValueError("xy must have shape (2, N)")
        N = xy.shape[1]
        vals = np.zeros(N)
        for i in range(N):
            vals[i] = self.evaluate_physical(xy[0, i], xy[1, i], xlim, ylim)
        return vals

    def elastic_modulus_from_density(self, rho: float,
                                     E0: float = 17.0e3,
                                     power: float = 2.0) -> float:
        """
        由骨密度计算弹性模量。

        Carter-Hayes 幂律关系：
            E(rho) = E0 * (rho / rho_max)^power

        Parameters
        ----------
        rho : float
            归一化骨密度 [0, 1]
        E0 : float
            最大弹性模量 (MPa)
        power : float
            幂律指数

        Returns
        -------
        float
            弹性模量 (MPa)
        """
        rho_clip = max(0.0, min(1.0, rho))
        return E0 * (rho_clip ** power)

    def set_coefficients(self, coeffs: np.ndarray):
        """
        手动设置切比雪夫系数。
        """
        if coeffs.shape != (self.nx_cheb, self.ny_cheb):
            raise ValueError(f"Expected shape ({self.nx_cheb}, {self.ny_cheb}), got {coeffs.shape}")
        self.coeffs = coeffs.copy()

    def compute_l2_norm(self, xlim: Tuple[float, float] = (0.0, 20.0),
                        ylim: Tuple[float, float] = (0.0, 30.0)) -> float:
        """
        计算密度场在物理域上的 L2 范数。

        ||rho||_{L2} = sqrt( ∫∫_Omega rho(x,y)^2 dx dy )
        """
        x_min, x_max = xlim
        y_min, y_max = ylim

        def integrand(y: float, x: float) -> float:
            val = self.evaluate_physical(x, y, xlim, ylim)
            return val * val

        val, err = integrate.dblquad(integrand, x_min, x_max,
                                     lambda x: y_min, lambda x: y_max,
                                     epsabs=1e-10, epsrel=1e-10)
        if err > 1e-6:
            raise RuntimeError(f"L2 norm integration failed with error {err}")
        return np.sqrt(val)
