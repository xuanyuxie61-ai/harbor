"""
正交多项式基变换模块
实现不同正交多项式之间的转换

数学背景：
正交多项式在 Berry curvature 的高精度逼近中起关键作用。
常用多项式族：
- Legendre: P_n(x) on [-1,1], w(x)=1
- Chebyshev: T_n(x) on [-1,1], w(x)=1/sqrt(1-x²)
- Gegenbauer: C_n^λ(x), 广义 Legendre
- Hermite: H_n(x) on (-∞,∞), w(x)=exp(-x²)

Berry curvature 在 Weyl 点附近可展开为：
Ω(k) ≈ Σ_{n,m,l} c_{nml} P_n(kx) P_m(ky) P_l(kz)
"""

import numpy as np
from typing import Tuple, List, Callable


class PolynomialBasisConverter:
    """
    正交多项式基变换器

    实现不同多项式基之间的转换矩阵
    """

    def __init__(self, max_degree: int = 10):
        """
        初始化

        Parameters:
        -----------
        max_degree : int
            最大多项式阶数
        """
        self.max_degree = max_degree

    def legendre_to_monomial(self, n: int) -> np.ndarray:
        """
        Legendre 多项式到单项式基的转换

        P_n(x) = Σ_k a_k x^k

        递推关系：
        (n+1)P_{n+1}(x) = (2n+1)xP_n(x) - nP_{n-1}(x)

        Parameters:
        -----------
        n : int
            多项式阶数

        Returns:
        --------
        np.ndarray
            转换系数 [a_0, a_1, ..., a_n]
        """
        if n == 0:
            return np.array([1.0])
        elif n == 1:
            return np.array([0.0, 1.0])

        # 使用递推
        coeffs = np.zeros(n + 1)
        P_prev2 = np.array([1.0])  # P_0
        P_prev1 = np.array([0.0, 1.0])  # P_1

        for k in range(1, n):
            # (k+1)P_{k+1} = (2k+1)xP_k - kP_{k-1}
            xP_k = np.zeros(len(P_prev1) + 1)
            xP_k[1:] = P_prev1

            P_next = ((2*k + 1) * xP_k - k * np.pad(P_prev2, (0, max(0, len(xP_k) - len(P_prev2))))) / (k + 1)
            P_prev2 = P_prev1
            P_prev1 = P_next

        return P_prev1

    def monomial_to_legendre(self, coeffs: np.ndarray) -> np.ndarray:
        """
        单项式基到 Legendre 基的转换

        x^n = Σ_k b_k P_k(x)

        Parameters:
        -----------
        coeffs : np.ndarray
            单项式系数

        Returns:
        --------
        np.ndarray
            Legendre 系数
        """
        n = len(coeffs) - 1
        legendre_coeffs = np.zeros(n + 1)

        for k in range(n + 1):
            P_k = self.legendre_to_monomial(k)
            # 投影
            if len(P_k) > 0:
                legendre_coeffs[k] = coeffs[k] / P_k[k] if k < len(P_k) else 0

        return legendre_coeffs

    def chebyshev_to_monomial(self, n: int) -> np.ndarray:
        """
        Chebyshev 多项式到单项式基

        T_n(x) = cos(n arccos x)

        递推：T_{n+1}(x) = 2xT_n(x) - T_{n-1}(x)
        """
        if n == 0:
            return np.array([1.0])
        elif n == 1:
            return np.array([0.0, 1.0])

        T_prev2 = np.array([1.0])
        T_prev1 = np.array([0.0, 1.0])

        for k in range(1, n):
            xT_k = np.zeros(len(T_prev1) + 1)
            xT_k[1:] = T_prev1
            T_next = 2 * xT_k - np.pad(T_prev2, (0, max(0, len(xT_k) - len(T_prev2))))
            T_prev2 = T_prev1
            T_prev1 = T_next

        return T_prev1

    def gegenbauer_to_monomial(self, n: int, lam: float = 1.0) -> np.ndarray:
        """
        Gegenbauer 多项式到单项式基

        C_n^λ(x) 满足递推：
        nC_n^λ = 2(n+λ-1)xC_{n-1}^λ - (n+2λ-2)C_{n-2}^λ

        Parameters:
        -----------
        n : int
            阶数
        lam : float
            参数 λ (λ=1 为 Chebyshev 第二类, λ=1/2 为 Legendre)
        """
        if n == 0:
            return np.array([1.0])
        elif n == 1:
            return np.array([0.0, 2*lam])

        C_prev2 = np.array([1.0])
        C_prev1 = np.array([0.0, 2*lam])

        for k in range(1, n):
            xC_k = np.zeros(len(C_prev1) + 1)
            xC_k[1:] = C_prev1

            term1 = 2 * (k + lam - 1) * xC_k
            term2 = (k + 2*lam - 2) * np.pad(C_prev2, (0, max(0, len(xC_k) - len(C_prev2))))

            C_next = (term1 - term2) / (k + 1)
            C_prev2 = C_prev1
            C_prev1 = C_next

        return C_prev1

    def hermite_to_monomial(self, n: int, probabilist: bool = False) -> np.ndarray:
        """
        Hermite 多项式到单项式基

        物理学家版本：H_{n+1} = 2xH_n - 2nH_{n-1}
        概率论版本：He_{n+1} = xHe_n - nHe_{n-1}
        """
        if n == 0:
            return np.array([1.0])
        elif n == 1:
            return np.array([0.0, 2.0]) if not probabilist else np.array([0.0, 1.0])

        H_prev2 = np.array([1.0])
        if probabilist:
            H_prev1 = np.array([0.0, 1.0])
        else:
            H_prev1 = np.array([0.0, 2.0])

        for k in range(1, n):
            xH_k = np.zeros(len(H_prev1) + 1)
            xH_k[1:] = H_prev1

            if probabilist:
                H_next = xH_k - k * np.pad(H_prev2, (0, max(0, len(xH_k) - len(H_prev2))))
            else:
                H_next = 2 * xH_k - 2*k * np.pad(H_prev2, (0, max(0, len(xH_k) - len(H_prev2))))

            H_prev2 = H_prev1
            H_prev1 = H_next

        return H_prev1

    def laguerre_to_monomial(self, n: int, alpha: float = 0.0) -> np.ndarray:
        """
        广义 Laguerre 多项式到单项式基

        L_n^α(x) 满足递推：
        (n+1)L_{n+1}^α = (2n+1+α-x)L_n^α - (n+α)L_{n-1}^α
        """
        if n == 0:
            return np.array([1.0])
        elif n == 1:
            return np.array([1 + alpha, -1.0])

        L_prev2 = np.array([1.0])
        L_prev1 = np.array([1 + alpha, -1.0])

        for k in range(1, n):
            xL_k = np.zeros(len(L_prev1) + 1)
            xL_k[1:] = -L_prev1  # -x 项

            term1 = (2*k + 1 + alpha) * np.pad(L_prev1, (0, max(0, len(xL_k) - len(L_prev1))))
            term2 = (k + alpha) * np.pad(L_prev2, (0, max(0, len(xL_k) - len(L_prev2))))

            L_next = (term1 + xL_k - term2) / (k + 1)
            L_prev2 = L_prev1
            L_prev1 = L_next

        return L_prev1

    def conversion_matrix(self, from_basis: str, to_basis: str,
                          n_max: int = None) -> np.ndarray:
        """
        构建基变换矩阵

        Parameters:
        -----------
        from_basis : str
            源基 ('legendre', 'chebyshev', 'hermite', 'laguerre', 'gegenbauer')
        to_basis : str
            目标基 (同上)
        n_max : int
            最大阶数

        Returns:
        --------
        np.ndarray
            (n+1, n+1) 变换矩阵 M 使得 c_to = M @ c_from
        """
        if n_max is None:
            n_max = self.max_degree

        n = n_max + 1
        M = np.zeros((n, n))

        for i in range(n):
            # 第 i 个 from_basis 多项式
            if from_basis == 'legendre':
                coeffs = self.legendre_to_monomial(i)
            elif from_basis == 'chebyshev':
                coeffs = self.chebyshev_to_monomial(i)
            elif from_basis == 'hermite':
                coeffs = self.hermite_to_monomial(i)
            elif from_basis == 'laguerre':
                coeffs = self.laguerre_to_monomial(i)
            elif from_basis == 'gegenbauer':
                coeffs = self.gegenbauer_to_monomial(i, 1.0)
            else:
                raise ValueError(f"Unknown basis: {from_basis}")

            # 转换到目标基
            for j, c in enumerate(coeffs):
                if j < n:
                    M[j, i] = c

        return M

    def evaluate_polynomial(self, coeffs: np.ndarray, x: np.ndarray,
                           basis: str = 'legendre') -> np.ndarray:
        """
        在给定基底下计算多项式值

        Parameters:
        -----------
        coeffs : np.ndarray
            多项式系数
        x : np.ndarray
            计算点
        basis : str
            多项式基

        Returns:
        --------
        np.ndarray
            多项式值
        """
        result = np.zeros_like(x, dtype=float)

        for n, c in enumerate(coeffs):
            if basis == 'monomial':
                result += c * x**n
            elif basis == 'legendre':
                P_n = self._legendre_value(n, x)
                result += c * P_n
            elif basis == 'chebyshev':
                T_n = self._chebyshev_value(n, x)
                result += c * T_n
            else:
                # 其他基使用单项式展开
                mono_coeffs = getattr(self, f'{basis}_to_monomial')(n)
                for k, mc in enumerate(mono_coeffs):
                    if k < len(x.shape) or np.isscalar(x):
                        result += c * mc * x**k

        return result

    def _legendre_value(self, n: int, x: np.ndarray) -> np.ndarray:
        """计算 Legendre 多项式 P_n(x)"""
        if n == 0:
            return np.ones_like(x)
        elif n == 1:
            return x.copy()

        P_prev2 = np.ones_like(x)
        P_prev1 = x.copy()

        for k in range(1, n):
            P_next = ((2*k + 1) * x * P_prev1 - k * P_prev2) / (k + 1)
            P_prev2 = P_prev1
            P_prev1 = P_next

        return P_prev1

    def _chebyshev_value(self, n: int, x: np.ndarray) -> np.ndarray:
        """计算 Chebyshev 多项式 T_n(x)"""
        if n == 0:
            return np.ones_like(x)
        elif n == 1:
            return x.copy()

        T_prev2 = np.ones_like(x)
        T_prev1 = x.copy()

        for k in range(1, n):
            T_next = 2 * x * T_prev1 - T_prev2
            T_prev2 = T_prev1
            T_prev1 = T_next

        return T_prev1

    def fit_berry_curvature(self, k_data: np.ndarray, omega_data: np.ndarray,
                           basis: str = 'legendre', max_degree: int = 5) -> np.ndarray:
        """
        用正交多项式拟合 Berry curvature 数据

        Ω(k) ≈ Σ_{n,m,l} c_{nml} P_n(kx) P_m(ky) P_l(kz)

        Parameters:
        -----------
        k_data : np.ndarray
            k 点数据 (N, 3)
        omega_data : np.ndarray
            Berry curvature 数据 (N,) 或 (N, 3)
        basis : str
            多项式基
        max_degree : int
            最大阶数

        Returns:
        --------
        np.ndarray
            拟合系数
        """
        n_points = len(k_data)

        # 构建 Vandermonde 矩阵
        n_basis = (max_degree + 1)**3
        V = np.zeros((n_points, n_basis))

        idx = 0
        for i in range(max_degree + 1):
            for j in range(max_degree + 1):
                for l in range(max_degree + 1):
                    # 3D 基函数
                    if basis == 'legendre':
                        V[:, idx] = (self._legendre_value(i, k_data[:, 0]) *
                                    self._legendre_value(j, k_data[:, 1]) *
                                    self._legendre_value(l, k_data[:, 2]))
                    elif basis == 'chebyshev':
                        V[:, idx] = (self._chebyshev_value(i, k_data[:, 0]) *
                                    self._chebyshev_value(j, k_data[:, 1]) *
                                    self._chebyshev_value(l, k_data[:, 2]))
                    else:
                        V[:, idx] = (k_data[:, 0]**i * k_data[:, 1]**j * k_data[:, 2]**l)
                    idx += 1

        # 最小二乘拟合
        if omega_data.ndim == 1:
            coeffs, _, _, _ = np.linalg.lstsq(V, omega_data, rcond=None)
        else:
            coeffs = np.zeros((n_basis, omega_data.shape[1]))
            for d in range(omega_data.shape[1]):
                coeffs[:, d], _, _, _ = np.linalg.lstsq(V, omega_data[:, d], rcond=None)

        return coeffs
