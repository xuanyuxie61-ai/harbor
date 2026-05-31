"""
stability_analysis.py
=====================
稳定性分析与特征值计算模块

融合种子项目：
- 203_companion_matrix: Hermite 伴矩阵用于多项式求根
- 697_log_norm: 矩阵对数范数用于稳定性估计
- 700_logistic_bifurcation: 非线性动力学分叉分析

核心内容：
1. 线性稳定性分析：计算 Jacobian 矩阵特征值
2. 伴矩阵法求特征多项式根
3. 矩阵对数范数估计数值稳定性
4. 分叉分析与 Lyapunov 指数
5. 时间步长稳定性约束（CFL 条件等）

线性稳定性理论：
    对于系统 ∂u/∂t = L(u)，在平衡态 u* 附近线性化：
        ∂δu/∂t = J(u*) δu
    其中 J = ∂L/∂u|_{u*} 为 Jacobian 矩阵。

    若 J 的所有特征值实部 Re(λ) < 0，则平衡态稳定；
    若存在 Re(λ) > 0，则不稳定。

矩阵对数范数：
    μ_p(A) = lim_{h→0+} (||I + hA||_p - 1) / h

    对误差增长有界：
        ||exp(At)||_p ≤ exp(μ_p(A) t)
"""

import numpy as np


class CompanionMatrixEigenvalue:
    """
    基于伴矩阵的特征值计算。
    基于种子项目 203_companion_matrix。
    """

    @staticmethod
    def hermite_companion_matrix(coeffs):
        """
        构造 Hermite 基下的伴矩阵。

        对于多项式 p(x) = Σ_{k=0}^n c_k H_k(x)，其中 H_k 为 Hermite 多项式，
        其伴矩阵 A 满足 det(λI - A) = 0 的根即为 p(x) = 0 的解。

        Hermite 伴矩阵结构：
            A_{i,i+1} = 1/2    (i = 1, ..., n-1)
            A_{i,i-1} = i-1    (i = 2, ..., n)
            A_{n,j} = A_{n,j} - c_j / (2 c_n)   (j = 1, ..., n)

        Parameters
        ----------
        coeffs : ndarray
            Hermite 多项式系数 [c_0, c_1, ..., c_n]。

        Returns
        -------
        ndarray
            n×n 伴矩阵。
        """
        coeffs = np.asarray(coeffs, dtype=float)
        n = len(coeffs) - 1

        if n < 1:
            raise ValueError("多项式次数必须至少为 1")
        if abs(coeffs[-1]) < 1e-14:
            raise ValueError("首项系数不能为零")

        A = np.zeros((n, n))

        # 上对角线
        for i in range(n - 1):
            A[i, i + 1] = 0.5

        # 下对角线
        for i in range(1, n):
            A[i, i - 1] = i

        # 最后一行
        for j in range(n):
            A[n - 1, j] -= coeffs[j] / (2.0 * coeffs[n])

        return A

    @staticmethod
    def chebyshev_companion_matrix(coeffs):
        """
        构造 Chebyshev 基下的伴矩阵。

        Parameters
        ----------
        coeffs : ndarray
            Chebyshev 多项式系数 [c_0, ..., c_n]。

        Returns
        -------
        ndarray
            n×n 伴矩阵。
        """
        coeffs = np.asarray(coeffs, dtype=float)
        n = len(coeffs) - 1

        if n < 1:
            raise ValueError("多项式次数必须至少为 1")

        A = np.zeros((n, n))

        A[0, 1] = 1.0
        for i in range(1, n - 1):
            A[i, i - 1] = 0.5
            A[i, i + 1] = 0.5
        A[n - 1, :] -= coeffs[:-1] / (2.0 * coeffs[-1])
        if n >= 2:
            A[n - 1, n - 2] += 0.5

        return A

    @staticmethod
    def find_roots(coeffs, basis='power'):
        """
        求多项式的根。

        Parameters
        ----------
        coeffs : ndarray
            多项式系数 [c_0, ..., c_n]。
        basis : str
            基函数类型：'power', 'hermite', 'chebyshev'。

        Returns
        -------
        ndarray
            多项式的根。
        """
        if basis == 'power':
            return np.roots(coeffs[::-1])
        elif basis == 'hermite':
            A = CompanionMatrixEigenvalue.hermite_companion_matrix(coeffs)
            return np.linalg.eigvals(A)
        elif basis == 'chebyshev':
            A = CompanionMatrixEigenvalue.chebyshev_companion_matrix(coeffs)
            return np.linalg.eigvals(A)
        else:
            raise ValueError(f"不支持的基函数类型: {basis}")


class LogarithmicNorm:
    """
    矩阵对数范数计算。
    基于种子项目 697_log_norm。
    """

    @staticmethod
    def log_norm_l1(A):
        """
        L1 对数范数：
            μ_1(A) = max_j [ Re(a_{jj}) + Σ_{i≠j} |a_{ij}| ]

        Parameters
        ----------
        A : ndarray
            方阵。

        Returns
        -------
        float
            L1 对数范数。
        """
        A = np.asarray(A, dtype=complex)
        n = A.shape[0]

        B = np.abs(A) - np.diag(np.diag(np.abs(A)))
        c = np.real(np.diag(A))
        d = np.sum(B, axis=0)
        return np.max(c + d)

    @staticmethod
    def log_norm_l2(A):
        """
        L2 对数范数：
            μ_2(A) = λ_max((A + A^H) / 2)

        即 Hermitian 部分的最大特征值。

        Parameters
        ----------
        A : ndarray
            方阵。

        Returns
        -------
        float
            L2 对数范数。
        """
        A = np.asarray(A, dtype=complex)
        B = 0.5 * (A + A.conj().T)
        eigenvalues = np.linalg.eigvalsh(B)
        return np.max(eigenvalues)

    @staticmethod
    def log_norm_inf(A):
        """
        L∞ 对数范数：
            μ_∞(A) = max_i [ Re(a_{ii}) + Σ_{j≠i} |a_{ij}| ]

        Parameters
        ----------
        A : ndarray
            方阵。

        Returns
        -------
        float
            L∞ 对数范数。
        """
        A = np.asarray(A, dtype=complex)
        n = A.shape[0]

        B = np.abs(A) - np.diag(np.diag(np.abs(A)))
        c = np.real(np.diag(A))
        d = np.sum(B, axis=1)
        return np.max(c + d)

    @staticmethod
    def log_norm(A, p=2):
        """
        计算矩阵对数范数。

        Parameters
        ----------
        A : ndarray
            方阵。
        p : int or float
            范数类型：1, 2, 或 np.inf。

        Returns
        -------
        float
            对数范数。
        """
        if p == 1:
            return LogarithmicNorm.log_norm_l1(A)
        elif p == 2:
            return LogarithmicNorm.log_norm_l2(A)
        elif p == np.inf:
            return LogarithmicNorm.log_norm_inf(A)
        else:
            raise ValueError("p 必须为 1, 2 或 np.inf")


class LinearStabilityAnalysis:
    """
    线性稳定性分析。
    """

    @staticmethod
    def compute_jacobian_1d_diffusion_reaction(n, dx, D, reaction_derivative):
        """
        构造一维扩散-反应方程离散化的 Jacobian 矩阵。

        方程：∂u/∂t = D ∂²u/∂x² + R(u)
        在均匀网格上离散，Jacobian：
            J = D * L + diag(R'(u))

        其中 L 为离散 Laplacian（三对角矩阵）。

        Parameters
        ----------
        n : int
            网格点数。
        dx : float
            空间步长。
        D : float
            扩散系数。
        reaction_derivative : ndarray
            R'(u) 在各网格点的值。

        Returns
        -------
        ndarray
            n×n Jacobian 矩阵。
        """
        J = np.zeros((n, n))

        # 扩散部分（内部点 Dirichlet 边界）
        for i in range(1, n - 1):
            J[i, i - 1] = D / (dx ** 2)
            J[i, i] = -2.0 * D / (dx ** 2)
            J[i, i + 1] = D / (dx ** 2)

        # 反应部分
        for i in range(n):
            J[i, i] += reaction_derivative[i]

        return J

    @staticmethod
    def stability_criterion_eigenvalues(jacobian):
        """
        根据 Jacobian 特征值判断稳定性。

        Parameters
        ----------
        jacobian : ndarray
            Jacobian 矩阵。

        Returns
        -------
        dict
            {'stable': bool, 'max_real': float, 'eigenvalues': ndarray}
        """
        eigenvalues = np.linalg.eigvals(jacobian)
        max_real = np.max(np.real(eigenvalues))

        return {
            'stable': max_real < 0,
            'max_real': max_real,
            'eigenvalues': eigenvalues
        }

    @staticmethod
    def cfl_condition_1d_advection_diffusion(v, D, dx, safety_factor=0.5):
        """
        计算一维对流-扩散方程的 CFL 稳定性条件。

        对流 CFL：dt < dx / |v|
        扩散限制：dt < dx² / (2D)

        综合：dt < safety_factor * min(dx/|v|, dx²/(2D))

        Parameters
        ----------
        v : float
            对流速度。
        D : float
            扩散系数。
        dx : float
            空间步长。
        safety_factor : float
            安全因子。

        Returns
        -------
        float
            建议最大时间步长。
        """
        dt_adv = dx / max(abs(v), 1e-14)
        dt_diff = dx ** 2 / (2.0 * max(D, 1e-14))
        return safety_factor * min(dt_adv, dt_diff)

    @staticmethod
    def cfl_condition_2d_navier_stokes(vx_max, vy_max, nu, dx, dy, safety_factor=0.25):
        """
        计算二维 NS 方程的 CFL 条件。

        Parameters
        ----------
        vx_max, vy_max : float
            最大速度分量。
        nu : float
            运动粘度。
        dx, dy : float
            空间步长。
        safety_factor : float
            安全因子。

        Returns
        -------
        float
            建议最大时间步长。
        """
        dt_adv_x = dx / max(abs(vx_max), 1e-14)
        dt_adv_y = dy / max(abs(vy_max), 1e-14)
        dt_diff = 0.5 / (nu * (1.0 / dx ** 2 + 1.0 / dy ** 2))

        return safety_factor * min(dt_adv_x, dt_adv_y, dt_diff)


class BifurcationAnalysis:
    """
    分叉分析。
    基于种子项目 700_logistic_bifurcation 的思想。
    """

    @staticmethod
    def logistic_map(x, r):
        """
        Logistic 映射：x_{n+1} = r x_n (1 - x_n)。

        Parameters
        ----------
        x : float
            当前迭代值。
        r : float
            参数。

        Returns
        -------
        float
            下一迭代值。
        """
        return r * x * (1.0 - x)

    @staticmethod
    def find_attractors(r, x0=0.5, warmup=100, n_iter=500, tol=1e-5):
        """
        寻找 Logistic 映射的吸引子。

        Parameters
        ----------
        r : float
            Logistic 参数。
        x0 : float
            初始值。
        warmup : int
            预热迭代次数。
        n_iter : int
            记录迭代次数。
        tol : float
            吸引子识别容差。

        Returns
        -------
        ndarray
            吸引子集合。
        """
        x = x0
        for _ in range(warmup):
            x = BifurcationAnalysis.logistic_map(x, r)

        attractors = [x]
        for _ in range(n_iter):
            x = BifurcationAnalysis.logistic_map(x, r)
            # 检查是否已存在
            exists = False
            for a in attractors:
                if abs(x - a) < tol:
                    exists = True
                    break
            if not exists:
                attractors.append(x)

        return np.array(attractors)

    @staticmethod
    def lyapunov_exponent_logistic(r, x0=0.5, n_iter=10000):
        """
        计算 Logistic 映射的 Lyapunov 指数：
            λ = lim_{n→∞} (1/n) Σ_{i=0}^{n-1} ln|f'(x_i)|

        其中 f'(x) = r(1 - 2x)。

        λ > 0 表示混沌。

        Parameters
        ----------
        r : float
            Logistic 参数。
        x0 : float
            初始值。
        n_iter : int
            迭代次数。

        Returns
        -------
        float
            Lyapunov 指数。
        """
        x = x0
        lyap_sum = 0.0

        for _ in range(n_iter):
            x = BifurcationAnalysis.logistic_map(x, r)
            derivative = abs(r * (1.0 - 2.0 * x))
            if derivative < 1e-14:
                derivative = 1e-14
            lyap_sum += np.log(derivative)

        return lyap_sum / n_iter

    @staticmethod
    def phase_transition_bifurcation_parameter(T_undercooling, gamma, m_L, C0, D_l, k_p):
        """
        计算枝晶生长的分叉参数（Mullins-Sekerka 稳定性参数）。

        Mullins-Sekerka 不稳定性判据：
            当波长 λ > λ_c = 2π √(Γ / (V ΔT_0))
        界面失稳，形成枝晶。

        无量纲参数：
            σ* = 2D_l d_0 / (V λ²)
        其中 d_0 = Γ / ΔT_0 为毛细长度。

        Parameters
        ----------
        T_undercooling : float
            过冷度。
        gamma : float
            界面能。
        m_L : float
            液相线斜率。
        C0 : float
            初始浓度。
        D_l : float
            液相扩散系数。
        k_p : float
            分配系数。

        Returns
        -------
        dict
            分叉参数。
        """
        # 毛细长度
        delta_T0 = -m_L * C0 * (1.0 - k_p)
        if abs(delta_T0) < 1e-14:
            delta_T0 = 1e-14

        d0 = gamma / delta_T0

        # 特征速度（Ivantsov 解近似）
        V_tip = D_l * T_undercooling / (gamma)

        # 稳定性参数
        sigma_star = 2.0 * D_l * d0 * V_tip / (D_l ** 2)

        return {
            'capillary_length': d0,
            'characteristic_velocity': V_tip,
            'stability_parameter': sigma_star,
            'unstable': sigma_star < 0.025  # 经验临界值
        }
