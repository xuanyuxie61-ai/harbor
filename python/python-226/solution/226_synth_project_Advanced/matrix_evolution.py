"""
矩阵指数演化算子模块
===================
实现中微子传播的精确演化算符 U(x, x₀) = exp(-iH(x-x₀))。

三种矩阵指数算法（Moler & Van Loan, 2003）：
1. Padé近似 + 缩放平方法（标准方法，O(N³)）
2. Taylor级数法（小规模矩阵）
3. 本征值分解法（适用于正规矩阵）

核心数学公式：
1. 矩阵指数定义：
   exp(A) = Σ_{k=0}^∞ A^k / k! = I + A + A²/2! + A³/3! + ...

2. Padé近似 [p/q]：
   exp(A) ≈ [D_pq(A)]⁻¹ N_pq(A)
   N_pq(A) = Σ_{k=0}^p (p+q-k)!p! / ((p+q)!k!(p-k)!) A^k
   D_pq(A) = Σ_{k=0}^q (p+q-k)!q! / ((p+q)!k!(q-k)!) (-A)^k

3. 缩放平方法：
   exp(A) = [exp(A/2^s)]^{2^s}
   选择 s 使得 ||A/2^s|| < 1

4. 中微子演化：
   |ν(x)⟩ = U(x, x₀)|ν(x₀)⟩
   U(x, x₀) = exp(-i∫_{x₀}^x H(x')dx')
   对于分段常数H: U = Π_k exp(-iH_k Δx_k)

数据来源：
- 741_matrix_exponential: 三种矩阵指数算法的实现
"""

import numpy as np
from typing import Tuple, Optional
from scipy.linalg import expm


class MatrixExponential:
    """
    矩阵指数计算器。

    实现三种算法：
    1. Padé近似 + 缩放平方法（推荐）
    2. Taylor级数法
    3. 本征值分解法
    """

    def __init__(self, method: str = 'pade'):
        """
        初始化矩阵指数计算器。

        参数：
            method: 算法选择 ('pade', 'taylor', 'eigenvalue')
        """
        valid_methods = ['pade', 'taylor', 'eigenvalue']
        if method not in valid_methods:
            raise ValueError(f"无效方法: {method}, 可选: {valid_methods}")

        self.method = method

    def compute(self, A: np.ndarray) -> np.ndarray:
        """
        计算矩阵指数 exp(A)。

        参数：
            A: N×N矩阵

        返回：
            expA: exp(A)
        """
        if self.method == 'pade':
            return self._pade_scaling_squaring(A)
        elif self.method == 'taylor':
            return self._taylor_series(A)
        elif self.method == 'eigenvalue':
            return self._eigenvalue_decomposition(A)

    def _pade_scaling_squaring(self, A: np.ndarray, q: int = 6) -> np.ndarray:
        """
        Padé近似 + 缩放平方法。

        算法步骤：
        1. 计算缩放因子 s = max(0, ⌈log₂(||A||_∞)⌉ + 1)
        2. 缩放: Ã = A / 2^s
        3. 计算 [q/q] Padé近似: R = D⁻¹ N
           N = Σ_{k=0}^q c_k Ã^k
           D = Σ_{k=0}^q c_k (-Ã)^k
           c_k = (2q-k)!q! / ((2q)!k!(q-k)!)
        4. 平方还原: exp(A) = R^{2^s}

        参数：
            A: 输入矩阵
            q: Padé阶数（默认6）

        返回：
            expA: 矩阵指数
        """
        N = A.shape[0]

        # 步骤1: 计算缩放因子
        norm_A = np.linalg.norm(A, ord=np.inf)
        if norm_A < 1e-15:
            return np.eye(N, dtype=A.dtype)

        s = max(0, int(np.ceil(np.log2(norm_A))) + 1)

        # 步骤2: 缩放
        A_scaled = A / (2.0 ** s)

        # 步骤3: Padé近似
        # 计算系数 c_k
        c = np.zeros(q + 1)
        c[0] = 1.0
        for k in range(1, q + 1):
            c[k] = c[k-1] * (q - k + 1) / (k * (2 * q - k + 1))

        # 计算 N 和 D
        N_mat = np.eye(N, dtype=A.dtype)
        D_mat = np.eye(N, dtype=A.dtype)
        A_power = np.eye(N, dtype=A.dtype)

        for k in range(1, q + 1):
            A_power = A_power @ A_scaled
            N_mat += c[k] * A_power
            D_mat += c[k] * ((-1) ** k) * A_power

        # 求解 R = D⁻¹ N
        try:
            R = np.linalg.solve(D_mat, N_mat)
        except np.linalg.LinAlgError:
            # 使用伪逆作为后备
            R = np.linalg.lstsq(D_mat, N_mat, rcond=None)[0]

        # 步骤4: 平方还原
        for _ in range(s):
            R = R @ R

        return R

    def _taylor_series(self, A: np.ndarray, max_terms: int = 100,
                       tol: float = 1e-15) -> np.ndarray:
        """
        Taylor级数法。

        exp(A) = Σ_{k=0}^∞ A^k / k!

        递推计算：
        F_0 = I
        F_k = F_{k-1} × A / k
        E_k = E_{k-1} + F_k

        终止条件: ||E_k + F_k - E_k||_1 < tol

        参数：
            A: 输入矩阵
            max_terms: 最大项数
            tol: 收敛容差

        返回：
            expA: 矩阵指数
        """
        N = A.shape[0]
        E = np.eye(N, dtype=A.dtype)
        F = np.eye(N, dtype=A.dtype)

        for k in range(1, max_terms):
            F = F @ A / k
            E_new = E + F

            # 检查收敛
            if np.linalg.norm(E_new - E, ord=1) < tol:
                break

            E = E_new

        return E

    def _eigenvalue_decomposition(self, A: np.ndarray) -> np.ndarray:
        """
        本征值分解法。

        如果 A = V D V⁻¹（D为对角矩阵），则：
        exp(A) = V exp(D) V⁻¹
        exp(D) = diag(exp(d_1), ..., exp(d_N))

        适用于正规矩阵（A†A = AA†），对于非正规矩阵可能不稳定。

        参数：
            A: 输入矩阵

        返回：
            expA: 矩阵指数
        """
        # 检查是否为厄米特矩阵
        if np.allclose(A, A.conj().T, atol=1e-10):
            eigenvalues, V = np.linalg.eigh(A)
            exp_D = np.diag(np.exp(eigenvalues))
            return V @ exp_D @ V.conj().T
        else:
            eigenvalues, V = np.linalg.eig(A)
            exp_D = np.diag(np.exp(eigenvalues))
            try:
                V_inv = np.linalg.inv(V)
                return V @ exp_D @ V_inv
            except np.linalg.LinAlgError:
                # 退化为Taylor方法
                return self._taylor_series(A)


class NeutrinoEvolution:
    """
    中微子传播演化器。

    求解薛定谔型方程：
    i d/dx |ν(x)⟩ = H(x) |ν(x)⟩

    方法：
    1. 分段常数近似：
       U(x_{n+1}, x_n) = exp(-i H_n Δx)

    2.  Magnus展开（高阶修正）：
       U = exp(Ω₁ + Ω₂ + ...)
       Ω₁ = -i ∫ H dx
       Ω₂ = -½ ∫∫ [H(x₁), H(x₂)] dx₁dx₂

    3.  Runge-Kutta方法（有限差分）
    """

    def __init__(self, method: str = 'matrix_exp'):
        """
        初始化演化器。

        参数：
            method: 演化方法 ('matrix_exp', 'runge_kutta', 'magnus')
        """
        self.method = method
        self.mat_exp = MatrixExponential('pade')

    def evolve_constant_H(self, H: np.ndarray, dx: float) -> np.ndarray:
        """
        常数哈密顿量的精确演化。

        U(dx) = exp(-i H dx)

        参数：
            H: 哈密顿量矩阵 (N×N)
            dx: 传播距离

        返回：
            U: 演化算符 (N×N)
        """
        A = -1j * H * dx
        return self.mat_exp.compute(A)

    def evolve_varying_H(self, H_func, x_start: float, x_end: float,
                         N_steps: int, psi0: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        变哈密顿量的数值演化。

        使用分段常数近似：
        ψ(x_{n+1}) = exp(-i H(x_n) Δx) ψ(x_n)

        参数：
            H_func: 哈密顿量函数 H(x) → 矩阵
            x_start: 起始位置
            x_end: 终止位置
            N_steps: 步数
            psi0: 初态波函数

        返回：
            x_array: 位置数组
            psi_array: 波函数演化 (N_steps+1, N)
        """
        dx = (x_end - x_start) / N_steps
        x_array = np.linspace(x_start, x_end, N_steps + 1)

        N_dim = len(psi0)
        psi_array = np.zeros((N_steps + 1, N_dim), dtype=complex)
        psi_array[0] = psi0

        psi = psi0.copy()

        for n in range(N_steps):
            x_n = x_array[n]
            H_n = H_func(x_n)

            # 演化一步
            U_step = self.evolve_constant_H(H_n, dx)
            psi = U_step @ psi

            # 归一化（防止数值漂移）
            norm = np.linalg.norm(psi)
            if norm > 1e-15:
                psi /= norm

            psi_array[n + 1] = psi

        return x_array, psi_array

    def compute_oscillation_probabilities(self, H_func, x_start: float, x_end: float,
                                          N_steps: int, alpha: int = 0) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算振荡概率随传播距离的变化。

        P(να → νβ; x) = |⟨νβ|U(x, 0)|να⟩|²

        参数：
            H_func: 哈密顿量函数
            x_start, x_end: 传播范围
            N_steps: 步数
            alpha: 初态味指标 (0=e, 1=μ, 2=τ)

        返回：
            x_array: 位置数组
            probs: 概率数组 (N_steps+1, 3)
        """
        # 初态
        psi0 = np.zeros(3, dtype=complex)
        psi0[alpha] = 1.0

        # 演化
        x_array, psi_array = self.evolve_varying_H(H_func, x_start, x_end, N_steps, psi0)

        # 计算概率
        N_steps_actual = len(x_array)
        probs = np.zeros((N_steps_actual, 3))

        for n in range(N_steps_actual):
            psi = psi_array[n]
            for beta in range(3):
                probs[n, beta] = np.abs(psi[beta]) ** 2

            # 归一化检查
            total = np.sum(probs[n])
            if abs(total - 1.0) > 1e-10:
                probs[n] /= total  # 强制归一化

        return x_array, probs

    def magnus_expansion(self, H_func, x_start: float, x_end: float,
                         order: int = 2) -> np.ndarray:
        """
        Magnus展开计算演化算符。

        Ω₁ = -i ∫_{x₀}^{x} H(x') dx'
        Ω₂ = -½ ∫_{x₀}^{x} ∫_{x₀}^{x'} [H(x'), H(x'')] dx'' dx'

        U(x, x₀) ≈ exp(Ω₁ + Ω₂)

        参数：
            H_func: 哈密顿量函数
            x_start, x_end: 积分范围
            order: Magnus阶数 (1 或 2)

        返回：
            U: 演化算符
        """
        # 简化实现：使用梯形法则积分
        N_quad = 100
        x_quad = np.linspace(x_start, x_end, N_quad)
        dx = x_quad[1] - x_quad[0]

        # Ω₁
        Omega1 = np.zeros((3, 3), dtype=complex)
        for x in x_quad:
            Omega1 += H_func(x) * dx
        Omega1 *= -1j

        if order == 1:
            return self.mat_exp.compute(Omega1)

        # Ω₂ (对易子积分)
        Omega2 = np.zeros((3, 3), dtype=complex)
        for i, x1 in enumerate(x_quad):
            H1 = H_func(x1)
            for j, x2 in enumerate(x_quad[:i]):
                H2 = H_func(x2)
                commutator = H1 @ H2 - H2 @ H1
                Omega2 += commutator * dx * dx
        Omega2 *= -0.5

        Omega_total = Omega1 + Omega2
        return self.mat_exp.compute(Omega_total)


def vacuum_oscillation_length(delta_m2: float, energy_eV: float) -> float:
    """
    计算真空振荡长度。

    L_osc = 4πE / Δm² × ℏc

    物理意义：振荡概率的空间周期

    参数：
        delta_m2: 质量平方差 (eV²)
        energy_eV: 中微子能量 (eV)

    返回：
        L_osc: 振荡长度 (m)
    """
    if abs(delta_m2) < 1e-20:
        return np.inf

    # ℏc = 1.97327e-7 eV·m
    hbar_c = 1.97327e-7  # eV·m

    L_osc = 4 * np.pi * energy_eV / abs(delta_m2) * hbar_c

    return L_osc


def msW_resonance_energy(delta_m2: float, theta: float,
                         electron_density: float) -> float:
    """
    计算MSW共振能量。

    共振条件：2EV = Δm² cos(2θ)
    其中 V = √2 G_F N_e

    E_res = Δm² cos(2θ) / (2V)

    参数：
        delta_m2: 质量平方差 (eV²)
        theta: 混合角 (弧度)
        electron_density: 电子密度 (mol/cm³)

    返回：
        E_res: 共振能量 (eV)
    """
    from neutrino_physics import matter_potential

    # 计算物质势（单位归一化到1 eV）
    V = matter_potential(electron_density, 1.0)

    if abs(V) < 1e-20:
        return np.inf

    E_res = abs(delta_m2 * np.cos(2 * theta) / (2 * V))

    return E_res
