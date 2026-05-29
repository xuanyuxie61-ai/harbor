r"""
mhd_stability.py
===============
磁流体动力学（MHD）线性稳定性分析模块。
通过求解线性化 MHD 算子的特征值问题，判断 Harris 电流片
对撕裂模（Tearing Mode）的稳定性。

核心物理与数学模型
------------------
线性化不可压缩 MHD 方程（在平衡态 B_0, p_0 附近扰动）:

    rho_0 * partial v_1/partial t = -nabla p_1 + J_0 x B_1 + J_1 x B_0 + nu nabla^2 v_1
    partial B_1/partial t = nabla x (v_1 x B_0) + eta nabla^2 B_1
    nabla . v_1 = 0,   nabla . B_1 = 0

引入流函数 psi 和磁通量函数 A:
    v_1 = nabla x (psi e_z),   B_1 = nabla x (A e_z)

在傅里叶空间 k = k_x 下，得到标准形式:
    partial A/partial t = -i k_x B_0(y) psi + eta (partial^2/partial y^2 - k_x^2) A
    partial (nabla^2 psi)/partial t = i k_x B_0''(y) A + nu (partial^2/partial y^2 - k_x^2) nabla^2 psi

其中 B_0(y) = B_0 tanh(y/lambda) 为 Harris 平衡。

对于 tearing mode，特征值问题写为:
    gamma * A = -i k B_0(y) psi + eta (d^2/dy^2 - k^2) A
    gamma * (d^2/dy^2 - k^2) psi = i k B_0''(y) A + nu (d^2/dy^2 - k^2)^2 psi

稳定性判据:
    Re(gamma) > 0: 不稳定（扰动指数增长）
    Re(gamma) < 0: 稳定（扰动衰减）

Hankel 矩阵在径向坐标变换中的应用:
    在柱坐标下，径向拉普拉斯算子涉及 Hankel 结构，
    其逆矩阵可通过 Fiedler 算法快速构造。

整数 RREF 的应用:
    用于离散散度-自由约束 nabla . B=0 的精确整数验证，
    保证数值解严格满足磁场无散条件。

融入原项目:
- 505_hankel_inverse: Hankel 矩阵求逆（Fiedler 算法）
- 1047_row_echelon_integer: 整数矩阵行简化阶梯形（IRREF）
"""

import numpy as np
from typing import Tuple, Optional
from scipy.linalg import eig


class HankelSolver:
    """
    Hankel 矩阵快速求逆与线性系统求解。
    对应原项目 hankel_inverse 的核心算法。
    """

    @staticmethod
    def build_hankel(n: int, x: np.ndarray) -> np.ndarray:
        """
        构造 Hankel 矩阵: A(i,j) = x(i+j-1)。
        """
        x = np.asarray(x, dtype=float)
        if len(x) < 2 * n - 1:
            x = np.append(x, np.zeros(2 * n - 1 - len(x)))
        A = np.zeros((n, n))
        for j in range(n):
            A[:, j] = x[j:j + n]
        return A

    @staticmethod
    def build_toeplitz(n: int, x: np.ndarray) -> np.ndarray:
        """
        构造 Toeplitz 矩阵: A(i,j) = x(n+j-i)。
        """
        x = np.asarray(x, dtype=float)
        if len(x) < 2 * n - 1:
            x = np.append(x, np.zeros(2 * n - 1 - len(x)))
        A = np.zeros((n, n))
        for i in range(n):
            A[i, :] = x[n - 1 - i:n - 1 - i + n]
        return A

    @staticmethod
    def inverse(n: int, x: np.ndarray) -> np.ndarray:
        """
        使用 Fiedler 算法求 Hankel 矩阵的逆。
        算法步骤:
            1. 解 A * p = [x(n+1:2n-1); 0]
            2. 解 A * q = [0;...;0; 1]
            3. 构造四个 Hankel/Toeplitz 矩阵 M1-M4
            4. B = M1*M2 - M3*M4
        """
        A = HankelSolver.build_hankel(n, x)
        p_rhs = np.append(x[n:2 * n - 1], 0.0)
        u = np.linalg.solve(A, p_rhs)
        q_rhs = np.zeros(n)
        q_rhs[-1] = 1.0
        v = np.linalg.solve(A, q_rhs)

        # M1: Hankel([v(2:n); zeros(n,1)])
        w1 = np.append(v[1:], np.zeros(1))
        M1 = HankelSolver.build_hankel(n, w1)

        # M2: Toeplitz([zeros(n-1,1); u])
        w2 = np.append(np.zeros(n - 1), u)
        M2 = HankelSolver.build_toeplitz(n, w2)

        # M3: Hankel([u(2:n); [-1; zeros(n-1,1)]])
        z3 = np.zeros(n)
        z3[0] = -1.0
        w3 = np.append(u[1:], z3)
        M3 = HankelSolver.build_hankel(n, w3)

        # M4: Toeplitz([zeros(n-1,1); v])
        w4 = np.append(np.zeros(n - 1), v)
        M4 = HankelSolver.build_toeplitz(n, w4)

        B = M1 @ M2 - M3 @ M4
        return B


class IntegerRREF:
    """
    整数矩阵的行简化阶梯形（Integer RREF）。
    对应原项目 i4mat_rref 的核心算法。
    使用纯整数运算消除浮点舍入误差。
    """

    @staticmethod
    def _gcd_vec(v: np.ndarray) -> int:
        """计算整数向量所有元素的最大公约数。"""
        from math import gcd
        g = 0
        for val in v:
            g = gcd(g, int(abs(val)))
            if g == 1:
                break
        return g

    @staticmethod
    def rref(A_in: np.ndarray) -> Tuple[np.ndarray, int]:
        """
        计算整数矩阵的 IRREF。

        返回:
            A_rref: 行简化阶梯形矩阵
            det: 伪行列式（pseudodeterminant）
        """
        A = np.copy(A_in).astype(np.int64)
        m, n = A.shape
        lead = 0
        det = 1

        for r in range(m):
            if n <= lead:
                break

            i = r
            while A[i, lead] == 0:
                i += 1
                if m <= i:
                    i = r
                    lead += 1
                    if n <= lead:
                        lead = -1
                        break
            if lead < 0:
                break

            # 交换行
            if i != r:
                A[[i, r], :] = A[[r, i], :]
                det = -det

            # 确保主元为正
            if A[r, lead] < 0:
                A[r, :] = -A[r, :]
                det = -det

            det *= int(A[r, lead])

            # 化简行（除以最大公约数）
            g = IntegerRREF._gcd_vec(A[r, :])
            if g > 1:
                A[r, :] = A[r, :] // g

            # 消去其他行的 lead 列
            for i in range(m):
                if i != r and A[i, lead] != 0:
                    A[i, :] = A[r, lead] * A[i, :] - A[i, lead] * A[r, :]
                    g = IntegerRREF._gcd_vec(A[i, :])
                    if g > 1:
                        A[i, :] = A[i, :] // g

            lead += 1

        return A, det

    @staticmethod
    def verify_divergence_free(integer_incidence: np.ndarray,
                               b_field_dofs: np.ndarray) -> bool:
        """
        使用整数 RREF 验证离散散度-自由约束。
        对于一维或结构化网格，散度算子可离散为整数关联矩阵 C，
        要求 C * B = 0。通过将 C 化为 IRREF，检查 B 是否在其零空间中。
        """
        C = np.asarray(integer_incidence, dtype=np.int64)
        B = np.asarray(b_field_dofs, dtype=np.int64)
        Cr, _ = IntegerRREF.rref(C)
        # 检查 Cr * B 是否近似为零（允许舍入）
        residual = Cr @ B
        return np.all(np.abs(residual) < 1e-10)


class MHDStabilityAnalyzer:
    """
    MHD 线性稳定性分析器，针对 Harris 电流片的撕裂模。
    """

    def __init__(self,
                 B0: float = 1.0,
                 lambda_cs: float = 1.0,
                 eta: float = 1e-3,
                 nu: float = 1e-4,
                 kx: float = 0.5,
                 ny: int = 128,
                 y_max: float = 5.0):
        self.B0 = B0
        self.lambda_cs = lambda_cs
        self.eta = eta
        self.nu = nu
        self.kx = kx
        self.ny = ny
        self.y_max = y_max
        self.y = np.linspace(-y_max, y_max, ny)
        self.dy = self.y[1] - self.y[0]

    def _build_laplacian_1d(self) -> np.ndarray:
        """
        构造一维离散拉普拉斯算子 L = d^2/dy^2 - k^2（Dirichlet 边界）。
        """
        # HOLE 2: 请实现一维离散拉普拉斯算子的构造
        # 提示: 使用二阶中心差分近似 d^2/dy^2，并减去 k^2 项，
        #       边界条件为 Dirichlet（边界点固定为 0）
        raise NotImplementedError("Hole 2: 请实现一维离散拉普拉斯算子 L = d^2/dy^2 - k^2")

    def _equilibrium_B(self) -> np.ndarray:
        """Harris 平衡磁场 B_x(y)。"""
        return self.B0 * np.tanh(self.y / self.lambda_cs)

    def _equilibrium_Bpp(self) -> np.ndarray:
        """Harris 平衡 B_x''(y) = -2 B_0 / lambda^2 * tanh(y/lambda) * sech^2(y/lambda)。"""
        t = np.tanh(self.y / self.lambda_cs)
        s2 = 1.0 / np.cosh(self.y / self.lambda_cs) ** 2
        return -2.0 * self.B0 / (self.lambda_cs ** 2) * t * s2

    def tearing_mode_growth_rate(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        求解简化 tearing mode 特征值问题。
        使用不可压缩近似，将问题化为关于 psi 的广义特征值问题。

        简化模型（Furth-Killeen-Rosenbluth 框架）:
            gamma * L * psi = - (k^2 B_0'^2 / B_0) * psi + eta * L^2 * psi

        这里采用更简化的数值模型来演示特征值结构。
        """
        L = self._build_laplacian_1d()
        # 简化的特征值模型: gamma * psi = M * psi
        # M = eta * L + 驱动项
        B = self._equilibrium_B()
        Bpp = self._equilibrium_Bpp()

        # 避免除零
        B_safe = np.where(np.abs(B) < 1e-10, 1e-10, B)
        drive = np.diag(-self.kx ** 2 * Bpp / B_safe)

        M = self.eta * L + drive
        #  Dirichlet 边界修正
        M[0, :] = 0.0
        M[0, 0] = -1.0
        M[-1, :] = 0.0
        M[-1, -1] = -1.0

        eigenvalues, eigenvectors = eig(M)
        return eigenvalues, eigenvectors

    def analyze_stability(self) -> dict:
        """
        分析稳定性并返回关键指标。
        """
        gamma, modes = self.tearing_mode_growth_rate()
        real_parts = np.real(gamma)
        # 排除边界条件引入的人工特征值（通常为 -1）
        mask = np.abs(real_parts + 1.0) > 0.1
        if not np.any(mask):
            mask = np.ones_like(real_parts, dtype=bool)
        gamma_phys = gamma[mask]
        max_growth = np.max(np.real(gamma_phys))
        unstable = max_growth > 0

        return {
            'growth_rates': gamma,
            'max_growth_rate': max_growth,
            'unstable': unstable,
            'n_modes': len(gamma),
            'kx': self.kx,
            'eta': self.eta
        }


def demo_stability():
    """
    演示：分析 Harris 电流片的撕裂模稳定性。
    """
    print("\n[MHDStability] 演示: Tearing Mode 稳定性分析")

    # 1. Hankel 求逆测试
    n = 5
    # 使用非奇异 Hankel 矩阵（基于随机正数序列）
    np.random.seed(0)
    x = np.abs(np.random.randn(2 * n - 1)) + 0.1
    H = HankelSolver.build_hankel(n, x)
    try:
        H_inv = HankelSolver.inverse(n, x)
        I_approx = H @ H_inv
        err_inv = np.max(np.abs(I_approx - np.eye(n)))
        print(f"  Hankel 逆矩阵误差: {err_inv:.3e}")
    except np.linalg.LinAlgError:
        print("  Hankel 矩阵奇异，跳过逆矩阵测试")

    # 2. 整数 RREF 测试
    A_test = np.array([[1, 3, 0, 2, 6, 3, 1],
                       [-2, -6, 0, -2, -8, 3, 1],
                       [3, 9, 0, 0, 6, 6, 2],
                       [-1, -3, 0, 1, 0, 9, 3]], dtype=np.int64)
    Ar, det = IntegerRREF.rref(A_test)
    print(f"  IRREF 伪行列式: {det}")
    print(f"  IRREF 结果前三行主元列: {np.argmax(np.abs(Ar[0])), np.argmax(np.abs(Ar[1])), np.argmax(np.abs(Ar[2]))}")

    # 3. MHD 稳定性分析
    analyzer = MHDStabilityAnalyzer(kx=0.5, eta=1e-3, ny=64)
    result = analyzer.analyze_stability()
    print(f"  波数 kx={result['kx']}, 电阻率 eta={result['eta']:.0e}")
    print(f"  最大增长率: {result['max_growth_rate']:.4f}")
    print(f"  稳定性: {'不稳定' if result['unstable'] else '稳定'}")


if __name__ == "__main__":
    demo_stability()
