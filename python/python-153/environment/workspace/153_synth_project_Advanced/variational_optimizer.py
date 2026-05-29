"""
variational_optimizer.py
基于项目 120_broyden (Broyden拟牛顿法) 与 1374_unstable_ode (不稳定ODE)
的量子变分参数优化模块。

核心数学模型:
1. Broyden 拟牛顿法:
   近似 Jacobian 逆矩阵 B_k ≈ J_k^{-1} 的低秩更新:
   B_{k+1} = B_k + (s_k - B_k y_k) s_k^T B_k / (s_k^T B_k y_k)
   其中 s_k = x_{k+1} - x_k, y_k = F(x_{k+1}) - F(x_k)

   本实现采用递归步存储形式 (Sherman-Morrison-Woodbury):
   不显式存储 n x n 矩阵，而是存储 n x maxdim 的历史步。

2. 收敛判据:
   ||F(x*)|| <= atol + rtol * ||F(x_0)||

3. 不稳定 ODE 系统的稳定性分析:
   dy/dt = A y,  A = [[mu, 1/mu], [-1/mu, mu]]
   特征值: lambda = mu ± i/mu
   当 mu > 0 时，系统不稳定 (指数增长 + 高频振荡)。
   用于量子变分 landscape 中的鞍点与局部不稳定性分析。

4. 变分量子本征求解器 (VQE) 能量最小化:
   E(theta) = <psi(theta)| H |psi(theta)>
   通过 Broyden 方法求解 grad E(theta) = 0。
"""

import numpy as np
from typing import Callable, Tuple, Optional


def unstable_ode_system(t: float, y: np.ndarray, mu: float = 0.1) -> np.ndarray:
    """
    经典的不稳定 ODE 系统 (来自 1374_unstable_ode)。
    dy/dt = A y, 其中 A = [[mu, 1/mu], [-1/mu, mu]]

    特征值: lambda_{1,2} = mu ± i/mu
    当 mu = 0.1 时，实部为正 (不稳定)，虚部为 ±10i (高频振荡)。
    """
    if mu <= 0:
        raise ValueError("mu must be positive")
    if len(y) != 2:
        raise ValueError("State vector must have dimension 2")

    A = np.array([[mu, 1.0 / mu], [-1.0 / mu, mu]], dtype=np.float64)
    return A @ y


def unstable_exact_solution(t: float, mu: float = 0.1) -> np.ndarray:
    """
    不稳定 ODE 的精确解析解。
    y_1(t) = exp(mu*t) * (cos(t/mu) - mu^2 * sin(t/mu))
    y_2(t) = mu * exp(mu*t) * (cos(t/mu) - mu^2 * sin(t/mu))
            + exp(mu*t) * (-sin(t/mu)/mu - mu*cos(t/mu))
    """
    if mu <= 0:
        raise ValueError("mu must be positive")

    exp_term = np.exp(mu * t)
    cos_term = np.cos(t / mu)
    sin_term = np.sin(t / mu)

    y1 = exp_term * (cos_term - mu * mu * sin_term)
    y2 = mu * exp_term * (cos_term - mu * mu * sin_term)
    y2 += exp_term * (-sin_term / mu - mu * cos_term)

    return np.array([y1, y2])


def broyden_quasi_newton(
    F: Callable[[np.ndarray], np.ndarray],
    x0: np.ndarray,
    atol: float = 1e-8,
    rtol: float = 1e-6,
    maxit: int = 100,
    maxdim: int = 10
) -> Tuple[np.ndarray, int]:
    """
    Broyden 拟牛顿法求解非线性方程组 F(x) = 0。

    参数:
        F: 非线性向量值函数
        x0: 初始猜测
        atol: 绝对误差容限
        rtol: 相对误差容限
        maxit: 最大迭代次数
        maxdim: Broyden 历史步上限 (重启前)

    返回:
        (解向量, 终止标志: 0=成功, 1=失败)
    """
    x = x0.copy().astype(np.float64)
    n = len(x)

    fc = F(x)
    fnrm = np.linalg.norm(fc) / np.sqrt(n)
    if fnrm < 1e-15:
        return x, 0

    stop_tol = atol + rtol * fnrm

    # 预分配历史步数组
    stp = np.zeros((n, maxdim))
    stp[:, 0] = -fc
    stp_nrm = np.zeros(maxdim)
    stp_nrm[0] = np.dot(stp[:, 0], stp[:, 0])

    nbroy = 0
    itc = 0

    while itc < maxit:
        fnrmo = fnrm
        nbroy += 1
        itc += 1

        # 更新解
        if nbroy < maxdim:
            x = x + stp[:, nbroy - 1]
        else:
            x = x + stp[:, maxdim - 1]

        fc = F(x)
        fnrm = np.linalg.norm(fc) / np.sqrt(n)

        # 收敛检查
        if fnrm <= stop_tol:
            return x, 0

        # 单调下降保护
        if fnrmo <= fnrm and itc > 1:
            # 回退并重启
            x = x - stp[:, min(nbroy - 1, maxdim - 1)]
            nbroy = 0
            stp[:, 0] = -fc
            stp_nrm[0] = np.dot(stp[:, 0], stp[:, 0])
            continue

        # 构造下一步 (递归 Sherman-Morrison-Woodbury)
        if nbroy + 1 < maxdim:
            z = -fc
            for kbr in range(nbroy - 1):
                if stp_nrm[kbr] < 1e-15:
                    continue
                z = z + stp[:, kbr + 1] * np.dot(stp[:, kbr], z) / stp_nrm[kbr]

            if stp_nrm[nbroy - 1] > 1e-15:
                zz = np.dot(stp[:, nbroy - 1], z) / stp_nrm[nbroy - 1]
                denom = 1.0 - zz
                if abs(denom) > 1e-15:
                    stp[:, nbroy] = z / denom
                    stp_nrm[nbroy] = np.dot(stp[:, nbroy], stp[:, nbroy])
                else:
                    # 奇异，重启
                    nbroy = 0
                    stp[:, 0] = -fc
                    stp_nrm[0] = np.dot(stp[:, 0], stp[:, 0])
            else:
                nbroy = 0
                stp[:, 0] = -fc
                stp_nrm[0] = np.dot(stp[:, 0], stp[:, 0])
        else:
            # 达到 maxdim，重启
            nbroy = 0
            stp[:, 0] = -fc
            stp_nrm[0] = np.dot(stp[:, 0], stp[:, 0])

    # 最终检查
    fc = F(x)
    fnrm = np.linalg.norm(fc) / np.sqrt(n)
    if fnrm <= stop_tol:
        return x, 0
    return x, 1


class VariationalQuantumOptimizer:
    """
    变分量子电路参数优化器。
    使用 Broyden 拟牛顿法最小化能量泛函 E(theta)。
    """

    def __init__(
        self,
        energy_func: Optional[Callable[[np.ndarray], float]] = None,
        gradient_func: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        atol: float = 1e-7,
        rtol: float = 1e-5,
        maxit: int = 200,
        maxdim: int = 15
    ):
        self.energy_func = energy_func
        self.gradient_func = gradient_func
        self.atol = atol
        self.rtol = rtol
        self.maxit = maxit
        self.maxdim = maxdim
        self.history: list = []

    def _numerical_gradient(self, theta: np.ndarray, eps: float = 1e-7) -> np.ndarray:
        """中心差分计算数值梯度。"""
        grad = np.zeros_like(theta)
        for i in range(len(theta)):
            theta_plus = theta.copy()
            theta_minus = theta.copy()
            theta_plus[i] += eps
            theta_minus[i] -= eps
            grad[i] = (self.energy_func(theta_plus) - self.energy_func(theta_minus)) / (2.0 * eps)
        return grad

    def optimize(self, theta0: np.ndarray) -> Tuple[np.ndarray, float, int]:
        """
        优化量子电路参数。

        返回:
            (最优参数, 最优能量, 终止标志)
        """
        theta0 = np.array(theta0, dtype=np.float64)

        def F(theta):
            if self.gradient_func is not None:
                return self.gradient_func(theta)
            return self._numerical_gradient(theta)

        theta_opt, ierr = broyden_quasi_newton(
            F, theta0, self.atol, self.rtol, self.maxit, self.maxdim
        )

        energy_opt = self.energy_func(theta_opt)
        self.history.append({"theta": theta_opt.copy(), "energy": energy_opt, "ierr": ierr})

        return theta_opt, energy_opt, ierr

    def vqe_minimize(
        self,
        hamiltonian: np.ndarray,
        ansatz_func: Callable[[np.ndarray], np.ndarray],
        theta0: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """
        变分量子本征求解器 (VQE) 能量最小化。
        E(theta) = <psi(theta)| H |psi(theta)>

        参数:
            hamiltonian: 厄米哈密顿量矩阵
            ansatz_func: 参数化量子电路，输入参数 theta，输出量子态向量
            theta0: 初始参数
        """
        H = np.array(hamiltonian, dtype=np.complex128)
        if not np.allclose(H, H.conj().T, atol=1e-10):
            raise ValueError("Hamiltonian must be Hermitian")

        def energy(theta):
            psi = ansatz_func(theta)
            psi = psi / (np.linalg.norm(psi) + 1e-15)
            E = np.vdot(psi, H @ psi).real
            return E

        self.energy_func = energy
        self.gradient_func = None

        theta_opt, E_opt, ierr = self.optimize(theta0)
        return theta_opt, E_opt
