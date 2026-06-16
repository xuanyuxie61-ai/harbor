"""
reliability.py - 结构可靠性算法：FORM / SORM / Hasofer-Lind

本模块实现结构可靠性的核心算法：

1. FORM (First Order Reliability Method):
   在标准正态空间中，将非线性极限状态面在验算点处线性化，
   通过 Hasofer-Lind 迭代找到设计验算点（最可几点, MPP）。
   可靠度指标 beta = ||u*||,  P_f ≈ Phi(-beta).

2. SORM (Second Order Reliability Method):
   在设计验算点处，将极限状态面二次近似，
   利用主曲率对FORM结果进行 Breitung 修正。
   P_f ≈ Phi(-beta) * prod_i (1 + beta*kappa_i)^{-1/2}

3. Newton-Raphson 逆向通信:
   用于求解非线性方程组 F(u) = 0（如MPP optimality conditions）。
   Jacobian 通过有限差分近似。

核心公式 (Hasofer-Lind 迭代):
  u^{(k+1)} = [grad_g(u^k) . u^k - g(u^k)] / ||grad_g(u^k)||^2 * grad_g(u^k)

核心公式 (Breitung SORM):
  P_f ≈ Phi(-beta) * prod_{i=1}^{n-1} (1 + beta * kappa_i)^{-1/2}

种子项目映射:
  802_newton_rc → Newton逆向通信求解器
  468_geometry   → 极限状态面几何（曲率计算）
  855_pdflib     → 正态CDF/PPF
"""

import numpy as np
from random_variables import std_normal_pdf, std_normal_cdf, std_normal_ppf


class NewtonRaphsonRC:
    """Newton-Raphson 逆向通信求解器

    种子项目 802_newton_rc 的 Python 实现。
    通过逆向通信模式实现：调用者负责计算函数值 F(x)，
    求解器负责 Jacobian 近似和步长计算。

    用法:
      ido, x = 0, x0
      fx = F(x)
      while ido != 0:
          ido, x = solver.step(ido, x, fx)
          if ido > 0:
              fx = F(x)

    数学原理:
      x_{k+1} = x_k - J^{-1}(x_k) * F(x_k)
      J ≈ (F(x + e_j*delta) - F(x)) / delta   (有限差分)
    """

    def __init__(self, n, tol=None, max_iter=50):
        self.n = n
        self.tol = tol or np.sqrt(np.sqrt(np.finfo(float).eps))
        self.max_iter = max_iter
        self._reset()

    def _reset(self):
        self._delxj = 0.0
        self._epsilon = np.sqrt(np.sqrt(np.finfo(float).eps))
        self._fprime = np.zeros((self.n, self.n))
        self._fxcall = 0
        self._fxnrm = 0.0
        self._fxnrm0 = 0.0
        self._fxold = np.zeros(self.n)
        self._j = 0
        self._ncall = 0
        self._xold = np.zeros(self.n)

    def step(self, ido, x, fx):
        """执行一步逆向通信

        Parameters
        ----------
        ido : int
            0 = 新问题; 1 = 返回 F(x+delX_j); 2 = 返回 F(x)
        x : np.ndarray (n,)
        fx : np.ndarray (n,)

        Returns
        -------
        ido_new : int  (0=收敛, 1=需要F(x+dx), 2=需要F(x), -1=失败)
        x_new : np.ndarray (n,)
        """
        x = np.asarray(x, dtype=float).ravel()
        fx = np.asarray(fx, dtype=float).ravel()
        n = self.n

        if ido == 0:
            self._reset()
            self._fxnrm = np.linalg.norm(fx)
            self._fxnrm0 = self._fxnrm
            self._fxold = fx.copy()
            self._xold = x.copy()

            if self._fxnrm <= self.tol * (self._fxnrm0 + 1.0):
                return 0, x.copy()

            ido_new = 1
            self._j = 0
            self._delxj = self._epsilon * (abs(x[0]) + 1.0)
            self._xold = x[0]
            x_new = x.copy()
            x_new[0] += self._delxj
            return ido_new, x_new

        elif ido == 1:
            self._ncall += 1
            if self._ncall > 1000:
                return -1, x.copy()

            j = self._j
            self._fprime[:, j] = (fx - self._fxold) / self._delxj
            x_new = x.copy()
            x_new[j] = self._xold

            if j < n - 1:
                self._j = j + 1
                self._delxj = self._epsilon * (abs(x_new[self._j]) + 1.0)
                self._xold = x_new[self._j]
                x_new[self._j] += self._delxj
                return 1, x_new
            else:
                rcond = 1.0 / (np.linalg.cond(self._fprime) + 1e-30)
                if rcond < 1e-15:
                    pass  # Jacobian 病态，继续但警告
                try:
                    delx = np.linalg.solve(self._fprime, fx)
                except np.linalg.LinAlgError:
                    delx = np.linalg.lstsq(self._fprime, fx, rcond=None)[0]
                x_new[:n] -= delx[:n]
                return 2, x_new

        elif ido == 2:
            self._ncall += 1
            self._fxcall += 1
            if self._ncall > 1000:
                return -1, x.copy()

            fxnrm = np.linalg.norm(fx)
            if fxnrm <= self.tol * (self._fxnrm0 + 1.0):
                return 0, x.copy()

            if self._fxcall > 15 and fxnrm > 0.95 * self._fxnrm:
                return -1, x.copy()

            self._fxnrm = fxnrm
            ido_new = 1
            self._fxold = fx.copy()
            self._j = 0
            self._delxj = self._epsilon * (abs(x[0]) + 1.0)
            self._xold = x[0]
            x_new = x.copy()
            x_new[0] += self._delxj
            return ido_new, x_new

        else:
            return -1, x.copy()


def hasofer_lind_form(lsf, n_dim, u0=None, tol=1e-6, max_iter=100):
    """Hasofer-Lind FORM 算法

    在标准正态空间中寻找最可几点(MPP):
      min |u|  s.t.  g(u) = 0

    算法 (Rackwitz-Fiessler 迭代):
      alpha_k = -grad_g(u_k) / ||grad_g(u_k)||
      u_{k+1} = (alpha_k . u_k - g(u_k)/||grad_g||) * alpha_k
      等价: u_{k+1} = [grad^T u - g] / ||grad||^2 * grad

    Parameters
    ----------
    lsf : LimitStateFunction 对象
    n_dim : int
    u0 : np.ndarray, 初始点 (默认原点)
    tol : float, 收敛容差
    max_iter : int

    Returns
    -------
    beta : float, 可靠度指标
    u_star : np.ndarray, 设计验算点 (MPP)
    alpha : np.ndarray, 方向余弦
    converged : bool
    history : dict, 迭代历史
    """
    if u0 is None:
        u = np.zeros(n_dim)
        # 若原点处梯度为零, 使用小偏移
        grad0 = lsf.gradient(u)
        if np.linalg.norm(grad0) < 1e-14:
            u = np.ones(n_dim) * 0.1
    else:
        u = np.asarray(u0, dtype=float).copy()

    history = {'u': [], 'beta': [], 'g': []}

    converged = False
    for iteration in range(max_iter):
        g_val = float(np.atleast_1d(lsf.evaluate(u.reshape(1, -1)))[0])
        grad = lsf.gradient(u)
        grad_norm = np.linalg.norm(grad)

        if grad_norm < 1e-14:
            break

        beta_k = np.linalg.norm(u)
        history['u'].append(u.copy())
        history['beta'].append(beta_k)
        history['g'].append(g_val)

        # 方向余弦 (负梯度方向)
        alpha = -grad / grad_norm

        # Hasofer-Lind 更新公式
        # 在新切平面 u_{k+1} = [grad^T u_k - g(u_k)] / ||grad||^2 * grad
        numerator = np.dot(grad, u) - g_val
        denominator = grad_norm ** 2

        if abs(denominator) < 1e-30:
            break

        u_new = (numerator / denominator) * grad

        # 步长阻尼: 若新点 g 值偏离过大, 缩减步长
        g_new = float(np.atleast_1d(lsf.evaluate(u_new.reshape(1, -1)))[0])
        if iteration > 2 and abs(g_new) > 10.0 * max(abs(g_val), 1.0):
            for t in [0.75, 0.5, 0.25, 0.1]:
                u_try = u + t * (u_new - u)
                g_try = float(np.atleast_1d(lsf.evaluate(u_try.reshape(1, -1)))[0])
                if abs(g_try) < abs(g_val):
                    u_new = u_try
                    break

        # 检查收敛
        u_change = np.linalg.norm(u_new - u)
        u_norm = max(np.linalg.norm(u_new), 1e-15)

        u = u_new

        if u_change / u_norm < tol and abs(g_val) < max(tol * 100, 1e-4):
            converged = True
            break

    beta = np.linalg.norm(u)
    g_final = float(np.atleast_1d(lsf.evaluate(u.reshape(1, -1)))[0])
    grad_final = lsf.gradient(u)
    grad_norm_final = np.linalg.norm(grad_final)
    alpha_final = -grad_final / (grad_norm_final + 1e-30)

    history['u'].append(u.copy())
    history['beta'].append(beta)
    history['g'].append(g_final)

    return beta, u, alpha_final, converged, history


def sorm_breitung(lsf, u_star, beta):
    """SORM Breitung 修正

    在设计验算点 u* 处计算极限状态面的主曲率 kappa_i,
    然后应用 Breitung 渐近公式:

    P_f^{SORM} = Phi(-beta) * prod_{i=1}^{n-1} (1 + beta * kappa_i)^{-1/2}

    曲率通过旋转Hessian到切平面坐标系获得:
    B = R^T * H_g / ||grad_g|| * R
    其中 R 为将 e_n 映射到 alpha 的旋转矩阵。

    Parameters
    ----------
    lsf : LimitStateFunction
    u_star : np.ndarray, FORM 设计点
    beta : float, FORM 可靠度指标

    Returns
    -------
    pf_sorm : float
    curvatures : np.ndarray, n-1 个主曲率
    correction : float, 修正因子
    """
    n_dim = len(u_star)
    grad = lsf.gradient(u_star)
    grad_norm = np.linalg.norm(grad)
    H = lsf.hessian(u_star)

    if grad_norm < 1e-14:
        return std_normal_cdf(-beta), np.zeros(max(n_dim - 1, 1)), 1.0

    # 单位法向量 (指向失效域)
    alpha = -grad / grad_norm

    # 构造旋转矩阵 R: R 将 e_n 映射到 alpha
    # 使用 Gram-Schmidt 构造正交基
    R = np.eye(n_dim)
    R[:, -1] = alpha
    for i in range(n_dim - 1):
        v = R[:, i].copy()
        for j in range(i):
            v -= np.dot(R[:, j], v) * R[:, j]
        v -= np.dot(alpha, v) * alpha
        norm_v = np.linalg.norm(v)
        if norm_v > 1e-12:
            R[:, i] = v / norm_v
        else:
            # 构造随机向量
            e = np.zeros(n_dim)
            e[i] = 1.0
            e -= np.dot(alpha, e) * alpha
            for j in range(i):
                e -= np.dot(R[:, j], e) * R[:, j]
            norm_e = np.linalg.norm(e)
            if norm_e > 1e-12:
                R[:, i] = e / norm_e

    # 旋转 Hessian 到切平面
    R_tangent = R[:, :n_dim - 1]
    A = R_tangent.T @ H @ R_tangent / grad_norm

    # 主曲率 = A 的特征值
    try:
        eigvals = np.linalg.eigvalsh(A)
    except np.linalg.LinAlgError:
        eigvals = np.zeros(n_dim - 1)

    curvatures = np.real(eigvals)

    # Breitung 修正因子
    correction = 1.0
    for ki in curvatures:
        factor = 1.0 + beta * ki
        if factor > 1e-10:
            correction *= factor ** (-0.5)
        else:
            correction *= 1e5  # 极限情况

    pf_form = std_normal_cdf(-beta)
    pf_sorm = pf_form * correction

    return float(pf_sorm), curvatures, float(correction)


class FORMSolver:
    """FORM 求解器：封装 Hasofer-Lind 算法与结果后处理"""

    def __init__(self, lsf, n_dim, tol=1e-6, max_iter=200):
        self.lsf = lsf
        self.n_dim = n_dim
        self.tol = tol
        self.max_iter = max_iter
        self.beta = None
        self.u_star = None
        self.alpha = None
        self.pf = None
        self.converged = False
        self.history = None

    def solve(self, u0=None):
        """执行 FORM 分析"""
        self.beta, self.u_star, self.alpha, self.converged, self.history = \
            hasofer_lind_form(self.lsf, self.n_dim, u0, self.tol, self.max_iter)
        self.pf = std_normal_cdf(-self.beta)
        return self

    def importance_factors(self):
        """方向余弦的平方 = 各变量的重要性因子

        alpha_i^2 表示第 i 个变量对可靠度指标的贡献比例
        sum(alpha_i^2) = 1
        """
        if self.alpha is None:
            return None
        return self.alpha ** 2

    def sensitivity_to_mean(self, dmean=None):
        """可靠度对均值的灵敏度

        dbeta/dmu_i ≈ -alpha_i / sigma_i (对正态变量)
        这里简化为 dbeta/du_i = -alpha_i
        """
        if self.alpha is None:
            return None
        return -self.alpha
