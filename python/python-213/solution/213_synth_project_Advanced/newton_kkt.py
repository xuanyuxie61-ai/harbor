"""
newton_kkt.py - KKT 系统 Newton 求解器
=======================================

融合种子项目:
  - 404_fem2d_heat_rectangle: 带状矩阵 LU 分解 (DGB_FA/DGB_SL)
  - 963_r83_np: 三对角矩阵行列式计算
  - 104_boundary_locus: 稳定域分析

核心数学: Newton 步与 KKT 系统

Interior Point Method 的每一步需要求解 KKT 系统:

  [ H + Theta    A^T   -I    I  ] [ dx ]   [ r_d ]
  [ A            0      0    0  ] [ dy ] = [ r_p ]
  [ S            0      X    0  ] [ dz_l]  [ r_cl ]
  [ -Z_u         0      0    X_u] [ dz_u]  [ r_cu ]

其中:
  H = nabla^2 f(x) + nabla^2_x L   (Lagrangian Hessian)
  Theta = Z_l X_l^{-1} + Z_u X_u^{-1}  (障碍 Hessian 贡献)
  A = 等式约束 Jacobian
  X_l = diag(x - l),  Z_l = diag(z_l)  (下界)
  X_u = diag(u - x),  Z_u = diag(z_u)  (上界)

通过消除 dz_l 和 dz_u, 系统简化为:

  [ H + Theta_bar   A^T ] [ dx ]   [ r_bar ]
  [ A               0   ] [ dy ] = [ r_p   ]

其中:
  Theta_bar = Theta + Z_l X_l^{-1} + Z_u X_u^{-1}
  r_bar = r_d + X_l^{-1} r_cl - X_u^{-1} r_cu

这是一个对称不定系统 (saddle point), 需要特殊处理.

稳定域分析 (融合 104_boundary_locus):
  对 Newton 步进行线搜索时, 需要确保步长在稳定域内.
  类比 ODE 方法 y' = lambda y 的绝对稳定域:
    R(z) = amplification factor
    稳定域 = { z in C : |R(z)| <= 1 }

  对内点法, 稳定域对应于使 x + alpha dx > 0 的 alpha 范围.
"""

import numpy as np
from scipy import sparse
from scipy.sparse import linalg as splinalg
from typing import Tuple, Dict, Optional
from dataclasses import dataclass


@dataclass
class KKTResult:
    """KKT 系统求解结果."""
    dx: np.ndarray          # 原始变量步
    dy: np.ndarray          # 等式对偶步
    dz_lower: np.ndarray    # 下界对偶步
    dz_upper: np.ndarray    # 上界对偶步
    converged: bool         # 是否收敛
    residual_norm: float    # 残差范数
    iterations: int         # 迭代次数


class KKTSystem:
    """
    KKT 系统的构建与求解.

    使用增广系统方法 (reduced KKT system):
      (H + Theta) dx + A^T dy = r_bar
      A dx = r_p

    通过 Schur 补消去 dx:
      A (H + Theta)^{-1} A^T dy = A (H + Theta)^{-1} r_bar - r_p

    若 H + Theta 是正定的 (内点法中通常如此),
    则 Schur 补系统也是正定的, 可用 CG 求解.

    Parameters
    ----------
    n : int
        原始变量维度
    m : int
        等式约束个数
    """

    def __init__(self, n: int, m: int):
        self.n = n
        self.m = m
        self._factorization = None

    def build_reduced_system(self, H: np.ndarray, A: sparse.spmatrix,
                              theta: np.ndarray,
                              r_d: np.ndarray, r_p: np.ndarray,
                              r_cl: np.ndarray, r_cu: np.ndarray,
                              x_slack_l: np.ndarray, x_slack_u: np.ndarray,
                              z_lower: np.ndarray, z_upper: np.ndarray,
                              bounded_mask: np.ndarray
                              ) -> Tuple[np.ndarray, np.ndarray]:
        """
        构建约化 KKT 系统.

        消除 dz_l 和 dz_u 后的系统:
          (H + Theta) dx + A^T dy = r_bar
          A dx = r_p

        其中:
          Theta_i = z_{l,i} / (x_i - l_i) + z_{u,i} / (u_i - x_i)  (对有界分量)
          r_bar = r_d + Z_l (X-L)^{-1} r_cl - Z_u (U-X)^{-1} r_cu

        Parameters
        ----------
        H : ndarray
            Lagrangian Hessian
        A : sparse matrix
            等式约束矩阵
        theta : ndarray
            障碍 Hessian 对角贡献
        r_d, r_p : ndarray
            对偶和原始残差
        r_cl, r_cu : ndarray
            互补残差
        x_slack_l, x_slack_u : ndarray
            原始松弛 (x - l, u - x)
        z_lower, z_upper : ndarray
            对偶变量
        bounded_mask : ndarray of bool
            有界变量标记

        Returns
        -------
        H_bar : ndarray
            约化 Hessian H + Theta
        r_bar : ndarray
            约化右端
        """
        n = self.n

        # 构建 Theta 对角矩阵
        theta_diag = np.zeros(n)
        theta_diag += theta

        if np.any(bounded_mask):
            # 下界贡献: z_l / (x - l)
            x_l_safe = np.maximum(x_slack_l[bounded_mask], 1.0e-30)
            theta_diag[bounded_mask] += z_lower[bounded_mask] / x_l_safe

            # 上界贡献: z_u / (u - x)
            x_u_safe = np.maximum(x_slack_u[bounded_mask], 1.0e-30)
            theta_diag[bounded_mask] += z_upper[bounded_mask] / x_u_safe

        # H_bar = H + diag(theta)
        H_bar = H + np.diag(theta_diag)

        # r_bar = r_d + Z_l (X-L)^{-1} r_cl - Z_u (U-X)^{-1} r_cu
        r_bar = r_d.copy()
        if np.any(bounded_mask):
            x_l_safe = np.maximum(x_slack_l[bounded_mask], 1.0e-30)
            x_u_safe = np.maximum(x_slack_u[bounded_mask], 1.0e-30)
            r_bar[bounded_mask] += (
                z_lower[bounded_mask] * r_cl[bounded_mask] / x_l_safe -
                z_upper[bounded_mask] * r_cu[bounded_mask] / x_u_safe
            )

        return H_bar, r_bar

    def solve(self, H_bar: np.ndarray, A: sparse.spmatrix,
              r_bar: np.ndarray, r_p: np.ndarray,
              regularization: float = 1.0e-8) -> KKTResult:
        """
        求解约化 KKT 系统 (直接法).

        使用增广矩阵直接求解:
          [ H_bar   A^T ] [ dx ]   [ r_bar ]
          [ A       0   ] [ dy ] = [ r_p   ]

        对小/中规模问题, 直接 LU 分解最稳定.
        正则化确保鞍点系统非奇异:
          [ H_bar + eps*I    A^T ] [ dx ]   [ r_bar ]
          [ A            -eps*I ] [ dy ] = [ r_p   ]

        Parameters
        ----------
        H_bar : ndarray
            约化 Hessian
        A : sparse matrix
            等式约束矩阵
        r_bar, r_p : ndarray
            右端向量
        regularization : float
            正则化参数

        Returns
        -------
        KKTResult
            求解结果 (dx, dy)
        """
        n = self.n
        m = self.m
        total = n + m

        # 构建增广系统
        aug = np.zeros((total, total))
        aug[:n, :n] = H_bar + regularization * np.eye(n)
        A_dense = A.toarray() if sparse.issparse(A) else A
        aug[:n, n:] = A_dense.T
        aug[n:, :n] = A_dense
        aug[n:, n:] = -regularization * np.eye(m)

        rhs = np.concatenate([r_bar, r_p])

        try:
            sol = np.linalg.solve(aug, rhs)
            dx = sol[:n]
            dy = sol[n:]
            converged = True
            res_norm = float(np.linalg.norm(aug @ sol - rhs))
        except np.linalg.LinAlgError:
            dx = np.zeros(n)
            dy = np.zeros(m)
            converged = False
            res_norm = np.inf

        return KKTResult(
            dx=dx, dy=dy,
            dz_lower=np.zeros(n), dz_upper=np.zeros(n),
            converged=converged,
            residual_norm=res_norm,
            iterations=1
        )

    def recover_dual_steps(self, dx: np.ndarray,
                            x_slack_l: np.ndarray, x_slack_u: np.ndarray,
                            z_lower: np.ndarray, z_upper: np.ndarray,
                            r_cl: np.ndarray, r_cu: np.ndarray,
                            bounded_mask: np.ndarray
                            ) -> Tuple[np.ndarray, np.ndarray]:
        """
        从 dx 恢复对偶步 dz_l, dz_u.

        从互补条件:
          S dx_slack + X dz = -r_c
          => dz = -X^{-1} (r_c + S dx_slack)

        下界: dz_l = -(r_cl + Z_l * dx) / X_l
        上界: dz_u = -(r_cu - Z_u * dx) / X_u

        Parameters
        ----------
        dx : ndarray
            原始步
        x_slack_l, x_slack_u : ndarray
            原始松弛
        z_lower, z_upper : ndarray
            当前对偶变量
        r_cl, r_cu : ndarray
            互补残差
        bounded_mask : ndarray
            有界标记

        Returns
        -------
        dz_lower, dz_upper : ndarray
            对偶步
        """
        n = self.n
        dz_lower = np.zeros(n)
        dz_upper = np.zeros(n)

        if np.any(bounded_mask):
            x_l_safe = np.maximum(x_slack_l[bounded_mask], 1.0e-30)
            x_u_safe = np.maximum(x_slack_u[bounded_mask], 1.0e-30)

            dz_lower[bounded_mask] = -(
                r_cl[bounded_mask] + z_lower[bounded_mask] * dx[bounded_mask]
            ) / x_l_safe

            dz_upper[bounded_mask] = -(
                r_cu[bounded_mask] - z_upper[bounded_mask] * dx[bounded_mask]
            ) / x_u_safe

        return dz_lower, dz_upper

    def stability_check(self, dx: np.ndarray, ds: np.ndarray,
                         x: np.ndarray, s: np.ndarray,
                         alpha: float) -> bool:
        """
        检查 Newton 步的稳定性 (融合 104_boundary_locus).

        类比 ODE 方法的绝对稳定域分析:
          对试验方程 x' = lambda * x,
          数值方法给出 x_{n+1} = R(z) * x_n,
          z = dt * lambda.

          绝对稳定域 = { z in C : |R(z)| <= 1 }

        对内点法:
          步长 alpha 必须使 (x + alpha*dx, s + alpha*ds) > 0
          "稳定域" = { alpha : x + alpha*dx > 0, s + alpha*ds > 0 }

        Parameters
        ----------
        dx, ds : ndarray
            原始和对偶步
        x, s : ndarray
            当前点
        alpha : float
            步长

        Returns
        -------
        bool
            是否稳定 (保持正性)
        """
        x_new = x + alpha * dx
        s_new = s + alpha * ds
        return bool(np.all(x_new > 0) and np.all(s_new > 0))


def compute_lagrangian_hessian(Q: Optional[sparse.spmatrix],
                                 x_slack_l: np.ndarray,
                                 x_slack_u: np.ndarray,
                                 z_lower: np.ndarray,
                                 z_upper: np.ndarray,
                                 bounded_mask: np.ndarray) -> np.ndarray:
    """
    计算 Lagrangian Hessian.

    对二次规划:
      nabla^2 L = Q + Theta

    其中 Theta 来自障碍函数的 Hessian:
      Theta_ii = z_{l,i}/(x_i-l_i) + z_{u,i}/(u_i-x_i)

    对非线性问题:
      nabla^2 L = nabla^2 f(x) + sum y_i nabla^2 h_i(x) + Theta

    在反应-扩散最优控制中:
      nabla^2 f = w_track * I + w_control * gamma * I
      nabla^2 h = FEM 刚度矩阵

    Parameters
    ----------
    Q : sparse matrix or None
        目标 Hessian
    x_slack_l, x_slack_u : ndarray
        原始松弛
    z_lower, z_upper : ndarray
        对偶变量
    bounded_mask : ndarray
        有界标记

    Returns
    -------
    ndarray
        Lagrangian Hessian 的密集形式
    """
    n = len(x_slack_l)

    if Q is not None:
        H = Q.toarray() if sparse.issparse(Q) else Q.copy()
    else:
        H = np.zeros((n, n))

    # 加障碍 Hessian 贡献
    theta = np.zeros(n)
    if np.any(bounded_mask):
        x_l_safe = np.maximum(x_slack_l[bounded_mask], 1.0e-30)
        x_u_safe = np.maximum(x_slack_u[bounded_mask], 1.0e-30)
        theta[bounded_mask] = (z_lower[bounded_mask] / x_l_safe +
                                z_upper[bounded_mask] / x_u_safe)

    H += np.diag(theta)
    return H
