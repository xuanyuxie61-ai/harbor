"""
spatial_operators.py
空间微分算子离散化：有限差分、DG 风格通量、有限元质量/刚度矩阵

融合种子项目:
  - 434_fisher_pde_ftcs: FTCS (Forward Time Central Space) 差分模板
  - 272_dg1d_burgers: 间断 Galerkin 数值通量与限制器
  - 404_fem2d_heat_rectangle: 有限元质量与刚度矩阵组装
  - 065_ball_and_stick_display: Lax-Wendroff 双曲型格式思想

科学背景:
  考虑一维对流-扩散-反应算子:
      L[u] = epsilon * d^2u/dx^2 - v * du/dx + r * u * (1 - u)
  扩散项采用中心差分 (二阶精度):
      D2[u]_i = (u_{i+1} - 2 u_i + u_{i-1}) / dx^2
  对流项根据局部 Peclet 数 Pe = |v| dx / epsilon 选择离散格式:
      - Pe < 2: 中心差分 (du/dx ~ (u_{i+1} - u_{i-1})/(2dx))
      - Pe >= 2: 迎风差分 或 Lax-Wendroff 格式
  Lax-Wendroff 格式 (双曲型):
      u_i^{n+1} = u_i^n - c/2 (u_{i+1}^n - u_{i-1}^n)
                  + c^2/2 (u_{i+1}^n - 2 u_i^n + u_{i-1}^n)
      其中 c = v dt / dx 为 Courant 数。
  反应项为局部非线性: f(u) = r u (1 - u/K)

  有限元视角 (Galerkin 投影):
      M_{ij} = int phi_i phi_j dx   (质量矩阵)
      K_{ij} = int epsilon phi_i' phi_j' dx + int v phi_i phi_j' dx   (刚度矩阵)
  对于分段线性基函数 (hat functions) 在均匀网格上，M 为三对角:
      M = dx/6 * tridiag(1, 4, 1)
  采用 lumped mass 近似: M ~ dx * I，可将隐式系统简化为 band 系统。
"""

import numpy as np
from typing import Tuple, Optional


class SpatialDiscretization1D:
    """
    一维空间算子离散化容器。
    """

    def __init__(self,
                 x: np.ndarray,
                 epsilon: float = 0.01,
                 velocity: float = 0.5,
                 reaction_rate: float = 1.0,
                 carrying_capacity: float = 1.0):
        if x.ndim != 1 or len(x) < 3:
            raise ValueError("x must be 1D with length >= 3")
        if np.any(np.diff(x) <= 0):
            raise ValueError("x must be strictly increasing")
        self.x = x.copy()
        self.nx = len(x)
        self.dx = np.diff(x)
        self.epsilon = epsilon
        self.v = velocity
        self.r = reaction_rate
        self.K = carrying_capacity

        # 局部 Peclet 数（基于相邻网格间距）
        self.pe_local = self._compute_local_peclet()

    def _compute_local_peclet(self) -> np.ndarray:
        """
        局部 Peclet 数: Pe_i = |v| * h_i / epsilon
        """
        h = np.zeros(self.nx, dtype=np.float64)
        h[0] = self.dx[0]
        h[-1] = self.dx[-1]
        h[1:-1] = 0.5 * (self.dx[:-1] + self.dx[1:])
        pe = np.abs(self.v) * h / self.epsilon
        return pe

    def diffusion_operator(self, u: np.ndarray) -> np.ndarray:
        """
        二阶中心差分扩散算子，考虑非均匀网格。
        使用调和平均处理变网格:
            (d2u/dx2)_i = 2/(h_{i-1}+h_i) * [ (u_{i+1}-u_i)/h_i - (u_i-u_{i-1})/h_{i-1} ]
        """
        if len(u) != self.nx:
            raise ValueError("u length mismatch")
        d2u = np.zeros(self.nx, dtype=np.float64)
        for i in range(1, self.nx - 1):
            hp = self.x[i + 1] - self.x[i]
            hm = self.x[i] - self.x[i - 1]
            if hp <= 0 or hm <= 0:
                raise ValueError("Non-positive mesh spacing detected")
            d2u[i] = 2.0 / (hp + hm) * ((u[i + 1] - u[i]) / hp - (u[i] - u[i - 1]) / hm)
        # Neumann 边界：二阶外推
        d2u[0] = d2u[1]
        d2u[-1] = d2u[-2]
        return self.epsilon * d2u

    def advection_operator(self, u: np.ndarray, scheme: str = "auto") -> np.ndarray:
        """
        对流算子离散，根据局部 Peclet 数自适应选择格式。
        scheme: "centered", "upwind", "lax_wendroff", "auto"
        """
        if len(u) != self.nx:
            raise ValueError("u length mismatch")
        adv = np.zeros(self.nx, dtype=np.float64)

        for i in range(1, self.nx - 1):
            hp = self.x[i + 1] - self.x[i]
            hm = self.x[i] - self.x[i - 1]
            pe = self.pe_local[i]
            sel = scheme
            if sel == "auto":
                sel = "centered" if pe < 2.0 else "lax_wendroff"

            if sel == "centered":
                du = (u[i + 1] - u[i - 1]) / (hp + hm)
            elif sel == "upwind":
                if self.v > 0:
                    du = (u[i] - u[i - 1]) / hm
                else:
                    du = (u[i + 1] - u[i]) / hp
            elif sel == "lax_wendroff":
                # Lax-Wendroff 人工粘性格式
                du_center = (u[i + 1] - u[i - 1]) / (hp + hm)
                # 附加二阶耗散项系数
                nu = np.abs(self.v) * max(hp, hm) * 0.5
                diff_artificial = nu * ((u[i + 1] - u[i]) / hp - (u[i] - u[i - 1]) / hm) / (0.5 * (hp + hm))
                du = du_center - diff_artificial / self.v if self.v != 0 else du_center
            else:
                du = (u[i + 1] - u[i - 1]) / (hp + hm)
            adv[i] = self.v * du

        # 边界外推
        adv[0] = adv[1]
        adv[-1] = adv[-2]
        return adv

    def reaction_operator(self, u: np.ndarray) -> np.ndarray:
        """
        KPP-Fisher 反应项: f(u) = r * u * (1 - u/K)
        """
        u = np.clip(u, 0.0, None)  # 物理非负约束
        return self.r * u * (1.0 - u / self.K)

    def full_rhs_deterministic(self, u: np.ndarray, scheme: str = "auto") -> np.ndarray:
        """
        确定性右端项总和: L[u] + f(u)
        """
        return self.diffusion_operator(u) - self.advection_operator(u, scheme=scheme) + self.reaction_operator(u)

    def assemble_fem_matrices(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        组装 lumped-mass 近似下的质量矩阵 M 和刚度矩阵 K。
        返回 dense 矩阵 (nx, nx)。

        质量矩阵 ( lumped ):
            M_i = 0.5 * (h_{i-1} + h_i)  (内部)
        刚度矩阵 ( Galerkin with upwind stabilization ):
            K_ii = epsilon * (1/h_{i-1} + 1/h_i) + |v|/2
            K_{i,i+1} = -epsilon/h_i - v/2 * (1 + sign(v))
            K_{i,i-1} = -epsilon/h_{i-1} + v/2 * (1 - sign(v))
        """
        n = self.nx
        M = np.zeros(n, dtype=np.float64)
        K = np.zeros((n, n), dtype=np.float64)

        for i in range(n):
            if i == 0:
                h = self.dx[0]
                M[i] = 0.5 * h
                K[i, i] = self.epsilon / h + abs(self.v) / 2.0
                K[i, i + 1] = -self.epsilon / h - self.v / 2.0 * (1.0 + np.sign(self.v))
            elif i == n - 1:
                h = self.dx[-1]
                M[i] = 0.5 * h
                K[i, i] = self.epsilon / h + abs(self.v) / 2.0
                K[i, i - 1] = -self.epsilon / h + self.v / 2.0 * (1.0 - np.sign(self.v))
            else:
                hm = self.dx[i - 1]
                hp = self.dx[i]
                M[i] = 0.5 * (hm + hp)
                K[i, i] = self.epsilon * (1.0 / hm + 1.0 / hp) + abs(self.v) / 2.0
                K[i, i + 1] = -self.epsilon / hp - self.v / 2.0 * (1.0 + np.sign(self.v))
                K[i, i - 1] = -self.epsilon / hm + self.v / 2.0 * (1.0 - np.sign(self.v))

        return M, K

    def dg_numerical_flux(self, u_left: float, u_right: float) -> float:
        """
        1D 间断 Galerkin 风格的局部 Lax-Friedrichs 数值通量。
        用于 burgers 型非线性对流:
            f(u) = v * u - epsilon * du/dx
        LLF 通量:
            f_hat = 0.5 * (f_L + f_R) - 0.5 * alpha * (u_R - u_L)
        其中 alpha = max(|f'(u)|) 为局部最大波速。
        """
        f_L = self.v * u_left
        f_R = self.v * u_right
        alpha = abs(self.v)
        flux = 0.5 * (f_L + f_R) - 0.5 * alpha * (u_right - u_left)
        return flux
