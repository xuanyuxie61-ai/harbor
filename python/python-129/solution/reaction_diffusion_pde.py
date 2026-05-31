"""
reaction_diffusion_pde.py
基于 1170_stochastic_heat2d 的二维有限差分框架与 011_annulus_rule 的
环形区域数值积分思想，构建血管横截面上凝血因子的反应-扩散-对流模型。

物理模型：
    在柱坐标 (r, θ) 下，凝血因子浓度 c(r,θ,t) 满足反应-扩散方程：
        ∂c/∂t = D * (1/r) ∂/∂r(r ∂c/∂r) + (D/r²) ∂²c/∂θ² + R(c) + S(r,θ,t)

    其中 D 为有效扩散张量，R(c) 为级联反应源项（由 coagulation_ode 提供），
    S 为血管壁损伤处的释放源项。

    计算区域为血管横截面的环形区域：
        r ∈ [r_i, r_o],   θ ∈ [0, 2π)
    r_i 为血管中心，r_o 为血管壁半径。

数值离散：
    采用五点差分格式，对 r 方向使用变步长以适应柱坐标奇异性：
        (1/r_i) (r_{i+1/2} (c_{i+1}-c_i)/Δr - r_{i-1/2} (c_i-c_{i-1})/Δr) / Δr
"""

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve


class AnnularDiffusionSolver:
    """
    环形区域上的反应-扩散方程求解器。
    融合 011_annulus_rule 的极坐标数值积分与 1170_stochastic_heat2d 的
    有限差分稀疏矩阵技术。
    """

    def __init__(self, r_inner, r_outer, nr, ntheta, D_coef, source_func=None):
        """
        参数:
            r_inner    : float, 内半径 (μm)
            r_outer    : float, 外半径 (μm)
            nr         : int, 径向网格数
            ntheta     : int, 角向网格数
            D_coef     : float or ndarray, 扩散系数 (μm²/s)
            source_func: callable, 源项函数 S(r, θ, t, c)
        """
        if r_inner <= 0 or r_outer <= r_inner:
            raise ValueError("必须满足 0 < r_inner < r_outer")
        if nr < 3 or ntheta < 3:
            raise ValueError("nr 和 ntheta 必须 >= 3")

        self.r_inner = r_inner
        self.r_outer = r_outer
        self.nr = nr
        self.ntheta = ntheta
        self.D = float(D_coef)
        self.source_func = source_func

        # 构造网格
        self.r = np.linspace(r_inner, r_outer, nr)
        self.theta = np.linspace(0, 2 * np.pi, ntheta, endpoint=False)
        self.dr = self.r[1] - self.r[0]
        self.dtheta = self.theta[1] - self.theta[0]

        # 预构造 Laplacian 稀疏矩阵
        self.L = self._build_laplacian_matrix()

    def _build_laplacian_matrix(self):
        """
        构造环形区域 Laplacian 的稀疏矩阵表示。
        使用五点格式，包含径向和角向贡献。
        """
        nr = self.nr
        nt = self.ntheta
        n_total = nr * nt

        row_idx = []
        col_idx = []
        data = []

        def add_entry(i, j, val):
            row_idx.append(i)
            col_idx.append(j)
            data.append(val)

        for ir in range(nr):
            r_i = self.r[ir]
            r_half_plus = r_i + 0.5 * self.dr if ir < nr - 1 else r_i
            r_half_minus = r_i - 0.5 * self.dr if ir > 0 else r_i

            # 径向系数 (考虑 r 因子)
            if ir == 0:
                alpha_r = 0.0
                beta_r = (r_half_plus / r_i) / (self.dr ** 2)
                gamma_r = -beta_r
            elif ir == nr - 1:
                alpha_r = (r_half_minus / r_i) / (self.dr ** 2)
                beta_r = 0.0
                gamma_r = -alpha_r
            else:
                alpha_r = (r_half_minus / r_i) / (self.dr ** 2)
                beta_r = (r_half_plus / r_i) / (self.dr ** 2)
                gamma_r = -(alpha_r + beta_r)

            # 角向系数
            dtheta2 = self.dtheta ** 2
            alpha_theta = 1.0 / (r_i ** 2 * dtheta2)
            beta_theta = 1.0 / (r_i ** 2 * dtheta2)
            gamma_theta = -(alpha_theta + beta_theta)

            for it in range(nt):
                idx = ir * nt + it
                itp = (it + 1) % nt
                itm = (it - 1) % nt

                # 中心
                add_entry(idx, idx, self.D * (gamma_r + gamma_theta))
                # 径向邻居
                if ir > 0:
                    add_entry(idx, (ir - 1) * nt + it, self.D * alpha_r)
                if ir < nr - 1:
                    add_entry(idx, (ir + 1) * nt + it, self.D * beta_r)
                # 角向邻居
                add_entry(idx, ir * nt + itp, self.D * beta_theta)
                add_entry(idx, ir * nt + itm, self.D * alpha_theta)

        L = csr_matrix((data, (row_idx, col_idx)), shape=(n_total, n_total))
        return L

    def solve_steady_state(self, reaction_term_func, c_guess=None, max_iter=100, tol=1e-10):
        """
        求解稳态反应-扩散方程：
            D ∇²c + R(c) + S = 0
        使用带阻尼的 Picard 迭代。
        """
        n_total = self.nr * self.ntheta
        if c_guess is None:
            c = np.ones(n_total) * 1e-6
        else:
            c = np.asarray(c_guess, dtype=float).flatten()

        omega_relax = 0.3
        for it in range(max_iter):
            rhs = np.zeros(n_total)
            if self.source_func is not None:
                for i in range(n_total):
                    ir = i // self.ntheta
                    it = i % self.ntheta
                    rhs[i] = self.source_func(self.r[ir], self.theta[it], 0.0, c[i])

            R_vec = reaction_term_func(c)
            rhs = rhs + R_vec

            try:
                c_new = spsolve(self.L, -rhs)
            except Exception:
                c_new = np.linalg.lstsq(self.L.toarray(), -rhs, rcond=None)[0]

            c_new = np.clip(c_new, 0.0, 1e6)
            c_new = omega_relax * c_new + (1.0 - omega_relax) * c

            diff = np.linalg.norm(c_new - c, ord=np.inf)
            c = c_new
            if diff < tol:
                break
        return c.reshape((self.nr, self.ntheta))

    def integrate_over_annulus(self, field):
        """
        基于 011_annulus_rule 的数值积分思想，
        对定义在环形网格上的场进行面积积分。

        积分公式：
            ∫_Ω f(r,θ) r dr dθ ≈ Σ_i Σ_j f_{ij} r_i Δr Δθ
        """
        field = np.asarray(field)
        if field.shape != (self.nr, self.ntheta):
            raise ValueError("field 形状必须匹配网格 (nr, ntheta)")

        # 使用梯形法则的权重
        r_weights = np.zeros(self.nr)
        r_weights[0] = 0.5 * (self.r[0] + self.r[1]) * self.dr
        r_weights[-1] = 0.5 * (self.r[-2] + self.r[-1]) * self.dr
        for i in range(1, self.nr - 1):
            r_weights[i] = self.r[i] * self.dr

        theta_weight = self.dtheta
        total = 0.0
        for i in range(self.nr):
            for j in range(self.ntheta):
                total += field[i, j] * r_weights[i] * theta_weight
        return total


def wound_source_term(r, theta, t, c, r_wound=45.0, theta_wound=0.0,
                      strength=50.0, sigma_r=3.0, sigma_theta=0.5):
    """
    血管壁损伤处的 TF 释放源项，采用高斯分布：
        S(r,θ) = S0 * exp( -(r-r_wound)²/(2σ_r²) - (θ-θ_wound)²/(2σ_θ²) )
    """
    dr = r - r_wound
    dtheta = theta - theta_wound
    # 角向周期性处理
    dtheta = ((dtheta + np.pi) % (2 * np.pi)) - np.pi
    val = strength * np.exp(-0.5 * (dr ** 2 / sigma_r ** 2 + dtheta ** 2 / sigma_theta ** 2))
    return val


def demo_pde_solver():
    """
    演示：求解血管壁损伤处的凝血酶空间分布。
    """
    solver = AnnularDiffusionSolver(
        r_inner=1.0, r_outer=50.0, nr=40, ntheta=32,
        D_coef=8.0e-3,
        source_func=lambda r, th, t, c: wound_source_term(r, th, t, c)
    )

    def reaction(c):
        # 简化的 Michaelis-Menten 汇项
        return -0.01 * c / (c + 10.0 + 1e-12)

    c_steady = solver.solve_steady_state(reaction)
    total = solver.integrate_over_annulus(c_steady)
    print(f"稳态总量: {total:.4f}")
    print(f"最大浓度: {c_steady.max():.4f} at r={solver.r[np.unravel_index(c_steady.argmax(), c_steady.shape)[0]]:.2f}")
    return solver, c_steady


if __name__ == "__main__":
    demo_pde_solver()
