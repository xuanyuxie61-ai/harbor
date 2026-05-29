"""
fem_uncertainty_field.py
基于有限元方法的定位不确定性场分析

核心数学模型：
1. 不确定性场 PDE：
   将机器人定位不确定性 u(x,y) 建模为椭圆型偏微分方程的解：
   
   -∇ · (a(x,y) ∇u) + c(x,y) u = f(x,y),  (x,y) ∈ Ω
   u = 0,                                  (x,y) ∈ ∂Ω
   
   其中：
   - a(x,y) 为扩散系数（表示不确定性传播速率）
   - c(x,y) 为反应系数（表示观测信息对不确定性的抑制）
   - f(x,y) 为源项（表示运动噪声引入的不确定性）

2. 弱形式与 Galerkin 离散：
   求 u_h ∈ V_h 使得 ∀ v_h ∈ V_h：
   ∫_Ω a ∇u_h · ∇v_h dx + ∫_Ω c u_h v_h dx = ∫_Ω f v_h dx
   
   其中 V_h 为分片线性有限元空间

3. 线性三角形（T3）基函数：
   对于三角形 T = (p1, p2, p3)，面积 = 0.5 * |det([1,x1,y1; 1,x2,y2; 1,x3,y3])|
   
   基函数：φ_i(p_j) = δ_ij
   
   ∇φ_1 = [(y2-y3), (x3-x2)] / (2A)
   ∇φ_2 = [(y3-y1), (x1-x3)] / (2A)
   ∇φ_3 = [(y1-y2), (x2-x1)] / (2A)

4. 单元刚度矩阵：
   K_{ij}^{(e)} = ∫_{T_e} (a ∇φ_i · ∇φ_j + c φ_i φ_j) dx dy
   
   对于常数 a, c：
   K^{(e)} = a * A * [∇φ_i · ∇φ_j]_{3×3} + c * (A/12) * [[2,1,1],[1,2,1],[1,1,2]]

5. 环形区域数值积分（融合 annulus_rule 项目）：
   用于在机器人周围环形区域估计协方差传播：
   
   ∫_{环形} f(x,y) dx dy ≈ Σ_{i=1}^{Nr} Σ_{j=1}^{Nt} w_{ij} f(x_{ij}, y_{ij})
   
   其中：
   - 径向采用 Legendre-Gauss 求积
   - 角度方向均匀采样
   - 权重包含雅可比行列式 r dr dθ
   
   坐标变换：
   x = cx + r * cos(θ)
   y = cy + r * sin(θ)
   
   径向节点：r_j = sqrt( (r2^2 - r1^2) * (ξ_j + 1)/2 + r1^2 )
   其中 ξ_j 为 Legendre 节点
"""

import numpy as np


class FEMUncertaintyField:
    """
    有限元不确定性场求解器
    融合 fem2d_bvp_linear 和 fem2d_sample 项目
    """

    def __init__(self, nx=20, ny=20):
        """
        Parameters
        ----------
        nx, ny : int
            规则网格节点数
        """
        self.nx = max(int(nx), 2)
        self.ny = max(int(ny), 2)

    def solve_uncertainty_field(self, domain, a_func, c_func, f_func):
        """
        在矩形区域上求解不确定性场 PDE
        
        Parameters
        ----------
        domain : tuple ((xmin, xmax), (ymin, ymax))
        a_func : callable
            扩散系数 a(x,y)
        c_func : callable
            反应系数 c(x,y)
        f_func : callable
            源项 f(x,y)
        
        Returns
        -------
        u : ndarray, shape (nx, ny)
            节点上的不确定性值
        x : ndarray
        y : ndarray
        """
        (xmin, xmax), (ymin, ymax) = domain
        x = np.linspace(xmin, xmax, self.nx)
        y = np.linspace(ymin, ymax, self.ny)

        mn = self.nx * self.ny
        A_mat = np.zeros((mn, mn), dtype=np.float64)
        b_vec = np.zeros(mn, dtype=np.float64)

        # 高斯求积节点（3点Gauss-Legendre）
        quad_xi = np.array([-0.7745966692414834, 0.0, 0.7745966692414834])
        quad_w = np.array([0.5555555555555556, 0.8888888888888889, 0.5555555555555556])

        for ex in range(self.nx - 1):
            xw = x[ex]
            xe = x[ex + 1]
            hx = xe - xw

            for ey in range(self.ny - 1):
                ys = y[ey]
                yn = y[ey + 1]
                hy = yn - ys

                # 四个节点索引
                sw = ey * self.nx + ex
                se = ey * self.nx + ex + 1
                nw = (ey + 1) * self.nx + ex
                ne = (ey + 1) * self.nx + ex + 1
                nodes = [sw, se, nw, ne]

                # 在每个四边形单元上用双线性基函数 + 高斯积分
                for qx in range(3):
                    xi = quad_xi[qx]
                    xq = 0.5 * ((1.0 - xi) * xw + (1.0 + xi) * xe)
                    wx = quad_w[qx] * hx * 0.5

                    for qy in range(3):
                        eta = quad_xi[qy]
                        yq = 0.5 * ((1.0 - eta) * ys + (1.0 + eta) * yn)
                        wy = quad_w[qy] * hy * 0.5
                        wq = wx * wy

                        # 双线性基函数在参考单元 [-1,1]^2 上
                        # N1 = (1-xi)(1-eta)/4  (sw)
                        # N2 = (1+xi)(1-eta)/4  (se)
                        # N3 = (1-xi)(1+eta)/4  (nw)
                        # N4 = (1+xi)(1+eta)/4  (ne)
                        N = np.array([
                            0.25 * (1.0 - xi) * (1.0 - eta),
                            0.25 * (1.0 + xi) * (1.0 - eta),
                            0.25 * (1.0 - xi) * (1.0 + eta),
                            0.25 * (1.0 + xi) * (1.0 + eta)
                        ], dtype=np.float64)

                        # 梯度（对物理坐标）
                        dN_dxi = np.array([-0.25 * (1.0 - eta), 0.25 * (1.0 - eta),
                                           -0.25 * (1.0 + eta), 0.25 * (1.0 + eta)])
                        dN_deta = np.array([-0.25 * (1.0 - xi), -0.25 * (1.0 + xi),
                                            0.25 * (1.0 - xi), 0.25 * (1.0 + xi)])

                        # 雅可比
                        J = np.array([
                            [dN_dxi @ np.array([xw, xe, xw, xe]),
                             dN_dxi @ np.array([ys, ys, yn, yn])],
                            [dN_deta @ np.array([xw, xe, xw, xe]),
                             dN_deta @ np.array([ys, ys, yn, yn])]
                        ])
                        detJ = J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0]
                        if abs(detJ) < 1e-14:
                            continue

                        # 梯度变换
                        invJ = np.array([[J[1, 1], -J[0, 1]], [-J[1, 0], J[0, 0]]]) / detJ
                        gradN = np.zeros((4, 2))
                        for i in range(4):
                            gradN[i] = invJ @ np.array([dN_dxi[i], dN_deta[i]])

                        aq = a_func(xq, yq)
                        cq = c_func(xq, yq)
                        fq = f_func(xq, yq)

                        # 组装单元矩阵
                        for i in range(4):
                            for j in range(4):
                                A_mat[nodes[i], nodes[j]] += wq * (
                                    aq * np.dot(gradN[i], gradN[j]) +
                                    cq * N[i] * N[j]
                                )
                            b_vec[nodes[i]] += wq * fq * N[i]

        # 边界条件：Dirichlet u=0
        for iy in range(self.ny):
            for ix in range(self.nx):
                k = iy * self.nx + ix
                if ix == 0 or ix == self.nx - 1 or iy == 0 or iy == self.ny - 1:
                    A_mat[k, :] = 0.0
                    A_mat[:, k] = 0.0
                    A_mat[k, k] = 1.0
                    b_vec[k] = 0.0

        # 求解
        try:
            u = np.linalg.solve(A_mat, b_vec)
        except np.linalg.LinAlgError:
            u = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]

        u_grid = u.reshape(self.ny, self.nx).T
        return u_grid, x, y

    @staticmethod
    def sample_field_at_points(u_grid, x_grid, y_grid, query_points):
        """
        在任意点采样有限元场值
        
        融合 fem2d_sample 的双线性插值思想
        
        Parameters
        ----------
        u_grid : ndarray, shape (nx, ny)
        x_grid, y_grid : 1D arrays
        query_points : ndarray, shape (M, 2)
        
        Returns
        -------
        values : ndarray, shape (M,)
        """
        query_points = np.asarray(query_points, dtype=np.float64)
        values = np.zeros(query_points.shape[0], dtype=np.float64)

        nx = len(x_grid)
        ny = len(y_grid)
        dx = x_grid[1] - x_grid[0] if nx > 1 else 1.0
        dy = y_grid[1] - y_grid[0] if ny > 1 else 1.0

        for idx, (px, py) in enumerate(query_points):
            # 找到所在单元
            ix = int((px - x_grid[0]) / dx)
            iy = int((py - y_grid[0]) / dy)
            ix = max(0, min(ix, nx - 2))
            iy = max(0, min(iy, ny - 2))

            # 局部坐标
            xi = 2.0 * (px - x_grid[ix]) / dx - 1.0
            eta = 2.0 * (py - y_grid[iy]) / dy - 1.0
            xi = np.clip(xi, -1.0, 1.0)
            eta = np.clip(eta, -1.0, 1.0)

            # 双线性插值
            N1 = 0.25 * (1.0 - xi) * (1.0 - eta)
            N2 = 0.25 * (1.0 + xi) * (1.0 - eta)
            N3 = 0.25 * (1.0 - xi) * (1.0 + eta)
            N4 = 0.25 * (1.0 + xi) * (1.0 + eta)

            values[idx] = (N1 * u_grid[ix, iy] +
                           N2 * u_grid[ix + 1, iy] +
                           N3 * u_grid[ix, iy + 1] +
                           N4 * u_grid[ix + 1, iy + 1])

        return values


class AnnularCovarianceEstimator:
    """
    环形区域协方差估计器
    融合 annulus_rule 项目的数值积分思想
    """

    def __init__(self, nr=8, nt=32):
        self.nr = max(int(nr), 1)
        self.nt = max(int(nt), 4)

    @staticmethod
    def legendre_gauss_nodes(n):
        """
        计算 Legendre-Gauss 求积节点和权重（使用 numpy 的 leggauss）
        """
        try:
            xi, w = np.polynomial.legendre.leggauss(n)
        except Exception:
            # 退化到简单中点规则
            xi = np.linspace(-1, 1, n, endpoint=False) + 1.0 / n
            w = np.full(n, 2.0 / n)
        return xi, w

    def integrate_annular_covariance(self, center, r1, r2, covariance_func):
        """
        在环形区域 [r1, r2] 上积分协方差
        
        积分区域：r1^2 <= (x-cx)^2 + (y-cy)^2 <= r2^2
        
        坐标变换：
        x = cx + r * cos(θ)
        y = cy + r * sin(θ)
        
        雅可比：J = r dr dθ
        
        径向变换：ρ = (r^2 - r1^2) / (r2^2 - r1^2) ∈ [0,1]
        r = sqrt( ρ*(r2^2-r1^2) + r1^2 )
        dr = (r2^2 - r1^2) / (2*r) dρ
        
        Parameters
        ----------
        center : tuple (cx, cy)
        r1, r2 : float
            内外半径
        covariance_func : callable
            f(x, y) -> ndarray(2,2) 或 float
        
        Returns
        -------
        integral : ndarray(2,2) or float
        """
        cx, cy = center
        r1 = float(r1)
        r2 = float(r2)
        if r2 <= r1:
            r2 = r1 + 1e-6

        area = np.pi * (r2 * r2 - r1 * r1)

        # Legendre 节点
        ra, rw = self.legendre_gauss_nodes(self.nr)

        # 径向变量变换：[-1,1] -> [r1^2, r2^2]
        r_sq_nodes = 0.5 * ((r2 ** 2 - r1 ** 2) * ra + (r2 ** 2 + r1 ** 2))
        r_nodes = np.sqrt(r_sq_nodes)
        # 权重变换
        w_r = rw * 0.5 * (r2 ** 2 - r1 ** 2)
        # 除以 (r2+r1)(r2-r1) = r2^2-r1^2 归一化
        w_r = w_r / (r2 ** 2 - r1 ** 2)

        tw = 1.0 / self.nt

        # 采样协方差类型
        sample_val = covariance_func(cx + r_nodes[0] * np.cos(0.0),
                                      cy + r_nodes[0] * np.sin(0.0))
        if np.isscalar(sample_val):
            integral = 0.0
            is_scalar = True
        else:
            integral = np.zeros((2, 2), dtype=np.float64)
            is_scalar = False

        for i in range(self.nt):
            theta = 2.0 * np.pi * i / self.nt
            c_t = np.cos(theta)
            s_t = np.sin(theta)
            for j in range(self.nr):
                r_val = r_nodes[j]
                px = cx + r_val * c_t
                py = cy + r_val * s_t
                val = covariance_func(px, py)
                weight = area * tw * w_r[j]
                if is_scalar:
                    integral += weight * val
                else:
                    integral += weight * np.asarray(val, dtype=np.float64)

        return integral
