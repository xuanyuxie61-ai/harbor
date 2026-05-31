"""
electrostatics.py
酶活性位点静电势计算模块

核心功能：
- 二维 Poisson 方程有限差分解
- 粒子云网格（Particle-in-Cell, PIC）电荷密度分配
- Gauss-Seidel 迭代求解电势
- 电场计算与离子受力

科学背景：
酶催化反应过渡态的静电稳定化能是催化效率的核心因素：
    ΔG‡_elec = ⟨ψ|Ĥ|ψ⟩_TS - ⟨ψ|Ĥ|ψ⟩_R

Poisson-Boltzmann 方程（线性化形式）：
    ∇·[ε(r)∇φ(r)] = -4πρ(r) + κ²ε(r)φ(r)

在二维活性位面近似下（z=常数截面）：
    -∇²φ = ρ/ε_0    （真空区域）
    -∇²φ + κ²φ = ρ/ε_0    （溶剂区域）

其中：
    φ: 静电势（V）
    ρ: 电荷密度（C/m³）
    ε_0: 真空介电常数 = 8.854×10⁻¹² F/m
    κ: Debye-Hückel 屏蔽参数，κ² = 2000 N_A e² I / (ε_0 ε_r k_B T)
    I: 离子强度（mol/L）

本模块采用有限差分法在规则网格上求解：
    (φ_{i+1,j} + φ_{i-1,j} + φ_{i,j+1} + φ_{i,j-1} - 4φ_{i,j}) / h² = -ρ_{i,j}/ε_0

边界条件：
    - Dirichlet: φ = φ_0 在蛋白表面
    - Neumann: ∂φ/∂n = 0 在对称边界
"""

import numpy as np


class Poisson2DSolver:
    """
    二维 Poisson 方程求解器
    采用五点差分模板和 Gauss-Seidel 迭代
    """

    def __init__(self, nx, ny, dh, boundary_box=None):
        """
        参数：
            nx, ny: x 和 y 方向格点数
            dh: 网格间距（Å）
            boundary_box: Dirichlet 边界框 [[x1,x2],[y1,y2]]
        """
        self.nx = nx
        self.ny = ny
        self.dh = dh
        self.nn = nx * ny
        self.boundary_box = boundary_box

        # 构建有限差分模板矩阵 A
        self.A = self._build_stencil()

    def _build_stencil(self):
        """
        构建五点差分模板矩阵 A

        对于内部节点 (i,j)：
            A(u,u)    = -4/h²
            A(u,u-1)  =  1/h²
            A(u,u+1)  =  1/h²
            A(u,u-nx) =  1/h²
            A(u,u+nx) =  1/h²

        边界处理：
            - y=0: Neumann, A(u,u)=-1/h, A(u,u+nx)=1/h
            - y=height: Neumann
            - x=width: Neumann
            - x=0: Dirichlet, A(u,u)=1
        """
        nx, ny, dh = self.nx, self.ny, self.dh
        nn = nx * ny
        A = np.zeros((nn, nn), dtype=float)
        h2 = dh * dh

        # 内部节点
        for j in range(1, ny - 1):
            for i in range(1, nx - 1):
                u = j * nx + i
                A[u, u] = -4.0 / h2
                A[u, u - 1] = 1.0 / h2
                A[u, u + 1] = 1.0 / h2
                A[u, u - nx] = 1.0 / h2
                A[u, u + nx] = 1.0 / h2

        # y=0 Neumann 边界
        j = 0
        for i in range(nx):
            u = j * nx + i
            A[u, u] = -1.0 / dh
            A[u, u + nx] = 1.0 / dh

        # y=height Neumann 边界
        j = ny - 1
        for i in range(nx):
            u = j * nx + i
            A[u, u - nx] = 1.0 / dh
            A[u, u] = -1.0 / dh

        # x=width Neumann 边界
        i = nx - 1
        for j in range(ny):
            u = j * nx + i
            A[u, :] = 0.0
            A[u, u - 1] = 1.0 / dh
            A[u, u] = -1.0 / dh

        # x=0 Dirichlet 边界
        i = 0
        for j in range(ny):
            u = j * nx + i
            A[u, :] = 0.0
            A[u, u] = 1.0

        # 障碍物（蛋白板）Dirichlet 边界
        if self.boundary_box is not None:
            bx1, bx2 = self.boundary_box[0]
            by1, by2 = self.boundary_box[1]
            for j in range(by1, by2 + 1):
                for i in range(bx1, bx2 + 1):
                    u = j * nx + i
                    A[u, :] = 0.0
                    A[u, u] = 1.0

        return A

    def solve_gs(self, phi_init, den, n0, phi0, te, phi_p, eps0, qe, max_iter=2000, tol=0.1):
        """
        Gauss-Seidel 迭代求解电势

        迭代公式：
            b = -qe/ε_0 * [den - n0 * exp((phi - phi0)/te)]
            phi_i = [b_i - Σ_{j<i} A_{ij} phi_j - Σ_{j>i} A_{ij} phi_j] / A_{ii}

        参数：
            phi_init: 初始电势猜测 (nx, ny)
            den: 电荷密度 (nx, ny)
            n0: 参考粒子密度
            phi0: 参考电势
            te: 电子温度 (eV)
            phi_p: 壁面电势
            eps0: 真空介电常数
            qe: 元电荷
        """
        nx, ny = self.nx, self.ny
        nn = self.nn
        phi = phi_init.copy().ravel()
        den_vec = den.copy().ravel()

        for it in range(1, max_iter + 1):
            # 重新计算右端项（包含 Boltzmann 电子项）
            # 限制 phi 范围防止指数溢出
            phi_clipped = np.clip(phi, phi0 - 10.0 * te, phi0 + 10.0 * te)
            b = den_vec - n0 * np.exp((phi_clipped - phi0) / te)
            b = -b * qe / eps0

            # 边界条件
            b[0:nx] = 0.0
            b[nn - nx:nn] = 0.0
            b[nx - 1:nn:nx] = 0.0
            b[0:nn:nx] = phi0

            if self.boundary_box is not None:
                bx1, bx2 = self.boundary_box[0]
                by1, by2 = self.boundary_box[1]
                for j in range(by1, by2 + 1):
                    b[bx1 + j * nx:bx2 + 1 + j * nx] = phi_p

            # Gauss-Seidel 更新
            for i in range(nn):
                phi[i] = (b[i] - np.dot(self.A[i, 0:i], phi[0:i])
                          - np.dot(self.A[i, i + 1:nn], phi[i + 1:nn])) / self.A[i, i]

            # 残差检查
            if it % 10 == 0:
                res = np.linalg.norm(b - self.A @ phi)
                if res <= tol:
                    break

        return phi.reshape((nx, ny))

    def compute_electric_field(self, phi):
        """
        计算电场 E = -∇φ

        中心差分（内部）：
            E_x(i,j) = [φ(i-1,j) - φ(i+1,j)] / (2h)
            E_y(i,j) = [φ(i,j-1) - φ(i,j+1)] / (2h)

        前向/后向差分（边界）
        """
        nx, ny, dh = self.nx, self.ny, self.dh
        efx = np.zeros((nx, ny), dtype=float)
        efy = np.zeros((nx, ny), dtype=float)

        # 内部中心差分
        efx[1:nx - 1, :] = (phi[0:nx - 2, :] - phi[2:nx, :]) / (2.0 * dh)
        efy[:, 1:ny - 1] = (phi[:, 0:ny - 2] - phi[:, 2:ny]) / (2.0 * dh)

        # 边界前向/后向差分
        efx[0, :] = (phi[0, :] - phi[1, :]) / dh
        efx[nx - 1, :] = (phi[nx - 2, :] - phi[nx - 1, :]) / dh
        efy[:, 0] = (phi[:, 0] - phi[:, 1]) / dh
        efy[:, ny - 1] = (phi[:, ny - 2] - phi[:, ny - 1]) / dh

        return efx, efy


def pic_charge_density(nx, ny, dh, part_x, part_v, np_part, spwt, mp_q):
    """
    粒子云网格（PIC）电荷密度分配

    采用双线性权重分配（Cloud-in-Cell, CIC）：
        粒子位于 (x_p, y_p)，所在网格单元左下角 (i,j)
        hx = (x_p - x_i) / dx, hy = (y_p - y_j) / dy
        ρ(i,j)     += q_p * (1-hx)*(1-hy) / (dx*dy)
        ρ(i+1,j)   += q_p * hx*(1-hy) / (dx*dy)
        ρ(i,j+1)   += q_p * (1-hx)*hy / (dx*dy)
        ρ(i+1,j+1) += q_p * hx*hy / (dx*dy)

    参数：
        nx, ny: 网格维度
        dh: 网格间距
        part_x: 粒子位置数组 (np, 2)
        part_v: 粒子速度数组 (np, 2)
        np_part: 实际粒子数
        spwt: 比权重（真实粒子/宏粒子）
        mp_q: 宏粒子电荷
    返回：
        den: 电荷密度 (nx, ny)
    """
    chg = np.zeros((nx, ny), dtype=float)

    for p in range(np_part):
        fi = 1.0 + part_x[p, 0] / dh
        i = int(np.floor(fi))
        hx = fi - i

        fj = 1.0 + part_x[p, 1] / dh
        j = int(np.floor(fj))
        hy = fj - j

        # 边界保护
        if i < 0 or i >= nx - 1 or j < 0 or j >= ny - 1:
            continue

        chg[i, j] += (1.0 - hx) * (1.0 - hy)
        chg[i + 1, j] += hx * (1.0 - hy)
        chg[i, j + 1] += (1.0 - hx) * hy
        chg[i + 1, j + 1] += hx * hy

    # 密度计算
    den = spwt * mp_q * chg / (dh * dh)

    # 边界密度加倍（CIC 修正）
    den[0, :] *= 2.0
    den[nx - 1, :] *= 2.0
    den[:, 0] *= 2.0
    den[:, ny - 1] *= 2.0

    # 密度下限
    den = den + 10000.0

    return den


def poisson_2d_exact_solution(x, y):
    """
    二维 Poisson 方程的解析解（用于验证）

    精确解：
        u(x,y) = 2(1+y) / [(3+x)² + (1+y)²]

    对应右端项：
        f(x,y) = -u_{xx} - u_{yy}

    返回：
        u, ux, uy, uxx, uxy, uyy
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    denom = (3.0 + x) ** 2 + (1.0 + y) ** 2
    u = 2.0 * (1.0 + y) / denom
    ux = (-2.0 * x - 6.0) * (2.0 * y + 2.0) / (denom ** 2)
    uy = (-2.0 * y - 2.0) * (2.0 * y + 2.0) / (denom ** 2) + 2.0 / denom
    uxx = 4.0 * (y + 1.0) * (4.0 * (x + 3.0) ** 2 / denom - 1.0) / (denom ** 2)
    uxy = 4.0 * (x + 3.0) * (4.0 * (y + 1.0) ** 2 / denom - 1.0) / (denom ** 2)
    uyy = 4.0 * (y + 1.0) * (4.0 * (y + 1.0) ** 2 / denom - 3.0) / (denom ** 2)

    return u, ux, uy, uxx, uxy, uyy


def electrostatic_stabilization_energy(phi, rho, dx, dy):
    """
    计算静电稳定化能

    公式：
        E_stab = (1/2) ∫∫ ρ(r) φ(r) dr
               ≈ (1/2) Σ_{i,j} ρ_{ij} φ_{ij} dx dy

    在过渡态理论中，该能量是活化自由能的重要组成：
        ΔG‡ = ΔG‡_intrinsic + ΔG‡_elec + ΔG‡_solvent
    """
    return 0.5 * np.sum(rho * phi) * dx * dy
