"""
spectral_discretization.py
间断Galerkin (DG) 谱元方法求解海洋内波传播方程

融合项目:
- 271_dg1d_advection: 1D DG对流求解器

核心科学:
将一维对流方程推广至内波传播方程:
    ∂u/∂t + a · ∂u/∂x = S(x,t,u)

其中 S 包含非线性源项、浮力修正和耗散。

采用 nodal DG 方法，基于 Jacobi-Gauss-Lobatto 点上的 Legendre 多项式展开。
数值通量采用 upwind 格式。
"""

import numpy as np
from numpy.polynomial.legendre import leggauss


def jacobi_gauss_lobatto(N):
    """
    计算 Jacobi-Gauss-Lobatto (JGL) 节点和权重
    
    用于 nodal DG 的配点。JGL 点包含区间端点 [-1, 1]。
    
    参数:
        N: 多项式阶数
    
    返回:
        r: JGL 节点 (在 [-1, 1] 上)
        w: 权重
    """
    if N == 0:
        return np.array([-1.0, 1.0]), np.array([1.0, 1.0])
    
    # JGL 点是 (1-x²) P'_{N}(x) 的零点
    # 使用 Gauss-Legendre 点 + 端点
    # 标准方法: 使用对称三对角矩阵的特征值
    
    # 构造 Jacobi 矩阵 (Legendre 情况 α=β=0)
    n = N
    J = np.zeros((n + 1, n + 1))
    
    for i in range(1, n + 1):
        beta = i / np.sqrt(4.0 * i * i - 1.0)
        J[i, i-1] = beta
        J[i-1, i] = beta
    
    # JGL 点: 修改矩阵使得 [-1, 1] 是特征值
    # 简化实现: 使用端点 + Legendre Gauss 点的扩展
    
    # 使用 numpy 的 leggauss 并加入端点
    gauss_r, gauss_w = leggauss(n)
    
    r = np.concatenate([[-1.0], gauss_r, [1.0]])
    w = np.concatenate([[2.0 / (n * (n + 1))], gauss_w, [2.0 / (n * (n + 1))]])
    
    # 确保单调性
    r = np.sort(r)
    
    return r, w


def vandermonde_1d(N, r):
    """
    构建1D Vandermonde矩阵
    
    V_{ij} = P_j(r_i)  其中 P_j 为第 j 阶 Legendre 多项式
    
    参数:
        N: 多项式阶数
        r: 节点坐标
    
    返回:
        V: Vandermonde矩阵
    """
    V = np.zeros((len(r), N + 1))
    
    # 使用三项递推公式计算 Legendre 多项式
    V[:, 0] = 1.0
    if N >= 1:
        V[:, 1] = r
    
    for j in range(1, N):
        V[:, j+1] = ((2.0 * j + 1.0) * r * V[:, j] - j * V[:, j-1]) / (j + 1.0)
    
    return V


def grad_vandermonde_1d(N, r):
    """
    构建1D梯度Vandermonde矩阵
    
    Vr_{ij} = dP_j/dr(r_i)
    """
    Vr = np.zeros((len(r), N + 1))
    
    Vr[:, 0] = 0.0
    if N >= 1:
        Vr[:, 1] = 1.0
    
    # 使用递推: dP_j/dr = j/(1-r²) [r P_j - P_{j-1}]
    V = vandermonde_1d(N, r)
    
    for j in range(1, N):
        Vr[:, j+1] = Vr[:, j-1] + (2.0 * j + 1.0) * V[:, j]
    
    return Vr


def d_matrix_1d(N, r):
    """
    构建1D微分矩阵
    
    D = Vr · V^{-1}
    """
    V = vandermonde_1d(N, r)
    Vr = grad_vandermonde_1d(N, r)
    
    try:
        V_inv = np.linalg.inv(V)
    except np.linalg.LinAlgError:
        V_inv = np.linalg.pinv(V)
    
    D = Vr @ V_inv
    return D


def lift_matrix_1d(N, r):
    """
    构建1D DG提升算子
    
    LIFT = V · V^T · Emat
    """
    V = vandermonde_1d(N, r)
    n_nodes = len(r)
    
    # 面质量矩阵 (仅端点)
    Emat = np.zeros((n_nodes, 2))
    Emat[0, 0] = 1.0   # 左端点
    Emat[-1, 1] = 1.0  # 右端点
    
    lift = V @ V.T @ Emat
    return lift


def mesh_gen_1d(xmin, xmax, K):
    """
    生成1D均匀网格
    
    参数:
        xmin, xmax: 区间边界
        K: 单元数量
    
    返回:
        VX: 顶点坐标
        EToV: 单元到顶点映射
    """
    VX = np.linspace(xmin, xmax, K + 1)
    EToV = np.zeros((K, 2), dtype=int)
    
    for k in range(K):
        EToV[k, 0] = k
        EToV[k, 1] = k + 1
    
    return VX, EToV


class DGInternalWaveSolver:
    """
    1D间断Galerkin求解器用于内波传播
    
    求解方程:
        ∂u/∂t + c · ∂u/∂x = S(u,x,t)
    
    使用低存储5级Runge-Kutta时间积分。
    """
    
    def __init__(self, N=4, K=20, xmin=0.0, xmax=2000.0,
                 wave_speed=1.0, N_buoyancy=0.01):
        """
        初始化DG求解器
        
        参数:
            N: 每个单元上的多项式阶数
            K: 单元数量
            xmin, xmax: 空间域 [m]
            wave_speed: 内波波速 [m/s]
            N_buoyancy: 浮力频率 [rad/s]
        """
        self.N = N
        self.K = K
        self.wave_speed = wave_speed
        self.N_buoyancy = N_buoyancy
        
        # 参考单元节点
        self.r, _ = jacobi_gauss_lobatto(N)
        self.Np = len(self.r)  # 每个单元的节点数
        
        # 微分矩阵
        self.Dr = d_matrix_1d(N, self.r)
        
        # 提升算子
        self.LIFT = lift_matrix_1d(N, self.r)
        
        # 网格
        self.VX, self.EToV = mesh_gen_1d(xmin, xmax, K)
        
        # 物理坐标
        self.x = np.zeros((K, self.Np))
        self.dx = (xmax - xmin) / K
        
        for k in range(K):
            x_l = self.VX[self.EToV[k, 0]]
            x_r = self.VX[self.EToV[k, 1]]
            self.x[k, :] = 0.5 * (x_r + x_l) + 0.5 * (x_r - x_l) * self.r
        
        # 几何因子
        self.rx = 2.0 / self.dx  # dr/dx
        self.J = self.dx / 2.0   # 雅可比
        
        # 面缩放因子
        self.Fscale = 1.0 / self.J
        
        # 边界映射
        self.vmapM = np.zeros((K, 2), dtype=int)  # 内部面节点
        self.vmapP = np.zeros((K, 2), dtype=int)  # 外部面节点
        
        for k in range(K):
            self.vmapM[k, 0] = k * self.Np         # 左面
            self.vmapM[k, 1] = k * self.Np + N     # 右面
        
        # 构建全局映射
        self.vmapP[0, 0] = (K - 1) * self.Np + N   # 周期边界: 左=右
        self.vmapP[0, 1] = 1 * self.Np             # 右=下一单元左
        
        for k in range(1, K - 1):
            self.vmapP[k, 0] = (k - 1) * self.Np + N
            self.vmapP[k, 1] = (k + 1) * self.Np
        
        self.vmapP[K-1, 0] = (K - 2) * self.Np + N
        self.vmapP[K-1, 1] = 0                     # 周期边界: 右=左
        
        # 低存储RK系数
        self.rk4a = np.array([0.0, -567301805773.0/1357537059087.0,
                              -2404267990393.0/2016746695238.0,
                              -3550918686646.0/2091501179385.0,
                              -1275806237668.0/842570457699.0])
        self.rk4b = np.array([1432997174477.0/9575080441755.0,
                              5161836677717.0/13612068292357.0,
                              1720146321549.0/2090206949498.0,
                              3134564353537.0/4481467310338.0,
                              2277821191437.0/14882151754819.0])
        self.rk4c = np.array([0.0, 1432997174477.0/9575080441755.0,
                              2526269341429.0/6820363962896.0,
                              2006345519317.0/3224310063776.0,
                              2802321613138.0/2924317926251.0])
    
    def initial_condition(self, mode='sech2'):
        """
        初始条件
        
        参数:
            mode: 'sech2' (孤立波), 'sin' (正弦波)
        """
        if mode == 'sech2':
            u = np.zeros((self.K, self.Np))
            for k in range(self.K):
                u[k, :] = 2.0 / np.cosh((self.x[k, :] - 1000.0) / 100.0)**2
        elif mode == 'sin':
            u = np.zeros((self.K, self.Np))
            for k in range(self.K):
                u[k, :] = np.sin(2.0 * np.pi * self.x[k, :] / 2000.0)
        else:
            u = np.zeros((self.K, self.Np))
        
        return u
    
    def source_term(self, u, x, t):
        """
        内波源项
        
        S(u,x,t) = -N² · sin(kx - ωt) - ν · ∇⁴u
        
        参数:
            u: 状态变量
            x: 空间坐标
            t: 时间
        
        返回:
            S: 源项
        """
        # 非线性浮力修正
        S_buoyancy = -self.N_buoyancy**2 * np.sin(2.0 * np.pi * x / 2000.0 - 0.1 * t)
        
        # 高阶耗散 (超粘性)
        S_diss = -1.0e-6 * u
        
        S = S_buoyancy + S_diss
        return S
    
    def rhs_dg(self, u, t):
        """
        计算DG右端项
        
        rhs = -a · rx · (Dr @ u) + LIFT · (Fscale · du_flux)
        """
        # 内部导数
        dudx = np.zeros_like(u)
        for k in range(self.K):
            dudx[k, :] = self.rx * (self.Dr @ u[k, :])
        
        # 面通量 (upwind)
        du = np.zeros((self.K, 2))
        for k in range(K := self.K):
            # 左面
            idxM = self.vmapM[k, 0]
            idxP = self.vmapP[k, 0]
            kM = idxM // self.Np
            iM = idxM % self.Np
            kP = idxP // self.Np
            iP = idxP % self.Np
            
            if kM < self.K and kP < self.K:
                uM = u[kM, iM]
                uP = u[kP, iP]
            else:
                uM = u[k, 0]
                uP = u[k, 0]
            
            # upwind flux
            alpha = 1.0
            du[k, 0] = self.wave_speed * (uM - uP) * 0.5 * (1.0 - alpha * np.sign(self.wave_speed))
        
            # 右面
            idxM = self.vmapM[k, 1]
            idxP = self.vmapP[k, 1]
            kM = idxM // self.Np
            iM = idxM % self.Np
            kP = idxP // self.Np
            iP = idxP % self.Np
            
            if kM < self.K and kP < self.K:
                uM = u[kM, iM]
                uP = u[kP, iP]
            else:
                uM = u[k, -1]
                uP = u[k, -1]
            
            du[k, 1] = self.wave_speed * (uM - uP) * 0.5 * (1.0 - alpha * np.sign(self.wave_speed))
        
        # 源项
        S = np.zeros_like(u)
        for k in range(self.K):
            S[k, :] = self.source_term(u[k, :], self.x[k, :], t)
        
        # 组装RHS
        rhs = np.zeros_like(u)
        for k in range(self.K):
            flux_term = self.LIFT @ (self.Fscale * du[k, :])
            rhs[k, :] = -self.wave_speed * dudx[k, :] + flux_term + S[k, :]
        
        return rhs
    
    def solve(self, t_final=100.0, dt=0.5):
        """
        低存储RK-45时间积分
        
        参数:
            t_final: 终止时间 [s]
            dt: 时间步长 [s]
        
        返回:
            t_history: 时间历史
            u_history: 解历史
        """
        u = self.initial_condition()
        
        nsteps = int(t_final / dt)
        t_history = np.zeros(nsteps + 1)
        u_history = np.zeros((nsteps + 1, self.K, self.Np))
        u_history[0, :, :] = u
        
        time = 0.0
        
        for n in range(nsteps):
            resu = np.zeros_like(u)
            
            for INTRK in range(5):
                rhsu = self.rhs_dg(u, time)
                resu = self.rk4a[INTRK] * resu + dt * rhsu
                u = u + self.rk4b[INTRK] * resu
            
            time += dt
            t_history[n+1] = time
            u_history[n+1, :, :] = u
            
            # 边界处理
            u = np.clip(u, -10.0, 10.0)
        
        return t_history, u_history
