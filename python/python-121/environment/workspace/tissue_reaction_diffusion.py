"""
tissue_reaction_diffusion.py
心脏组织反应扩散方程求解模块

融入原项目:
- 369_fd2d_predator_prey: 2D反应扩散方程的有限差分求解
- 114_box_flow: 有限元方法中的质量/刚度矩阵组装、时间步进
- 988_r8pbu: 共轭梯度法求解线性系统

功能:
1. 单域模型（Monodomain）的有限差分离散化
2. 双域模型（Bidomain）的简化求解
3. 时间步进（前向欧拉、Crank-Nicolson、ADI）
4. 边界条件处理（Dirichlet/Neumann/No-flux）

核心方程:
Monodomain: ∂V_m/∂t = ∇·(D∇V_m) - I_ion/C_m
Bidomain: 
  ∇·(σ_i∇V_m) + ∇·(σ_i∇φ_e) = χ(C_m∂V_m/∂t + I_ion)
  ∇·((σ_i+σ_e)∇φ_e) + ∇·(σ_i∇V_m) = 0
"""

import numpy as np
from math import sqrt


# ============================================================================
# 扩散张量构建
# ============================================================================

def build_diffusion_tensor(D_f, D_t, fiber_angle):
    """
    构建各向异性扩散张量
    
    心肌纤维方向 e_f = (cos θ, sin θ)
    横纤维方向 e_t = (-sin θ, cos θ)
    
    扩散张量:
    D = D_f * e_f ⊗ e_f + D_t * e_t ⊗ e_t
      = [[D_f*cos²θ + D_t*sin²θ, (D_f-D_t)*sinθ*cosθ],
         [(D_f-D_t)*sinθ*cosθ, D_f*sin²θ + D_t*cos²θ]]
    
    参数:
        D_f: 纤维方向扩散系数
        D_t: 横纤维方向扩散系数
        fiber_angle: 纤维角度场 (rad)
    返回:
        Dxx, Dxy, Dyy: 扩散张量分量场
    """
    c = np.cos(fiber_angle)
    s = np.sin(fiber_angle)
    
    Dxx = D_f * c ** 2 + D_t * s ** 2
    Dxy = (D_f - D_t) * s * c
    Dyy = D_f * s ** 2 + D_t * c ** 2
    
    return Dxx, Dxy, Dyy


# ============================================================================
# 有限差分拉普拉斯算子（各向异性）
# ============================================================================

def anisotropic_laplacian_5point(u, Dxx, Dxy, Dyy, dx, dy):
    """
    各向异性五点差分拉普拉斯算子
    
    对于方程: ∇·(D∇u) = ∂/∂x(Dxx ∂u/∂x + Dxy ∂u/∂y) + ∂/∂y(Dxy ∂u/∂x + Dyy ∂u/∂y)
    
    离散形式（基于通量守恒）:
    (∇·(D∇u))_{i,j} ≈ 
      [Dxx_{i+1/2,j}*(u_{i+1,j}-u_{i,j}) - Dxx_{i-1/2,j}*(u_{i,j}-u_{i-1,j})] / dx²
    + [Dyy_{i,j+1/2}*(u_{i,j+1}-u_{i,j}) - Dyy_{i,j-1/2}*(u_{i,j}-u_{i,j-1})] / dy²
    + [Dxy_{i+1/2,j+1/2}*(u_{i+1,j+1}-u_{i,j}) - ...] / (2*dx*dy)
    
    参数:
        u: (nx, ny) 标量场
        Dxx, Dxy, Dyy: (nx, ny) 扩散张量分量
        dx, dy: 网格间距
    返回:
        lap: (nx, ny) 拉普拉斯结果
    """
    # TODO: Hole 2 - 实现各向异性五点差分拉普拉斯算子
    # 基于通量守恒形式离散化:
    #   ∇·(D∇u) = ∂/∂x(Dxx ∂u/∂x + Dxy ∂u/∂y) + ∂/∂y(Dxy ∂u/∂x + Dyy ∂u/∂y)
    # 需要处理:
    #   - x方向通量 (Dxx_{i±1/2,j} * (u_{i±1,j} - u_{i,j}) / dx²)
    #   - y方向通量 (Dyy_{i,j±1/2} * (u_{i,j±1} - u_{i,j}) / dy²)
    #   - 交叉项 Dxy * (u_{i+1,j+1} - u_{i+1,j-1} - u_{i-1,j+1} + u_{i-1,j-1}) / (4*dx*dy)
    #   - 无通量边界条件 (Neumann)
    raise NotImplementedError("Hole 2: anisotropic_laplacian_5point 待实现")


def isotropic_laplacian_5point(u, dx, dy):
    """
    各向同性五点差分拉普拉斯算子
    
    ∇²u ≈ (u_{i+1,j} - 2u_{i,j} + u_{i-1,j})/dx² + (u_{i,j+1} - 2u_{i,j} + u_{i,j-1})/dy²
    """
    nx, ny = u.shape
    lap = np.zeros_like(u)
    
    lap[1:nx - 1, 1:ny - 1] = (
        (u[2:nx, 1:ny - 1] - 2.0 * u[1:nx - 1, 1:ny - 1] + u[0:nx - 2, 1:ny - 1]) / (dx ** 2) +
        (u[1:nx - 1, 2:ny] - 2.0 * u[1:nx - 1, 1:ny - 1] + u[1:nx - 1, 0:ny - 2]) / (dy ** 2)
    )
    
    # 边界条件
    lap[0, :] = lap[1, :]
    lap[nx - 1, :] = lap[nx - 2, :]
    lap[:, 0] = lap[:, 1]
    lap[:, ny - 1] = lap[:, ny - 2]
    
    return lap


# ============================================================================
# 反应扩散方程时间步进（源自 369_fd2d_predator_prey）
# ============================================================================

def forward_euler_step(u, v, D, dx, dy, dt, reaction_func, params):
    """
    前向欧拉时间步进
    
    u^{n+1} = u^n + dt * [D*∇²u^n + f(u^n, v^n)]
    v^{n+1} = v^n + dt * g(u^n, v^n)
    
    稳定性条件（CFL）:
    dt ≤ dx² / (4*D)  (2D各向同性)
    
    参数:
        u, v: 当前场
        D: 扩散系数或扩散张量
        dx, dy: 空间步长
        dt: 时间步长
        reaction_func: 反应函数 f(u,v), g(u,v)
        params: 反应参数
    返回:
        u_new, v_new: 更新后的场
    """
    # 计算拉普拉斯
    if isinstance(D, tuple) and len(D) == 3:
        Dxx, Dxy, Dyy = D
        lap_u = anisotropic_laplacian_5point(u, Dxx, Dxy, Dyy, dx, dy)
    else:
        lap_u = isotropic_laplacian_5point(u, dx, dy) * D
    
    # 计算反应项
    f, g = reaction_func(u, v, **params)
    
    # 更新
    u_new = u + dt * (lap_u + f)
    v_new = v + dt * g
    
    # 边界处理
    u_new = apply_boundary_conditions(u_new)
    v_new = apply_boundary_conditions(v_new)
    
    return u_new, v_new


def crank_nicolson_step(u, v, D, dx, dy, dt, reaction_func, params,
                        cg_tol=1e-10, max_cg_iter=500):
    """
    Crank-Nicolson隐式时间步进（用于扩散项）
    
    (I - 0.5*dt*D*∇²) u^{n+1} = (I + 0.5*dt*D*∇²) u^n + dt*f(u^n, v^n)
    
    使用共轭梯度法求解隐式系统
    
    参数:
        u, v: 当前场
        D: 扩散系数（标量）
        dx, dy: 空间步长
        dt: 时间步长
        reaction_func: 反应函数
        params: 反应参数
        cg_tol: CG收敛容差
        max_cg_iter: 最大CG迭代次数
    返回:
        u_new, v_new: 更新后的场
    """
    nx, ny = u.shape
    n = nx * ny
    
    # 构建带状矩阵（简化：各向同性）
    mu = nx  # 带宽
    a = np.zeros((mu + 1, n))
    
    coeff = 0.5 * dt * D
    
    for j in range(ny):
        for i in range(nx):
            idx = j * nx + i
            # 对角元
            diag_val = 1.0
            if i > 0:
                diag_val += coeff / (dx ** 2)
                a[mu - 1, idx + 1] = -coeff / (dx ** 2)
            if i < nx - 1:
                diag_val += coeff / (dx ** 2)
            if j > 0:
                diag_val += coeff / (dy ** 2)
                a[mu - nx, idx + nx] = -coeff / (dy ** 2)
            if j < ny - 1:
                diag_val += coeff / (dy ** 2)
            a[mu, idx] = diag_val
    
    # 右端项
    lap_u = isotropic_laplacian_5point(u, dx, dy)
    f, g = reaction_func(u, v, **params)
    
    rhs = u + 0.5 * dt * D * lap_u + dt * f
    rhs_flat = rhs.flatten()
    
    # 使用共轭梯度法求解
    from linear_algebra_core import r8pbu_cg
    u_new_flat, res, iters = r8pbu_cg(n, mu, a, rhs_flat, np.zeros(n),
                                      tol=cg_tol, max_iter=max_cg_iter)
    
    u_new = u_new_flat.reshape((nx, ny))
    v_new = v + dt * g
    
    u_new = apply_boundary_conditions(u_new)
    v_new = apply_boundary_conditions(v_new)
    
    return u_new, v_new


def adi_step(u, v, D, dx, dy, dt, reaction_func, params):
    """
    ADI（交替方向隐式）时间步进
    
    第一步（隐式x，显式y）:
    (I - 0.5*dt*D*∂²/∂x²) u* = (I + 0.5*dt*D*∂²/∂y²) u^n + 0.5*dt*f(u^n,v^n)
    
    第二步（隐式y，显式x）:
    (I - 0.5*dt*D*∂²/∂y²) u^{n+1} = (I + 0.5*dt*D*∂²/∂x²) u* + 0.5*dt*f(u*,v*)
    
    ADI是无条件稳定的，每步只需解三对角线性系统
    """
    nx, ny = u.shape
    f, g = reaction_func(u, v, **params)
    
    # 第一步: 隐式x
    rx = 0.5 * dt * D / (dx ** 2)
    u_star = np.zeros_like(u)
    
    for j in range(ny):
        # 构建三对角系统
        a_tri = np.full(nx, -rx)
        b_tri = np.full(nx, 1.0 + 2.0 * rx)
        c_tri = np.full(nx, -rx)
        
        # 右端项
        d_tri = np.zeros(nx)
        for i in range(nx):
            d_tri[i] = u[i, j]
            if j > 0:
                d_tri[i] += 0.5 * dt * D * (u[i, j - 1] - 2 * u[i, j] + u[i, j + 1 if j < ny - 1 else j]) / (dy ** 2)
            d_tri[i] += 0.5 * dt * f[i, j]
        
        # 边界条件
        b_tri[0] = 1.0
        c_tri[0] = 0.0
        d_tri[0] = u[0, j]
        a_tri[nx - 1] = 0.0
        b_tri[nx - 1] = 1.0
        d_tri[nx - 1] = u[nx - 1, j]
        
        u_star[:, j] = _solve_tridiagonal(a_tri, b_tri, c_tri, d_tri)
    
    # 第二步: 隐式y
    ry = 0.5 * dt * D / (dy ** 2)
    u_new = np.zeros_like(u)
    
    f_star, g_star = reaction_func(u_star, v, **params)
    
    for i in range(nx):
        a_tri = np.full(ny, -ry)
        b_tri = np.full(ny, 1.0 + 2.0 * ry)
        c_tri = np.full(ny, -ry)
        
        d_tri = np.zeros(ny)
        for j in range(ny):
            d_tri[j] = u_star[i, j]
            if i > 0:
                d_tri[j] += 0.5 * dt * D * (u_star[i - 1, j] - 2 * u_star[i, j] +
                                               u_star[i + 1 if i < nx - 1 else i, j]) / (dx ** 2)
            d_tri[j] += 0.5 * dt * f_star[i, j]
        
        # 边界条件
        b_tri[0] = 1.0
        c_tri[0] = 0.0
        d_tri[0] = u_star[i, 0]
        a_tri[ny - 1] = 0.0
        b_tri[ny - 1] = 1.0
        d_tri[ny - 1] = u_star[i, ny - 1]
        
        u_new[i, :] = _solve_tridiagonal(a_tri, b_tri, c_tri, d_tri)
    
    v_new = v + dt * g
    
    u_new = apply_boundary_conditions(u_new)
    v_new = apply_boundary_conditions(v_new)
    
    return u_new, v_new


def _solve_tridiagonal(a, b, c, d):
    """
    求解三对角线性系统（Thomas算法）
    
    矩阵形式:
    |b0 c0         | |x0|   |d0|
    |a1 b1 c1      | |x1|   |d1|
    |   a2 b2 c2   | |x2| = |d2|
    |      ...     | |...|  |...|
    |         an bn| |xn|   |dn|
    
    算法复杂度: O(n)
    """
    n = len(d)
    cp = np.zeros(n)
    dp = np.zeros(n)
    x = np.zeros(n)
    
    cp[0] = c[0] / b[0]
    dp[0] = d[0] / b[0]
    
    for i in range(1, n):
        denom = b[i] - a[i] * cp[i - 1]
        if abs(denom) < 1e-15:
            denom = 1e-15
        cp[i] = c[i] / denom if i < n - 1 else 0.0
        dp[i] = (d[i] - a[i] * dp[i - 1]) / denom
    
    x[n - 1] = dp[n - 1]
    for i in range(n - 2, -1, -1):
        x[i] = dp[i] - cp[i] * x[i + 1]
    
    return x


def apply_boundary_conditions(field):
    """
    应用无通量边界条件（Neumann）
    
    通过镜像法: 边界外推等于边界值
    """
    nx, ny = field.shape
    field[0, :] = field[1, :]
    field[nx - 1, :] = field[nx - 2, :]
    field[:, 0] = field[:, 1]
    field[:, ny - 1] = field[:, ny - 2]
    return field


# ============================================================================
# 反应扩散方程求解器主函数
# ============================================================================

def solve_reaction_diffusion_2d(u0, v0, D, dx, dy, dt, T,
                                 reaction_func, reaction_params,
                                 solver='forward_euler',
                                 stimulus_func=None,
                                 stimulus_region=None):
    """
    求解2D反应扩散方程
    
    参数:
        u0, v0: 初始场
        D: 扩散系数或扩散张量
        dx, dy: 空间步长
        dt: 时间步长
        T: 总时间
        reaction_func: 反应函数
        reaction_params: 反应参数
        solver: 'forward_euler', 'crank_nicolson', 'adi'
        stimulus_func: 刺激函数
        stimulus_region: 刺激区域掩码
    返回:
        u_history: 膜电位历史（稀疏采样）
        v_history: 恢复变量历史
        t_history: 时间历史
    """
    n_steps = int(T / dt)
    nx, ny = u0.shape
    
    u = u0.copy()
    v = v0.copy()
    
    # 稀疏保存历史
    save_interval = max(1, n_steps // 100)
    n_saved = n_steps // save_interval + 1
    
    u_history = np.zeros((n_saved, nx, ny))
    v_history = np.zeros((n_saved, nx, ny))
    t_history = np.zeros(n_saved)
    
    u_history[0] = u
    v_history[0] = v
    t_history[0] = 0.0
    
    save_idx = 1
    
    for step in range(1, n_steps + 1):
        # 应用刺激
        if stimulus_func is not None and stimulus_region is not None:
            stim = stimulus_func(step * dt)
            u += stimulus_region * stim * dt
        
        # 时间步进
        if solver == 'forward_euler':
            u, v = forward_euler_step(u, v, D, dx, dy, dt,
                                       reaction_func, reaction_params)
        elif solver == 'crank_nicolson':
            u, v = crank_nicolson_step(u, v, D, dx, dy, dt,
                                        reaction_func, reaction_params)
        elif solver == 'adi':
            u, v = adi_step(u, v, D, dx, dy, dt,
                            reaction_func, reaction_params)
        else:
            u, v = forward_euler_step(u, v, D, dx, dy, dt,
                                       reaction_func, reaction_params)
        
        # 保存历史
        if step % save_interval == 0 and save_idx < n_saved:
            u_history[save_idx] = u
            v_history[save_idx] = v
            t_history[save_idx] = step * dt
            save_idx += 1
    
    # 截取有效保存的数据
    u_history = u_history[:save_idx]
    v_history = v_history[:save_idx]
    t_history = t_history[:save_idx]
    
    return u_history, v_history, t_history


# ============================================================================
# 纤维角度场生成
# ============================================================================

def generate_fiber_angle_field(nx, ny, model='parallel'):
    """
    生成心肌纤维角度场
    
    模型:
    - 'parallel': 平行纤维，角度恒定
    - 'rotational': 旋转型纤维（从心外膜到心内膜旋转）
    - 'radial': 放射状纤维
    
    参数:
        nx, ny: 网格尺寸
        model: 纤维模型
    返回:
        angle: (nx, ny) 纤维角度场 (rad)
    """
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    if model == 'parallel':
        angle = np.zeros((nx, ny))
    elif model == 'rotational':
        # 从心外膜(-60°)到心内膜(+60°)线性旋转
        angle = -np.pi / 3.0 + (2.0 * np.pi / 3.0) * Y
    elif model == 'radial':
        # 放射状
        angle = np.arctan2(Y - 0.5, X - 0.5)
    else:
        angle = np.zeros((nx, ny))
    
    return angle
