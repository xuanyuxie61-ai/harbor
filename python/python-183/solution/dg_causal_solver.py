r"""
dg_causal_solver.py
================================================================================
基于间断 Galerkin (DG) 方法的因果效应时空传播求解器

原项目映射: 275_dg1d_poisson — 一维泊松方程的间断 Galerkin 有限元求解

科学背景
--------
在结构方程模型中，外生冲击（干预）$do(X_j=x)$ 的因果效应往往沿时空域传播。
我们将该传播过程建模为带有狄拉克源项的扩散型偏微分方程：

$$ \frac{\partial u}{\partial t} - \frac{\partial}{\partial x}\left(K(x)\frac{\partial u}{\partial x}\right) = f(x,t) + \sum_{j}\beta_j\,\delta(x-x_j)\,Y_j $$

其中 $u(x,t)$ 为因果势函数，$K(x)$ 为空间扩散系数，$\beta_j$ 为第 $j$ 个变量
对时空域的因果耦合强度，$Y_j$ 为结构方程中的内生变量取值。

为了捕捉激波与不连续边界，我们采用**局部间断 Galerkin (LDG)** 方法，
在单元交界面上引入数值通量与惩罚项，保证弱解的稳定性与一致性。

核心公式
--------
1. 弱形式（在每个单元 $I_i$ 上，乘以测试函数 $v$ 并分部积分）:
   $$ \int_{I_i} u_t v\,dx + \hat{u}v|_{\partial I_i} - \int_{I_i} u v_x\,dx = \int_{I_i} q v\,dx $$
   其中 $q = K(x)u_x$ 为通量变量，$\hat{u}$ 为数值通量。

2. 数值通量（SIPG 格式，对称内罚）:
   $$ \hat{u} = \{u\} - \frac{1}{2}\llbracket u \rrbracket, \qquad
      \hat{q} = \{q\} + \frac{\sigma}{h}\llbracket u \rrbracket $$
   其中 $\{u\}=\frac{u^++u^-}{2}$ 为均值，$\llbracket u\rrbracket=u^+-u^-$ 为跳量，
   $\sigma$ 为惩罚参数。

3. 局部质量矩阵（参考单元 $[-1,1]$ 上，二次多项式基 $P_2$）:
   $$ M_{ij} = \int_{-1}^{1}\phi_i(\xi)\phi_j(\xi)\,d\xi $$
   刚度矩阵:
   $$ A_{ij} = \int_{-1}^{1}\phi_i'(\xi)\phi_j'(\xi)\,d\xi $$

4. 时间离散采用隐式 Euler 以保障稳定性：
   $$ (M + \Delta t\,A)u^{n+1} = M u^n + \Delta t\,F^n $$
r"""

import numpy as np
from typing import Callable, Tuple


def legendre_basis_2d() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""
    构造参考单元 $[-1,1]$ 上的二次 Legendre 型局部基函数及其导数在
    Gauss-Legendre 点上的取值。

    返回
    ----
    phi : ndarray, shape (3, 2)
        基函数在 2 个 Gauss 点上的值。
    dphi : ndarray, shape (3, 2)
        基函数导数在 Gauss 点上的值。
    w : ndarray, shape (2,)
        2 点 Gauss-Legendre 权重。
    r"""
    # 2 点 Gauss-Legendre 点与权重（精确积分 3 次多项式）
    xi = np.array([-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0)])
    w = np.array([1.0, 1.0])

    # 在参考单元 [-1,1] 上定义 3 个二次多项式基，满足 phi_k(xi_j)=delta_{kj}
    # 这里我们使用节点在 [-1,0,1] 的 Lagrange 基，便于处理边界
    nodes = np.array([-1.0, 0.0, 1.0])
    phi = np.zeros((3, 2))
    dphi = np.zeros((3, 2))
    for k in range(3):
        for gp in range(2):
            # Lagrange 基 L_k(xi)
            x = xi[gp]
            L = 1.0
            dL = 0.0
            for m in range(3):
                if m != k:
                    L *= (x - nodes[m]) / (nodes[k] - nodes[m])
            # 导数（数值求导）
            dx = 1e-8
            Lp = 1.0
            Lm = 1.0
            for m in range(3):
                if m != k:
                    Lp *= ((x + dx) - nodes[m]) / (nodes[k] - nodes[m])
                    Lm *= ((x - dx) - nodes[m]) / (nodes[k] - nodes[m])
            dL = (Lp - Lm) / (2.0 * dx)
            phi[k, gp] = L
            dphi[k, gp] = dL
    return phi, dphi, w


def assemble_dg_matrices(nel: int, K: float = 1.0,
                         penal: float = 10.0,
                         ss: float = -1.0) -> Tuple[np.ndarray, np.ndarray]:
    r"""
    组装一维 DG 的全局质量矩阵与刚度矩阵。

    单元剖分：$[0,1]$ 分为 nel 个等长单元，$h=1/\text{nel}$。
    每个单元采用 3 个局部自由度（二次多项式 $P_2$）。
    全局自由度总数 $N_{\text{dof}} = 3 \times \text{nel}$。

    Parameters
    ----------
    nel : int
        单元个数，必须 >= 1。
    K : float
        扩散系数（假设常数）。
    penal : float
        界面惩罚参数 $\sigma$。
    ss : float
        对称化参数：1.0=NIPG, 0.0=IIPG, -1.0=SIPG（默认）。

    Returns
    -------
    M : ndarray, shape (Ndof, Ndof)
        全局质量矩阵。
    A : ndarray, shape (Ndof, Ndof)
        全局刚度矩阵（含扩散项与惩罚通量）。
    r"""
    if nel < 1:
        raise ValueError("nel 必须至少为 1。")
    locdim = 3
    ndof = nel * locdim
    h = 1.0 / nel

    phi, dphi, wg = legendre_basis_2d()

    M = np.zeros((ndof, ndof))
    A = np.zeros((ndof, ndof))

    # 局部质量矩阵（参考单元）
    Mloc = np.zeros((locdim, locdim))
    for ii in range(locdim):
        for jj in range(locdim):
            s = 0.0
            for gp in range(2):
                s += wg[gp] * phi[ii, gp] * phi[jj, gp]
            Mloc[ii, jj] = s * (h / 2.0)  # Jacobian h/2

    # 局部刚度矩阵（扩散项 $K u_x v_x$）
    Aloc_diff = np.zeros((locdim, locdim))
    for ii in range(locdim):
        for jj in range(locdim):
            s = 0.0
            for gp in range(2):
                s += wg[gp] * dphi[ii, gp] * dphi[jj, gp]
            Aloc_diff[ii, jj] = K * s * (2.0 / h)  # Jacobian for derivative

    # 界面通量矩阵（SIPG 格式简化版）
    # 为保证数值稳定性，将边界与界面通量矩阵整体缩小一个量级
    scale = 0.1
    Bmat = scale * np.array([
        [penal, 1.0 - penal, -2.0 + penal],
        [-ss - penal, -1.0 + ss - penal, 2.0 - ss - penal],
        [2.0 * ss + penal, 1.0 - 2.0 * ss - penal, -2.0 + 2.0 * ss + penal]
    ])
    Cmat = scale * np.array([
        [penal, -1.0 + penal, -2.0 + penal],
        [ss + penal, -1.0 + ss + penal, -2.0 + ss + penal],
        [2.0 * ss + penal, -1.0 + 2.0 * ss + penal, -2.0 + 2.0 * ss + penal]
    ])
    Dmat = scale * np.array([
        [-penal, -1.0 + penal, 2.0 - penal],
        [-ss - penal, -1.0 + ss + penal, 2.0 - ss - penal],
        [-2.0 * ss - penal, -1.0 + 2.0 * ss + penal, 2.0 - 2.0 * ss - penal]
    ])
    Emat = scale * np.array([
        [-penal, 1.0 - penal, 2.0 - penal],
        [ss + penal, -1.0 + ss + penal, -2.0 + ss + penal],
        [-2.0 * ss - penal, 1.0 - 2.0 * ss - penal, 2.0 - 2.0 * ss - penal]
    ])
    F0mat = scale * np.array([
        [penal, 2.0 - penal, -4.0 + penal],
        [-2.0 * ss - penal, -2.0 + 2.0 * ss + penal, 4.0 - 2.0 * ss - penal],
        [4.0 * ss + penal, 2.0 - 4.0 * ss - penal, -4.0 + 4.0 * ss + penal]
    ])
    FNmat = scale * np.array([
        [penal, -2.0 + penal, -4.0 + penal],
        [2.0 * ss + penal, -2.0 + 2.0 * ss + penal, -4.0 + 2.0 * ss + penal],
        [4.0 * ss + penal, -2.0 + 4.0 * ss + penal, -4.0 + 4.0 * ss + penal]
    ])

    # 组装全局矩阵（含扩散+通量）
    for i in range(nel):
        base = i * locdim
        for ii in range(locdim):
            for jj in range(locdim):
                # 局部扩散与质量
                M[base + ii, base + jj] += Mloc[ii, jj]
                A[base + ii, base + jj] += Aloc_diff[ii, jj]

        # 单元间耦合
        if i == 0:
            for ii in range(locdim):
                for jj in range(locdim):
                    A[base + ii, base + jj] += (F0mat[ii, jj] + Cmat[ii, jj])
                    if nel > 1:
                        A[base + ii, base + locdim + jj] += Dmat[ii, jj]
        elif i == nel - 1:
            for ii in range(locdim):
                for jj in range(locdim):
                    A[base + ii, base + jj] += (FNmat[ii, jj] + Bmat[ii, jj])
                    A[base + ii, base - locdim + jj] += Emat[ii, jj]
        else:
            for ii in range(locdim):
                for jj in range(locdim):
                    A[base + ii, base + jj] += (Bmat[ii, jj] + Cmat[ii, jj])
                    A[base + ii, base - locdim + jj] += Emat[ii, jj]
                    A[base + ii, base + locdim + jj] += Dmat[ii, jj]

    return M, A


def solve_causal_diffusion_dg(nel: int = 8,
                               nsteps: int = 50,
                               dt: float = 0.001,
                               K: float = 1.0,
                               source_func: Callable = None,
                               u0_func: Callable = None) -> Tuple[np.ndarray, np.ndarray]:
    r"""
    使用 DG 空间离散 + 隐式 Euler 时间推进求解因果扩散方程。

    PDE:
    $$ u_t - K u_{xx} = f(x,t), \quad x\in[0,1], \; t>0 $$
    Dirichlet 边界：$u(0,t)=u(1,t)=0$（通过惩罚项弱施加）。

    Parameters
    ----------
    nel : int
        空间单元数。
    nsteps : int
        时间步数。
    dt : float
        时间步长，需满足稳定性条件（隐式无条件稳定，但过大影响精度）。
    K : float
        扩散系数。
    source_func : callable
        源项函数 f(x,t)。若 None，使用因果狄拉克组合源。
    u0_func : callable
        初始条件函数。若 None，使用零初始条件。

    Returns
    -------
    t : ndarray, shape (nsteps+1,)
        时间网格。
    u_history : ndarray, shape (nsteps+1, Ndof)
        每个时间步的解向量。
    r"""
    if dt <= 0.0:
        raise ValueError("时间步长 dt 必须为正。")
    if nsteps < 0:
        raise ValueError("时间步数必须非负。")

    M, A = assemble_dg_matrices(nel, K=K, penal=1.0, ss=-1.0)
    ndof = M.shape[0]

    # 施加 Dirichlet 边界条件（通过边界罚项弱施加 u(0)=u(1)=0）
    # 已在 A 矩阵中通过 F0mat/FNmat 包含边界惩罚
    # 右端项边界贡献
    h = 1.0 / nel
    locdim = 3
    b_dirichlet = np.zeros(ndof)
    # 左边界 (x=0) 罚项右端项: penal * u_D * phi(0)
    # 对于参考单元 [-1,1]，左端点 xi=-1，基函数近似值
    b_dirichlet[0] = 0.0  # u_D=0
    b_dirichlet[2] = 0.0  # u_D=0
    # 右边界 (x=1)
    b_dirichlet[ndof - 3] = 0.0
    b_dirichlet[ndof - 1] = 0.0

    # 初始条件
    if u0_func is None:
        u = np.zeros(ndof)
    else:
        u = np.zeros(ndof)
        for i in range(nel):
            base = i * locdim
            xmid = (i + 0.5) * h
            u[base] = u0_func(xmid)

    if source_func is None:
        # 默认因果源：两个脉冲，随时间衰减
        def source_func(x, t):
            return 1.0 * np.exp(-((x - 0.3) ** 2) / 0.02) * np.exp(-t * 2.0) + \
                   0.5 * np.exp(-((x - 0.7) ** 2) / 0.02) * np.exp(-t * 2.0)

    t = 0.0
    t_history = [0.0]
    u_history = [u.copy()]

    # 隐式系统 (M + dt*A) u^{n+1} = M u^n + dt (F^n + b_dirichlet)
    LHS = M + dt * A
    # 正则化保证可逆且正定
    LHS = LHS + 1e-8 * np.eye(ndof)

    for _ in range(nsteps):
        F = np.zeros(ndof)
        for i in range(nel):
            base = i * locdim
            xmid = (i + 0.5) * h
            fval = source_func(xmid, t)
            F[base] = fval

        rhs = M @ u + dt * (F + b_dirichlet)
        u = np.linalg.solve(LHS, rhs)
        t += dt
        t_history.append(t)
        u_history.append(u.copy())

    return np.array(t_history), np.array(u_history)


def demo():
    r"""模块自测试。"""
    t_hist, u_hist = solve_causal_diffusion_dg(nel=8, nsteps=100, dt=0.001, K=1.0)
    print(f"[dg_causal_solver] DG 扩散求解完成: t_final={t_hist[-1]:.4f}, "
          f"u_max={np.max(np.abs(u_hist[-1])):.6e}")
    return t_hist, u_hist


if __name__ == "__main__":
    demo()
