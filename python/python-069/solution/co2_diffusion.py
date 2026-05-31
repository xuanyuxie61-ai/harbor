"""
CO2 反应扩散模块：基于 fem2d_predator_prey_fast 思想，
将冠层内 CO2 扩散-吸收过程建模为反应扩散系统。

核心方程（类比 predator-prey）：
  CO2 浓度 C(x,y,t) 类比 prey，光合吸收项类比 predator。

  dC/dt = D * nabla^2 C + R_soil - V_max * C / (K_m + C) * LAI(x,y)

其中：
  D: CO2 湍流扩散系数 (m^2/s)
  R_soil: 土壤呼吸源项
  V_max * C / (K_m + C): Michaelis-Menten 型光合吸收
  LAI(x,y): 空间分布的叶面积指数

空间离散：有限差分，周期边界条件。
时间离散：向后 Euler + 固定点迭代。
"""
import numpy as np
from scipy.sparse import diags, csr_matrix
from scipy.sparse.linalg import spsolve


def build_laplacian_2d(J, h):
    """
    构建二维 Laplacian 矩阵（5 点格式），周期边界条件。
    J: 每维网格点数
    h: 空间步长
    """
    n = J * J
    mu = 1.0 / (h ** 2)
    # 主对角线
    main = np.full(n, -4.0)
    # 左右
    off1 = np.ones(n - 1)
    for i in range(1, J):
        off1[i * J - 1] = 0.0
    # 上下
    offJ = np.ones(n - J)
    # 周期边界
    # x 方向周期
    px = np.zeros(n)
    for i in range(J):
        px[i * J] = 1.0
        px[i * J + J - 1] = 1.0
    # y 方向周期
    py = np.zeros(n)
    py[:J] = 1.0
    py[n - J:] = 1.0

    diagonals = [main, off1, off1, offJ, offJ, px, py]
    offsets = [0, -1, 1, -J, J, -(J - 1), J - 1]
    L = diags(diagonals, offsets, shape=(n, n), format='lil')
    # 修正周期连接
    for i in range(J):
        L[i * J, i * J + J - 1] = 1.0
        L[i * J + J - 1, i * J] = 1.0
    for j in range(J):
        L[j, n - J + j] = 1.0
        L[n - J + j, j] = 1.0
    return mu * L.tocsr()


def co2_diffusion_solver(J, h, D, dt, n_steps, C0, R_soil, V_max, K_m, LAI_grid):
    """
    求解 CO2 反应扩散方程。
    J: 每维网格点数
    h: 空间步长 (m)
    D: 扩散系数 (m^2/s)
    dt: 时间步长 (s)
    n_steps: 时间步数
    C0: 初始 CO2 浓度 (umol/mol)
    R_soil: 土壤呼吸 (umol/m^2/s)
    V_max: 最大吸收速率
    K_m: Michaelis 常数
    LAI_grid: (J,J) 叶面积指数空间分布
    返回: C 随时间的演化列表
    """
    n = J * J
    C = np.full(n, C0, dtype=float)
    L = build_laplacian_2d(J, h)
    I_mat = diags([np.ones(n)], [0], shape=(n, n), format='csr')
    B = I_mat - dt * D * L

    LAI_vec = LAI_grid.ravel()
    results = [C.copy()]

    for _ in range(n_steps):
        # 非线性吸收项：Picard 迭代线性化
        C_old = C.copy()
        for _ in range(3):
            absorption = V_max * C_old / (K_m + np.maximum(C_old, 1e-3)) * LAI_vec
            rhs = C + dt * (R_soil - absorption)
            C_new = spsolve(B, rhs)
            C_new = np.clip(C_new, 380.0, 2000.0)
            if np.max(np.abs(C_new - C_old)) < 1e-3:
                break
            C_old = C_new
        C = C_new
        results.append(C.copy())
    return results
