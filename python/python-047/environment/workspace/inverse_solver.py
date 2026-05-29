"""
inverse_solver.py
重力异常反演与正则化求解模块

本模块实现从观测重力异常到深部密度结构的反演算法。
核心问题是求解第一类Fredholm积分方程的离散化形式：

    d = G * m + epsilon

其中：
  - d: (N_obs,) 观测重力异常向量 [mGal]
  - G: (N_obs, N_param) 格林函数/灵敏度矩阵
  - m: (N_param,) 待求密度异常参数 [kg/m^3]
  - epsilon: 观测噪声

由于问题的病态性（ill-posed），需要引入正则化：

    min_m { ||G*m - d||^2 + alpha^2 * ||L*m||^2 }

其正规方程为：
    (G^T G + alpha^2 L^T L) * m = G^T d

融合以下种子项目的核心算法：
  - 1003_r8utt：上三角Toeplitz矩阵快速求解
  - 056_asa314：模运算矩阵求逆（用于整数约束预条件子）
  - 444_football_dynamic：组合计数用于正则化路径评估
"""

import numpy as np
from matrix_kernels import r8utt_solve, r8utt_to_dense, r8utt_inverse, \
    toeplitz_matvec, tikhonov_preconditioner_toeplitz, football_combination_count, \
    sparse_approximate_inverse_mod


def build_sensitivity_matrix(obs_points, grid_centers, grid_volumes, use_toeplitz=False):
    """
    构造重力异常的灵敏度（格林函数）矩阵。
    
    对于棱柱体网格，元素 G[i,j] 表示第 j 个网格单元对第 i 个观测点的
    重力异常贡献（垂直分量）：
        G[i,j] = G_const * V_j * (z_i - z_j) / r_{ij}^3 * 1e5 [mGal / (kg/m^3)]
    
    参数：
        obs_points: (N_obs, 3) 观测点坐标 [m]
        grid_centers: (N_param, 3) 网格单元中心 [m]
        grid_volumes: (N_param,) 网格单元体积 [m^3]
        use_toeplitz: 是否返回Toeplitz结构描述而非完整矩阵
    返回：
        G: (N_obs, N_param) 灵敏度矩阵 或 (N_param,) Toeplitz第一行
    """
    G_CONST = 6.67430e-11
    
    obs = np.asarray(obs_points, dtype=float)
    centers = np.asarray(grid_centers, dtype=float)
    vols = np.asarray(grid_volumes, dtype=float)
    
    N_obs = obs.shape[0]
    N_param = centers.shape[0]
    
    if use_toeplitz:
        # 假设均匀网格，只返回第一行
        first_row = np.zeros(N_param)
        for j in range(N_param):
            dx = obs[0, 0] - centers[j, 0]
            dy = obs[0, 1] - centers[j, 1]
            dz = obs[0, 2] - centers[j, 2]
            r = np.sqrt(dx**2 + dy**2 + dz**2)
            r = max(r, 1e-6)
            first_row[j] = G_CONST * vols[j] * dz / (r**3) * 1e5
        return first_row
    
    # TODO(Hole_2): 实现重力灵敏度矩阵的格林函数核计算
    # 物理公式：
    #   G[i,j] = G_const * V_j * (z_i - z_j) / r_{ij}^3 * 1e5
    # 其中：
    #   - G_const = 6.67430e-11 [m^3 kg^-1 s^-2] 为万有引力常数
    #   - V_j 为第 j 个网格单元的体积 [m^3]
    #   - dz = obs[i,2] - centers[j,2] 为垂向距离 [m]
    #   - r_{ij} = |obs_i - center_j| 为欧氏距离 [m]
    #   - 1e5 为单位转换因子（m/s^2 -> mGal）
    # 注意：
    #   1. 需避免 r_{ij} -> 0 时的除零（最小截断 1e-6）
    #   2. 输出矩阵维度为 (N_obs, N_param)
    #   3. 该公式与 forward_model.py 中 prism_gravity_anomaly 的物理核一致
    raise NotImplementedError("Hole_2: build_sensitivity_matrix 格林函数核待实现")


def tikhonov_solve_dense(G, d, alpha, order=1):
    """
    稠密矩阵的Tikhonov正则化求解。
    
    正则化方程：
        (G^T G + alpha^2 L^T L) m = G^T d
    
    差分正则化矩阵 L（一阶）：
        L_{i,i} = 1, L_{i,i+1} = -1 (对于 i < n-1)
    
    参数：
        G: (N_obs, N_param) 灵敏度矩阵
        d: (N_obs,) 观测数据
        alpha: 正则化参数
        order: 差分阶数 (1 或 2)
    返回：
        m: (N_param,) 反演密度异常
        residual: 残差范数
        reg_term: 正则化项范数
    """
    G = np.asarray(G, dtype=float)
    d = np.asarray(d, dtype=float)
    N_obs, N_param = G.shape
    
    # 构造正则化矩阵 L
    if order == 1:
        L = np.zeros((N_param - 1, N_param))
        for i in range(N_param - 1):
            L[i, i] = 1.0
            L[i, i + 1] = -1.0
    elif order == 2:
        L = np.zeros((N_param - 2, N_param))
        for i in range(N_param - 2):
            L[i, i] = 1.0
            L[i, i + 1] = -2.0
            L[i, i + 2] = 1.0
    else:
        raise ValueError("order must be 1 or 2")
    
    # 正规方程矩阵
    A = G.T @ G + alpha**2 * (L.T @ L)
    b = G.T @ d
    
    # 求解（使用SVD以处理病态性）
    try:
        # 尝试直接求解
        m = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        # 奇异时使用伪逆
        m = np.linalg.lstsq(A, b, rcond=1e-10)[0]
    
    residual = np.linalg.norm(G @ m - d)
    reg_term = np.linalg.norm(L @ m)
    
    return m, residual, reg_term


def tikhonov_solve_toeplitz(green_row, d, alpha, grid_shape, dx, dy, dz):
    """
    利用Toeplitz结构加速的Tikhonov正则化求解。
    
    融合 1003_r8utt 的核心算法。
    
    对于规则网格，G^T G 具有Toeplitz-like结构。
    这里使用一维等价Toeplitz矩阵近似求解。
    
    参数：
        green_row: (N_param,) 格林函数Toeplitz第一行
        d: (N_obs,) 观测数据
        alpha: 正则化参数
        grid_shape: (nx, ny, nz)
        dx, dy, dz: 网格间距
    返回：
        m: (N_param,) 反演结果
    """
    N_param = len(green_row)
    
    # 构造近似Toeplitz正规方程
    # A = G^T G + alpha^2 I
    a = np.zeros(N_param)
    for lag in range(N_param):
        s = 0.0
        for k in range(N_param - lag):
            s += green_row[k] * green_row[k + lag]
        a[lag] = s
    a[0] += alpha**2
    
    # 右端项 b = G^T d（近似）
    # 假设所有观测点等效，b_j = sum_i G[i,j] * d[i]
    b = np.zeros(N_param)
    for j in range(N_param):
        b[j] = green_row[j] * np.sum(d)
    
    # 使用Toeplitz矩阵求解
    # 将上三角Toeplitz近似用于后向替换
    try:
        m = r8utt_solve(N_param, a, b)
    except ValueError:
        a[0] += 1e-12
        m = r8utt_solve(N_param, a, b)
    
    return m


def l_curve_criterion(G, d, alphas, order=1):
    """
    使用L曲线准则选择最优正则化参数。
    
    L曲线绘制 (log||G*m - d||, log||L*m||) 随 alpha 的变化。
    最优 alpha 位于曲线拐角处（最大曲率）。
    
    参数：
        G: 灵敏度矩阵
        d: 观测数据
        alphas: 待测试的正则化参数列表
        order: 差分阶数
    返回：
        best_alpha: 最优正则化参数
        residuals: 各alpha对应的残差
        reg_terms: 各alpha对应的正则化项
        curvatures: 曲率
    """
    residuals = []
    reg_terms = []
    
    for alpha in alphas:
        m, res, reg = tikhonov_solve_dense(G, d, alpha, order)
        residuals.append(res)
        reg_terms.append(reg)
    
    residuals = np.array(residuals)
    reg_terms = np.array(reg_terms)
    
    # 对数坐标
    log_res = np.log10(residuals + 1e-15)
    log_reg = np.log10(reg_terms + 1e-15)
    
    # 数值计算曲率（简化）
    n = len(alphas)
    curvatures = np.zeros(n)
    for i in range(1, n - 1):
        dx1 = log_res[i] - log_res[i - 1]
        dy1 = log_reg[i] - log_reg[i - 1]
        dx2 = log_res[i + 1] - log_res[i]
        dy2 = log_reg[i + 1] - log_reg[i]
        
        ds1 = np.sqrt(dx1**2 + dy1**2)
        ds2 = np.sqrt(dx2**2 + dy2**2)
        
        if ds1 > 1e-15 and ds2 > 1e-15:
            ddx = dx2 / ds2 - dx1 / ds1
            ddy = dy2 / ds2 - dy1 / ds1
            curvatures[i] = abs(ddx * dy1 / ds1 - ddy * dx1 / ds1)
    
    best_idx = np.argmax(curvatures)
    best_alpha = alphas[best_idx]
    
    return best_alpha, residuals, reg_terms, curvatures


def gcv_criterion(G, d, alphas, order=1):
    """
    广义交叉验证（GCV）准则选择正则化参数。
    
    GCV函数：
        GCV(alpha) = ||G*m - d||^2 / (N - tr(H))^2
    其中 H = G * (G^T G + alpha^2 L^T L)^{-1} * G^T 是hat矩阵。
    
    最优 alpha 使 GCV(alpha) 最小。
    """
    N_obs = G.shape[0]
    gcv_values = []
    residuals = []
    
    for alpha in alphas:
        m, res, reg = tikhonov_solve_dense(G, d, alpha, order)
        residuals.append(res)
        
        # 估计 hat 矩阵的迹（简化估计）
        # tr(H) ~ N_obs * sum_j (s_j^2 / (s_j^2 + alpha^2)) / N_param
        # 使用SVD奇异值的简化估计
        try:
            s = np.linalg.svd(G, compute_uv=False)
            s = s[s > 1e-12]
            trace_h = np.sum(s**2 / (s**2 + alpha**2))
        except:
            trace_h = N_obs * 0.5
        
        denom = max(N_obs - trace_h, 1e-3)
        gcv = res**2 / (denom**2)
        gcv_values.append(gcv)
    
    best_idx = np.argmin(gcv_values)
    best_alpha = alphas[best_idx]
    
    return best_alpha, np.array(gcv_values), np.array(residuals)


def iterative_tikhonov_cg(G, d, alpha, order=1, max_iter=500, tol=1e-6):
    """
    使用共轭梯度法迭代求解大规模Tikhonov正则化问题。
    
    求解：
        (G^T G + alpha^2 L^T L) m = G^T d
    
    CG算法避免显式构造 A = G^T G + alpha^2 L^T L，
    只需矩阵-向量乘积，适合大规模问题。
    
    参数：
        G: (N_obs, N_param)
        d: (N_obs,)
        alpha: 正则化参数
        order: 差分阶数
        max_iter: 最大迭代次数
        tol: 收敛容差
    返回：
        m: (N_param,) 解
        iterations: 实际迭代次数
        residual_norms: 残差范数历史
    """
    G = np.asarray(G, dtype=float)
    d = np.asarray(d, dtype=float)
    N_obs, N_param = G.shape
    
    # 构造正则化矩阵作用函数
    def apply_L(v):
        if order == 1:
            Lv = np.zeros(N_param - 1)
            for i in range(N_param - 1):
                Lv[i] = v[i] - v[i + 1]
            return Lv
        else:
            Lv = np.zeros(N_param - 2)
            for i in range(N_param - 2):
                Lv[i] = v[i] - 2.0 * v[i + 1] + v[i + 2]
            return Lv
    
    def apply_LT(w):
        if order == 1:
            LTw = np.zeros(N_param)
            LTw[0] = w[0]
            for i in range(1, N_param - 1):
                LTw[i] = w[i] - w[i - 1]
            LTw[-1] = -w[-1]
            return LTw
        else:
            LTw = np.zeros(N_param)
            LTw[0] = w[0]
            LTw[1] = -2.0 * w[0] + w[1]
            for i in range(2, N_param - 2):
                LTw[i] = w[i - 2] - 2.0 * w[i - 1] + w[i]
            LTw[-2] = w[-3] - 2.0 * w[-2]
            LTw[-1] = w[-2]
            return LTw
    
    def matvec(v):
        return G.T @ (G @ v) + alpha**2 * apply_LT(apply_L(v))
    
    b = G.T @ d
    m = np.zeros(N_param)
    r = b - matvec(m)
    p = r.copy()
    
    residual_norms = [np.linalg.norm(r)]
    
    for k in range(max_iter):
        Ap = matvec(p)
        rTr = np.dot(r, r)
        pAp = np.dot(p, Ap)
        
        if abs(pAp) < 1e-15:
            break
        
        alpha_cg = rTr / pAp
        m = m + alpha_cg * p
        r_new = r - alpha_cg * Ap
        rTr_new = np.dot(r_new, r_new)
        
        residual_norms.append(np.sqrt(rTr_new))
        
        if np.sqrt(rTr_new) < tol * np.linalg.norm(b):
            break
        
        beta_cg = rTr_new / rTr
        p = r_new + beta_cg * p
        r = r_new
    
    return m, len(residual_norms), np.array(residual_norms)


def resolution_matrix_analysis(G, alpha, order=1):
    """
    计算反演分辨率矩阵和数据分辨率矩阵。
    
    模型分辨率矩阵：
        R_m = (G^T G + alpha^2 L^T L)^{-1} G^T G
    数据分辨率矩阵：
        R_d = G (G^T G + alpha^2 L^T L)^{-1} G^T
    
    R_m 接近单位矩阵表示模型参数可较好分辨。
    """
    N_obs, N_param = G.shape
    
    if order == 1:
        L = np.zeros((N_param - 1, N_param))
        for i in range(N_param - 1):
            L[i, i] = 1.0
            L[i, i + 1] = -1.0
    else:
        L = np.zeros((N_param - 2, N_param))
        for i in range(N_param - 2):
            L[i, i] = 1.0
            L[i, i + 1] = -2.0
            L[i, i + 2] = 1.0
    
    A = G.T @ G + alpha**2 * (L.T @ L)
    try:
        A_inv = np.linalg.inv(A)
    except np.linalg.LinAlgError:
        A_inv = np.linalg.pinv(A)
    
    R_m = A_inv @ (G.T @ G)
    R_d = G @ A_inv @ G.T
    
    # 计算分辨率指标
    spread_m = np.sum((R_m - np.eye(N_param))**2)
    trace_d = np.trace(R_d)
    
    return R_m, R_d, spread_m, trace_d
