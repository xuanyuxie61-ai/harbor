"""
utils.py
SLAM 系统通用工具函数

包含：
- SE(2) 李群/李代数运算
- 数值稳定性工具
- 统计检验函数
- 日志与输出格式化
"""

import numpy as np


def se2_exp(v):
    """
    SE(2) 指数映射
    
    v = [vx, vy, vtheta]
    
    对于小角度：exp(v) ≈ I + v^∧
    对于一般角度：
    T = [[cosθ, -sinθ, Vx],
         [sinθ,  cosθ, Vy],
         [0,     0,    1 ]]
    
    其中：
    若 |vtheta| < epsilon：
       Vx = vx, Vy = vy
    否则：
       Vx = (vx*sinθ + vy*(cosθ-1)) / θ
       Vy = (vy*sinθ + vx*(1-cosθ)) / θ
    """
    vx, vy, vtheta = v
    theta = vtheta

    if abs(theta) < 1e-8:
        Vx = vx
        Vy = vy
    else:
        s = np.sin(theta)
        c = np.cos(theta)
        Vx = (vx * s + vy * (c - 1.0)) / theta
        Vy = (vy * s + vx * (1.0 - c)) / theta

    c = np.cos(theta)
    s = np.sin(theta)
    return np.array([[c, -s, Vx],
                     [s,  c, Vy],
                     [0.0, 0.0, 1.0]], dtype=np.float64)


def se2_log(T):
    """
    SE(2) 对数映射
    
    T = [[R, t], [0, 1]]
    θ = atan2(R[1,0], R[0,0])
    
    若 |θ| < epsilon：
       v = [t_x, t_y, θ]
    否则：
       A = sinθ/θ, B = (1-cosθ)/θ
       V = 1/θ * [[A, B], [-B, A]] * t
    """
    R = T[0:2, 0:2]
    t = T[0:2, 2]
    theta = np.arctan2(R[1, 0], R[0, 0])

    if abs(theta) < 1e-8:
        vx = t[0]
        vy = t[1]
    else:
        A = np.sin(theta) / theta
        B = (1.0 - np.cos(theta)) / theta
        det = A * A + B * B
        if abs(det) < 1e-15:
            vx = t[0]
            vy = t[1]
        else:
            vx = (A * t[0] + B * t[1]) / det
            vy = (-B * t[0] + A * t[1]) / det

    return np.array([vx, vy, theta], dtype=np.float64)


def normalize_angle(angle):
    """归一化角度到 [-pi, pi]"""
    while angle > np.pi:
        angle -= 2.0 * np.pi
    while angle < -np.pi:
        angle += 2.0 * np.pi
    return angle


def is_positive_semidefinite(M, tol=1e-10):
    """检验矩阵是否半正定"""
    M = np.asarray(M, dtype=np.float64)
    if not np.allclose(M, M.T, atol=tol):
        return False
    eigvals = np.linalg.eigvalsh(M)
    return np.min(eigvals) >= -tol


def nearest_positive_semidefinite(M):
    """计算最近的半正定矩阵（谱投影）"""
    M = np.asarray(M, dtype=np.float64)
    M_sym = 0.5 * (M + M.T)
    eigvals, eigvecs = np.linalg.eigh(M_sym)
    eigvals = np.maximum(eigvals, 0.0)
    return eigvecs @ np.diag(eigvals) @ eigvecs.T


def mahalanobis_distance(x, mu, Sigma):
    """
    马氏距离
    
    d_M(x, μ) = sqrt( (x-μ)^T Σ^{-1} (x-μ) )
    """
    diff = np.asarray(x) - np.asarray(mu)
    try:
        Sigma_inv = np.linalg.inv(Sigma)
    except np.linalg.LinAlgError:
        Sigma_inv = np.linalg.pinv(Sigma)
    return np.sqrt(max(diff @ Sigma_inv @ diff, 0.0))


def chi2_confidence_interval(dim, confidence=0.95):
    """
    卡方分布置信区间（近似值）
    
    对于维度 dim，置信水平 confidence 的阈值
    """
    # 使用 Wilson-Hilferty 变换近似
    # χ²_dim ≈ dim * (1 - 2/(9*dim) + z*sqrt(2/(9*dim)))^3
    from math import sqrt
    # 标准正态分位数（95% 约 1.96）
    z = 1.96 if confidence == 0.95 else 2.576 if confidence == 0.99 else 1.645
    if dim <= 0:
        return 0.0
    approx = dim * (1.0 - 2.0 / (9.0 * dim) + z * sqrt(2.0 / (9.0 * dim))) ** 3
    return max(approx, 0.0)


def compute_trajectory_ate(estimated, ground_truth):
    """
    计算绝对轨迹误差 (ATE)
    
    ATE = sqrt( 1/N Σ ||trans(T_est_i) - trans(T_gt_i)||^2 )
    """
    est = np.asarray(estimated, dtype=np.float64)
    gt = np.asarray(ground_truth, dtype=np.float64)
    if est.shape != gt.shape:
        raise ValueError("shape mismatch")
    if est.ndim == 1:
        est = est.reshape(1, -1)
        gt = gt.reshape(1, -1)
    diffs = est[:, 0:2] - gt[:, 0:2]
    rmse = np.sqrt(np.mean(np.sum(diffs ** 2, axis=1)))
    return rmse


def robust_loss(residual, huber_delta=1.0):
    """
    Huber 鲁棒损失函数
    
    ρ(r) = 0.5 * r^2          , |r| <= δ
           δ * (|r| - 0.5*δ)  , |r| > δ
    """
    r = abs(float(residual))
    d = float(huber_delta)
    if r <= d:
        return 0.5 * r * r
    else:
        return d * (r - 0.5 * d)


def format_matrix_latex(M, name="M", precision=4):
    """格式化矩阵为 LaTeX 字符串（用于文档生成）"""
    rows = []
    for row in M:
        rows.append(" & ".join(f"{v:.{precision}f}" for v in row))
    body = " \\\\\n".join(rows)
    return f"\\[{name} = \\begin{{bmatrix}}\n{body}\n\\end{{bmatrix}}\\]"
