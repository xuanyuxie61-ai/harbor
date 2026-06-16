"""
uncertainty_tracking.py — 分布函数演化不确定性追踪 (GPC 思想)
============================================================

种子项目映射: 1262_aybo_gpc-self-location — 生成式预测编码.

原项目核心:
  使用 Kalman 滤波框架追踪自位置的不确定性.
  状态估计: x̂_{k+1} = A x̂_k + K (z_k - H x̂_k)
  协方差更新: P_{k+1} = (I - KH) P_k (I - KH)^T + KRK^T
  关键操作:
    - symmetrise(S) = 0.5(S + S^T)
    - stabilise_cov(S, jitter) — 确保协方差正定

本项目映射:
  将 GPC 的 Kalman 滤波框架应用于 Fokker-Planck 方程的
  不确定性量化 (UQ).

  状态向量: 分布函数 f(v) 的离散值 = [f_1, ..., f_N]
  预测步: f̂_{n+1} = f̂_n + Δt · C[f̂_n]  (FP 时间推进)
  观测: f 的矩 (n, T) 可精确计算
  不确定性: Σ = Cov(f) — 分布函数的协方差矩阵

  物理含义:
  - 协方差矩阵的对角元 Σ_{ii} = Var(f(v_i)) 给出每个速度点的方差
  - 非对角元 Σ_{ij} = Cov(f(v_i), f(v_j)) 给出速度点间的相关性
  - 追踪 Σ 的演化可以量化碰撞过程中不确定性的传播和衰减

  关键创新:
  - 碰撞算子的耗散性 → 不确定性应随时间衰减
  - 协方差稳定化 → 确保 Σ 始终正定 (映射自 stabilise_cov)
  - 精度权重 → 不同速度区域的不确定性权重不同 (映射自 π_ego/π_allo)
"""

import numpy as np


# ===========================================================================
#  §1  协方差稳定化  (源自 gpc-self-location)
# ===========================================================================
def symmetrise(S):
    """对称化矩阵: S_sym = 0.5(S + S^T).

    源自 gpc-self-location 中的 symmetrise().
    数值误差可能使协方差矩阵轻微不对称, 需要周期对称化.
    """
    return 0.5 * (S + S.T)


def stabilise_cov(S, jitter=1.0e-7):
    """确保协方差矩阵对称正定.

    源自 gpc-self-location 中的 stabilise_cov():
      1. 对称化
      2. 检查最小特征值
      3. 如果 λ_min < jitter, 添加 (jitter - λ_min) I

    Parameters
    ----------
    S : ndarray(n, n)  协方差矩阵
    jitter : float  最小特征值下界

    Returns
    -------
    S_stable : ndarray(n, n)  稳定化的协方差矩阵
    """
    S = symmetrise(np.array(S, dtype=np.float64))
    eigenvalues = np.linalg.eigvalsh(S)
    min_eig = float(np.min(eigenvalues))

    if min_eig < jitter:
        S = S + np.eye(S.shape[0]) * (jitter - min_eig + 1.0e-9)

    return symmetrise(S)


# ===========================================================================
#  §2  不确定性传播器
# ===========================================================================
class UncertaintyTracker:
    """分布函数演化的不确定性追踪器.

    使用简化的 Kalman 滤波框架:
      预测: f̂_{n+1} = f̂_n + dt · C[f̂_n]
      协方差预测: P_{n+1} = F P_n F^T + Q
      其中 F = I + dt · ∂C/∂f ≈ 切线线性化算子
            Q = 过程噪声 (数值误差模型)

    映射自 GPC 中的自位置不确定性追踪:
      状态 → 分布函数
      转移矩阵 → FP 算子的线性化
      观测 → 矩约束
      精度 → 不同速度区域的可信度
    """

    def __init__(self, N_velocity, initial_covariance_scale=1e-4):
        """
        Parameters
        ----------
        N_velocity : int  速度网格点数
        initial_covariance_scale : float  初始协方差缩放
        """
        self.N = N_velocity
        # 初始协方差: 对角矩阵, 表示独立的速度点不确定性
        self.P = initial_covariance_scale * np.eye(N_velocity)
        self.P = stabilise_cov(self.P)

        # 过程噪声 (数值截断误差模型)
        self.Q_scale = 1e-8
        self.Q = self.Q_scale * np.eye(N_velocity)

        # 精度权重 (映射自 π_ego / π_allo)
        # 高速区域精度较低 (分布函数值小, 相对误差大)
        self.precision = np.ones(N_velocity)

    def set_precision_from_distribution(self, f):
        """基于分布函数值设置精度权重.

        π(v) ∝ f(v) / max(f)  (分布函数值越大, 精度越高)
        低速核心区域: π ≈ 1
        高速尾部: π << 1

        映射自 GPC 中的 ego/allo 精度权重.
        """
        f_max = np.max(f)
        if f_max > 1e-30:
            self.precision = np.maximum(f / f_max, 0.01)
        else:
            self.precision = np.ones(self.N)

    def predict(self, f_predicted, jacobian_approx=None):
        """预测步: 更新协方差.

        P_{n+1} = F P_n F^T + Q

        Parameters
        ----------
        f_predicted : ndarray  预测的分布函数
        jacobian_approx : ndarray(N,N) or None  Jacobian 近似
            如果为 None, 使用 F ≈ I (简单传播)
        """
        if jacobian_approx is not None:
            F = np.eye(self.N) + jacobian_approx
        else:
            # 简化: F ≈ I (假设一步变化小)
            F = np.eye(self.N)

        # 协方差传播
        self.P = F @ self.P @ F.T + self.Q

        # 精度加权
        pi_diag = np.diag(self.precision)
        self.P = pi_diag @ self.P @ pi_diag

        # 稳定化
        self.P = stabilise_cov(self.P)

        # 更新精度
        self.set_precision_from_distribution(f_predicted)

    def get_uncertainty(self):
        """获取当前不确定性信息.

        Returns
        -------
        info : dict
        """
        diag = np.diag(self.P)
        eigenvalues = np.linalg.eigvalsh(self.P)
        return {
            "variance_per_point": diag,
            "mean_variance": float(np.mean(diag)),
            "max_variance": float(np.max(diag)),
            "min_variance": float(np.min(diag)),
            "total_uncertainty": float(np.trace(self.P)),
            "log_determinant": float(np.sum(np.log(np.maximum(eigenvalues, 1e-30)))),
            "condition_number": float(np.max(eigenvalues) / max(np.min(eigenvalues), 1e-30)),
        }

    def confidence_interval(self, f, confidence=0.95):
        """计算分布函数的置信区间.

        假设 Gaussian 不确定性:
          f ± z_{α/2} · √diag(P)

        Parameters
        ----------
        f : ndarray  当前分布函数估计
        confidence : float  置信水平 (默认 95%)

        Returns
        -------
        lower, upper : ndarray  置信区间上下界
        """
        from scipy.stats import norm
        z = norm.ppf(0.5 + confidence / 2.0)
        std = np.sqrt(np.maximum(np.diag(self.P), 0.0))
        return f - z * std, f + z * std


# ===========================================================================
#  §3  不确定性诊断
# ===========================================================================
def uncertainty_diagnosis(tracker, f_current):
    """综合不确定性诊断.

    Returns
    -------
    report : dict
    """
    unc = tracker.get_uncertainty()
    lower, upper = tracker.confidence_interval(f_current)

    # 检测不确定度异常增长
    mean_var = unc["mean_variance"]
    cond = unc["condition_number"]

    warnings = []
    if cond > 1e6:
        warnings.append(f"协方差条件数过大: {cond:.2e} (> 1e6)")
    if mean_var > 0.1:
        warnings.append(f"平均方差过大: {mean_var:.4e} (> 0.1)")

    return {
        "uncertainty_metrics": unc,
        "confidence_lower": lower,
        "confidence_upper": upper,
        "warnings": warnings,
    }
