"""
kalman_engine.py — 扩展 Kalman 滤波与平滑引擎
===============================================

本模块实现高能物理径迹重建的核心 Kalman 滤波算法:

    1. 前向滤波 (Forward Filter)
    2. 后向平滑 (Backward Smoother) — 基于 [064] 的向后 Euler 思想
    3. 参数更新与 χ² 递推

状态向量: x = (κ, tan λ, φ, d, z₀)^T
    κ     = q/p_T  横向曲率 [1/mm]
    tan λ         偶极角正切
    φ             方位角 [rad]
    d             横向冲击参数 [mm]
    z₀            纵向冲击参数 [mm]

Kalman 滤波递推:

[Prediction]
    x̃_{k|k-1} = f(x̂_{k-1})                    (状态传播)
    C̃_{k|k-1} = F_k · Ĉ_{k-1} · F_k^T + Q_k  (协方差传播)

[Update]
    r_k = m_k - H_k · x̃_{k|k-1}               (残差/创新)
    G_k = H_k · C̃_{k|k-1} · H_k^T + V_k       (创新协方差)
    K_k = C̃_{k|k-1} · H_k^T · G_k^{-1}         (Kalman 增益)
    x̂_k = x̃_{k|k-1} + K_k · r_k               (状态更新)
    Ĉ_k = (I - K_k · H_k) · C̃_{k|k-1}          (协方差更新)
    χ²_k += r_k^T · G_k^{-1} · r_k              (χ² 递推)

[Joseph 形式 (数值稳定)]
    Ĉ_k = (I - K_k · H_k) · C̃ · (I - K_k · H_k)^T + K_k · V_k · K_k^T

[Backward Smoothing]
    x̃_{k|N} = x̂_k + A_k · (x̃_{k+1|N} - x̃_{k+1|k})
    其中 A_k = Ĉ_k · F_{k+1}^T · C̃_{k+1|k}^{-1}  (平滑增益)
"""

import math
import numpy as np
from typing import List, Dict, Optional, Tuple


# ============================================================
# 状态向量操作
# ============================================================
class TrackState:
    """
    5维径迹状态向量

    x = (κ, tan_lambda, phi, d, z0)

    提供:
    - 状态传播 (在磁场中)
    - 测量预测 (从状态到击中位置)
    - 雅可比矩阵计算
    """

    def __init__(self, params, covariance=None):
        """
        Parameters
        ----------
        params : array_like, shape (5,)
            [kappa, tan_lambda, phi, d, z0]
        covariance : array_like, shape (5, 5), optional
            协方差矩阵
        """
        self.params = np.array(params, dtype=float)
        if covariance is not None:
            self.covariance = np.array(covariance, dtype=float)
        else:
            self.covariance = np.eye(5) * 1e-4

    @property
    def kappa(self):
        return self.params[0]

    @property
    def tan_lambda(self):
        return self.params[1]

    @property
    def phi(self):
        return self.params[2]

    @property
    def d(self):
        return self.params[3]

    @property
    def z0(self):
        return self.params[4]

    @property
    def pt_gev(self):
        """横向动量 [GeV/c]"""
        if abs(self.kappa) < 1e-15:
            return float('inf')
        return abs(0.3 * 2.0 / (self.kappa * 1e3))  # B=2T, κ in 1/mm

    @property
    def theta(self):
        """极角 θ [rad]"""
        return math.atan2(1.0, max(self.tan_lambda, -10.0))

    @property
    def eta(self):
        """赝快度 η = -ln(tan(θ/2))"""
        theta = self.theta
        if theta < 1e-6 or theta > math.pi - 1e-6:
            return 10.0  # 极限值
        return -math.log(math.tan(theta / 2.0))

    def copy(self):
        """深拷贝"""
        return TrackState(
            self.params.copy(),
            self.covariance.copy()
        )

    def predict_to_layer(self, current_radius, target_radius, bfield_t=2.0):
        """
        将状态从当前半径传播到目标半径

        在均匀场中，螺旋线径迹的参数变化:
            Δs ≈ Δr / cos(λ)          (路径长度)
            Δφ ≈ κ · Δs              (方位角变化)
            Δd ≈ 0                    (冲击参数在传播中不变)
            Δz₀ ≈ tan(λ) · Δs        (纵向位置变化)

        更精确的传播使用圆弧公式:
            Δφ = arcsin(κ · Δr / (1 + κ · d))
            新 φ = φ + Δφ
            新 d = d + (1 - cos(Δφ)) / κ  (近似)

        Parameters
        ----------
        current_radius : float
            当前径向位置 [mm]
        target_radius : float
            目标径向位置 [mm]
        bfield_t : float
            磁场 [T]

        Returns
        -------
        tuple : (new_state, jacobian_F, process_noise_Q)
        """
        dr = target_radius - current_radius
        kappa = self.kappa
        tan_lam = self.tan_lambda
        phi = self.phi

        # 防止除零
        if abs(kappa) < 1e-12:
            kappa_safe = 1e-12 * (1.0 if kappa >= 0 else -1.0)
        else:
            kappa_safe = kappa

        # 传播 (圆弧近似)
        # Δφ = arcsin(κ·Δr / (1 + κ·d))
        denom = 1.0 + kappa * self.d
        if abs(denom) < 1e-10:
            denom = 1e-10 * (1.0 if denom >= 0 else -1.0)

        sin_dphi = kappa * dr / denom
        sin_dphi = max(-1.0, min(1.0, sin_dphi))  # 限制在 [-1, 1]
        dphi = math.asin(sin_dphi)

        # 路径长度
        ds = dr / max(math.cos(math.atan(tan_lam)), 0.01)

        # 新参数
        new_kappa = kappa  # 均匀场中曲率不变
        new_tan_lambda = tan_lam  # 均匀场中偶极角不变
        new_phi = phi + dphi
        new_d = self.d + dr  # 简化
        new_z0 = self.z0 + tan_lam * ds

        new_params = np.array([new_kappa, new_tan_lambda, new_phi, new_d, new_z0])

        # 雅可比矩阵 F = ∂x_new/∂x_old
        F = np.eye(5)
        # ∂(new_phi)/∂(kappa) = dr / (denom · cos(dphi))
        cos_dphi = math.cos(dphi)
        if abs(cos_dphi) < 1e-10:
            cos_dphi = 1e-10
        F[2, 0] = dr / (denom * cos_dphi)
        # ∂(new_phi)/∂(d) = -kappa² · dr / (denom² · cos(dphi))
        F[2, 3] = -kappa**2 * dr / (denom**2 * cos_dphi)
        # ∂(new_phi)/∂(phi) = 1 (已在对角线)
        # ∂(new_d)/∂(d) = 1 (已在对角线)
        # ∂(new_z0)/∂(tan_lambda) = ds
        F[4, 1] = ds
        # ∂(new_z0)/∂(z0) = 1 (已在对角线)

        # 过程噪声 (简化)
        Q = np.eye(5) * 1e-8

        new_state = TrackState(new_params, self.covariance.copy())
        return new_state, F, Q


# ============================================================
# 测量模型
# ============================================================
def measurement_model(state, layer_radius):
    """
    从状态向量预测测量值

    对于圆柱层，测量为 (r, z):
        h(x) = (layer_radius, z0 + tan(λ) · s)

    简化: 假设测量就是 (d, z0) 本身 (局部坐标系)

    Parameters
    ----------
    state : TrackState
    layer_radius : float

    Returns
    -------
    ndarray : 预测测量 [2]
    """
    return np.array([state.d, state.z0])


def measurement_jacobian(state, layer_radius):
    """
    测量雅可比矩阵 H = ∂h/∂x

    对于局部测量 (d, z0):
        H = [[0, 0, 0, 1, 0],    # ∂d/∂x
             [0, 0, 0, 0, 1]]    # ∂z0/∂x

    Parameters
    ----------
    state : TrackState
    layer_radius : float

    Returns
    -------
    ndarray, shape (2, 5)
    """
    H = np.zeros((2, 5))
    H[0, 3] = 1.0  # ∂d_meas/∂d
    H[1, 4] = 1.0  # ∂z_meas/∂z0
    return H


# ============================================================
# Kalman 滤波核心
# ============================================================
class KalmanFilter:
    """
    扩展 Kalman 滤波器

    实现前向滤波 + Joseph 形式协方差更新
    """

    def __init__(self, bfield_t=2.0):
        self.bfield = bfield_t
        self.chi2 = 0.0
        self.ndf = 0
        self.filter_states = []  # 滤波后的状态历史
        self.predicted_states = []  # 预测状态历史
        self.gains = []  # Kalman 增益历史
        self.chi2_per_hit = []  # 每击中的 χ² 贡献

    def reset(self):
        """重置滤波器状态"""
        self.chi2 = 0.0
        self.ndf = 0
        self.filter_states = []
        self.predicted_states = []
        self.gains = []
        self.chi2_per_hit = []

    def predict(self, state, current_radius, target_radius):
        """
        预测步骤

        x̃_{k|k-1} = f(x̂_{k-1})
        C̃_{k|k-1} = F_k · Ĉ_{k-1} · F_k^T + Q_k

        Parameters
        ----------
        state : TrackState
            当前滤波状态
        current_radius : float
        target_radius : float

        Returns
        -------
        TrackState : 预测状态
        """
        pred_state, F, Q = state.predict_to_layer(
            current_radius, target_radius, self.bfield)

        # 协方差传播
        pred_cov = F @ state.covariance @ F.T + Q

        # 对称化 (数值稳定)
        pred_cov = 0.5 * (pred_cov + pred_cov.T)

        pred_state.covariance = pred_cov
        self.predicted_states.append(pred_state.copy())

        return pred_state

    def update(self, predicted_state, measurement, measurement_cov,
               layer_radius):
        """
        更新步骤 (Joseph 形式)

        r_k = m_k - H_k · x̃_{k|k-1}
        G_k = H_k · C̃ · H_k^T + V_k
        K_k = C̃ · H_k^T · G_k^{-1}
        x̂_k = x̃ + K_k · r_k
        Ĉ_k = (I - K·H) · C̃ · (I - K·H)^T + K · V · K^T  (Joseph)

        Parameters
        ----------
        predicted_state : TrackState
        measurement : ndarray, shape (2,)
            [r_meas, z_meas] 或 [d_meas, z_meas]
        measurement_cov : ndarray, shape (2, 2)
        layer_radius : float

        Returns
        -------
        TrackState : 滤波后状态
        """
        # 测量雅可比
        H = measurement_jacobian(predicted_state, layer_radius)

        # 预测测量
        h_pred = measurement_model(predicted_state, layer_radius)

        # 残差 (创新)
        residual = measurement - h_pred

        # 创新协方差
        G = H @ predicted_state.covariance @ H.T + measurement_cov

        # 安全求逆 (2×2)
        try:
            G_inv = np.linalg.inv(G)
        except np.linalg.LinAlgError:
            G_inv = np.linalg.pinv(G)

        # Kalman 增益
        K = predicted_state.covariance @ H.T @ G_inv

        # 状态更新
        new_params = predicted_state.params + K @ residual

        # Joseph 形式协方差更新 (数值稳定)
        I_KH = np.eye(5) - K @ H
        new_cov = (I_KH @ predicted_state.covariance @ I_KH.T +
                   K @ measurement_cov @ K.T)

        # 对称化
        new_cov = 0.5 * (new_cov + new_cov.T)

        # χ² 递推
        chi2_contrib = float(residual.T @ G_inv @ residual)
        self.chi2 += max(chi2_contrib, 0.0)
        self.ndf += 2  # 每个击中 2 个测量
        self.chi2_per_hit.append(max(chi2_contrib, 0.0))

        # 构建滤波后状态
        filtered_state = TrackState(new_params, new_cov)
        self.filter_states.append(filtered_state.copy())
        self.gains.append(K.copy())

        return filtered_state

    def get_results(self):
        """获取滤波结果"""
        chi2_ndof = self.chi2 / max(self.ndf - 5, 1)
        return {
            'chi2': self.chi2,
            'ndf': self.ndf,
            'chi2_ndof': chi2_ndof,
            'chi2_per_hit': self.chi2_per_hit,
            'n_hits': len(self.filter_states),
            'final_state': self.filter_states[-1] if self.filter_states else None,
        }


# ============================================================
# 后向平滑 (Backward Smoother)
# ============================================================
class KalmanSmoother:
    """
    Raugh-Tung 后向平滑器

    基于 [064_backward_euler_fixed] 的向后积分思想:
    原算法从终值向后积分 ODE; 这里从最后一个击中向后平滑

    平滑公式:
        A_k = Ĉ_k · F_{k+1}^T · C̃_{k+1|k}^{-1}   (平滑增益)
        x̂^s_k = x̂_k + A_k · (x̂^s_{k+1} - x̃_{k+1|k})
        Ĉ^s_k = Ĉ_k + A_k · (Ĉ^s_{k+1} - C̃_{k+1|k}) · A_k^T
    """

    def __init__(self, kalman_filter):
        self.kf = kalman_filter

    def smooth(self, layer_radii):
        """
        执行后向平滑

        Parameters
        ----------
        layer_radii : list of float
            各层半径 [mm]

        Returns
        -------
        list of TrackState : 平滑后的状态
        """
        kf = self.kf
        n = len(kf.filter_states)

        if n == 0:
            return []

        # 初始化: 最后一个滤波状态 = 平滑状态
        smoothed = [None] * n
        smoothed[-1] = kf.filter_states[-1].copy()

        # 向后递推
        for k in range(n - 2, -1, -1):
            # 获取 k+1 步的预测状态和滤波状态
            pred_next = kf.predicted_states[k + 1]
            filt_curr = kf.filter_states[k]

            # 需要传播雅可比 F_{k+1}
            # 从 predicted_states 重新计算
            r_curr = layer_radii[k] if k < len(layer_radii) else 0
            r_next = layer_radii[k + 1] if k + 1 < len(layer_radii) else 0

            _, F, _ = filt_curr.predict_to_layer(r_curr, r_next, kf.bfield)

            # 预测协方差 C̃_{k+1|k}
            C_pred = pred_next.covariance

            # 平滑增益 A_k
            try:
                C_pred_inv = np.linalg.inv(C_pred)
            except np.linalg.LinAlgError:
                C_pred_inv = np.linalg.pinv(C_pred)

            A = filt_curr.covariance @ F.T @ C_pred_inv

            # 平滑状态
            dx = smoothed[k + 1].params - pred_next.params
            smoothed_params = filt_curr.params + A @ dx

            # 平滑协方差
            dC = smoothed[k + 1].covariance - C_pred
            smoothed_cov = filt_curr.covariance + A @ dC @ A.T

            # 对称化
            smoothed_cov = 0.5 * (smoothed_cov + smoothed_cov.T)

            smoothed[k] = TrackState(smoothed_params, smoothed_cov)

        return smoothed


# ============================================================
# 完整径迹拟合流程
# ============================================================
def fit_track(initial_state, hits, detector_layers, noise_model=None,
              bfield_t=2.0):
    """
    完整径迹拟合流程

    1. 初始化 Kalman 滤波器
    2. 按层顺序前向滤波
    3. 后向平滑
    4. 计算 χ² 和残差

    Parameters
    ----------
    initial_state : TrackState
        初始状态猜测
    hits : list of dict
        击中列表 (每层一个或零个)
    detector_layers : list
        DetectorLayer 列表
    noise_model : optional
    bfield_t : float

    Returns
    -------
    dict : 拟合结果
    """
    kf = KalmanFilter(bfield_t)
    kf.reset()

    # 前向滤波
    current_state = initial_state.copy()
    current_radius = detector_layers[0].radius_mm * 0.5  # 起始半径

    fitted_hits = []
    fitted_predictions = []
    fitted_residuals = []

    for layer_idx, layer in enumerate(detector_layers):
        if layer_idx >= len(hits) or hits[layer_idx] is None:
            continue

        hit = hits[layer_idx]

        # 预测到当前层
        pred_state = kf.predict(current_state, current_radius, layer.radius_mm)

        # 测量值和协方差
        meas = np.array([hit.get('r_meas', layer.radius_mm),
                         hit.get('z_meas', 0.0)])
        meas_cov = np.diag([layer.sigma_r_mm**2, layer.sigma_z_mm**2])

        # 更新
        current_state = kf.update(pred_state, meas, meas_cov, layer.radius_mm)

        current_radius = layer.radius_mm

        # 记录
        fitted_hits.append(hit)
        h_pred = measurement_model(pred_state, layer.radius_mm)
        fitted_predictions.append(h_pred.tolist())
        fitted_residuals.append((meas - h_pred).tolist())

    # 后向平滑
    active_radii = [layer.radius_mm for layer in detector_layers
                    if detector_layers.index(layer) < len(hits) and hits[detector_layers.index(layer)] is not None]
    smoother = KalmanSmoother(kf)
    smoothed_states = smoother.smooth(active_radii)

    # 计算平滑后的残差
    smoothed_residuals = []
    for i, state in enumerate(smoothed_states):
        if i < len(fitted_hits):
            h_smooth = measurement_model(state, active_radii[i] if i < len(active_radii) else 0)
            meas = np.array([fitted_hits[i].get('r_meas', 0),
                             fitted_hits[i].get('z_meas', 0)])
            smoothed_residuals.append((meas - h_smooth).tolist())

    # 结果汇总
    results = kf.get_results()
    results['smoothed_states'] = smoothed_states
    results['fitted_hits'] = fitted_hits
    results['predictions'] = fitted_predictions
    results['residuals'] = fitted_residuals
    results['smoothed_residuals'] = smoothed_residuals
    results['filter_states'] = kf.filter_states

    return results
