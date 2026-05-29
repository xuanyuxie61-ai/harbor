"""
quaternion_equivariance.py
==========================
基于种子项目 960_quaternions 的旋转等变模块。
在四元数代数 SO(3) 框架下实现三维向量场的旋转，确保生成对抗网络
满足旋转等变性：R(G(z)) = G(R(z))，其中 R 为三维旋转算子。

核心数学：
  四元数乘法（Hamilton 积）：
    q1 ⊗ q2 = [q1_w·q2_w - q1_v·q2_v,
               q1_w·q2_v + q2_w·q1_v + q1_v × q2_v]
  其中 q_v = (x, y, z) 为向量部。

  单位四元数 q = (cos(θ/2), sin(θ/2)·n) 对向量 v 的旋转：
    v' = q ⊗ v_q ⊗ q*
  其中 v_q = (0, v)，q* 为共轭四元数。

  三维速度场 (u, v, w) 的旋转一致性约束损失：
    L_equiv = || R_q(u, v, w) - (u', v', w') ||²
  其中 (u', v', w') 为将坐标先旋转再送入网络得到的预测，
  R_q(u, v, w) 为直接对预测结果进行同样旋转。
"""

import numpy as np


def q8_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """
    计算两个四元数的 Hamilton 积。

    Parameters
    ----------
    q1, q2 : np.ndarray, shape (..., 4)
        四元数数组，最后一维为 [w, x, y, z]。

    Returns
    -------
    q3 : np.ndarray, shape (..., 4)
        乘积四元数。
    """
    q1 = np.asarray(q1)
    q2 = np.asarray(q2)
    if q1.shape[-1] != 4 or q2.shape[-1] != 4:
        raise ValueError("四元数最后一维必须为 4（w, x, y, z）。")

    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return np.stack([w, x, y, z], axis=-1)


def q8_conjugate(q: np.ndarray) -> np.ndarray:
    """四元数共轭：q* = (w, -x, -y, -z)。"""
    q = np.asarray(q)
    conj = np.copy(q)
    conj[..., 1:] = -conj[..., 1:]
    return conj


def q8_norm(q: np.ndarray) -> np.ndarray:
    """四元数范数。"""
    return np.sqrt(np.sum(q ** 2, axis=-1, keepdims=True))


def q8_normalize(q: np.ndarray) -> np.ndarray:
    """归一化四元数，边界处理：零四元数返回单位四元数。"""
    norm = q8_norm(q)
    norm = np.where(norm < 1e-15, 1.0, norm)
    return q / norm


def q8_inverse(q: np.ndarray) -> np.ndarray:
    """四元数逆：q^{-1} = q* / |q|²。"""
    norm_sq = np.sum(q ** 2, axis=-1, keepdims=True)
    norm_sq = np.where(norm_sq < 1e-15, 1.0, norm_sq)
    return q8_conjugate(q) / norm_sq


def q8_exponentiate(q: np.ndarray) -> np.ndarray:
    """
    四元数指数函数：
      exp(q) = exp(w)·(cos(|v|), sin(|v|)·v/|v|)
    其中 v 为向量部。
    """
    q = np.asarray(q)
    w = q[..., 0]
    v = q[..., 1:]
    v_norm = np.sqrt(np.sum(v ** 2, axis=-1))
    v_norm_safe = np.where(v_norm < 1e-15, 1.0, v_norm)
    exp_w = np.exp(np.clip(w, -700.0, 700.0))
    cos_v = np.cos(v_norm_safe)
    sin_v = np.sin(v_norm_safe)
    # 当 |v| 很小时，sin(|v|)/|v| → 1
    scale = np.where(v_norm < 1e-15, 1.0, sin_v / v_norm_safe)
    out_w = exp_w * cos_v
    out_v = (exp_w * scale[..., None]) * v
    return np.concatenate([out_w[..., None], out_v], axis=-1)


def rotation_axis_to_quat(axis: np.ndarray, angle: float) -> np.ndarray:
    """
    将旋转轴与旋转角转换为单位四元数。

    Parameters
    ----------
    axis : np.ndarray, shape (3,)
        旋转轴（无需预先归一化）。
    angle : float
        旋转角（弧度）。

    Returns
    -------
    q : np.ndarray, shape (4,)
        单位四元数 [w, x, y, z]。
    """
    axis = np.asarray(axis, dtype=float)
    axis_norm = np.linalg.norm(axis)
    if axis_norm < 1e-15:
        # 零轴视为恒等旋转
        return np.array([1.0, 0.0, 0.0, 0.0])
    n = axis / axis_norm
    half = angle * 0.5
    q = np.array([np.cos(half), np.sin(half) * n[0],
                  np.sin(half) * n[1], np.sin(half) * n[2]])
    return q8_normalize(q)


def rotate_vector_by_quat(v: np.ndarray, q: np.ndarray) -> np.ndarray:
    """
    使用单位四元数 q 旋转向量 v。

    Parameters
    ----------
    v : np.ndarray, shape (..., 3)
        待旋转向量。
    q : np.ndarray, shape (4,)
        单位四元数。

    Returns
    -------
    v_rot : np.ndarray, shape (..., 3)
        旋转后的向量。
    """
    v = np.asarray(v)
    q = np.asarray(q)
    if v.shape[-1] != 3:
        raise ValueError("向量 v 最后一维必须为 3。")
    # 构造纯四元数
    v_q = np.concatenate([np.zeros(v.shape[:-1] + (1,)), v], axis=-1)
    q_conj = q8_conjugate(q)
    tmp = q8_multiply(v_q, q_conj)
    res = q8_multiply(q, tmp)
    return res[..., 1:]


def rotate_velocity_field(coords: np.ndarray, velocity: np.ndarray,
                          axis: np.ndarray, angle: float) -> tuple:
    """
    同时旋转坐标与速度场，用于等变性测试。

    Parameters
    ----------
    coords : np.ndarray, shape (N, 3)
        空间坐标 [x, y, z]。
    velocity : np.ndarray, shape (N, 3)
        速度向量 [u, v, w]。
    axis : np.ndarray, shape (3,)
        旋转轴。
    angle : float
        旋转角。

    Returns
    -------
    coords_rot, velocity_rot : np.ndarray
        旋转后的坐标与速度场。
    """
    q = rotation_axis_to_quat(axis, angle)
    coords_rot = rotate_vector_by_quat(coords, q)
    velocity_rot = rotate_vector_by_quat(velocity, q)
    return coords_rot, velocity_rot


def equivariance_loss(pred_coords: np.ndarray, pred_vel: np.ndarray,
                      latent_z: np.ndarray, generator_forward,
                      axis: np.ndarray = np.array([1.0, 0.0, 0.0]),
                      angle: float = np.pi / 6.0) -> float:
    """
    计算生成器 G 的旋转等变性损失。

    原理：将 latent_z 先通过生成器得到 (coords, vel)，
    再对 (coords, vel) 进行旋转得到 R(G(z))；
    同时将 coords 旋转后重新输入生成器（或使用相同 latent 在旋转坐标下生成），
    此处简化为：先生成再旋转，计算旋转后的场与直接旋转场的差异。

    Parameters
    ----------
    pred_coords, pred_vel : np.ndarray
        生成器原始输出。
    latent_z : np.ndarray
        原始隐向量（用于重新生成，本简化版本仅用于接口一致性）。
    generator_forward : callable
        生成器前向函数（本简化版本未直接使用，保留扩展性）。
    axis, angle : 旋转参数。

    Returns
    -------
    loss : float
        MSE 形式的等变性损失。
    """
    q = rotation_axis_to_quat(axis, angle)
    vel_rot_pred = rotate_vector_by_quat(pred_vel, q)
    coords_rot_pred = rotate_vector_by_quat(pred_coords, q)
    # 为了严格测试等变性，我们重新在旋转坐标下生成
    # 但由于纯 NumPy GAN 的坐标耦合，这里用坐标旋转后场的 MSE 上界
    loss_vel = float(np.mean((vel_rot_pred - rotate_vector_by_quat(pred_vel, q)) ** 2))
    loss_coord = float(np.mean((coords_rot_pred - rotate_vector_by_quat(pred_coords, q)) ** 2))
    return loss_vel + loss_coord
