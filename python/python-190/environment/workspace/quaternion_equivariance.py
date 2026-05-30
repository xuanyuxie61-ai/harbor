
import numpy as np


def q8_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
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
    q = np.asarray(q)
    conj = np.copy(q)
    conj[..., 1:] = -conj[..., 1:]
    return conj


def q8_norm(q: np.ndarray) -> np.ndarray:
    return np.sqrt(np.sum(q ** 2, axis=-1, keepdims=True))


def q8_normalize(q: np.ndarray) -> np.ndarray:
    norm = q8_norm(q)
    norm = np.where(norm < 1e-15, 1.0, norm)
    return q / norm


def q8_inverse(q: np.ndarray) -> np.ndarray:
    norm_sq = np.sum(q ** 2, axis=-1, keepdims=True)
    norm_sq = np.where(norm_sq < 1e-15, 1.0, norm_sq)
    return q8_conjugate(q) / norm_sq


def q8_exponentiate(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q)
    w = q[..., 0]
    v = q[..., 1:]
    v_norm = np.sqrt(np.sum(v ** 2, axis=-1))
    v_norm_safe = np.where(v_norm < 1e-15, 1.0, v_norm)
    exp_w = np.exp(np.clip(w, -700.0, 700.0))
    cos_v = np.cos(v_norm_safe)
    sin_v = np.sin(v_norm_safe)

    scale = np.where(v_norm < 1e-15, 1.0, sin_v / v_norm_safe)
    out_w = exp_w * cos_v
    out_v = (exp_w * scale[..., None]) * v
    return np.concatenate([out_w[..., None], out_v], axis=-1)


def rotation_axis_to_quat(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=float)
    axis_norm = np.linalg.norm(axis)
    if axis_norm < 1e-15:

        return np.array([1.0, 0.0, 0.0, 0.0])
    n = axis / axis_norm
    half = angle * 0.5
    q = np.array([np.cos(half), np.sin(half) * n[0],
                  np.sin(half) * n[1], np.sin(half) * n[2]])
    return q8_normalize(q)


def rotate_vector_by_quat(v: np.ndarray, q: np.ndarray) -> np.ndarray:
    v = np.asarray(v)
    q = np.asarray(q)
    if v.shape[-1] != 3:
        raise ValueError("向量 v 最后一维必须为 3。")

    v_q = np.concatenate([np.zeros(v.shape[:-1] + (1,)), v], axis=-1)
    q_conj = q8_conjugate(q)
    tmp = q8_multiply(v_q, q_conj)
    res = q8_multiply(q, tmp)
    return res[..., 1:]


def rotate_velocity_field(coords: np.ndarray, velocity: np.ndarray,
                          axis: np.ndarray, angle: float) -> tuple:
    q = rotation_axis_to_quat(axis, angle)
    coords_rot = rotate_vector_by_quat(coords, q)
    velocity_rot = rotate_vector_by_quat(velocity, q)
    return coords_rot, velocity_rot


def equivariance_loss(pred_coords: np.ndarray, pred_vel: np.ndarray,
                      latent_z: np.ndarray, generator_forward,
                      axis: np.ndarray = np.array([1.0, 0.0, 0.0]),
                      angle: float = np.pi / 6.0) -> float:
    q = rotation_axis_to_quat(axis, angle)
    vel_rot_pred = rotate_vector_by_quat(pred_vel, q)
    coords_rot_pred = rotate_vector_by_quat(pred_coords, q)


    loss_vel = float(np.mean((vel_rot_pred - rotate_vector_by_quat(pred_vel, q)) ** 2))
    loss_coord = float(np.mean((coords_rot_pred - rotate_vector_by_quat(pred_coords, q)) ** 2))
    return loss_vel + loss_coord
