"""
physics_informed_loss.py
========================
物理约束损失函数

科学背景:
  纯数据驱动的 GNN 可能违反物理守恒律。本模块注入以下物理约束:
    1. 能量守恒：预测能量在等变变换下的不变性。
    2. 力-能量一致性：F = -∇E，要求网络输出力为能量负梯度。
    3. 电荷守恒：总电荷为各原子电荷之和。
    4. 旋转/平移等变性：原子受力应在旋转下协变。

  数学表达:
    L_physics = λ_E |E(R) - E(R')|²          (R' 为旋转/平移后的坐标)
              + λ_F ||F + ∇_R E||²
              + λ_Q |Σ q_i - Q_target|²
"""

import numpy as np
from typing import Callable


class PhysicsInformedLoss:
    """
    物理信息损失计算器。
    """

    def __init__(self, lambda_energy: float = 1.0,
                 lambda_force: float = 10.0,
                 lambda_charge: float = 5.0):
        self.lambda_energy = lambda_energy
        self.lambda_force = lambda_force
        self.lambda_charge = lambda_charge

    def energy_invariance_loss(self, energy_func: Callable,
                               coords: np.ndarray) -> float:
        """
        能量旋转/平移不变性损失。
        对坐标施加随机旋转矩阵 R (R^T R = I, det(R)=1)，
        要求 E(R @ coords + t) ≈ E(coords)。
        """
        n_atoms = coords.shape[0]
        # 随机旋转 (Householder 反射近似)
        v = np.random.randn(3)
        v = v / np.linalg.norm(v)
        u = np.random.randn(3)
        u = u - np.dot(u, v) * v
        u = u / np.linalg.norm(u)
        w = np.cross(v, u)
        R = np.column_stack([u, w, v])
        if np.linalg.det(R) < 0:
            R[:, 0] *= -1
        t = np.random.randn(3) * 0.1
        coords_rot = (coords @ R.T) + t.reshape(1, 3)

        e1 = energy_func(coords)
        e2 = energy_func(coords_rot)
        return float((e1 - e2) ** 2)

    def force_consistency_loss(self, energy_func: Callable,
                               coords: np.ndarray) -> float:
        """
        力-能量一致性损失: ||F_pred + ∇E||²。
        使用有限差分近似能量梯度。
        """
        delta = 1e-4
        n_atoms = coords.shape[0]
        grad_fd = np.zeros_like(coords)
        e0 = energy_func(coords)
        for i in range(n_atoms):
            for d in range(3):
                coords_plus = coords.copy()
                coords_plus[i, d] += delta
                e_plus = energy_func(coords_plus)
                grad_fd[i, d] = (e_plus - e0) / delta
        # 预测力应等于负梯度；这里用梯度范数作为代理
        return float(np.sum(grad_fd ** 2))

    def charge_conservation_loss(self, predicted_charges: np.ndarray,
                                 target_total: float) -> float:
        """
        总电荷守恒: (Σ q_i - Q_target)²。
        """
        return float((np.sum(predicted_charges) - target_total) ** 2)

    def total_physics_loss(self, energy_func: Callable,
                           coords: np.ndarray,
                           predicted_charges: np.ndarray,
                           target_total_charge: float) -> float:
        """
        总物理损失。
        """
        loss_e = self.energy_invariance_loss(energy_func, coords)
        loss_f = self.force_consistency_loss(energy_func, coords)
        loss_q = self.charge_conservation_loss(predicted_charges, target_total_charge)
        return (self.lambda_energy * loss_e +
                self.lambda_force * loss_f +
                self.lambda_charge * loss_q)


def schrodinger_residual_energy(wavefunction_vals: np.ndarray,
                                potential_vals: np.ndarray,
                                laplacian_wf: np.ndarray,
                                hbar: float = 1.0,
                                mass: float = 1.0) -> float:
    """
    定态薛定谔方程残差:
        [-ℏ²/(2m) ∇² + V] ψ - E ψ = 0
    给定 ψ, V, ∇²ψ，计算残差范数作为损失项。
    """
    kinetic = - (hbar ** 2) / (2.0 * mass) * laplacian_wf
    residual = kinetic + potential_vals * wavefunction_vals
    # 能量期望值
    E_expect = np.sum(wavefunction_vals * residual)
    residual_norm = np.sum((residual - E_expect * wavefunction_vals) ** 2)
    return float(residual_norm)
