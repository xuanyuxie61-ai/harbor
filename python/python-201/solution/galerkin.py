"""
galerkin.py — 随机 Galerkin 投影模块
=======================================
实现多项式混沌展开的随机 Galerkin 投影,
将随机偏微分方程转化为确定性耦合系统。

核心公式:
  随机 Galerkin 投影:
    对随机 PDE: L(u; ξ) = f(ξ)
    代入 PCE: u(x,ξ) ≈ Σ_k û_k(x) Ψ_k(ξ)
    投影: <L(Σ_k û_k Ψ_k; ξ), Ψ_j>_w = <f, Ψ_j>_w

  耦合系数:
    C_{ijk} = <Ψ_i Ψ_j, Ψ_k>

  非线性项处理 (Cahn-Hilliard c^3):
    (c^3)_j ≈ Σ_{i,k,l} ĉ_i ĉ_k ĉ_l C_{iklj}

映射种子项目:
  - 738_matrix_assemble_parfor: 矩阵装配 → Galerkin 系统装配
  - 986_r8ncf: 稀疏矩阵 → 稀疏 Galerkin 系统
  - 211_continuity_exact: 无散度约束 → Galerkin 相容性
"""

import numpy as np
from scipy import sparse, linalg
from typing import Tuple, Optional, Dict
from config import GlobalConfig, BasisConfig
from polynomial_basis import OrthogonalPolynomialBasis
from measure import gauss_quadrature, tensor_product_quadrature
from utils import (SparseMatrixCOO, assemble_stochastic_galerkin_matrix,
                   multi_index_set)


class StochasticGalerkinProjector:
    """
    随机 Galerkin 投影器。

    管理三重乘积张量和非线性项的投影。

    映射 738_matrix_assemble_parfor: 将并行矩阵装配
    推广到 Galerkin 耦合系统的块结构装配。

    映射 986_r8ncf: 利用稀疏 COO 格式
    高效存储大规模 Galerkin 系统。
    """

    def __init__(self, config: GlobalConfig,
                 basis: OrthogonalPolynomialBasis):
        self.config = config
        self.basis = basis
        self.P = basis.n_basis
        self.d = basis.n_dim

        # 求积规则
        n_quad = max(self.P + 5, 2 * self.P)
        self.quad_nodes, self.quad_weights = tensor_product_quadrature(
            config.measures, min(n_quad, 12))

        # 预计算三重乘积张量
        self.coupling_tensor = None
        self._compute_coupling_tensor()

        # 预计算二次乘积张量 (用于双线性项)
        self.quadratic_tensor = None
        self._compute_quadratic_tensor()

    def _compute_coupling_tensor(self):
        """计算三重乘积张量 C_{ijk}"""
        self.coupling_tensor = self.basis.compute_coupling_tensor(
            self.quad_nodes, self.quad_weights)

    def _compute_quadratic_tensor(self):
        """
        计算二次乘积张量 (用于 c^2 或 c * ∇c 项)。

        Q_{ij}^k = <Ψ_i Ψ_j, Ψ_k> = C_{ijk}

        实际上就是三重乘积张量, 但按不同方式索引。
        """
        self.quadratic_tensor = self.coupling_tensor.copy()

    def project_linear(self, coeffs: np.ndarray,
                       physical_operator) -> np.ndarray:
        """
        对线性项进行 Galerkin 投影。

        对 L(u) = a(ξ) * K * u:
          (L_Galerkin)_{j} = Σ_k a_k * Σ_i C_{jik} * K * û_i

        参数:
            coeffs: 随机系数 a 的 PCE 系数, shape (P,)
            physical_operator: 物理空间算子 (矩阵或函数)

        返回:
            projected: 投影后的耦合系统矩阵/向量
        """
        P = self.P

        # 对每个 j, 计算 Σ_k a_k Σ_i C_{jik} K û_i
        # 简化: 返回 Galerkin 耦合矩阵
        result = np.zeros((P, P))

        for k in range(P):
            if abs(coeffs[k]) < 1e-16:
                continue
            for j in range(P):
                for i in range(P):
                    result[j, i] += (coeffs[k] *
                                     self.coupling_tensor[j, i, k])

        return result

    def project_nonlinear_cubic(self, c_hat: np.ndarray
                                ) -> np.ndarray:
        """
        对非线性三次项 c^3 进行 Galerkin 投影。

        (c^3)_j ≈ Σ_{i,k,l} ĉ_i ĉ_k ĉ_l <Ψ_i Ψ_k Ψ_l, Ψ_j>

        使用四重乘积张量的简化近似:
          <Ψ_i Ψ_k Ψ_l, Ψ_j> ≈ Σ_m C_{ikm} C_{mlj}

        参数:
            c_hat: PCE 系数, shape (P,)

        返回:
            cubic_proj: shape (P,), 三次项的 Galerkin 投影
        """
        P = self.P
        C = self.coupling_tensor

        # 计算 c^2 的 PCE 系数 (Galerkin 近似)
        # (c^2)_m ≈ Σ_{i,k} ĉ_i ĉ_k C_{ikm}
        c2_hat = np.zeros(P)
        for m in range(P):
            for i in range(P):
                if abs(c_hat[i]) < 1e-16:
                    continue
                for k in range(P):
                    c2_hat[m] += c_hat[i] * c_hat[k] * C[i, k, m]

        # 计算 c^3 = c * c^2 的 Galerkin 投影
        # (c^3)_j ≈ Σ_{m,l} ĉ_l (c^2)_m C_{lmj}
        cubic_proj = np.zeros(P)
        for j in range(P):
            for l in range(P):
                if abs(c_hat[l]) < 1e-16:
                    continue
                for m in range(P):
                    cubic_proj[j] += (c_hat[l] * c2_hat[m] *
                                      C[l, m, j])

        return cubic_proj

    def project_nonlinear_quadratic(self, c_hat: np.ndarray
                                    ) -> np.ndarray:
        """
        对非线性二次项 c^2 进行 Galerkin 投影。

        (c^2)_j = Σ_{i,k} ĉ_i ĉ_k C_{ikj}

        参数:
            c_hat: PCE 系数, shape (P,)

        返回:
            quad_proj: shape (P,)
        """
        C = self.coupling_tensor
        P = self.P
        result = np.zeros(P)

        for j in range(P):
            for i in range(P):
                if abs(c_hat[i]) < 1e-16:
                    continue
                for k in range(P):
                    result[j] += c_hat[i] * c_hat[k] * C[i, k, j]

        return result

    def compute_diffusion_coupling(self, kappa_hat: np.ndarray
                                   ) -> sparse.csr_matrix:
        """
        计算随机扩散项的 Galerkin 耦合矩阵。

        对 -∇·(κ(ξ) ∇c):
          Galerkin 矩阵 A_{ji} = Σ_k κ_k C_{jik} * K^{phys}

        其中 K^{phys} 是物理空间的 Laplacian 矩阵。

        参数:
            kappa_hat: κ 的 PCE 系数, shape (P,)

        返回:
            A: shape (P, P), Galerkin 耦合系数矩阵
        """
        A = np.zeros((self.P, self.P))
        for k in range(self.P):
            if abs(kappa_hat[k]) < 1e-16:
                continue
            for j in range(self.P):
                for i in range(self.P):
                    A[j, i] += kappa_hat[k] * self.coupling_tensor[j, i, k]
        return sparse.csr_matrix(A)

    def verify_galerkin_symmetry(self) -> float:
        """
        验证 Galerkin 投影的对称性。

        对自伴随问题, Galerkin 矩阵应为对称的:
          A_{ji} = A_{ij}

        返回:
            ||A - A^T||_F / ||A||_F
        """
        C = self.coupling_tensor
        # 检查 C_{ijk} = C_{jik} (对第一个两指标的对称性)
        asymmetry = 0.0
        norm = 0.0
        for i in range(self.P):
            for j in range(self.P):
                for k in range(self.P):
                    asymmetry += (C[i, j, k] - C[j, i, k]) ** 2
                    norm += C[i, j, k] ** 2
        return np.sqrt(asymmetry) / max(np.sqrt(norm), 1e-15)

    def compute_sensitivity_coupling(self) -> np.ndarray:
        """
        计算灵敏度耦合矩阵。

        对每个随机变量 ξ_d:
          S_{j}^{(d)} = ∂/∂ξ_d (Σ_k û_k Ψ_k) |_{ξ=μ}
                      = Σ_k û_k Ψ_k'(μ_d) Π_{l≠d} Ψ_{k_l}(μ_l)

        返回:
            S: shape (d, P), 各维度的灵敏度系数
        """
        # 在均值点求值
        mu = np.array([m.mean for m in self.config.measures])
        mu_2d = mu.reshape(1, -1)

        d = self.d
        P = self.P
        S = np.zeros((d, P))

        for p_idx in range(P):
            multi_idx = self.basis.indices[p_idx]
            for dim in range(d):
                deg = multi_idx[dim]
                if deg == 0:
                    S[dim, p_idx] = 0.0
                    continue

                # 计算 Ψ'_{deg}(μ_d) 使用差分近似
                eps = 1e-7
                mu_plus = mu_2d.copy()
                mu_plus[0, dim] += eps
                mu_minus = mu_2d.copy()
                mu_minus[0, dim] -= eps

                psi_plus = self.basis.evaluate_1d(
                    dim, deg, mu_plus[0, dim:dim + 1])
                psi_minus = self.basis.evaluate_1d(
                    dim, deg, mu_minus[0, dim:dim + 1])
                dpsi = (psi_plus - psi_minus) / (2 * eps)

                # 乘以其他维度的值
                prod = float(dpsi[0])
                for other_dim in range(d):
                    if other_dim == dim:
                        continue
                    other_deg = multi_idx[other_dim]
                    val = self.basis.evaluate_1d(
                        other_dim, other_deg,
                        np.array([mu[other_dim]]))
                    prod *= float(val[0])

                S[dim, p_idx] = prod

        return S


class GalerkinSystemAssembler:
    """
    Galerkin 系统装配器。

    将随机 Galerkin 投影的结果组装为可求解的线性/非线性系统。

    映射 738_matrix_assemble_parfor: 并行装配块矩阵。
    映射 986_r8ncf: 稀疏矩阵管理。
    """

    def __init__(self, projector: StochasticGalerkinProjector,
                 n_physical_dof: int):
        self.projector = projector
        self.n_phys = n_physical_dof
        self.P = projector.P

    def assemble_mass_matrix(self) -> sparse.csr_matrix:
        """
        装配随机质量矩阵。

        M_{SG} = M^{phys} ⊗ I^{stoch}

        因为 <Ψ_i, Ψ_j> = δ_{ij}。
        """
        # 单位矩阵 (正交基)
        I_stoch = sparse.eye(self.P, format='csr')
        # 物理质量矩阵 (简化为单位阵)
        M_phys = sparse.eye(self.n_phys, format='csr')
        return sparse.kron(I_stoch, M_phys, format='csr')

    def assemble_stiffness_matrix(
        self, kappa_hat: np.ndarray
    ) -> sparse.csr_matrix:
        """
        装配随机刚度矩阵。

        K_{SG} = K_coupling(κ̂) ⊗ K^{phys}

        参数:
            kappa_hat: κ 的 PCE 系数
        """
        K_coupling = self.projector.compute_diffusion_coupling(
            kappa_hat)
        # 物理刚度矩阵 (简化: 一维 Laplacian 的特征值)
        K_phys_vals = np.array(
            [4 * np.sin(np.pi * k / (2 * self.n_phys)) ** 2
             for k in range(self.n_phys)])
        K_phys = sparse.diags(K_phys_vals, format='csr')

        return sparse.kron(K_coupling, K_phys, format='csr')

    def summary(self) -> str:
        """返回系统摘要"""
        total_dof = self.P * self.n_phys
        return (f"Galerkin 系统: P={self.P}, "
                f"N_phys={self.n_phys}, "
                f"Total DOF={total_dof}")
