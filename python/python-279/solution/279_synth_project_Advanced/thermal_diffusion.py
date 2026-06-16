"""
thermal_diffusion.py
====================
热传导方程稀疏矩阵求解器。

科学背景 (源自 405_fem2d_heat_sparse):
  材料中的热传导由抛物型 PDE 描述:

    rho * c_p * dT/dt = div(kappa * grad(T)) + Q

  其中:
    rho: 密度 [kg/m^3]
    c_p: 比热容 [J/(kg*K)]
    kappa: 热导率 [W/(m*K)]
    Q: 体积热源 [W/m^3] (焦耳热, 反应热等)

  热扩散率:
    alpha = kappa / (rho * c_p)  [m^2/s]

  边界条件:
    Dirichlet: T = T_bc (等温边界)
    Neumann: -kappa * dT/dn = q (热流边界)
    Robin: -kappa * dT/dn = h*(T - T_inf) (对流散热)

  焦耳热源 (电池内部):
    Q_joule = sigma * |E|^2 = sigma * |grad(phi)|^2

  反应热 (电化学反应):
    Q_rxn = j * eta  (j: 电流密度, eta: 过电位)

离散化:
  2D FEM:
    M * dT/dt + K * T = F
  其中:
    M_ij = integral(rho*c_p * N_i * N_j dA)  (质量矩阵)
    K_ij = integral(kappa * grad(N_i) . grad(N_j) dA)  (刚度矩阵)
    F_i = integral(Q * N_i dA) + BC 项

  Backward Euler:
    (M + dt*K) * T^{n+1} = M * T^n + dt * F^{n+1}

 Fourier 热传导方程 (瞬态):
    dT/dt = alpha * laplacian(T) + Q/(rho*c_p)

  稳态 (Laplace):
    laplacian(T) = -Q/kappa

  Biot 数:
    Bi = h*L/kappa  (h: 对流换热系数)
    Bi << 1: 内部热阻控制 (集总热容法)
    Bi >> 1: 表面热阻控制

 Stefan-Boltzmann 辐射:
    q_rad = eps_s * sigma_SB * (T^4 - T_env^4)
    sigma_SB = 5.670374419e-8 W/(m^2*K^4)
"""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from material_constants import (
    LLZO_THERMAL_CONDUCTIVITY, LLZO_DENSITY, LLZO_SPECIFIC_HEAT,
    LLZO_THERMAL_DIFFUSIVITY, SMALL_NUMBER,
)


# Stefan-Boltzmann 常数
SIGMA_SB = 5.670374419e-8


class ThermalDiffusion2D:
    """
    2D 热传导方程稀疏矩阵求解器 (源自 405_fem2d_heat_sparse)。

    使用三角形线性元 (P1) 离散。
    """
    def __init__(self, node_xy, triangle_node, kappa=None, rho=None, cp=None):
        """
        参数:
            node_xy: [2, n_node] 节点坐标
            triangle_node: [3, n_tri] 三角形节点 (1-indexed)
            kappa: 热导率 [W/(m*K)], 默认 LLZO
            rho: 密度 [kg/m^3]
            cp: 比热容 [J/(kg*K)]
        """
        self.node_xy = np.asarray(node_xy, dtype=np.float64)
        self.triangle_node = np.asarray(triangle_node, dtype=np.int64)
        if self.triangle_node.shape[0] != 3:
            self.triangle_node = self.triangle_node.T

        self.n_node = self.node_xy.shape[1]
        self.n_tri = self.triangle_node.shape[1]

        self.kappa = kappa if kappa is not None else LLZO_THERMAL_CONDUCTIVITY
        self.rho = rho if rho is not None else LLZO_DENSITY
        self.cp = cp if cp is not None else LLZO_SPECIFIC_HEAT
        self.alpha = self.kappa / (self.rho * self.cp)

        # 组装矩阵
        self.M, self.K = self._assemble_matrices()

        # 温度场
        self.T = np.ones(self.n_node) * 300.0  # 初始 300K

    def _triangle_area(self, n1, n2, n3):
        """三角形面积 (叉积)"""
        v1 = self.node_xy[:, n1 - 1]
        v2 = self.node_xy[:, n2 - 1]
        v3 = self.node_xy[:, n3 - 1]
        cross = (v2[0]-v1[0])*(v3[1]-v1[1]) - (v2[1]-v1[1])*(v3[0]-v1[0])
        return 0.5 * abs(cross)

    def _assemble_matrices(self):
        """
        组装全局质量矩阵 M 和刚度矩阵 K。

        对 P1 三角形元:
          刚度矩阵 (单元):
            K^e_{ij} = kappa * (grad N_i . grad N_j) * Area
          质量矩阵 (单元, 集中化):
            M^e_{ii} = rho * cp * Area / 3

          grad N_i = (1/(2*Area)) * (y_j - y_k, x_k - x_j)  (i, j, k 逆时针)
        """
        K_rows, K_cols, K_vals = [], [], []
        M_rows, M_cols, M_vals = [], [], []

        for e in range(self.n_tri):
            n1, n2, n3 = self.triangle_node[:, e]
            v1 = self.node_xy[:, n1 - 1]
            v2 = self.node_xy[:, n2 - 1]
            v3 = self.node_xy[:, n3 - 1]

            area = 0.5 * abs((v2[0]-v1[0])*(v3[1]-v1[1]) - (v2[1]-v1[1])*(v3[0]-v1[0]))
            if area < SMALL_NUMBER:
                continue

            # 形函数梯度
            # grad N1 = ((y2-y3), (x3-x2)) / (2*area), 等等
            b1 = v2[1] - v3[1]
            b2 = v3[1] - v1[1]
            b3 = v1[1] - v2[1]
            c1 = v3[0] - v2[0]
            c2 = v1[0] - v3[0]
            c3 = v2[0] - v1[0]
            inv2A = 1.0 / (2.0 * area)

            # 单元刚度矩阵 (3x3)
            b_vec = np.array([b1, b2, b3]) * inv2A
            c_vec = np.array([c1, c2, c3]) * inv2A
            K_local = self.kappa * area * (
                np.outer(b_vec, b_vec) + np.outer(c_vec, c_vec)
            )

            # 单元质量矩阵 (一致质量, 3x3)
            M_local = self.rho * self.cp * area / 12.0 * (
                2.0 * np.eye(3) + np.ones((3, 3))
            )

            # 组装到全局
            nodes = [n1 - 1, n2 - 1, n3 - 1]
            for i_loc in range(3):
                for j_loc in range(3):
                    I = nodes[i_loc]
                    J = nodes[j_loc]
                    K_rows.append(I)
                    K_cols.append(J)
                    K_vals.append(K_local[i_loc, j_loc])
                    M_rows.append(I)
                    M_cols.append(J)
                    M_vals.append(M_local[i_loc, j_loc])

        K = sparse.csr_matrix((K_vals, (K_rows, K_cols)), shape=(self.n_node, self.n_node))
        M = sparse.csr_matrix((M_vals, (M_rows, M_cols)), shape=(self.n_node, self.n_node))
        return M, K

    def set_boundary_dirichlet(self, node_list, T_bc):
        """
        施加 Dirichlet 边界条件 (大数法)。

        方法: 在 K 和 M 中对角元放大 1e30, 右端 = 1e30 * T_bc
        """
        penalty = 1e30
        self.bc_nodes = np.asarray(node_list, dtype=np.int64)
        self.bc_values = np.ones(len(node_list)) * T_bc
        self.penalty = penalty

    def set_boundary_neumann(self, edge_flux_dict):
        """
        施加 Neumann 边界: -kappa * dT/dn = q [W/m^2]

        edge_flux_dict: {node_idx: flux_value}
        """
        self.neumann_nodes = list(edge_flux_dict.keys())
        self.neumann_flux = list(edge_flux_dict.values())

    def _apply_boundary_conditions(self, K_mod, F_mod):
        """应用边界条件到系统"""
        if hasattr(self, 'bc_nodes'):
            for i, node in enumerate(self.bc_nodes):
                K_mod[node, :] = 0
                K_mod[:, node] = 0
                K_mod[node, node] = self.penalty
                F_mod[node] = self.penalty * self.bc_values[i]
        if hasattr(self, 'neumann_nodes'):
            for node, flux in zip(self.neumann_nodes, self.neumann_flux):
                F_mod[node] += flux
        return K_mod, F_mod

    def solve_steady_state(self, Q_source=None):
        """
        稳态求解: K * T = F

        Q_source: [n_node] 热源项 [W/m^3]
        F_i = integral(Q * N_i dA) 通过 M / (rho*cp) @ Q 近似
        """
        K_mod = self.K.copy().tolil()
        F = np.zeros(self.n_node)
        if Q_source is not None:
            # 热源积分: F_i = integral(Q * N_i dA)
            # 使用质量矩阵/ (rho*cp) 近似
            M_geom = self.M / (self.rho * self.cp)  # 几何质量矩阵
            F = M_geom.dot(Q_source)

        K_mod, F = self._apply_boundary_conditions(K_mod, F)
        self.T = spsolve(K_mod.tocsr(), F)
        return self.T

    def step_backward_euler(self, dt, Q_source=None):
        """
        Backward Euler 时间步进:

          (M + dt*K) * T^{n+1} = M * T^n + dt * F^{n+1}
        """
        A = self.M + dt * self.K
        F = self.M.dot(self.T)
        if Q_source is not None:
            M_geom = self.M / (self.rho * self.cp)
            F += dt * M_geom.dot(Q_source)

        A_mod = A.copy().tolil()
        A_mod, F = self._apply_boundary_conditions(A_mod, F)
        self.T = spsolve(A_mod.tocsr(), F)
        return self.T

    def joule_heating_source(self, phi_field, sigma_e):
        """
        计算焦耳热源:
          Q = sigma * |grad(phi)|^2

        简化: 假设 phi_field 为节点电势
        """
        Q = np.zeros(self.n_node)
        for e in range(self.n_tri):
            n1, n2, n3 = self.triangle_node[:, e]
            v1 = self.node_xy[:, n1 - 1]
            v2 = self.node_xy[:, n2 - 1]
            v3 = self.node_xy[:, n3 - 1]
            area = 0.5 * abs((v2[0]-v1[0])*(v3[1]-v1[1]) - (v2[1]-v1[1])*(v3[0]-v1[0]))
            if area < SMALL_NUMBER:
                continue
            b1, b2, b3 = v2[1]-v3[1], v3[1]-v1[1], v1[1]-v2[1]
            c1, c2, c3 = v3[0]-v2[0], v1[0]-v3[0], v2[0]-v1[0]
            inv2A = 1.0 / (2.0 * area)
            phi_e = np.array([phi_field[n1-1], phi_field[n2-1], phi_field[n3-1]])
            grad_phi_x = (b1*phi_e[0] + b2*phi_e[1] + b3*phi_e[2]) * inv2A
            grad_phi_y = (c1*phi_e[0] + c2*phi_e[1] + c3*phi_e[2]) * inv2A
            Q_e = sigma_e * (grad_phi_x**2 + grad_phi_y**2)
            Q[n1-1] += Q_e * area / 3
            Q[n2-1] += Q_e * area / 3
            Q[n3-1] += Q_e * area / 3
        return Q

    def total_heat_flux(self):
        """
        总热流量 (通过边界):
          Q_total = integral(-kappa * dT/dn ds)
        简化: 由能量守恒 Q_total = integral(Q_source dV)
        """
        return np.sum(self.kappa * self.K.dot(self.T))

    def biot_number(self, h_conv, L_char):
        """
        Biot 数:
          Bi = h * L / kappa
        """
        return h_conv * L_char / max(self.kappa, SMALL_NUMBER)

    def thermal_diffusion_time(self, L_char):
        """
        热扩散特征时间:
          tau = L^2 / alpha
        """
        return L_char**2 / max(self.alpha, SMALL_NUMBER)


def thermal_conductivity_mixture(kappa_1, kappa_2, vol_frac_1, model='maxwell'):
    """
    复合材料有效热导率模型。

    Maxwell-Garnett:
      kappa_eff = kappa_2 * (kappa_1 + 2*kappa_2 + 2*f_1*(kappa_1-kappa_2))
                       / (kappa_1 + 2*kappa_2 - f_1*(kappa_1-kappa_2))

    串联 (Reuss):
      1/kappa_eff = f_1/kappa_1 + f_2/kappa_2

    并联 (Voigt):
      kappa_eff = f_1*kappa_1 + f_2*kappa_2

    Bruggeman 对称:
      f_1*(kappa_1-kappa_eff)/(kappa_1+2*kappa_eff)
      + f_2*(kappa_2-kappa_eff)/(kappa_2+2*kappa_eff) = 0
    """
    f1 = vol_frac_1
    f2 = 1.0 - f1
    if model == 'voigt':
        return f1 * kappa_1 + f2 * kappa_2
    elif model == 'reuss':
        return 1.0 / (f1/max(kappa_1, SMALL_NUMBER) + f2/max(kappa_2, SMALL_NUMBER))
    elif model == 'maxwell':
        num = kappa_1 + 2*kappa_2 + 2*f1*(kappa_1 - kappa_2)
        den = kappa_1 + 2*kappa_2 - f1*(kappa_1 - kappa_2)
        return kappa_2 * num / max(den, SMALL_NUMBER)
    else:
        raise ValueError(f"Unknown model: {model}")
